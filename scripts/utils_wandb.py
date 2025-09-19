"""
Typed, cache-aware client for Weights & Biases (W&B) run data and helpers.

Features:
- List runs and fetch per-run config, summary, and full history DataFrame
- Local caching to reduce API calls (Parquet preferred, CSV fallback)
- Robust JSON conversion for W&B/numpy objects
- Convenience utilities to collect histories and map run names to groups

Environment:
- Ensure you're logged in (wandb.login()) or set WANDB_API_KEY
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings
import pandas as pd

try:
    import wandb
    from wandb.apis.public import Api, Run  # type: ignore
except Exception as exc:  # pragma: no cover - import-time hint
    raise RuntimeError(
        "wandb is required to use scripts.wandb_client. Install with `pip install wandb`."
    ) from exc


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _hash_parts(parts: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"|")
    return hasher.hexdigest()[:16]


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary objects to a JSON-serializable structure.

    Handles W&B summary/config, numpy scalars/arrays, and mappings/sequences.
    Falls back to repr() for unknown objects.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Numpy types (scalars/arrays)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
    except Exception:
        pass

    # W&B types: try to_json / to_dict if present
    for attr in ("to_json", "to_dict", "_json_dict"):
        try:
            obj = getattr(value, attr)
        except Exception:
            continue
        try:
            data = obj() if callable(obj) else obj
            if isinstance(data, str):
                return json.loads(data)
            return _to_jsonable(data)
        except Exception:
            pass

    # Mappings
    try:
        from collections.abc import Mapping

        if isinstance(value, Mapping):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
    except Exception:
        pass

    # Sequences (but not strings/bytes)
    from collections.abc import Sequence as Seq

    if isinstance(value, Seq) and not isinstance(value, (bytes, bytearray, str)):
        return [_to_jsonable(v) for v in value]

    return repr(value)


@dataclass(frozen=True)
class RunKey:
    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


@dataclasses.dataclass
class RunData:
    """Container for core data used by plots.

    - config: static run configuration (dict-like)
    - summary: last reported summary metrics
    - history_df: time-series metrics (pandas.DataFrame)
    - run: the underlying wandb Run object
    """

    key: RunKey
    config: Dict[str, Any]
    summary: Dict[str, Any]
    history_df: "Any"  # pandas.DataFrame, import avoided in types
    run: Run


class WandbDataClient:
    """Thin client for retrieving W&B run data with optional local caching."""

    def __init__(
        self,
        entity: str,
        project: str,
        *,
        cache_dir: Optional[os.PathLike[str] | str] = None,
        cache_ttl_seconds: int = 6 * 60 * 60,
        use_cache: bool = True,
        api: Optional[Api] = None,
    ) -> None:
        self.entity = entity
        self.project = project
        self.api = api or wandb.Api()
        self.use_cache = use_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.logger = logging.getLogger(__name__)

        # Default cache directory: XDG_CACHE_HOME/evo-ued/wandb or ~/.cache/evo-ued/wandb
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            base = Path(xdg) if xdg else Path.home() / ".cache"
            self.cache_dir = base / "evo-ued" / "wandb"
        _ensure_dir(self.cache_dir)

    # ----------------------------- Listing & lookup ----------------------------
    def list_runs(
        self,
        *,
        group: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        order: str = "-created_at",
        per_page: int = 200,
    ) -> List[Run]:
        """List runs for the configured entity/project."""
        query_filters: Dict[str, Any] = filters.copy() if filters else {}
        if group is not None:
            query_filters.setdefault("group", group)
        runs = self.api.runs(
            f"{self.entity}/{self.project}",
            filters=query_filters,
            order=order,
            per_page=per_page,
        )
        return list(runs)

    def get_run(self, run_id_or_path: str) -> Run:
        """Return a W&B Run by id or full path."""
        if "/" in run_id_or_path:
            return self.api.run(run_id_or_path)
        return self.api.run(f"{self.entity}/{self.project}/{run_id_or_path}")

    # --------------------------------- Fetching --------------------------------
    def fetch_run_data(
        self,
        run_id_or_path: str,
        *,
        keys: Optional[Sequence[str]] = None,
        samples: Optional[int] = None,
        refresh: bool = False,
        max_rows: Optional[int] = None,
    ) -> RunData:
        """Fetch config, summary, and history used by W&B plots for a run."""
        run = self.get_run(run_id_or_path)
        key = RunKey(run.entity, run.project, run.id)

        config = self._get_or_cache_json(key, "config.json", run.config, refresh)
        summary = self._get_or_cache_json(key, "summary.json", run.summary, refresh)
        history_df = self._get_or_cache_history(
            key, run, keys=keys, samples=samples, refresh=refresh
        )

        if max_rows is not None and len(history_df) > max_rows:
            history_df = history_df.tail(max_rows)

        return RunData(
            key=key, config=config, summary=summary, history_df=history_df, run=run
        )

    # --------------------------------- Caching ---------------------------------
    def _run_cache_dir(self, key: RunKey) -> Path:
        d = self.cache_dir / key.entity / key.project / key.run_id
        _ensure_dir(d)
        return d

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists() or not self.use_cache:
            return False
        age = time.time() - path.stat().st_mtime
        return age < self.cache_ttl_seconds

    def _get_or_cache_json(
        self, key: RunKey, filename: str, value_provider: Any, refresh: bool
    ) -> Dict[str, Any]:
        path = self._run_cache_dir(key) / filename
        if not refresh and self._is_fresh(path):
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                self.logger.warning(
                    "Failed to read cache %s (%s); refetching", path, exc
                )

        # Allow value_provider to be a dict or a callable that returns a dict
        raw: Any
        if callable(value_provider):
            raw = value_provider()  # type: ignore[assignment]
        else:
            raw = value_provider

        value = _to_jsonable(raw)
        if not isinstance(value, dict):
            value = {"value": value}

        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f)
        return value

    def _get_or_cache_history(
        self,
        key: RunKey,
        run: Run,
        *,
        keys: Optional[Sequence[str]] = None,
        samples: Optional[int] = None,
        refresh: bool,
    ) -> "Any":
        selector = [] if not keys else list(keys)
        hashed = _hash_parts(
            ["all" if not selector else ",".join(sorted(selector)), str(samples)]
        )
        parquet_path = self._run_cache_dir(key) / f"history_{hashed}.parquet"
        csv_path = self._run_cache_dir(key) / f"history_{hashed}.csv"

        if not refresh and self._is_fresh(parquet_path):
            import pandas as pd  # type: ignore

            return pd.read_parquet(parquet_path)
        if not refresh and self._is_fresh(csv_path):
            import pandas as pd  # type: ignore

            return pd.read_csv(csv_path)

        # Fetch from API
        df = run.history(keys=list(keys) if keys else None, samples=samples or 0)

        # Prefer parquet if fastparquet/pyarrow is available, otherwise CSV
        try:
            import pandas as pd  # type: ignore

            assert isinstance(df, pd.DataFrame)
            df.to_parquet(parquet_path)
        except Exception as exc:
            self.logger.info(
                "Parquet write failed (%s); saving CSV to %s", exc, csv_path
            )
            try:
                df.to_csv(csv_path, index=False)  # type: ignore[attr-defined]
            except Exception:
                pass
        return df


__all__ = [
    "WandbDataClient",
    "RunData",
    "RunKey",
]


# ------------------------------ History collection ------------------------------


@dataclass
class HistoryCollectionConfig:
    entity: str
    project: str
    groups: Sequence[str]
    step_key: str
    metric_key: str
    samples: Optional[int] = None
    max_rows: Optional[int] = None
    use_cache: bool = True
    refresh: bool = False
    cache_ttl_seconds: int = 6 * 60 * 60


def _choose_step_key(columns: Any, preferred: str) -> Optional[str]:
    """Handle differences in step key names between trainings/loggers.

    Tries the preferred key first, then common alternatives observed in W&B logs.
    """
    candidates = [
        preferred,
        "num_updates",
        "Step",
        "_step",
        "global_step",
        "num_env_steps",
    ]
    for c in candidates:
        if c in columns:
            return c
    return None


def _process_run_df(
    df: pd.DataFrame,
    run_id: str,
    group_label: str,
    step_key: str,
    metric_key: str,
) -> Optional[pd.DataFrame]:
    """Normalize a run's history into tidy columns.

    Returns a DataFrame with columns [step_key, value, run_id, group] or None if
    required columns are missing.
    """
    step_col = _choose_step_key(df.columns, step_key)
    if step_col is None or metric_key not in df.columns:
        return None
    return (
        df[[step_col, metric_key]]
        .dropna()
        .rename({metric_key: "value", step_col: step_key}, axis=1)
        .assign(run_id=run_id, group=group_label)
    )


def _process_run_df_multi(
    df: pd.DataFrame,
    run_id: str,
    group_label: str,
    step_key: str,
    metric_keys: Sequence[str],
) -> Optional[pd.DataFrame]:
    """Normalize a run's history with multiple metrics into long form.

    Returns a DataFrame with columns [step_key, metric, value, run_id, group]
    or None if required columns are missing.
    """
    step_col = _choose_step_key(df.columns, step_key)
    if step_col is None:
        return None
    present_metrics: List[str] = [m for m in metric_keys if m in df.columns]
    if not present_metrics:
        return None
    wide = df[[step_col] + present_metrics].copy()
    long = wide.melt(id_vars=[step_col], var_name="metric", value_name="value")
    long = long.dropna(subset=["value"]).rename({step_col: step_key}, axis=1)
    return long.assign(run_id=run_id, group=group_label)


def _fetch_history_with_retry(
    client: WandbDataClient,
    run_id: str,
    keys: Sequence[str],
    *,
    samples: Optional[int],
    max_rows: Optional[int],
    refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Fetch run history, retrying once with refresh on failure."""
    try:
        data = client.fetch_run_data(
            run_id,
            keys=keys,
            samples=samples,
            max_rows=max_rows,
            refresh=refresh,
        )
        return data.history_df
    except Exception:
        if not refresh:
            try:
                data = client.fetch_run_data(
                    run_id,
                    keys=keys,
                    samples=samples,
                    max_rows=max_rows,
                    refresh=True,
                )
                return data.history_df
            except Exception:
                return None
        return None


def _collect_from_runs(
    client: WandbDataClient,
    runs_and_labels: Sequence[Tuple[Any, str]],
    step_key: str,
    metric_key: str,
    *,
    samples: Optional[int],
    max_rows: Optional[int],
    refresh: bool = False,
) -> pd.DataFrame:
    """Shared collection path for any iterable of (run, group_label)."""
    frames: List[pd.DataFrame] = []
    keys = [step_key, metric_key]
    for run, label in runs_and_labels:
        run_id = getattr(run, "id", str(run))
        df = _fetch_history_with_retry(
            client,
            run_id,
            keys,
            samples=samples,
            max_rows=max_rows,
            refresh=refresh,
        )
        if df is None:
            continue
        slim = _process_run_df(df, run_id, label, step_key, metric_key)
        if slim is not None:
            frames.append(slim)

    if not frames:
        return pd.DataFrame(
            {
                step_key: pd.Series(dtype="float64"),
                "value": pd.Series(dtype="float64"),
                "run_id": pd.Series(dtype="object"),
                "group": pd.Series(dtype="object"),
            }
        )
    return pd.concat(frames, ignore_index=True)


def _collect_from_runs_multi(
    client: WandbDataClient,
    runs_and_labels: Sequence[Tuple[Any, str]],
    step_key: str,
    metric_keys: Sequence[str],
    *,
    samples: Optional[int],
    max_rows: Optional[int],
    refresh: bool = False,
) -> pd.DataFrame:
    """Collect multiple metrics per run and return long-form rows per metric.

    Output columns: [step_key, metric, value, run_id, group]
    """
    frames: List[pd.DataFrame] = []
    # Fetch a superset once per run
    keys = [step_key] + list(dict.fromkeys(metric_keys))
    for run, label in runs_and_labels:
        run_id = getattr(run, "id", str(run))
        df = _fetch_history_with_retry(
            client,
            run_id,
            keys,
            samples=samples,
            max_rows=max_rows,
            refresh=refresh,
        )
        if df is None:
            continue
        slim = _process_run_df_multi(df, run_id, label, step_key, metric_keys)
        if slim is not None:
            frames.append(slim)

    if not frames:
        return pd.DataFrame(
            {
                step_key: pd.Series(dtype="float64"),
                "metric": pd.Series(dtype="object"),
                "value": pd.Series(dtype="float64"),
                "run_id": pd.Series(dtype="object"),
                "group": pd.Series(dtype="object"),
            }
        )
    return pd.concat(frames, ignore_index=True)


def _collect_group(
    client: WandbDataClient,
    group: str,
    step_key: str,
    metric_key: str,
    *,
    samples: Optional[int],
    max_rows: Optional[int],
) -> pd.DataFrame:
    runs = client.list_runs(group=group)
    runs_and_labels = [(run, group) for run in runs]
    return _collect_from_runs(
        client,
        runs_and_labels,
        step_key,
        metric_key,
        samples=samples,
        max_rows=max_rows,
        refresh=False,
    )


def collect_histories(cfg: HistoryCollectionConfig) -> pd.DataFrame:
    """Collect histories for multiple W&B groups into one tidy DataFrame.

    Returns columns: [step_key, value, run_id, group].
    """
    client = WandbDataClient(
        cfg.entity,
        cfg.project,
        cache_ttl_seconds=cfg.cache_ttl_seconds,
        use_cache=cfg.use_cache,
    )

    runs_and_labels: List[Tuple[Any, str]] = []
    for group in cfg.groups:
        for run in client.list_runs(group=group):
            runs_and_labels.append((run, group))

    return _collect_from_runs(
        client,
        runs_and_labels,
        cfg.step_key,
        cfg.metric_key,
        samples=cfg.samples,
        max_rows=cfg.max_rows,
        refresh=cfg.refresh,
    )


# ----------------------------- Mapping-based grouping -----------------------------


def load_runname_group_map_from_file(path: str | Path) -> Dict[str, str]:
    """Load mapping {run_name -> group_label} from JSON config with format:

    { "groups": { "AlgoA": ["run_a", ...], "AlgoB": ["run_b", ...] }, ... }
    """
    import json

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    groups = data.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(
            "Config must contain a 'groups' mapping of {algo: [run_names...]}."
        )

    mapping: Dict[str, str] = {}
    for algo, run_list in groups.items():
        if not isinstance(run_list, list):
            continue
        for rn in run_list:
            mapping[str(rn)] = str(algo)
    return mapping


def _extract_original_name(run: Any) -> Optional[str]:
    cfg = getattr(run, "config", None)
    if cfg is not None:
        try:
            val = cfg.get("run_name")
            if isinstance(val, str) and val:
                return val
        except (AttributeError, TypeError, ValueError, KeyError):
            pass
    name = getattr(run, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _select_runs_by_mapping(
    client: WandbDataClient,
    runname_to_group: Dict[str, str],
    step_key: str,
) -> List[Tuple[Any, str]]:
    """Choose one run per original name based on completeness; attach labels."""
    all_runs = list(client.list_runs())
    candidates_by_name: Dict[str, List[Any]] = {}
    label_by_name: Dict[str, str] = {}

    for run in all_runs:
        orig_name = _extract_original_name(run)
        if not orig_name:
            continue
        label = None
        for k in (
            orig_name,
            getattr(run, "name", None),
            getattr(run, "id", None),
            getattr(run, "group", None),
        ):
            if isinstance(k, str) and k in runname_to_group:
                label = runname_to_group[k]
                break
        if label is None:
            continue
        candidates_by_name.setdefault(orig_name, []).append(run)
        label_by_name[orig_name] = label

    def completeness_score(r: Any) -> float:
        # Prefer summary step (robust to W&B Summary proxy behavior)
        s_obj = getattr(r, "summary", None)
        if s_obj is not None:
            keys = (step_key, "num_updates", "_step", "Step", "global_step")
            get_attr = None
            try:
                get_attr = s_obj.get  # type: ignore[attr-defined]
            except (AttributeError, KeyError):
                get_attr = None
            if callable(get_attr):
                for k in keys:
                    try:
                        v = get_attr(k, None)  # type: ignore[misc]
                        if isinstance(v, (int, float)):
                            return float(v)
                    except (TypeError, KeyError, ValueError):
                        continue
            # Fallback to item access
            for k in keys:
                try:
                    v = s_obj[k]  # type: ignore[index]
                    if isinstance(v, (int, float)):
                        return float(v)
                except (KeyError, TypeError, ValueError):
                    continue
        # Fallback to history size/max(step)
        try:
            dfh = r.history(keys=[step_key], samples=500)
            if step_key in dfh.columns and not dfh.empty:
                return float(dfh[step_key].max())
            return float(len(dfh))
        except (AttributeError, RuntimeError, ValueError):
            return 0.0

    chosen: List[Tuple[Any, str]] = []
    for name, runs in candidates_by_name.items():
        if len(runs) == 1:
            chosen.append((runs[0], label_by_name[name]))
            continue
        scored = sorted(
            ((completeness_score(r), r) for r in runs), key=lambda x: x[0], reverse=True
        )
        best_score, best = scored[0]
        others = ", ".join(f"{getattr(r, 'id', '?')}:{s}" for s, r in scored[1:])
        warnings.warn(
            f"Duplicate runs for '{name}'. Using {getattr(best, 'id', '?')} (score={best_score}). Others: {others}"
        )
        chosen.append((best, label_by_name[name]))
    return chosen


def collect_histories_from_run_mapping(
    entity: str,
    project: str,
    runname_to_group: Dict[str, str],
    *,
    step_key: str,
    metric_key: str,
    samples: Optional[int] = None,
    max_rows: Optional[int] = None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_ttl_seconds: int = 6 * 60 * 60,
) -> pd.DataFrame:
    """Collect histories by mapping specific run identifiers to custom groups."""
    client = WandbDataClient(
        entity,
        project,
        cache_ttl_seconds=cache_ttl_seconds,
        use_cache=use_cache,
    )

    chosen = _select_runs_by_mapping(client, runname_to_group, step_key)

    return _collect_from_runs(
        client,
        chosen,
        step_key,
        metric_key,
        samples=samples,
        max_rows=max_rows,
        refresh=refresh,
    )


# New: multi-metric variant
def collect_multi_histories_from_run_mapping(
    entity: str,
    project: str,
    runname_to_group: Dict[str, str],
    *,
    step_key: str,
    metric_keys: Sequence[str],
    samples: Optional[int] = None,
    max_rows: Optional[int] = None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_ttl_seconds: int = 6 * 60 * 60,
) -> pd.DataFrame:
    """Collect histories for multiple metrics per mapped run.

    Returns columns: [step_key, metric, value, run_id, group].
    """
    client = WandbDataClient(
        entity,
        project,
        cache_ttl_seconds=cache_ttl_seconds,
        use_cache=use_cache,
    )

    chosen = _select_runs_by_mapping(client, runname_to_group, step_key)

    return _collect_from_runs_multi(
        client,
        chosen,
        step_key,
        metric_keys,
        samples=samples,
        max_rows=max_rows,
        refresh=refresh,
    )


# -------------------------- Step alignment diagnostics --------------------------


def summarize_max_steps(df: pd.DataFrame, step_key: str) -> pd.DataFrame:
    """Return one row per run with its maximum recorded step."""
    if df is None or df.empty:
        return pd.DataFrame(
            {
                "group": pd.Series(dtype="object"),
                "run_id": pd.Series(dtype="object"),
                "max_step": pd.Series(dtype="float64"),
            }
        )

    max_by_run = (
        df.groupby("run_id")[step_key]
        .max()
        .reset_index()
        .rename({step_key: "max_step"}, axis=1)
    )
    group_by_run = df.groupby("run_id")["group"].first().reset_index()
    out = max_by_run.merge(group_by_run, on="run_id")[["group", "run_id", "max_step"]]
    out = out.set_index(["group", "run_id"]).sort_index().reset_index()
    return out


def check_update_step_alignment(
    df: pd.DataFrame,
    step_key: str,
    *,
    per_group: bool = True,
    atol: float = 0.0,
) -> pd.DataFrame:
    """Warn if runs end at different update steps; returns per-run max steps."""
    steps = summarize_max_steps(df, step_key)
    if steps.empty:
        return steps

    def _warn_if_inconsistent(scope: str, sub: pd.DataFrame) -> None:
        vals = sub["max_step"].to_numpy()
        if len(vals) <= 1:
            return
        ref = float(vals[0])
        mismatched = [v for v in vals if abs(float(v) - ref) > float(atol)]
        if mismatched:
            counts = sub.groupby("max_step").size().sort_index()
            warnings.warn(
                f"Inconsistent max {step_key} across {scope}: "
                f"{dict(counts.to_dict())}. Consider aligning by a fixed step."
            )

    _warn_if_inconsistent("all runs", steps)
    if per_group:
        for g, sub in steps.groupby("group"):
            _warn_if_inconsistent(f"group '{g}'", sub)
    return steps
