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
        hashed_samples = "all" if samples is None else str(samples)
        hashed = _hash_parts(
            ["all" if not selector else ",".join(sorted(selector)), hashed_samples]
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
        _hist_kwargs: Dict[str, Any] = {"keys": list(keys) if keys else None}
        if samples is not None:
            _hist_kwargs["samples"] = int(samples)
        df = run.history(**_hist_kwargs)

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
    "list_runs_metadata",
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
    # Optional: attach selected run metadata (from config/summary) to each row
    attach_config_keys: Optional[Sequence[str]] = None
    attach_summary_keys: Optional[Sequence[str]] = None


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


def _fetch_run_with_retry(
    client: WandbDataClient,
    run_id: str,
    keys: Sequence[str],
    *,
    samples: Optional[int],
    max_rows: Optional[int],
    refresh: bool = False,
) -> Optional[RunData]:
    """Fetch full RunData with retry (config, summary, history)."""
    try:
        data = client.fetch_run_data(
            run_id,
            keys=keys,
            samples=samples,
            max_rows=max_rows,
            refresh=refresh,
        )
        return data
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
                return data
            except Exception:
                return None
        return None


def _get_nested(dct: Dict[str, Any], dotted_key: str) -> Any:
    """Safely get nested dict value using dot-separated key.

    Returns None if any part is missing or not a mapping.
    """
    cur: Any = dct
    for part in str(dotted_key).split("."):
        try:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        except Exception:
            return None
    return cur


def _collect_from_runs(
    client: WandbDataClient,
    runs_and_labels: Sequence[Tuple[Any, str]],
    step_key: str,
    metric_key: str,
    *,
    samples: Optional[int],
    max_rows: Optional[int],
    refresh: bool = False,
    attach_config_keys: Optional[Sequence[str]] = None,
    attach_summary_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Shared collection path for any iterable of (run, group_label)."""
    frames: List[pd.DataFrame] = []
    keys = [step_key, metric_key]
    for run, label in runs_and_labels:
        run_id = getattr(run, "id", str(run))
        want_meta = bool(attach_config_keys) or bool(attach_summary_keys)
        if want_meta:
            rd = _fetch_run_with_retry(
                client,
                run_id,
                keys,
                samples=samples,
                max_rows=max_rows,
                refresh=refresh,
            )
            if rd is None:
                continue
            df = rd.history_df
            slim = _process_run_df(df, run_id, label, step_key, metric_key)
            if slim is not None:
                meta: Dict[str, Any] = {}
                if attach_config_keys:
                    for k in attach_config_keys:
                        meta[str(k)] = _to_jsonable(_get_nested(rd.config, str(k)))
                if attach_summary_keys:
                    for k in attach_summary_keys:
                        meta[f"summary.{k}"] = _to_jsonable(
                            _get_nested(rd.summary, str(k))
                        )
                if meta:
                    try:
                        slim = slim.assign(**meta)
                    except Exception:
                        # Fallback: ensure all meta values are present
                        for mk, mv in meta.items():
                            slim[mk] = mv
        else:
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
        base: Dict[str, Any] = {
            step_key: pd.Series(dtype="float64"),
            "value": pd.Series(dtype="float64"),
            "run_id": pd.Series(dtype="object"),
            "group": pd.Series(dtype="object"),
        }
        if attach_config_keys:
            for k in attach_config_keys:
                base[str(k)] = pd.Series(dtype="object")
        if attach_summary_keys:
            for k in attach_summary_keys:
                base[f"summary.{k}"] = pd.Series(dtype="object")
        return pd.DataFrame(base)
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
    attach_config_keys: Optional[Sequence[str]] = None,
    attach_summary_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Collect multiple metrics per run and return long-form rows per metric.

    Output columns: [step_key, metric, value, run_id, group]
    """
    frames: List[pd.DataFrame] = []
    # Fetch a superset once per run
    keys = [step_key] + list(dict.fromkeys(metric_keys))
    for run, label in runs_and_labels:
        run_id = getattr(run, "id", str(run))
        want_meta = bool(attach_config_keys) or bool(attach_summary_keys)
        if want_meta:
            rd = _fetch_run_with_retry(
                client,
                run_id,
                keys,
                samples=samples,
                max_rows=max_rows,
                refresh=refresh,
            )
            if rd is None:
                continue
            df = rd.history_df
            slim = _process_run_df_multi(df, run_id, label, step_key, metric_keys)
            if slim is not None:
                meta: Dict[str, Any] = {}
                if attach_config_keys:
                    for k in attach_config_keys:
                        meta[str(k)] = _to_jsonable(_get_nested(rd.config, str(k)))
                if attach_summary_keys:
                    for k in attach_summary_keys:
                        meta[f"summary.{k}"] = _to_jsonable(
                            _get_nested(rd.summary, str(k))
                        )
                if meta:
                    try:
                        slim = slim.assign(**meta)
                    except Exception:
                        for mk, mv in meta.items():
                            slim[mk] = mv
        else:
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
        base: Dict[str, Any] = {
            step_key: pd.Series(dtype="float64"),
            "metric": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
            "run_id": pd.Series(dtype="object"),
            "group": pd.Series(dtype="object"),
        }
        if attach_config_keys:
            for k in attach_config_keys:
                base[str(k)] = pd.Series(dtype="object")
        if attach_summary_keys:
            for k in attach_summary_keys:
                base[f"summary.{k}"] = pd.Series(dtype="object")
        return pd.DataFrame(base)
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
        attach_config_keys=cfg.attach_config_keys,
        attach_summary_keys=cfg.attach_summary_keys,
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
    attach_config_keys: Optional[Sequence[str]] = None,
    attach_summary_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Collect histories by mapping specific run identifiers to custom groups.

    This variant fetches ONE metric per run.

    - Inputs: step_key, metric_key (str)
    - Output columns: [step_key, value, run_id, group] plus any attached
      metadata columns provided via attach_config_keys / attach_summary_keys
    - Use when you only need a single metric per run (lighter and simpler)
    """
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
        attach_config_keys=attach_config_keys,
        attach_summary_keys=attach_summary_keys,
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
    attach_config_keys: Optional[Sequence[str]] = None,
    attach_summary_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Collect histories for multiple metrics per mapped run in one pass.

    This variant fetches MANY metrics per run and returns a long-form DataFrame
    with one row per (step, metric).

    - Inputs: step_key, metric_keys (Sequence[str])
    - Output columns: [step_key, metric, value, run_id, group] plus any attached
      metadata columns provided via attach_config_keys / attach_summary_keys
    - Use when you want multiple metrics available for downstream subsetting
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
        attach_config_keys=attach_config_keys,
        attach_summary_keys=attach_summary_keys,
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


# ------------------------------ Run meta inspection ------------------------------


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        # Some W&B summaries store epoch seconds
        try:
            return pd.to_datetime(float(value), unit="s", utc=True)
        except Exception:
            return None


def _run_file_stats(run: Any, *, max_files: Optional[int] = None) -> Dict[str, Any]:
    total_bytes: int = 0
    count: int = 0
    try:
        files_iter = run.files()  # paginated iterable
        for f in files_iter:
            try:
                size = _safe_getattr(f, "size", 0) or 0
                total_bytes += int(size)
                count += 1
                if max_files is not None and count >= int(max_files):
                    break
            except Exception:
                # Skip problematic file entries
                continue
    except Exception:
        pass
    mb = float(total_bytes) / (1024.0 * 1024.0)
    gb = float(total_bytes) / (1024.0 * 1024.0 * 1024.0)
    return {
        "num_files": count,
        "total_file_bytes": int(total_bytes),
        "total_file_mb": mb,
        "total_file_gb": gb,
    }


def _summary_get(summary_obj: Any, key: str, default: Any = None) -> Any:
    try:
        get = summary_obj.get  # type: ignore[attr-defined]
        if callable(get):
            return get(key, default)
    except Exception:
        pass
    try:
        return summary_obj[key]  # type: ignore[index]
    except Exception:
        return default


def list_runs_metadata(
    entity: str,
    project: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    per_page: int = 200,
    include_file_sizes: bool = False,
    max_files: Optional[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of key metadata for all runs in a project.

    Columns include identifiers, state, timing, grouping, tags, URL, and
    approximate storage usage (sum of run files) to help identify runs that
    are safe to delete to reclaim space.

    Parameters:
        entity: W&B entity/org/user
        project: W&B project name
        filters: Optional API filters (same as wandb.Api().runs filters)
        per_page: Pagination size for listing runs
        include_file_sizes: If True, sum sizes of files attached to each run
        max_files: Optional cap on files scanned per run (for speed)
        use_cache: Construct client with caching of JSON/history (not used here)

    Returns:
        pandas.DataFrame with one row per run.
    """
    client = WandbDataClient(
        entity,
        project,
        use_cache=use_cache,
    )

    runs = client.list_runs(filters=filters, per_page=per_page)
    rows: List[Dict[str, Any]] = []

    for run in runs:
        # Core identifiers
        run_id = _safe_getattr(run, "id")
        name = _safe_getattr(run, "name")
        path = f"{_safe_getattr(run, 'entity', entity)}/{_safe_getattr(run, 'project', project)}/{run_id}"
        url = _safe_getattr(run, "url")

        # Grouping / context
        group = _safe_getattr(run, "group")
        tags = list(_safe_getattr(run, "tags", []) or [])
        state = _safe_getattr(run, "state")
        user = _safe_getattr(run, "user")
        username = _safe_getattr(user, "username") if user is not None else None

        # Timing info
        created_at = _to_timestamp(_safe_getattr(run, "created_at"))
        updated_at = _to_timestamp(_safe_getattr(run, "updated_at"))

        # From summary
        summary = _safe_getattr(run, "summary")
        runtime_seconds = None
        finished_at = None
        try:
            get = summary.get  # type: ignore[attr-defined]
            runtime_seconds = get("_runtime")
            finished_at = _to_timestamp(get("_timestamp"))
        except Exception:
            # Fallback to item access
            try:
                runtime_seconds = summary["_runtime"]  # type: ignore[index]
            except Exception:
                runtime_seconds = None
            try:
                finished_at = _to_timestamp(summary["_timestamp"])  # type: ignore[index]
            except Exception:
                finished_at = None

        runtime_hours = float(runtime_seconds) / 3600.0 if runtime_seconds else None

        # Common summary metrics (fast to read; avoids heavy history fetches)
        summary_num_updates = (
            _summary_get(summary, "num_updates") if summary is not None else None
        )
        summary_num_env_steps = (
            _summary_get(summary, "num_env_steps") if summary is not None else None
        )
        summary_step = _summary_get(summary, "_step") if summary is not None else None
        summary_solve_mean = (
            _summary_get(summary, "solve_rate/mean") if summary is not None else None
        )
        summary_return_mean = (
            _summary_get(summary, "return/mean") if summary is not None else None
        )
        summary_eval_len_mean = (
            _summary_get(summary, "eval_ep_lengths/mean")
            if summary is not None
            else None
        )

        # Sweep info if present
        sweep_id = None
        try:
            sw = _safe_getattr(run, "sweep")
            if sw is not None:
                sweep_id = (
                    _safe_getattr(sw, "id") or _safe_getattr(sw, "name") or str(sw)
                )
        except Exception:
            sweep_id = None

        # Config/summary sizes
        config = _safe_getattr(run, "config", {}) or {}
        summary_dict: Dict[str, Any] = {}
        try:
            # Convert to basic dict where possible
            summary_dict = _to_jsonable(summary) if summary is not None else {}
        except Exception:
            summary_dict = {}
        config_keys = len(list(config.keys())) if isinstance(config, dict) else None
        summary_keys = (
            len(list(summary_dict.keys())) if isinstance(summary_dict, dict) else None
        )
        config_run_name = None
        try:
            if isinstance(config, dict):
                config_run_name = config.get("run_name")
        except Exception:
            config_run_name = None

        # File stats (approx storage)
        file_stats: Dict[str, Any] = {}
        if include_file_sizes:
            file_stats = _run_file_stats(run, max_files=max_files)

        rows.append(
            {
                "entity": _safe_getattr(run, "entity", entity),
                "project": _safe_getattr(run, "project", project),
                "run_id": run_id,
                "run_name": name,
                "config.run_name": config_run_name,
                "path": path,
                "url": url,
                "state": state,
                "group": group,
                "tags": tags,
                "username": username,
                "created_at": created_at,
                "updated_at": updated_at,
                "finished_at": finished_at,
                "runtime_seconds": runtime_seconds,
                "runtime_hours": runtime_hours,
                "summary.num_updates": summary_num_updates,
                "summary.num_env_steps": summary_num_env_steps,
                "summary._step": summary_step,
                "summary.solve_rate/mean": summary_solve_mean,
                "summary.return/mean": summary_return_mean,
                "summary.eval_ep_lengths/mean": summary_eval_len_mean,
                "sweep_id": sweep_id,
                "config_keys": config_keys,
                "summary_keys": summary_keys,
                **file_stats,
            }
        )

    df = pd.DataFrame(rows)
    # Useful default sort: largest first, then most recent
    sort_cols: List[str] = []
    if "total_file_bytes" in df.columns:
        sort_cols.append("total_file_bytes")
    if "updated_at" in df.columns:
        sort_cols.append("updated_at")
    if sort_cols:
        try:
            df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        except Exception:
            pass
    return df.reset_index(drop=True)
