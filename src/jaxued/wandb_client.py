"""
Utilities for accessing Weights & Biases (W&B) run data from code.

This module provides a small, typed client that fetches run metadata, summary
statistics and the full time-series history that powers W&B browser plots.

Features:
- Simple methods to list runs and fetch per-run data (config, summary, history)
- Optional local caching to avoid repeated API calls when iterating on plots
- Minimal, dependency-light implementation leveraging wandb's public API

Environment:
- Ensure you are logged into W&B (wandb.login()) or have WANDB_API_KEY set.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import logging


try:
    import wandb
    from wandb.apis.public import Api, Run  # type: ignore
except Exception as exc:  # pragma: no cover - import-time hint
    raise RuntimeError(
        "wandb is required to use jaxued.wandb_client. Install with `pip install wandb`."
    ) from exc


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _hash_parts(parts: Sequence[str]) -> str:
    m = hashlib.sha256()
    for part in parts:
        m.update(part.encode("utf-8"))
        m.update(b"|")
    return m.hexdigest()[:16]


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary objects to a JSON-serializable structure.

    Handles W&B summary/config objects, numpy scalars/arrays, and generic
    mappings/sequences. Falls back to repr() for unknown objects.
    """
    # Primitive types
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Numpy types (scalars/arrays)
    try:  # lazy import and duck-typed conversion
        import numpy as np  # type: ignore

        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
    except Exception:
        pass

    # W&B types: try to_json / to_dict if present. Some W&B objects override
    # attribute access to map to dict keys, so getattr may raise KeyError.
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

    # Sequences (but not strings/bytes which are handled above)
    from collections.abc import Sequence as Seq

    if isinstance(value, Seq) and not isinstance(value, (bytes, bytearray, str)):
        return [_to_jsonable(v) for v in value]

    # Fallback
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
    """Container for the core data used by W&B plots.

    - config: static run configuration (dict-like)
    - summary: last reported summary metrics
    - history: time-series metrics (pandas.DataFrame)
    - run: the underlying wandb Run object (for advanced usage)
    """

    key: RunKey
    config: Dict[str, Any]
    summary: Dict[str, Any]
    history_df: "Any"  # pandas.DataFrame, but avoid importing pandas in types
    run: Run


class WandbDataClient:
    """Thin client for retrieving W&B run data with optional local caching.

    Caching strategy:
    - Each run's config/summary are cached as JSON.
    - History is cached to Parquet when available; falls back to CSV.
    - Cache invalidation via TTL or explicit refresh=True.
    """

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

        # Default cache directory: XDG_CACHE_HOME/jaxued/wandb or ~/.cache/jaxued/wandb
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            base = Path(xdg) if xdg else Path.home() / ".cache"
            self.cache_dir = base / "jaxued" / "wandb"
        _ensure_dir(self.cache_dir)

    # ----------------------------- Listing & lookup ----------------------------
    def list_runs(
        self,
        *,
        group: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        order: Optional[str] = "-created_at",
        per_page: int = 200,
    ) -> List[Run]:
        """List runs for the configured entity/project.

        Args:
            group: Optional W&B group name to filter on.
            filters: Additional W&B public API filters (dict).
            order: Sort order string as used by W&B (default newest first).
            per_page: Pagination size for API results.
        """
        query_filters: Dict[str, Any] = filters.copy() if filters else {}
        if group is not None:
            query_filters.setdefault("group", group)
        runs = self.api.runs(
            f"{self.entity}/{self.project}", filters=query_filters, order=order, per_page=per_page
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
        """Fetch config, summary, and history used by W&B plots for a run.

        Args:
            run_id_or_path: Run id (e.g. 'abcd1234') or full path 'entity/project/id'.
            keys: Optional subset of history keys to pull. None fetches all.
            samples: Optional downsampling passed to wandb history() for speed.
            refresh: Ignore cache and re-download.
            max_rows: If provided, truncate the returned history to last N rows.
        """
        run = self.get_run(run_id_or_path)
        key = RunKey(run.entity, run.project, run.id)

        config = self._get_or_cache_json(key, "config.json", run.config, refresh)
        summary = self._get_or_cache_json(key, "summary.json", run.summary, refresh)
        history_df = self._get_or_cache_history(key, run, keys=keys, samples=samples, refresh=refresh)

        if max_rows is not None and len(history_df) > max_rows:
            history_df = history_df.tail(max_rows)

        return RunData(key=key, config=config, summary=summary, history_df=history_df, run=run)

    # --------------------------------- Caching ---------------------------------
    def _run_cache_dir(self, key: RunKey) -> Path:
        d = self.cache_dir / key.entity / key.project / key.run_id
        _ensure_dir(d)
        return d

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if not self.use_cache:
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
                self.logger.warning("Failed to read cache %s (%s); refetching", path, exc)

        # Allow value_provider to be a dict or a callable that returns a dict
        raw: Any
        if callable(value_provider):
            raw = value_provider()  # type: ignore[assignment]
        else:
            raw = value_provider

        value = _to_jsonable(raw)
        if not isinstance(value, dict):
            # Ensure mapping for our return type
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
        import pandas as pd  # Local import to keep module import light

        # File name depends on keys and sampling to avoid collisions
        selector = [] if not keys else list(keys)
        hashed = _hash_parts(["all" if not selector else ",".join(sorted(selector)), str(samples)])
        parquet_path = self._run_cache_dir(key) / f"history_{hashed}.parquet"
        csv_path = self._run_cache_dir(key) / f"history_{hashed}.csv"

        if not refresh and self._is_fresh(parquet_path):
            return pd.read_parquet(parquet_path)
        if not refresh and self._is_fresh(csv_path):
            return pd.read_csv(csv_path)

        # Fetch from API
        df = run.history(keys=list(keys) if keys else None, samples=samples)

        # Prefer parquet if fastparquet/pyarrow is available, otherwise CSV
        try:
            df.to_parquet(parquet_path)
        except Exception as exc:
            self.logger.info("Parquet write failed (%s); saving CSV to %s", exc, csv_path)
            df.to_csv(csv_path, index=False)
        return df


__all__ = [
    "WandbDataClient",
    "RunData",
    "RunKey",
]


