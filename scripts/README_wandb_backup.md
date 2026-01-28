# WandB Data Backup & Offline Analysis

This document describes how to back up wandb data to Google Drive and use it offline.

## What's Backed Up

The cache contains all data needed for analysis:

- `config.json` - Run configuration (hyperparameters, run name, etc.)
- `summary.json` - Final summary metrics
- `history_*.parquet` - Full training history (all 37 columns including per-maze metrics)

**Not included**: GIF artifacts (rendered maze videos) - these are large and not needed for analysis.

## Cache Location

Default location: `~/.cache/evo-ued/wandb/callumrlawson-pibbss/JAXUED_TEST/`

Each run has its own subdirectory named by wandb run ID (e.g., `0d8eu2lt/`).

## Downloading Full History

Before backing up, download complete history for all runs:

```bash
# Test with one run first to check file size
python scripts/download_full_histories.py --test

# Download all 108 runs
python scripts/download_full_histories.py
```

This creates `history_*.parquet` files with all 37 columns (not just the 4 columns
that may have been cached during previous analysis).

## Backing Up to Google Drive

1. Run the download script to ensure all runs have full history
2. Navigate to `\\wsl$\Ubuntu-22.04\home\callum\.cache\evo-ued\wandb\` in Windows Explorer
3. Drag the `callumrlawson-pibbss` folder to Google Drive

**Current backup location:** `My Drive > jaxued_checkpoints > wandb`

## Restoring from Backup

1. Download from Google Drive: `My Drive > jaxued_checkpoints > wandb`
2. Place it at `~/.cache/evo-ued/wandb/` (or any location you prefer)
3. Use `OfflineDataClient` pointing to that location

```python
from scripts.utils_wandb import OfflineDataClient

# Using default cache location (~/.cache/evo-ued/wandb/)
client = OfflineDataClient("callumrlawson-pibbss", "JAXUED_TEST")

# Or with custom location (e.g., downloaded Drive folder)
client = OfflineDataClient(
    "callumrlawson-pibbss",
    "JAXUED_TEST",
    data_dir="~/Downloads/wandb"
)
```

## Using OfflineDataClient

### Basic Usage

```python
from scripts.utils_wandb import OfflineDataClient

client = OfflineDataClient("callumrlawson-pibbss", "JAXUED_TEST")

# List all cached runs
run_ids = client.list_runs()
print(f"Found {len(run_ids)} cached runs")

# Load a specific run
data = client.fetch_run_data("0d8eu2lt")
print(data.config)
print(data.summary)
print(data.history_df.columns.tolist())
```

### Collecting Histories (like the online functions)

```python
from scripts.utils_wandb import (
    OfflineDataClient,
    collect_histories_offline,
    collect_multi_histories_offline,
    load_runname_group_map_from_file,
)

client = OfflineDataClient("callumrlawson-pibbss", "JAXUED_TEST")

# Load run mapping from config file
runname_to_group = load_runname_group_map_from_file(
    "config/eval250_updates30k_lr1e-4.json"
)

# Single metric
df = collect_histories_offline(
    client,
    runname_to_group,
    step_key="num_updates",
    metric_key="solve_rate/mean",
)

# Multiple metrics (long form)
df_multi = collect_multi_histories_offline(
    client,
    runname_to_group,
    step_key="num_updates",
    metric_keys=["solve_rate/mean", "return/mean", "eval_ep_lengths/mean"],
)
```

## Available Columns in Full History

When downloaded with `keys=None`, history includes all logged columns:

- Step: `_step`, `num_updates`, `num_env_steps`
- Core metrics: `solve_rate/mean`, `return/mean`, `eval_ep_lengths/mean`
- Per-maze solve rates: `solve_rate/maze_0` through `solve_rate/maze_49`
- Losses: `loss_total`, `loss_actor`, `loss_critic`, `loss_entropy`
- Learning stats: `learning_rate`, `grad_norm`, etc.

## Differences from Online Mode

The offline functions mirror the online API but with some simplifications:

| Feature | Online (WandbDataClient) | Offline (OfflineDataClient) |
|---------|--------------------------|----------------------------|
| Network | Required | Not required |
| `samples` param | Supported | Not supported (full history loaded) |
| `refresh` param | Supported | Not applicable |
| `cache_ttl` | Configurable | Not applicable |
| Run deduplication | By completeness score | First match only |

## Troubleshooting

**"Cache directory not found" error:**
- Check that the data_dir path is correct
- Ensure the entity/project subdirectories exist

**Missing runs:**
- Run `download_full_histories.py` to fetch missing runs
- Check that run names in your config file match `config.run_name` in the cache

**Empty history DataFrame:**
- The run may not have logged the requested columns
- Check available columns with `client.fetch_run_data(run_id).history_df.columns`
