# Evo‑UED (built on JaxUED)

<div align="center">
    <img src="figures/Labyrinth_299.gif" >
    <img src="figures/StandardMaze3_299.gif">
    <img src="figures/SixteenRooms_299.gif">
    <img src="figures/StandardMaze2_299.gif">
    <br>
    <img src="figures/SixteenRooms2_299.gif">
    <img src="figures/Labyrinth2_299.gif">
    <img src="figures/StandardMaze_299.gif">
    <img src="figures/LabyrinthFlipped_299.gif">
</div>

This repo contains our research‑specific additions on top of upstream JaxUED. We keep the repo small by:
- importing everything from upstream `jaxued`, and
- shipping only our changes in a tiny overlay module `evo_ued` plus scripts/configs for results.

What you get here is the minimum needed to reproduce our experiments and analysis.

### What’s new (vs upstream)
- **Score function**: `evo_ued.utils.negative_mean_reward`
  - Computes the time‑averaged reward per level over completed episodes, standardizes across the batch (z‑score), and returns the negative z‑score. Higher scores ≈ harder levels (lower rewards).
- **Result pipelines** for sweeps and checkpoint evaluation:
  - `scripts/extract_checkpoint_results.py` → convert each `results.npz` to a tidy Parquet table.
  - `scripts/combine_checkpoint_results.py` → combine many Parquet files using simple JSON configs in `config/` (grouped by algorithm/run names, optional checkpoint).
- **Configs**: ready‑to‑use JSONs in `config/` for 30k/50k steps and LR sweeps (e.g., `updates30k_lr_sweep.json`).
- **Examples**: Maze examples only. Craftax/Gymnax examples live upstream and are removed here to avoid drift.
- **Overlay only**: no vendored `src/jaxued`. We install upstream as a dependency and keep our changes isolated.

## Quickstart

1) Get upstream code (submodule) and install:
```bash
git submodule update --init --recursive
python -m pip install -e third_party/jaxued
```

2) Make your overlay importable (no install needed):
```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

3) Minimal run (maze, PLR‑style with our score):
```bash
PYTHONPATH=src WANDB_MODE=disabled \
python examples/maze_plr_egt.py \
  --num_updates 1 --num_steps 8 --num_train_envs 2 --num_minibatches 1 --epoch_ppo 1 \
  --score_function neg_mean_reward --prioritization topk --topk_k 1 \
  --level_buffer_capacity 8 --replay_prob 0.5 --minimum_fill_ratio 0.0
```

4) Alternative tiny run (DR):
```bash
PYTHONPATH=src WANDB_MODE=disabled \
python examples/maze_dr_egt.py \
  --num_updates 1 --num_steps 8 --num_train_envs 2 --num_minibatches 1 --epoch_ppo 1
```

### Using `neg_mean_reward`
- Enable in `examples/maze_plr_egt.py` via `--score_function neg_mean_reward`.
- Implementation is in `src/evo_ued/utils.py` and mirrors the logic in our diff (implemented on top of upstream `accumulate_rollout_stats`).

### Extract and combine results
1) Extract each checkpoint’s `results.npz` into a tidy Parquet file:
```bash
python -m scripts.extract_checkpoint_results --results_root ./results --force
```

2) Combine many runs into a single Parquet using a config from `config/`:
```bash
python -m scripts.combine_checkpoint_results \
  --config config/updates30k_lr_sweep.json \
  --force
```

Config format (example):
```json
{
  "checkpoint": 118,
  "output": "results/combined/eval250_updates30k_lr_sweep.parquet",
  "groups": {
    "DR (baseline)": ["dr_baseline_eval250_seed0_30000a", "dr_baseline_eval250_seed1_30000a"],
    "DR (evolutionary)": ["dr_softmin_eval250_seed1_30000a"],
    "ACCEL": ["accel_eval250_seed1_30000a"]
  }
}
```
The combiner searches under:
- `results/<run>/<seed>/<checkpoint>/results_extracted.parquet` (preferred), or
- legacy flat locations under `results/<run>/<seed>/`.
It concatenates all tables and tags each row with `run_name`, `seed`, and `checkpoint`.

### Repo layout (what matters here)
- `src/evo_ued/` — overlay (currently only `utils.py` with `negative_mean_reward`).
- `examples/` — Maze examples only: `maze_{dr,plr,paired}.py`, `maze_*_egt.py`.
- `scripts/` — result extraction/combination/analysis scripts.
- `config/` — run groups to combine (e.g., LR sweeps, checkpoints).
- `notebooks/` — analysis notebooks (performance curves, correlations, etc.).
- `third_party/jaxued/` — upstream code (submodule).
- `figures/` — GIFs/PNGs used by this README.

### Repro tips
- Pin the upstream submodule to the commit you trained against (already recorded in `.gitmodules`).
- When upgrading JAX/Flax, do a quick smoke run (like the command above) to catch API drift early.

### Troubleshooting
- If `evo_ued` isn’t found, add this repo to `PYTHONPATH` (or `pip install -e .`).
- If Parquet writes fail, install a Parquet engine, e.g. `pip install pyarrow`.
- Deprecation warnings like `jax.tree_map is deprecated` are benign for our runs.

### Licensing & attribution
- We rely on upstream JaxUED for most functionality. Please refer to their license and cite appropriately.
- This repo adds minimal glue and analysis around our experiments.
