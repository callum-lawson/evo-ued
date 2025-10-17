# Evo-UED

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

This repo provides initial algorithms and experiments to link **evolutionary game theory (EGT)** to **unsupervised environment design (UED)** in reinforcement learning. The code is a thin overlay on [JaxUED](https://github.com/DramaCow/jaxued), which provides the environments and core algorithms; this repo provides the evolutionary modifications (to the Domain Randomisation and Prioritised Level Replay algorithms), as well as run configs and analysis scripts.

A talk presenting the preliminary results can be found [here](https://youtu.be/XR-uPPCXGAs). 

---

## What we add

- **Evolutionary weighting**  
  - `evo_ued/utils.py` defines `negative_mean_reward`.  
  - This computes inverse average return per level, standardised across the batches. This is used to up-weight the importance or sampling frequency of "under-exploited" levels where the agent is currently weak.  
  - This is used in evolutionary versions of DR and PLR.

- **Evolutionary UED variants**  
  - `maze_dr_egt.py` and `maze_plr_egt.py` are evolutionary UED variants.  
  - Baselines (DR, PLR, PAIRED) are also here, for comparison.  
  - Only maze environments are included. Craftax and Gymnax examples remain upstream.

- **Result handling**  
  - `scripts/extract_checkpoint_results.py`: unpack each checkpoint into a Parquet file.  
  - `scripts/combine_checkpoint_results.py`: merge results across runs using JSON configs in `config/`.

- **Configs and notebooks**  
  - JSONs for 30k/50k updates and LR sweeps.  
  - Analysis notebooks for robustness, per-maze comparisons, and sensitivity tests.

---

## Quickstart

Clone and install upstream:

```bash
git submodule update --init --recursive
python -m pip install -e third_party/jaxued
```

Make the overlay importable:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Minimal test run (PLR with evolutionary weighting):

```bash
PYTHONPATH=src WANDB_MODE=disabled \
python examples/maze_plr_egt.py \
  --num_updates 1 --num_steps 8 --num_train_envs 2 --num_minibatches 1 --epoch_ppo 1 \
  --score_function neg_mean_reward --prioritization topk --topk_k 1 \
  --level_buffer_capacity 8 --replay_prob 0.5 --minimum_fill_ratio 0.0
```

Or with domain randomisation:

```bash
PYTHONPATH=src WANDB_MODE=disabled \
python examples/maze_dr_egt.py \
  --num_updates 1 --num_steps 8 --num_train_envs 2 --num_minibatches 1 --epoch_ppo 1
```

## Results pipeline

Extract results:

```bash
python -m scripts.extract_checkpoint_results --results_root ./results --force
```

Combine runs:

```bash
python -m scripts.combine_checkpoint_results \
  --config config/updates30k_lr_sweep.json \
  --force
```

Config example:

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

## Repo structure

- `src/evo_ued/` — overlay utilities.
- `examples/` — evolutionary + baseline maze runs.
- `scripts/` — result extraction and combination.
- `config/` — configs for sweeps and checkpoints.
- `notebooks/` — analysis notebooks.
- `third_party/jaxued/` — upstream code (submodule).
- `figures/` — GIFs of evaluation mazes for the README (from JaxUED).

## Notes

- Pin the upstream submodule commit you trained against.
- Check small runs after upgrading JAX/Flax.
- Install `pyarrow` if Parquet writes fail.
- `jax.tree_map` deprecation warnings can be ignored.
