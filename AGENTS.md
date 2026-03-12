# Repository Guidelines

## Project Structure & Module Organization
Core overlay code lives in `src/evo_ued/`. Example training entry points are in `examples/` (`maze_dr_egt.py`, `maze_plr_egt.py`, plus baseline scripts). Analysis and result-processing utilities live in `scripts/`. Tests are in `tests/`, with lightweight execution checks for example scripts. Keep generated artifacts such as `results/`, `checkpoints/`, and `wandb/` out of functional changes. Upstream dependencies are vendored under `third_party/`, especially `third_party/jaxued/`; treat those as external code unless a change explicitly targets the submodule copy.

## Build, Test, and Development Commands
Initialize and install the upstream package first:

```bash
git submodule update --init --recursive
python -m pip install -e third_party/jaxued
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Run a minimal local smoke test with:

```bash
PYTHONPATH=src WANDB_MODE=disabled python examples/maze_plr_egt.py --num_updates 1 --num_steps 8
```

Use the project conda environment for `python`, `pip`, and script execution. On this machine, `conda run -n evo-ued python` works for imports, but `pytest` may need to be installed separately before `pytest -q` will run. For results processing, use `python -m scripts.extract_checkpoint_results --results_root ./results --force` and `python -m scripts.combine_checkpoint_results --config config/updates30k_lr_sweep.json --force`.

For local analysis and checkpoint evaluation on this machine, prefer an activated `evo-ued` shell or `conda run -n evo-ued ...` over the bare system `python3`. The bare shell may be missing packages such as `numpy`/`pandas`, and example/eval commands may also need `PYTHONPATH=$PWD/src:$PWD/third_party/jaxued/src:$PYTHONPATH` so `jaxued` resolves correctly.

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python style consistent with the existing codebase. Follow descriptive, snake_case naming for modules, functions, and variables; keep example scripts named by environment and algorithm variant (`maze_plr.py`, `maze_plr_egt.py`). Prefer names that describe intent and outcome, such as `load_checkpoint` or `compute_loss`, and boolean names that read naturally (`is_ready`, `should_retry`). The repository already uses Black configuration in `pyproject.toml`; format Python edits with `black scripts tests examples src` when touching those areas. Avoid broad reformatting of `third_party/`.

## Testing Guidelines
Add or update `pytest` coverage for behavior changes when the environment supports it. Name tests `test_*.py` and prefer small, deterministic smoke tests over long training runs. When a change affects example scripts, verify they start successfully with `WANDB_MODE=disabled` and bounded runtime arguments.

For analysis-only work in `scripts/`, notebooks, or result aggregation code, lightweight validation is acceptable: run the specific command that changed, confirm output files are produced, and check key columns, shapes, or summary values. Do not describe that as full repository validation.

If a planned change touches training behavior, environment logic, shared library code, or example execution paths, call that out before editing and warn that broader testing is warranted. In those cases, do not treat analysis-command checks as a substitute for `pytest` or example smoke tests.

## Docstrings & Comments
Keep function docstrings to one sentence in plain language. Default to no inline comments; prefer clearer names, smaller helpers, and explicit intermediate variables. Add comments only for rationale, constraints, or algorithm references that are not obvious from the code.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Add results presentation to README` and `Reorganise training run analysis functions...`. Keep commits focused and use that style. PRs should describe the behavioral change, list validation performed (`pytest -q`, example command run, or analysis command checks), and link any related issue or experiment note. Include plots or screenshots only when notebook, figure, or analysis output changes.
