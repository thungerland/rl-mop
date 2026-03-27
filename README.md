# rl-mop
Code for an adaptive imitation learning model to investigate deep learning routing patterns for complex task learning e.g. gridworld.

## Workflow

### 1. Train a Model

**Local training:**
```bash
python train_mop.py --config config.yaml --task_id BabyAI-GoToObj-v0 --trial 0

# For complex multi-room environments, increase max steps to allow more exploration:
python train_mop.py --task_id BabyAI-GoToImpUnlock-v0 --trial 0 --max_steps 1500
```

**Modal (cloud) training:**
```bash
# Use --detach for running in the background
modal run --detach modal_app.py::main --task-ids "BabyAI-GoToObj-v0" --trials "0" --script train_mop.py

# Multiple tasks and trials in parallel
modal run --detach modal_app.py::main --task-ids "BabyAI-GoToObj-v0,BabyAI-UnlockPickup-v0" --trials "0,1,2"
```

**Multi-seed training with periodic checkpoints:**

Use `main_seeds` to train multiple random seeds for the same hyperparameter config. Each seed gets its own subdirectory (`seed_S/`) and, if `--checkpoint-interval` is set, numbered checkpoints are saved throughout training alongside the final alias.

```bash
# Train 10 seeds for a single trial, saving a checkpoint every 250 updates
modal run --detach modal_app.py::main_seeds \
  --task-ids "BabyAI-UnlockPickup-v0" \
  --trials "20" \
  --seeds "0,1,2,3,4,5,6,7,8,9" \
  --checkpoint-interval 250 \
  --extra-args "--lpc_alpha 1e-4 --num_updates 5000"

# Multiple trials (different alpha values) × seeds
modal run --detach modal_app.py::main_seeds \
  --task-ids "BabyAI-UnlockPickup-v0" \
  --trials "20,21" \
  --seeds "0,1,2,3,4,5,6,7,8,9" \
  --checkpoint-interval 250
```

Checkpoints are saved to:
```
/checkpoints/<task_id>/trial_<N>/seed_<S>/checkpoint_250.pt
/checkpoints/<task_id>/trial_<N>/seed_<S>/checkpoint_500.pt
...
/checkpoints/<task_id>/trial_<N>/seed_<S>/checkpoint_5000.pt
/checkpoints/<task_id>/trial_<N>/seed_<S>/checkpoint_final.pt  ← alias for the last step
```

Download to local after training:
```bash
modal volume get rl-mop /BabyAI-UnlockPickup-v0/trial_20 ./modal_checkpoints/BabyAI-UnlockPickup-v0/trial_20
```

### 2. Get Checkpoints

Local training saves checkpoints automatically to `checkpoints/`.

For Modal training, download checkpoints from the volume:
```bash
# List available checkpoints
modal volume ls rl-mop

# Download to local folder; --force to override existing file with the same name
modal volume get rl-mop / ./modal_checkpoints/

# Copy to checkpoints folder
cp -r modal_checkpoints/* checkpoints/
```

### 3. Evaluate

**Parallel evaluation on Modal (recommended):**

Compares `modal_checkpoints/` against the local `evaluation_results.csv` and spawns one GPU job per missing checkpoint in parallel — much faster than the sequential fallback.

```bash
# Evaluate all missing checkpoints (1000 episodes each, default)
modal run modal_app.py::eval_parallel

# Custom episode count
modal run modal_app.py::eval_parallel --num-episodes 500

# Force re-evaluate everything
modal run modal_app.py::eval_parallel --force

# Only evaluate a specific task's missing trials
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0"

# After completion, download results locally
modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv
modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache

# To download only a specific newly-evaluated task without overwriting everything else:
modal volume get rl-mop-eval /eval_output/evaluation_cache/BabyAI-UnlockPickup-v0 ./evaluation_cache/
```

**Selective evaluation of seeds and checkpoints:**

`eval_parallel` accepts `--seed` and `--update` filters so you can target a specific slice of the checkpoint tree without evaluating everything.

```bash
# Evaluate all seeds for one trial (all update steps)
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20

# Evaluate a single seed across all its update steps
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20 --seed 3

# Evaluate one specific update step across all seeds (useful for plateau analysis)
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20 --update 1500

# Evaluate checkpoints bracketing the plateau (run once per update step of interest)
for update in 1000 1250 1500 1750 2000; do
  modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20 --update $update
done

# Evaluate a specific seed at a specific update
modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20 --seed 0 --update 1500
```

The same filters work locally via `batch_eval.py`:
```bash
python batch_eval.py --task BabyAI-UnlockPickup-v0 --trial 20 --seed 0 --update 1500
```

Results are written to the `rl-mop-eval` Modal volume and downloaded on demand. The `evaluation_cache/` folder is what `routing_viz.py` and `logit_viz.py` read from.

> `eval_main` is a simpler alternative that runs all evaluations sequentially in a single container (`modal run modal_app.py::eval_main`).

**Local batch evaluation (alternative):**
```bash
# Evaluate all new checkpoints in modal_checkpoints/
python batch_eval.py

# Re-evaluate all checkpoints
python batch_eval.py --force

# Evaluate with custom episode count
python batch_eval.py --num_episodes 200

# Evaluate specific task, trial, seed, or update step
python batch_eval.py --task BabyAI-UnlockPickup-v0
python batch_eval.py --task BabyAI-UnlockPickup-v0 --trial 20
python batch_eval.py --task BabyAI-UnlockPickup-v0 --trial 20 --seed 0
python batch_eval.py --task BabyAI-UnlockPickup-v0 --trial 20 --update 1500

# Skip routing cache for faster evaluation
python batch_eval.py --skip_routing
```

Results are saved to `evaluation_results.csv` with columns:
- Metrics: `success_rate`, `path_ratio`, `mean_lpc`
- Training params: `num_updates`, `unroll_len`, `lr`, `lpc_alpha`, `max_steps`
- Architecture: `expert_hidden_sizes`, `intermediate_dim`, `router_hidden_size`

Routing data is cached to `evaluation_cache/` for fast visualization.

**View results:**
```bash
# Print full table
python -c "import pandas as pd; print(pd.read_csv('evaluation_results.csv').to_string())"

# Sort by success rate (includes seed and update columns)
python -c "
import pandas as pd
df = pd.read_csv('evaluation_results.csv')
print(df.sort_values('success_rate', ascending=False)[['task_id', 'trial', 'seed', 'update', 'success_rate', 'path_ratio', 'mean_lpc']].to_string())
"
```

**Single checkpoint evaluation:**
```bash
python eval_mop.py --checkpoint checkpoints/BabyAI-GoToObj-v0/trial_0/checkpoint_final.pt --num_episodes 100
```

**Focused analysis (recommended for single-task iteration):**
```bash
python analyze.py <task_id> <trial> [plot_type] [--seed S] [--update U]

# Examples
python analyze.py BabyAI-GoToDoor-v0 0
python analyze.py BabyAI-GoToDoor-v0 0 by_carrying_phase
python analyze.py BabyAI-GoToDoor-v0 0 action_frequency

# Seeded / update-specific checkpoints
python analyze.py BabyAI-UnlockPickup-v0 20 overall --seed 3
python analyze.py BabyAI-UnlockPickup-v0 20 kl_heatmap --seed 3 --update 1500
```
Generates and previews a plot, then prompts whether to save. Reads from `evaluation_cache/` (no marimo required). Available plot types:
- `overall` (default) — routing heatmap across all data
- `by_starting_room`, `by_door_location`, `by_door_and_box_row`, `by_carrying_phase`, `by_agent_and_target_quadrant` — grouped routing heatmaps
- `action_frequency`, `action_frequency_carrying`, `entropy_heatmap` — logit/distributional analyses

New plot types: add a function to `plotting_utils.py`, import and add a branch in `analyze.py`.

**Routing visualization (marimo notebook — broad exploration across tasks):**
```bash
marimo run routing_viz.py
```
Interactive notebook for exploring routing heatmaps across all cached tasks.

**Logit / distributional analysis (marimo notebook — broad exploration across tasks):**
```bash
marimo run logit_viz.py
```
Interactive notebook for action frequency and entropy heatmap plots across all cached tasks. Requires a v3 cache file (produced by `batch_eval.py` without `--skip_routing`).
