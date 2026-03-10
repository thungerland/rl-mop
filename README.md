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

# After completion, download results locally (--force)
modal volume get rl-mop-eval /evaluation_results.csv ./evaluation_results.csv
modal volume get rl-mop-eval /evaluation_cache .

# To download only a specific newly-evaluated task without overwriting everything else:
modal volume get rl-mop-eval /evaluation_cache/BabyAI-SomeTask-v0 ./evaluation_cache/
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

# Evaluate specific task or trial
python batch_eval.py --task BabyAI-GoToRedBall-v0
python batch_eval.py --task BabyAI-GoToRedBall-v0 --trial 0

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

# Sort by success rate
python -c "
import pandas as pd
df = pd.read_csv('evaluation_results.csv')
print(df.sort_values('success_rate', ascending=False)[['task_id', 'trial', 'success_rate', 'path_ratio', 'mean_lpc']].to_string())
"
```

**Single checkpoint evaluation:**
```bash
python eval_mop.py --checkpoint checkpoints/BabyAI-GoToObj-v0/trial_0/checkpoint_final.pt --num_episodes 100
```

**Focused analysis (recommended for single-task iteration):**
```bash
python analyze.py <task_id> <trial> [plot_type]

# Examples
python analyze.py BabyAI-GoToDoor-v0 0
python analyze.py BabyAI-GoToDoor-v0 0 by_carrying_phase
python analyze.py BabyAI-GoToDoor-v0 0 action_frequency
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
