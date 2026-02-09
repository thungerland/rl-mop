# Running Experiments

## Overview

Your code has been refactored to make it easy to run multiple experiments with different configurations:

1. **config.yaml** - Base configuration file with hyperparameter defaults
2. **train.py** - Modified to accept CLI arguments that override config.yaml
3. **modal_app.py** - Modal launcher that can run multiple experiments in parallel

## Quick Start

### Single Run
```bash
modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0"
```

### Multiple Trials for One Task
```bash
modal run modal_app.py --task-ids "BabyAI-ActionObjDoor-v0" --trials "0,1,2,3,4"
```

### Multiple Tasks, Multiple Trials
```bash
modal run modal_app.py \
  --task-ids "BabyAI-GoToObj-v0,BabyAI-ActionObjDoor-v0" \
  --trials "0,1,2"
```

This launches 2 tasks × 3 trials = 6 experiments in parallel on Modal.

### With Hyperparameter Overrides
```bash
modal run modal_app.py \
  --task-ids "BabyAI-GoToObj-v0" \
  --trials "0" \
  --hidden-size 256 \
  --unroll-len 40 \
  --lr 5e-4
```

## Configuration Management

### Using config.yaml (Recommended)
Edit [config.yaml](config.yaml) for your default hyperparameters:
- `task_id`: Default task
- `num_envs`: Number of parallel environments
- `hidden_size`: GRU hidden dimension
- `unroll_len`: BPTT unroll length
- `num_updates`: Total training updates
- `lr`: Learning rate

### Using Custom Configs
You can create multiple config files in `configs/`:
- [configs/quick_test.yaml](configs/quick_test.yaml) - Fast debugging runs
- [configs/large_scale.yaml](configs/large_scale.yaml) - Full-scale training

Then specify with `--config-path` (TODO: need to add this parameter to modal_app.py if you want it)

## Local Testing

You can also run locally without Modal:

```bash
# Using config.yaml defaults
python train.py

# With overrides
python train.py --task_id "BabyAI-GoToObj-v0" --trial 0 --hidden_size 128

# Using a different config
python train.py --config configs/quick_test.yaml --trial 1
```

## Available Hyperparameters

All parameters can be overridden via command line:
- `--task_id`: BabyAI environment ID
- `--trial`: Trial number (for WandB naming and random seed tracking)
- `--num_envs`: Number of parallel environments
- `--hidden_size`: GRU hidden dimension
- `--unroll_len`: BPTT unroll length
- `--num_updates`: Total training updates
- `--lr`: Learning rate
- `--seed`: Random seed
- `--log_interval`: How often to log metrics
- `--max_steps`: Max steps per episode before truncation (default: environment-specific, e.g. 576 for GoToImpUnlock). Increase for complex multi-room environments to allow more exploration.

## Tips

1. **Start small**: Test with `configs/quick_test.yaml` before full runs
2. **Monitor WandB**: All runs automatically log to your WandB project
3. **Parallel vs Sequential**: Modal runs all jobs in parallel by default
4. **Cost management**: Each Modal container uses a T4 GPU (check your Modal limits)
