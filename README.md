# rl-mop
Code for an adaptive imitation learning model to investigate deep learning routing patterns for complex task learning e.g. gridworld.

## Workflow

### 1. Train a Model

**Local training:**
```bash
python train_mop.py --config config.yaml --task_id BabyAI-GoToObj-v0 --trial 0
```

**Modal (cloud) training:**
```bash
modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0" --script train_mop.py
```

### 2. Get Checkpoints

Local training saves checkpoints automatically to `checkpoints/`.

For Modal training, download checkpoints from the volume:
```bash
# List available checkpoints
modal volume ls rl-mop

# Download to local folder
modal volume get rl-mop / ./modal_checkpoints/

# Copy to checkpoints folder
cp -r modal_checkpoints/* checkpoints/
```

### 3. Evaluate

**Basic evaluation (performance metrics):**
```bash
python eval_mop.py --checkpoint checkpoints/BabyAI-GoToObj-v0/trial_0/checkpoint_final.pt --num_episodes 100
```

**Routing visualization (marimo notebook):**
```bash
marimo edit routing_viz.py --watch
```
This opens an interactive notebook showing:
- Environment layout
- Expert routing heatmaps by grid position
- Routing weight distributions
