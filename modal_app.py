import modal

app = modal.App("babyai-gru-train")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("git+https://github.com/mila-iqia/babyai.git")
    .add_local_dir(".", remote_path="/root/project")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_run(
    task_id: str,
    trial: int,
    num_envs: int = None,
    hidden_size: int = None,
    unroll_len: int = None,
    num_updates: int = None,
    lr: float = None,
    seed: int = None,
    config_path: str = "config.yaml",
):
    """
    Run a single training experiment with specified parameters.

    All hyperparameters default to values in config.yaml unless explicitly overridden.

    Args:
        task_id: BabyAI task ID (required)
        trial: Trial number for this run (required)
        num_envs: Number of parallel environments (optional override)
        hidden_size: GRU hidden size (optional override)
        unroll_len: Unroll length for BPTT (optional override)
        num_updates: Number of training updates (optional override)
        lr: Learning rate (optional override)
        seed: Random seed (optional override)
        config_path: Path to config file (default: config.yaml)
    """
    import sys
    import subprocess
    sys.path.append("/root/project")

    # Build command-line arguments
    cmd = [
        sys.executable,
        "/root/project/train.py",
        "--config", f"/root/project/{config_path}",
        "--task_id", task_id,
        "--trial", str(trial),
    ]

    # Add optional overrides (only if specified)
    if num_envs is not None:
        cmd.extend(["--num_envs", str(num_envs)])
    if hidden_size is not None:
        cmd.extend(["--hidden_size", str(hidden_size)])
    if unroll_len is not None:
        cmd.extend(["--unroll_len", str(unroll_len)])
    if num_updates is not None:
        cmd.extend(["--num_updates", str(num_updates)])
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Run the training script
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


@app.local_entrypoint()
def main(
    task_ids: str,
    trials: str,
    num_envs: int = None,
    hidden_size: int = None,
    unroll_len: int = None,
    num_updates: int = None,
    lr: float = None,
):
    """
    Launch multiple training runs on Modal.

    All runs use config.yaml as the base configuration, with optional overrides.

    Usage examples:
        # Single run with config.yaml defaults
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0"

        # Multiple trials for one task
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0,1,2,3,4"

        # Multiple tasks, multiple trials
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0,BabyAI-ActionObjDoor-v0" --trials "0,1,2"

        # With hyperparameter overrides
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0" --hidden-size 256 --unroll-len 40

        # Override learning rate for all runs
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0,1,2" --lr 5e-4
    """
    # Parse comma-separated lists
    task_id_list = [t.strip() for t in task_ids.split(",")]
    trial_list = [int(t.strip()) for t in trials.split(",")]

    print(f"Launching {len(task_id_list) * len(trial_list)} experiments on Modal:")
    print(f"  Task IDs: {task_id_list}")
    print(f"  Trials: {trial_list}")

    # Show any hyperparameter overrides
    overrides = []
    if num_envs is not None:
        overrides.append(f"num_envs={num_envs}")
    if hidden_size is not None:
        overrides.append(f"hidden_size={hidden_size}")
    if unroll_len is not None:
        overrides.append(f"unroll_len={unroll_len}")
    if num_updates is not None:
        overrides.append(f"num_updates={num_updates}")
    if lr is not None:
        overrides.append(f"lr={lr}")

    if overrides:
        print(f"  Overrides: {', '.join(overrides)}")
    else:
        print(f"  Using config.yaml defaults")

    # Launch all runs in parallel using Modal
    # Each (task_id, trial) combination runs in a separate container
    from itertools import product

    jobs = []
    for task_id, trial in product(task_id_list, trial_list):
        print(f"  - Queuing: {task_id} trial {trial}")
        job = train_run.spawn(
            task_id=task_id,
            trial=trial,
            num_envs=num_envs,
            hidden_size=hidden_size,
            unroll_len=unroll_len,
            num_updates=num_updates,
            lr=lr,
        )
        jobs.append((task_id, trial, job))

    print(f"\n{len(jobs)} jobs queued. Waiting for completion...")

    # Wait for all jobs and report results
    for task_id, trial, job in jobs:
        try:
            return_code = job.get()
            status = "✓ Success" if return_code == 0 else f"✗ Failed (code {return_code})"
            print(f"{status}: {task_id} trial {trial}")
        except Exception as e:
            print(f"✗ Error: {task_id} trial {trial} - {e}")

    print("\nAll experiments completed!")
