import modal

app = modal.App("babyai-gru-train")

# Persistent volume for storing checkpoints
checkpoints_volume = modal.Volume.from_name("rl-mop", create_if_missing=True)
CHECKPOINTS_PATH = "/checkpoints"

# Persistent volume for storing evaluation outputs (cache + CSV)
eval_volume = modal.Volume.from_name("rl-mop-eval", create_if_missing=True)
EVAL_OUTPUT_PATH = "/eval_output"

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
    timeout=60 * 60 * 24, # 24 hours which is the max allowed by Modal for GPU functions - use checkpoints and reentry to handle longer experiments
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={CHECKPOINTS_PATH: checkpoints_volume},
)
def train_run(
    task_id: str,
    trial: int,
    script: str = "train.py",
    config_path: str = "config.yaml",
    extra_args: str = None,
):
    """
    Run a single training experiment with specified parameters.

    Args:
        task_id: BabyAI task ID (required)
        trial: Trial number for this run (required)
        script: Python script to run, e.g. "train.py" or "train_mop.py" (default: "train.py")
        config_path: Path to config file (default: config.yaml)
        extra_args: Additional CLI arguments as a string, e.g. "--num_envs 16 --lr 1e-3"
    """
    import sys
    import subprocess
    import shlex
    sys.path.append("/root/project")

    script_path = f"/root/project/{script}"

    # Build command-line arguments
    cmd = [
        sys.executable,
        script_path,
        "--config", f"/root/project/{config_path}",
        "--task_id", task_id,
        "--trial", str(trial),
        "--checkpoint_dir", "/checkpoints",
    ]

    # Add any extra arguments
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    # Run the training script
    result = subprocess.run(cmd, capture_output=False, text=True)

    # Commit the volume to persist checkpoints
    checkpoints_volume.commit()

    return result.returncode


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        CHECKPOINTS_PATH: checkpoints_volume,
        EVAL_OUTPUT_PATH: eval_volume,
    },
)
def eval_run(
    num_episodes: int = 1000,
    force: bool = False,
    task: str = None,
    trial: int = None,
    skip_routing: bool = False,
):
    """
    Run batch_eval.py on Modal, reading checkpoints from the rl-mop volume
    and writing evaluation_cache/ and evaluation_results.csv to the rl-mop-eval volume.
    """
    import sys
    import subprocess
    sys.path.append("/root/project")

    cmd = [
        sys.executable,
        "/root/project/batch_eval.py",
        "--checkpoint_dir", CHECKPOINTS_PATH,
        "--cache_dir", f"{EVAL_OUTPUT_PATH}/evaluation_cache",
        "--results_path", f"{EVAL_OUTPUT_PATH}/evaluation_results.csv",
        "--num_episodes", str(num_episodes),
    ]

    if force:
        cmd.append("--force")
    if task:
        cmd.extend(["--task", task])
    if trial is not None:
        cmd.extend(["--trial", str(trial)])
    if skip_routing:
        cmd.append("--skip_routing")

    result = subprocess.run(cmd, capture_output=False, text=True)

    # Commit results to the eval volume so they are retrievable
    eval_volume.commit()

    return result.returncode


@app.local_entrypoint()
def eval_main(
    num_episodes: int = 1000,
    force: bool = False,
    task: str = None,
    trial: int = None,
    skip_routing: bool = False,
):
    """
    Launch batch evaluation on Modal and print download instructions on completion.

    Usage examples:
        # Evaluate all checkpoints (1000 episodes each)
        modal run modal_app.py::eval_main

        # Force re-evaluation
        modal run modal_app.py::eval_main --force

        # Custom episode count
        modal run modal_app.py::eval_main --num-episodes 500

        # Evaluate a single task
        modal run modal_app.py::eval_main --task "BabyAI-UnlockPickup-v0"

        # Then download results locally
        modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache
        modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv
    """
    print(f"Launching batch evaluation on Modal:")
    print(f"  Episodes per checkpoint: {num_episodes}")
    print(f"  Force re-evaluate: {force}")
    if task:
        print(f"  Task filter: {task}")
    if trial is not None:
        print(f"  Trial filter: {trial}")
    if skip_routing:
        print(f"  Skipping routing cache")

    job = eval_run.spawn(
        num_episodes=num_episodes,
        force=force,
        task=task,
        trial=trial,
        skip_routing=skip_routing,
    )

    print("\nJob queued. Waiting for completion...")
    try:
        return_code = job.get()
        if return_code == 0:
            print("\n✓ Evaluation complete. Download results with:")
            print("  modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache")
            print("  modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv")
        else:
            print(f"\n✗ Evaluation failed (exit code {return_code}).")
    except Exception as e:
        print(f"\n✗ Job error: {e}")


@app.local_entrypoint()
def main(
    task_ids: str,
    trials: str,
    script: str = "train_mop.py",
    extra_args: str = None,
):
    """
    Launch multiple training runs on Modal.

    All runs use config.yaml as the base configuration, with optional overrides
    passed via --extra-args.

    Usage examples:
        # Single run with config.yaml defaults (train.py)
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0"

        # Run MoE training (train_mop.py)
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0" --script train_mop.py

        # Multiple trials for one task
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0,1,2,3,4"

        # Multiple tasks, multiple trials
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0,BabyAI-ActionObjDoor-v0" --trials "0,1,2"

        # With hyperparameter overrides via extra-args
        modal run modal_app.py --task-ids "BabyAI-GoToObj-v0" --trials "0" --extra-args "--hidden_size 256 --unroll_len 40"

        # MoE with custom architecture (single layer)
        modal run modal_app.py --task-ids BabyAI-GoToObj-v0 --trials 0 --script train_mop.py --extra-args "--expert_hidden_sizes [32,64,128]"

        # MoE with multi-layer architecture
        modal run modal_app.py --task-ids BabyAI-GoToObj-v0 --trials 0 --script train_mop.py --extra-args "--expert_hidden_sizes [[32,64],[64,128],[128,256]]"
    """
    # Parse comma-separated lists
    task_id_list = [t.strip() for t in task_ids.split(",")]
    trial_list = [int(t.strip()) for t in trials.split(",")]

    print(f"Launching {len(task_id_list) * len(trial_list)} experiments on Modal:")
    print(f"  Script: {script}")
    print(f"  Task IDs: {task_id_list}")
    print(f"  Trials: {trial_list}")
    if extra_args:
        print(f"  Extra args: {extra_args}")
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
            script=script,
            extra_args=extra_args,
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
