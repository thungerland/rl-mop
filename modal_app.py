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
    .add_local_dir(".", remote_path="/root/project", ignore=[
        "evaluation_cache",
        "evaluation_results.csv",
        "modal_checkpoints",
        "checkpoints",
        "plots",
        "wandb",
        "__pycache__",
        "__marimo__",
        "*.ipynb",
    ])
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
    seed: int = None,
    update: int = None,
    final_only: bool = False,
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
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if update is not None:
        cmd.extend(["--update", str(update)])
    if final_only:
        cmd.append("--final-only")
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
def eval_parallel(
    num_episodes: int = 1000,
    force: bool = False,
    task: str = None,
    trial: int = None,
    seed: int = None,
    update: int = None,
    final_only: bool = False,
    skip_routing: bool = False,
):
    """
    Discover unevaluated checkpoints locally and launch one Modal eval_run job
    per missing checkpoint in parallel.

    Compares modal_checkpoints/ against evaluation_results.csv using
    (task_id, trial, seed, update) as the canonical key to correctly handle the
    path prefix difference between local modal_checkpoints/ and Modal's /checkpoints/.

    Supports two checkpoint layouts:
      Old: task_id/trial_N/checkpoint_final.pt
      New: task_id/trial_N/seed_S/checkpoint_<U>.pt

    checkpoint_final.pt aliases are skipped (numbered files are canonical).

    Usage examples:
        # Evaluate all missing checkpoints (1000 episodes each)
        modal run modal_app.py::eval_parallel

        # Force re-evaluate everything
        modal run modal_app.py::eval_parallel --force

        # Only evaluate a specific task/trial/seed/update slice
        modal run modal_app.py::eval_parallel --task "BabyAI-UnlockPickup-v0" --trial 20 --update 1500

        # After completion, download results:
        modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv
        modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache
    """
    import csv
    from pathlib import Path

    # Discover all local checkpoints
    checkpoint_base = Path("modal_checkpoints")
    if not checkpoint_base.exists():
        print(f"Error: '{checkpoint_base}' directory not found. Download checkpoints first.")
        return

    glob_pattern = "**/checkpoint_final.pt" if final_only else "**/checkpoint_*.pt"
    local_checkpoints = []  # list of (task_id, trial_num, seed_num, update_num)
    for checkpoint_path in checkpoint_base.glob(glob_pattern):
        parts = checkpoint_path.relative_to(checkpoint_base).parts

        if len(parts) == 3:
            # Old layout: task_id/trial_N/checkpoint_final.pt
            task_id_p, trial_str, filename = parts
            seed_num = None
        elif len(parts) == 4:
            # New layout: task_id/trial_N/seed_S/checkpoint_<U>.pt
            task_id_p, trial_str, seed_str, filename = parts
            try:
                seed_num = int(seed_str.split("_")[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse seed from {checkpoint_path}, skipping.")
                continue
        else:
            print(f"Warning: Unexpected path depth {checkpoint_path}, skipping.")
            continue

        name = filename.replace(".pt", "")
        update_str = name.split("_", 1)[1]
        if update_str == "final":
            if len(parts) == 4 and not final_only:
                continue  # skip alias in default mode — numbered file is canonical
            # final_only mode or old 3-part: include; update resolved from config at eval time
            update_num = None
        else:
            if final_only:
                continue
            try:
                update_num = int(update_str)
            except ValueError:
                print(f"Warning: Could not parse update from {checkpoint_path}, skipping.")
                continue

        try:
            trial_num = int(trial_str.split("_")[1])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse trial from {checkpoint_path}, skipping.")
            continue

        # Apply optional filters
        if task and task_id_p != task:
            continue
        if trial is not None and trial_num != trial:
            continue
        if seed is not None and seed_num != seed:
            continue
        if update is not None and update_num != update:
            continue

        local_checkpoints.append((task_id_p, trial_num, seed_num, update_num))

    local_checkpoints.sort(key=lambda x: (x[0], x[1], x[2] if x[2] is not None else -1, x[3] if x[3] is not None else -1))

    # Load evaluated keys from CSV using (task_id, trial, seed, update) columns
    evaluated = set()
    csv_path = Path("evaluation_results.csv")
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                evaluated.add((
                    row["task_id"],
                    str(row["trial"]),
                    str(row.get("seed", "")),
                    str(row.get("update", "")),
                ))

    def _key(t, n, s, u):
        return (t, str(n), str(s) if s is not None else "", str(u) if u is not None else "")

    # Compute missing set
    if force:
        missing = list(local_checkpoints)
    else:
        missing = [c for c in local_checkpoints if _key(*c) not in evaluated]

    # Print summary
    print(f"Local checkpoints found: {len(local_checkpoints)}")
    print(f"Already evaluated:       {len(evaluated)}")
    print(f"Jobs to launch:          {len(missing)}")
    if force:
        print("  (--force: re-evaluating all)")
    print()

    if not missing:
        print("Nothing to evaluate. Use --force to re-evaluate existing results.")
        return

    print("Checkpoints to evaluate:")
    for task_id_p, trial_num, seed_num, update_num in missing:
        seed_part = f"/seed_{seed_num}" if seed_num is not None else ""
        update_part = f"@{update_num}" if update_num is not None else ""
        print(f"  - {task_id_p}/trial_{trial_num}{seed_part}{update_part}")
    print()

    # Spawn one eval_run per missing checkpoint
    jobs = []
    for task_id_p, trial_num, seed_num, update_num in missing:
        seed_part = f"/seed_{seed_num}" if seed_num is not None else ""
        update_part = f"@{update_num}" if update_num is not None else ""
        print(f"  Queuing: {task_id_p}/trial_{trial_num}{seed_part}{update_part}")
        job = eval_run.spawn(
            num_episodes=num_episodes,
            force=force,
            task=task_id_p,
            trial=trial_num,
            seed=seed_num,
            update=update_num,
            final_only=final_only,
            skip_routing=skip_routing,
        )
        jobs.append((task_id_p, trial_num, seed_num, update_num, job))

    print(f"\n{len(jobs)} job(s) queued. Waiting for completion...")
    print("(You can Ctrl+C to detach — jobs will continue running on Modal)\n")

    # Collect results
    failures = []
    for task_id_p, trial_num, seed_num, update_num, job in jobs:
        seed_part = f"/seed_{seed_num}" if seed_num is not None else ""
        update_part = f"@{update_num}" if update_num is not None else ""
        label = f"{task_id_p}/trial_{trial_num}{seed_part}{update_part}"
        try:
            return_code = job.get()
            if return_code == 0:
                print(f"  Success: {label}")
            else:
                print(f"  Failed (exit code {return_code}): {label}")
                failures.append(label)
        except Exception as e:
            print(f"  Error: {label} - {e}")
            failures.append(label)

    print()
    if not failures:
        print(f"All {len(jobs)} job(s) completed successfully.")
    else:
        print(f"{len(jobs) - len(failures)}/{len(jobs)} job(s) succeeded.")
        print("Failed checkpoints:")
        for label in failures:
            print(f"  - {label}")

    print("\nDownload results with:")
    print("  modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv")
    print("  modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache")


@app.local_entrypoint()
def main_seeds(
    task_ids: str,
    trials: str,
    seeds: str,
    script: str = "train_mop.py",
    extra_args: str = None,
    checkpoint_interval: int = None,
):
    """
    Launch training runs for every (task_id, trial, seed) combination in parallel.

    Usage examples:
        # Train 10 seeds for one task/trial with periodic checkpoints
        modal run modal_app.py::main_seeds \
          --task-ids "BabyAI-UnlockPickup-v0" \
          --trials "20" \
          --seeds "0,1,2,3,4,5,6,7,8,9" \
          --checkpoint-interval 250 \
          --extra-args "--lpc_alpha 1e-4 --num_updates 5000"

        # Multiple trials × seeds
        modal run modal_app.py::main_seeds \
          --task-ids "BabyAI-UnlockPickup-v0" \
          --trials "20,21" \
          --seeds "0,1,2,3"
    """
    from itertools import product as iproduct

    task_id_list = [t.strip() for t in task_ids.split(",")]
    trial_list = [int(t.strip()) for t in trials.split(",")]
    seed_list = [int(s.strip()) for s in seeds.split(",")]

    total = len(task_id_list) * len(trial_list) * len(seed_list)
    print(f"Launching {total} training runs on Modal:")
    print(f"  Script: {script}")
    print(f"  Task IDs: {task_id_list}")
    print(f"  Trials: {trial_list}")
    print(f"  Seeds: {seed_list}")
    if checkpoint_interval:
        print(f"  Checkpoint interval: every {checkpoint_interval} updates")
    if extra_args:
        print(f"  Extra args: {extra_args}")

    jobs = []
    for task_id, trial, seed_val in iproduct(task_id_list, trial_list, seed_list):
        run_extra = f"{extra_args or ''} --seed {seed_val}".strip()
        if checkpoint_interval:
            run_extra += f" --checkpoint_interval {checkpoint_interval}"
        print(f"  - Queuing: {task_id} trial {trial} seed {seed_val}")
        job = train_run.spawn(
            task_id=task_id,
            trial=trial,
            script=script,
            extra_args=run_extra,
        )
        jobs.append((task_id, trial, seed_val, job))

    print(f"\n{len(jobs)} jobs queued. Waiting for completion...")

    for task_id, trial, seed_val, job in jobs:
        try:
            return_code = job.get()
            status = "✓ Success" if return_code == 0 else f"✗ Failed (code {return_code})"
            print(f"{status}: {task_id} trial {trial} seed {seed_val}")
        except Exception as e:
            print(f"✗ Error: {task_id} trial {trial} seed {seed_val} - {e}")

    print("\nAll experiments completed!")


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
