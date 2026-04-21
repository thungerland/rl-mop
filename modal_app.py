import modal

app = modal.App("babyai-gru-train")

# Persistent volume for storing checkpoints
checkpoints_volume = modal.Volume.from_name("rl-mop", create_if_missing=True)
CHECKPOINTS_PATH = "/checkpoints"

# Persistent volume for storing evaluation outputs (cache + CSV)
eval_volume = modal.Volume.from_name("rl-mop-eval", create_if_missing=True)
EVAL_OUTPUT_PATH = "/eval_output"

# Separate volume for normalised-loss training runs — keeps them isolated from originals
checkpoints_normalised_volume = modal.Volume.from_name("rl-mop-normalised", create_if_missing=True)
CHECKPOINTS_NORMALISED_PATH = "/checkpoints_normalised"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("git+https://github.com/mila-iqia/babyai.git")
    .add_local_dir(".", remote_path="/root/project", ignore=[
        "evaluation_cache",
        "evaluation_cache_normalised",
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
    volumes={
        CHECKPOINTS_PATH: checkpoints_volume,
        CHECKPOINTS_NORMALISED_PATH: checkpoints_normalised_volume,
    },
)
def train_run(
    task_id: str,
    trial: int,
    script: str = "train.py",
    config_path: str = "config.yaml",
    extra_args: str = None,
    checkpoint_dir: str = "/checkpoints",
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
        "--checkpoint_dir", checkpoint_dir,
    ]

    # Add any extra arguments
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    # Run the training script
    result = subprocess.run(cmd, capture_output=False, text=True)

    # Commit only the volume that was actually written to
    if checkpoint_dir == CHECKPOINTS_NORMALISED_PATH:
        checkpoints_normalised_volume.commit()
    else:
        checkpoints_volume.commit()

    return result.returncode


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        CHECKPOINTS_PATH: checkpoints_volume,
        CHECKPOINTS_NORMALISED_PATH: checkpoints_normalised_volume,
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
    checkpoint_dir: str = CHECKPOINTS_PATH,
    cache_dir: str = f"{EVAL_OUTPUT_PATH}/evaluation_cache",
    results_path: str = f"{EVAL_OUTPUT_PATH}/evaluation_results.csv",
):
    """
    Run batch_eval.py on Modal, reading checkpoints from a volume and
    writing evaluation_cache/ and evaluation_results.csv to the rl-mop-eval volume.

    For normalised runs pass:
        checkpoint_dir=CHECKPOINTS_NORMALISED_PATH  ("/checkpoints_normalised")
        cache_dir="/eval_output/evaluation_cache_normalised"
        results_path="/eval_output/evaluation_results_normalised.csv"
    """
    import sys
    import subprocess
    sys.path.append("/root/project")

    cmd = [
        sys.executable,
        "/root/project/batch_eval.py",
        "--checkpoint_dir", checkpoint_dir,
        "--cache_dir", cache_dir,
        "--results_path", results_path,
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
    checkpoint_dir: str = CHECKPOINTS_PATH,
    cache_dir: str = f"{EVAL_OUTPUT_PATH}/evaluation_cache",
    results_path: str = f"{EVAL_OUTPUT_PATH}/evaluation_results.csv",
):
    """
    Launch batch evaluation on Modal and print download instructions on completion.

    Usage examples:
        # Evaluate all checkpoints (1000 episodes each)
        modal run modal_app.py::eval_main

        # Normalised runs:
        modal run modal_app.py::eval_main \
          --checkpoint-dir "/checkpoints_normalised" \
          --cache-dir "/eval_output/evaluation_cache_normalised" \
          --results-path "/eval_output/evaluation_results_normalised.csv"

        # Then download results locally
        modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache
        modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv
    """
    print(f"Launching batch evaluation on Modal:")
    print(f"  Episodes per checkpoint: {num_episodes}")
    print(f"  Force re-evaluate: {force}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Results path: {results_path}")
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
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        results_path=results_path,
    )

    print("\nJob queued. Waiting for completion...")
    try:
        return_code = job.get()
        if return_code == 0:
            print(f"\n✓ Evaluation complete. Download results with:")
            print(f"  modal volume get rl-mop-eval {cache_dir} ./{cache_dir.split('/')[-1]}")
            print(f"  modal volume get rl-mop-eval {results_path} ./{results_path.split('/')[-1]}")
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
    checkpoint_base_local: str = "modal_checkpoints",
    checkpoint_dir: str = CHECKPOINTS_PATH,
    cache_dir: str = f"{EVAL_OUTPUT_PATH}/evaluation_cache",
    results_path: str = f"{EVAL_OUTPUT_PATH}/evaluation_results.csv",
    results_path_local: str = "evaluation_results.csv",
):
    """
    Discover unevaluated checkpoints locally and launch one Modal eval_run job
    per missing checkpoint in parallel.

    Compares a local checkpoints folder against a local results CSV using
    (task_id, trial, seed, update) as the canonical key.

    Usage examples:
        # Evaluate all missing checkpoints (1000 episodes each)
        modal run modal_app.py::eval_parallel

        # Normalised runs:
        modal run modal_app.py::eval_parallel \
          --checkpoint-base-local "modal_checkpoints_normalised" \
          --checkpoint-dir "/checkpoints_normalised" \
          --cache-dir "/eval_output/evaluation_cache_normalised" \
          --results-path "/eval_output/evaluation_results_normalised.csv" \
          --results-path-local "evaluation_results_normalised.csv"

        # After completion, download results:
        modal volume get rl-mop-eval /eval_output/evaluation_results.csv ./evaluation_results.csv
        modal volume get rl-mop-eval /eval_output/evaluation_cache ./evaluation_cache
    """
    import csv
    from pathlib import Path

    # Discover all local checkpoints
    checkpoint_base = Path(checkpoint_base_local)
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
    csv_path = Path(results_path_local)
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
            checkpoint_dir=checkpoint_dir,
            cache_dir=cache_dir,
            results_path=results_path,
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
    print(f"  modal volume get rl-mop-eval {results_path} ./{results_path.split('/')[-1]}")
    print(f"  modal volume get rl-mop-eval {cache_dir} ./{cache_dir.split('/')[-1]}")


@app.function(
    image=image,
    volumes={EVAL_OUTPUT_PATH: eval_volume},
    timeout=60 * 60 * 3,  # 3 hours — no GPU needed, just CPU + volume I/O
)
def extract_seed_metrics(
    task_id: str = "BabyAI-UnlockPickup-v0",
    trials: list = None,  # default: [20, 21, 22, 23, 26]
    output_csv: str = "/eval_output/eval_metrics_unlockpickup.csv",
    cache_dir: str = "/eval_output/evaluation_cache",
    phase_system: str = None,  # None = auto-detect from task_id via TASK_PHASE_SYSTEM
    update_filter: int = None,  # if set, only process this update number
):
    """
    Read every routing_data.json for the given task/trials from the eval volume,
    compute all per-seed quantities in one pass, and write a compact CSV.

    Per-row quantities:
      - Scalar metrics: task_id, trial, seed, update, lpc_alpha,
                        success_rate, path_ratio, mean_lpc
      - policy_complexity: I(S;A) = sum_s P(s)*KL(pi_hat(.|s)||P_a), include_mask applied
      - Per-phase correlation r/p values for all (phase, corr_type) pairs used by
        seed_agg_plots: lpc_entropy, lpc_kl_local, lpc_kl_global × phases

    phase_system: 'key_phase' (UnlockPickup, 4 phases) or 'unlock_phase' (OpenTwoDoors, 2 phases).
                  Defaults to auto-detect from task_id via TASK_PHASE_SYSTEM.

    After running, download with:
        modal volume get rl-mop-eval /eval_output/eval_metrics_unlockpickup.csv ./eval_metrics_unlockpickup.csv

    Usage:
        modal run modal_app.py::run_extract_seed_metrics
        modal run modal_app.py::run_extract_seed_metrics --task-id BabyAI-UnlockPickup-v0 --trials "[20,21,22]"
        modal run modal_app.py::run_extract_seed_metrics --task-id BabyAI-OpenTwoDoors-v0 --trials "20,21,22,23,24,25,26" --output-csv "/eval_output/eval_metrics_opentwodoors.csv"
    """
    import json
    import pathlib
    import sys
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/root/project")
    from plotting_utils import build_routing_data_tuples, compute_empirical_entropy
    from corr_plots import compute_corr, PHASE_LIST, TASK_PHASE_SYSTEM, _SUBPLOT_LINES
    from stats import _filter_by_phase

    if trials is None:
        trials = [20, 21, 22, 23, 26]

    # Load hyperparams from evaluation_results.csv on the volume
    results_csv = pathlib.Path(EVAL_OUTPUT_PATH) / "evaluation_results.csv"
    if not results_csv.exists():
        print(f"[error] {results_csv} not found on volume")
        return

    eval_df = pd.read_csv(results_csv)
    eval_df = eval_df[eval_df["task_id"] == task_id]

    # Build lookup: (trial, seed, update) -> row dict of hyperparams
    def _key(trial, seed, update):
        return (int(trial), int(seed) if seed is not None else None, int(update) if update is not None else None)

    hyperparam_map = {}
    for _, row in eval_df.iterrows():
        seed_val = None if pd.isna(row.get("seed")) else int(row["seed"])
        update_val = None if pd.isna(row.get("update")) else int(row["update"])
        hyperparam_map[_key(int(row["trial"]), seed_val, update_val)] = row.to_dict()

    # Phases and corr types
    resolved_phase_system = phase_system or TASK_PHASE_SYSTEM.get(task_id, "key_phase")
    phases = PHASE_LIST[resolved_phase_system]
    print(f"  phase_system: {resolved_phase_system} → phases: {phases}")
    corr_types = [spec["corr"] for spec in _SUBPLOT_LINES]  # lpc_entropy, lpc_kl_local, lpc_kl_global

    cache_base = pathlib.Path(cache_dir) / task_id
    rows = []

    for trial in trials:
        trial_dir = cache_base / f"trial_{trial}"
        if not trial_dir.exists():
            print(f"[skip] trial_{trial}: directory not found")
            continue

        # Discover seed/update combos
        seed_dirs = sorted(trial_dir.glob("seed_*"))
        if not seed_dirs:
            print(f"[skip] trial_{trial}: no seed_* directories found")
            continue

        for seed_dir in seed_dirs:
            try:
                seed_num = int(seed_dir.name.split("_")[1])
            except (IndexError, ValueError):
                print(f"[warn] Could not parse seed from {seed_dir.name}, skipping")
                continue

            # Find the largest update dir (same logic as discover_seeded_caches)
            update_dirs = []
            for upd_dir in seed_dir.glob("update_*"):
                if not upd_dir.is_dir():
                    continue
                try:
                    upd_num = int(upd_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                candidate = upd_dir / "routing_data.json"
                if candidate.exists():
                    update_dirs.append((upd_num, candidate))

            # Sort by update number (ascending) to process ALL checkpoints for trial 20
            update_dirs.sort(key=lambda x: x[0])
            if update_filter is not None:
                update_dirs = [(u, p) for u, p in update_dirs if u == update_filter]

            for upd_num, cache_path in update_dirs:
                label = f"trial={trial} seed={seed_num} update={upd_num}"
                print(f"  Processing {label}...", end=" ", flush=True)

                try:
                    with open(cache_path) as f:
                        cache = json.load(f)
                    routing_data = build_routing_data_tuples(cache)
                except Exception as e:
                    print(f"ERROR loading: {e}")
                    continue

                print(f"{len(routing_data)} timesteps", end=" ", flush=True)

                # Scalar metrics from cache
                metrics = cache.get("metrics", {})
                success_rate = metrics.get("success_rate", float("nan"))
                path_ratio = metrics.get("path_ratio", float("nan"))
                mean_lpc = metrics.get("mean_lpc", float("nan"))

                # lpc_alpha from hyperparam_map (try to match, fallback to eval_df)
                hp = hyperparam_map.get(_key(trial, seed_num, upd_num))
                if hp is None:
                    # Try matching without update (for final-only entries)
                    trial_rows = eval_df[(eval_df["trial"] == trial)]
                    lpc_alpha = float(trial_rows["lpc_alpha"].iloc[0]) if not trial_rows.empty else float("nan")
                else:
                    lpc_alpha = float(hp.get("lpc_alpha", float("nan")))

                # Policy complexity: compute_empirical_entropy over all timesteps
                emp = compute_empirical_entropy(routing_data, alpha=0.5, min_visits=5)
                KL_s = emp["KL_s"]
                P_s = emp["P_s"]
                include_mask = emp["include_mask"]
                global_P_a = emp["P_a"]

                policy_complexity = float(sum(
                    P_s[s] * KL_s[s] for s in KL_s if include_mask.get(s, False)
                ))

                # Per-phase correlation r/p
                row = {
                    "task_id": task_id,
                    "trial": trial,
                    "seed": seed_num,
                    "update": upd_num,
                    "lpc_alpha": lpc_alpha,
                    "success_rate": success_rate,
                    "path_ratio": path_ratio,
                    "mean_lpc": mean_lpc,
                    "policy_complexity": policy_complexity,
                }

                for phase in phases:
                    for corr_type in corr_types:
                        try:
                            res = compute_corr(
                                routing_data, corr_type, phase,
                                dist_field=None, global_P_a=global_P_a,
                            )
                            row[f"r_{phase}_{corr_type}"] = res["r"]
                            row[f"p_{phase}_{corr_type}"] = res["p"]
                        except Exception as e:
                            print(f"\n    [warn] {label} phase={phase} corr={corr_type}: {e}")
                            row[f"r_{phase}_{corr_type}"] = float("nan")
                            row[f"p_{phase}_{corr_type}"] = float("nan")

                # Per-phase scalar metrics: I(S;A), H̄(A|S), mean LPC
                # global_P_a used as KL reference for consistency across phases/alphas
                for phase in phases:
                    phase_data = list(_filter_by_phase(routing_data, phase))
                    if not phase_data:
                        row[f"policy_complexity_{phase}"] = float("nan")
                        row[f"mean_entropy_{phase}"] = float("nan")
                        row[f"mean_lpc_{phase}"] = float("nan")
                        continue

                    emp_phase = compute_empirical_entropy(
                        phase_data, alpha=0.5, min_visits=5, P_a=global_P_a,
                    )
                    KL_s_p = emp_phase["KL_s"]
                    H_s_p  = emp_phase["H_s"]
                    P_s_p  = emp_phase["P_s"]
                    mask_p = emp_phase["include_mask"]

                    row[f"policy_complexity_{phase}"] = float(
                        sum(P_s_p[s] * KL_s_p[s] for s in KL_s_p if mask_p.get(s, False))
                    )
                    row[f"mean_entropy_{phase}"] = float(
                        sum(P_s_p[s] * H_s_p[s] for s in H_s_p if mask_p.get(s, False))
                    )
                    lpc_vals = [d["lpc"] for d in phase_data if d.get("lpc") is not None]
                    row[f"mean_lpc_{phase}"] = float(np.mean(lpc_vals)) if lpc_vals else float("nan")

                rows.append(row)
                print("done")

    if not rows:
        print("[error] No rows collected — check that seeded caches exist on the volume.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    eval_volume.commit()
    print(f"\nWrote {len(out_df)} rows to {output_csv}")
    print("\nDownload with:")
    print(f"  modal volume get rl-mop-eval {output_csv} ./{pathlib.Path(output_csv).name}")


@app.local_entrypoint()
def run_extract_seed_metrics(
    task_id: str = "BabyAI-UnlockPickup-v0",
    trials: str = "20,21,22,23,26",
    output_csv: str = "/eval_output/eval_metrics_unlockpickup.csv",
    cache_dir: str = "/eval_output/evaluation_cache",
    phase_system: str = None,  # None = auto-detect from task_id
    update: int = None,  # if set, only process this update number
):
    """
    Launch extract_seed_metrics on Modal.

    Usage:
        modal run modal_app.py::run_extract_seed_metrics
        modal run modal_app.py::run_extract_seed_metrics --trials "20,21,22"
        # Normalised runs:
        modal run modal_app.py::run_extract_seed_metrics \
          --cache-dir "/eval_output/evaluation_cache_normalised" \
          --output-csv "/eval_output/eval_metrics_unlockpickup_normalised.csv"
        # OpenTwoDoors:
        modal run modal_app.py::run_extract_seed_metrics \
          --task-id BabyAI-OpenTwoDoors-v0 \
          --trials "20,21,22,23,24,25,26" \
          --output-csv "/eval_output/eval_metrics_opentwodoors.csv"
    """
    trials_list = [int(t.strip()) for t in trials.split(",")]
    print(f"Launching extract_seed_metrics on Modal:")
    print(f"  task_id:      {task_id}")
    print(f"  trials:       {trials_list}")
    print(f"  cache_dir:    {cache_dir}")
    print(f"  output:       {output_csv}")
    print(f"  phase_system: {phase_system or 'auto'}")

    result = extract_seed_metrics.remote(
        task_id=task_id,
        trials=trials_list,
        output_csv=output_csv,
        cache_dir=cache_dir,
        phase_system=phase_system,
        update_filter=update,
    )

    from pathlib import Path
    print("\nDone. Download the CSV with:")
    print(f"  modal volume get rl-mop-eval {output_csv} ./{Path(output_csv).name}")


@app.local_entrypoint()
def main_seeds(
    task_ids: str,
    trials: str,
    seeds: str,
    script: str = "train_mop.py",
    extra_args: str = None,
    checkpoint_interval: int = None,
    checkpoint_dir: str = "/checkpoints",
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
    print(f"  Checkpoint dir: {checkpoint_dir}")

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
            checkpoint_dir=checkpoint_dir,
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
    checkpoint_dir: str = "/checkpoints",
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
    print(f"  Checkpoint dir: {checkpoint_dir}")

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
            checkpoint_dir=checkpoint_dir,
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
