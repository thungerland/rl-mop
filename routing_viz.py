import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from plotting_utils import (
        plot_overall_routing,
        plot_grouped_routing,
        group_routing_data,
        get_available_analyses,
    )

    return Path, get_available_analyses, group_routing_data, mo, np, plot_grouped_routing, plot_overall_routing, plt, torch


@app.cell
def _(Path, mo):
    # Find available checkpoints (check both directories)
    checkpoint_dirs = [Path("modal_checkpoints"), Path("checkpoints")]
    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            for cp in checkpoint_dir.glob("**/*.pt"):
                checkpoints.append((checkpoint_dir, cp))

    checkpoint_options = {
        str(cp.relative_to(base)): str(cp)
        for base, cp in checkpoints
    }

    # UI for checkpoint selection
    checkpoint_dropdown = mo.ui.dropdown(
        options=checkpoint_options,
        label="Select Checkpoint",
        value=list(checkpoint_options.keys())[0] if checkpoint_options else None
    )

    # Cache toggle
    use_cache_toggle = mo.ui.checkbox(
        label="Use cached routing data (if available)",
        value=True
    )

    # Number of episodes for evaluation (used when not using cache)
    num_episodes_slider = mo.ui.slider(
        start=10, stop=10000, value=50, step=100,
        label="Number of Episodes (for fresh evaluation)"
    )

    # Number of parallel environments
    num_envs_slider = mo.ui.slider(
        start=1, stop=16, value=8, step=1,
        label="Parallel Environments"
    )

    mo.vstack([
        mo.md("## Configuration"),
        checkpoint_dropdown,
        use_cache_toggle,
        num_episodes_slider,
        num_envs_slider,
    ])
    return checkpoint_dropdown, num_envs_slider, num_episodes_slider, use_cache_toggle


@app.cell
def _(Path, checkpoint_dropdown, mo, np, num_envs_slider, num_episodes_slider, torch, use_cache_toggle):
    import json
    from mixture_of_experts import MixtureOfExpertsPolicy, reset_hidden_on_done, compute_lpc
    from eval_mop import EvalVectorEnv, encode_obs_batch, load_checkpoint, evaluate

    mo.stop(not checkpoint_dropdown.value, mo.md("**Select a checkpoint above to begin.**"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to get config
    policy, config, _ = load_checkpoint(checkpoint_dropdown.value, device)
    task_id = config['task_id']
    trial = config['trial']

    # Check for cached routing data
    cache_path = Path("evaluation_cache") / task_id / f"trial_{trial}" / "routing_data.json"
    cache_exists = cache_path.exists()

    if use_cache_toggle.value and cache_exists:
        # Load from cache
        with open(cache_path) as f:
            cached = json.load(f)

        # Convert cached data back to expected format.
        # Support three formats:
        #   v1: per-timestep env_context embedded in each record
        #   v2: deduplicated episodes list, each record has an 'episode' index
        #   legacy: no env_context at all
        episodes = cached.get('episodes')
        routing_data = [
            (
                tuple(r['position']),
                {k: np.array(v) for k, v in r['layer_routing'].items()},
                r['lpc'],
                episodes[r['episode']] if episodes is not None else r.get('env_context', {}),
                r.get('carrying', 0),
            )
            for r in cached['routing_data']
        ]
        metrics = {
            'success_rate': cached['metrics']['success_rate'],
            'path_ratio': cached['metrics']['path_ratio'],
            'mean_lpc': cached['metrics']['mean_lpc'],
            'total_episodes': cached['num_episodes'],
        }
        data_source = f"Loaded from cache ({cached['evaluated_at'][:10]})"
    else:
        # Run fresh evaluation
        vec_env = EvalVectorEnv(task_id, num_envs_slider.value, device, lang_dim=config.get('lang_dim', 32))
        metrics, routing_data = evaluate(
            policy, vec_env, num_episodes_slider.value, device
        )
        data_source = "Fresh evaluation"

    cache_status = "available" if cache_exists else "not found"

    mo.md(f"""
    ## Evaluation Results
    - **Source**: {data_source}
    - **Cache**: {cache_status}
    - **Task**: {task_id}
    - **Episodes**: {metrics['total_episodes']}
    - **Success Rate**: {metrics['success_rate']:.1%}
    - **Path Ratio**: {metrics['path_ratio']:.2f}
    - **Mean LPC**: {metrics['mean_lpc']:.2f}
    - **Routing Samples**: {len(routing_data)}
    """)
    return routing_data, task_id, trial


@app.cell
def _(get_available_analyses, mo, routing_data):
    # Determine available analysis types based on the data
    available_analyses = get_available_analyses(routing_data)

    analysis_options = {
        'Overall (all data)': 'overall',
        'By starting room': 'by_starting_room',
        'By door location': 'by_door_location',
        'By door & box row': 'by_door_and_box_row',
        'By carrying phase': 'by_carrying_phase',
    }

    # Filter to only available analyses
    available_options = {k: v for k, v in analysis_options.items() if v in available_analyses}

    analysis_dropdown = mo.ui.dropdown(
        options=available_options,
        label="Analysis Type",
        value='Overall (all data)'
    )

    mo.vstack([
        mo.md("## Analysis Options"),
        analysis_dropdown,
    ])
    return (analysis_dropdown,)


@app.cell
def _(group_routing_data, routing_data, task_id):
    import gymnasium as gym

    sample_env = gym.make(task_id, render_mode="rgb_array")
    uw = sample_env.unwrapped

    # Collect room keys from routing data for per-room image generation
    groups = group_routing_data(routing_data, 'agent_start_room')
    target_rooms = set(groups.keys())

    # Collect door-location keys from routing data for per-door-location image generation
    door_groups = group_routing_data(routing_data, 'door_location')
    target_door_locations = set(door_groups.keys())

    # Collect combined door+box-row keys for per-combination image generation
    door_box_groups = group_routing_data(routing_data, 'door_and_box_row')
    target_door_box_locations = set(door_box_groups.keys())

    # Generate one env image per starting room / door location / door+box combo
    room_env_images = {}
    door_env_images = {}
    door_and_box_env_images = {}
    env_image = None
    env_mission = ""

    all_targets_found = lambda: (
        len(room_env_images) >= len(target_rooms)
        and len(door_env_images) >= len(target_door_locations)
        and len(door_and_box_env_images) >= len(target_door_box_locations)
    )

    for _ in range(1000):
        obs, _ = sample_env.reset()
        env_image = uw.get_frame(tile_size=32, agent_pov=False, highlight=False)
        env_mission = obs.get("mission", "") if isinstance(obs, dict) else ""

        if hasattr(uw, 'room_from_pos'):
            try:
                room = uw.room_from_pos(*uw.agent_pos)
                room_key = tuple(int(c) for c in room.top)
                if room_key in target_rooms and room_key not in room_env_images:
                    room_env_images[room_key] = (env_image, env_mission)
            except Exception:
                pass

        if target_door_locations or target_door_box_locations:
            door_positions = []
            box_positions_y = []
            for j in range(uw.grid.height):
                for i in range(uw.grid.width):
                    cell = uw.grid.get(i, j)
                    if cell is not None:
                        if cell.type == 'door':
                            door_positions.append((i, j))
                        elif cell.type == 'box':
                            box_positions_y.append(j)
            door_key = tuple(sorted(door_positions))
            if door_key in target_door_locations and door_key not in door_env_images:
                door_env_images[door_key] = (env_image, env_mission)
            if target_door_box_locations and box_positions_y:
                box_row_key = tuple(sorted(box_positions_y))
                combined_key = (door_key, box_row_key)
                if combined_key in target_door_box_locations and combined_key not in door_and_box_env_images:
                    door_and_box_env_images[combined_key] = (env_image, env_mission)

        if all_targets_found():
            break

    sample_env.close()
    return door_and_box_env_images, door_env_images, env_image, env_mission, room_env_images


@app.cell
def _(analysis_dropdown, door_and_box_env_images, door_env_images, env_image, env_mission, plot_grouped_routing, plot_overall_routing, room_env_images, routing_data):
    if analysis_dropdown.value == 'by_starting_room':
        fig_heatmap = plot_grouped_routing(
            routing_data=routing_data,
            group_by='agent_start_room',
            env_image=env_image,
            env_mission=env_mission,
            room_env_images=room_env_images,
        )
    elif analysis_dropdown.value == 'by_door_location':
        fig_heatmap = plot_grouped_routing(
            routing_data=routing_data,
            group_by='door_location',
            env_image=env_image,
            env_mission=env_mission,
            room_env_images=door_env_images,
        )
    elif analysis_dropdown.value == 'by_door_and_box_row':
        fig_heatmap = plot_grouped_routing(
            routing_data=routing_data,
            group_by='door_and_box_row',
            env_image=env_image,
            env_mission=env_mission,
            room_env_images=door_and_box_env_images,
            max_groups=16,
        )
    elif analysis_dropdown.value == 'by_carrying_phase':
        fig_heatmap = plot_grouped_routing(
            routing_data=routing_data,
            group_by='carrying_phase',
            env_image=env_image,
            env_mission=env_mission,
        )
    else:
        fig_heatmap = plot_overall_routing(
            routing_data=routing_data,
            env_image=env_image,
            env_mission=env_mission,
        )
    return (fig_heatmap,)


@app.cell
def _(fig_heatmap, mo):
    mo.mpl.interactive(fig_heatmap)
    return


@app.cell
def _(mo):
    # Save plots button
    save_button = mo.ui.run_button(label="Save Plots")
    save_button
    return (save_button,)


@app.cell
def _(Path, analysis_dropdown, fig_heatmap, mo, save_button, task_id, trial):
    # Handle save action
    _save_result = None
    if save_button.value:
        _plot_dir = Path("plots") / task_id / f"trial_{trial}"
        _plot_dir.mkdir(parents=True, exist_ok=True)
        if analysis_dropdown.value == "by_starting_room":
            _suffix = "_by_starting_room"
        elif analysis_dropdown.value == "by_door_location":
            _suffix = "_by_door_location"
        elif analysis_dropdown.value == "by_door_and_box_row":
            _suffix = "_by_door_and_box_row"
        elif analysis_dropdown.value == "by_carrying_phase":
            _suffix = "_by_carrying_phase"
        else:
            _suffix = ""
        _filename = f"routing_heatmap{_suffix}.png"
        fig_heatmap.savefig(_plot_dir / _filename, dpi=150, bbox_inches='tight')
        _save_result = f"Saved to {_plot_dir / _filename}"

    mo.md(f"**{_save_result}**") if _save_result else None
    return


if __name__ == "__main__":
    app.run()
