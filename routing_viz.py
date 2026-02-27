import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from plotting_utils import (
        plot_overall_routing,
        plot_grouped_routing,
        group_routing_data,
        get_available_analyses,
        pos_to_quadrant,
    )
    from eval_mop import _first_target_pos

    return Path, _first_target_pos, get_available_analyses, group_routing_data, mo, np, plot_grouped_routing, plot_overall_routing, plt, pos_to_quadrant


@app.cell
def _(Path, mo):
    # Find available tasks from the evaluation cache
    cache_base = Path("evaluation_cache")
    cache_options = {}
    if cache_base.exists():
        for task_dir in sorted(cache_base.iterdir()):
            if not task_dir.is_dir():
                continue
            for trial_dir in sorted(task_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                cache_file = trial_dir / "routing_data.json"
                if cache_file.exists():
                    label = f"{task_dir.name}/{trial_dir.name}"
                    cache_options[label] = str(cache_file)

    task_dropdown = mo.ui.dropdown(
        options=cache_options,
        label="Select Task / Trial",
        value=list(cache_options.keys())[0] if cache_options else None
    )

    mo.vstack([
        mo.md("## Configuration"),
        task_dropdown,
    ])
    return (task_dropdown,)


@app.cell
def _(Path, mo, np, task_dropdown):
    import json

    mo.stop(not task_dropdown.value, mo.md("**Select a task above to begin.**"))

    cache_path = Path(task_dropdown.value)
    with open(cache_path) as f:
        cached = json.load(f)

    task_id = cached['task_id']
    trial = cached['trial']

    # Support three cache formats:
    #   v1: per-timestep env_context embedded in each record
    #   v2: deduplicated episodes list, each record has an 'episode' index
    #   legacy: no env_context at all
    #   v3: includes 'action_logits' list of 7 floats per timestep
    episodes = cached.get('episodes')
    routing_data = [
        {
            'position': tuple(r['position']),
            'layer_routing': {k: np.array(v) for k, v in r['layer_routing'].items()},
            'lpc': r['lpc'],
            'env_context': episodes[r['episode']] if episodes is not None else r.get('env_context', {}),
            'carrying': r.get('carrying', 0),
            'action_logits': np.array(r['action_logits'], dtype=np.float32) if 'action_logits' in r else None,
        }
        for r in cached['routing_data']
    ]
    metrics = {
        'success_rate': cached['metrics']['success_rate'],
        'path_ratio': cached['metrics']['path_ratio'],
        'mean_lpc': cached['metrics']['mean_lpc'],
        'total_episodes': cached['num_episodes'],
    }

    mo.md(f"""
    ## Evaluation Results
    - **Source**: Cache ({cached['evaluated_at'][:10]})
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
        'By agent & target quadrant': 'by_agent_and_target_quadrant',
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
def _(_first_target_pos, group_routing_data, pos_to_quadrant, routing_data, task_id):
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

    # Collect agent+target quadrant keys for per-quadrant-combo image generation
    quadrant_groups = group_routing_data(routing_data, 'agent_and_target_quadrant')
    target_quadrant_combos = set(quadrant_groups.keys())

    # Generate one env image per starting room / door location / door+box combo / quadrant combo
    room_env_images = {}
    door_env_images = {}
    door_and_box_env_images = {}
    quadrant_env_images = {}
    env_image = None
    env_mission = ""

    all_targets_found = lambda: (
        len(room_env_images) >= len(target_rooms)
        and len(door_env_images) >= len(target_door_locations)
        and len(door_and_box_env_images) >= len(target_door_box_locations)
        and len(quadrant_env_images) >= len(target_quadrant_combos)
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

        if target_quadrant_combos:
            try:
                instr = getattr(uw, 'instrs', None)
                target_pos = _first_target_pos(instr) if instr is not None else None
                if target_pos is not None:
                    if hasattr(uw, 'room_from_pos'):
                        room = uw.room_from_pos(*uw.agent_pos)
                        rx, ry = room.top
                        rw, rh = room.size
                        room_bounds = (rx + 1, ry + 1, rx + rw - 2, ry + rh - 2)
                    else:
                        w, h = uw.grid.width, uw.grid.height
                        room_bounds = (1, 1, w - 2, h - 2)
                    aq = pos_to_quadrant(uw.agent_pos[0], uw.agent_pos[1], room_bounds)
                    tq = pos_to_quadrant(target_pos[0], target_pos[1], room_bounds)
                    quad_key = (aq, tq)
                    if quad_key in target_quadrant_combos and quad_key not in quadrant_env_images:
                        quadrant_env_images[quad_key] = (env_image, env_mission)
            except Exception:
                pass

        if all_targets_found():
            break

    sample_env.close()
    return door_and_box_env_images, door_env_images, env_image, env_mission, quadrant_env_images, room_env_images


@app.cell
def _(analysis_dropdown, door_and_box_env_images, door_env_images, env_image, env_mission, plot_grouped_routing, plot_overall_routing, quadrant_env_images, room_env_images, routing_data):
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
    elif analysis_dropdown.value == 'by_agent_and_target_quadrant':
        fig_heatmap = plot_grouped_routing(
            routing_data=routing_data,
            group_by='agent_and_target_quadrant',
            env_image=env_image,
            env_mission=env_mission,
            room_env_images=quadrant_env_images,
            max_groups=16,
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
        elif analysis_dropdown.value == "by_agent_and_target_quadrant":
            _suffix = "_by_agent_and_target_quadrant"
        else:
            _suffix = ""
        _filename = f"routing_heatmap{_suffix}.png"
        fig_heatmap.savefig(_plot_dir / _filename, dpi=150, bbox_inches='tight')
        _save_result = f"Saved to {_plot_dir / _filename}"

    mo.md(f"**{_save_result}**") if _save_result else None
    return


if __name__ == "__main__":
    app.run()
