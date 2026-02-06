import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from pathlib import Path
    from collections import defaultdict

    return Path, defaultdict, mo, np, plt, torch


@app.cell
def _(Path, mo):
    # Find available checkpoints
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("**/*.pt"))
    checkpoint_options = {str(cp.relative_to(checkpoint_dir)): str(cp) for cp in checkpoints}

    # UI for checkpoint selection
    checkpoint_dropdown = mo.ui.dropdown(
        options=checkpoint_options,
        label="Select Checkpoint",
        value=list(checkpoint_options.keys())[0] if checkpoint_options else None
    )

    # Number of episodes for evaluation
    num_episodes_slider = mo.ui.slider(
        start=10, stop=200, value=50, step=10,
        label="Number of Episodes"
    )

    # Number of parallel environments
    num_envs_slider = mo.ui.slider(
        start=1, stop=16, value=8, step=1,
        label="Parallel Environments"
    )

    mo.vstack([
        mo.md("## Configuration"),
        checkpoint_dropdown,
        num_episodes_slider,
        num_envs_slider,
    ])
    return checkpoint_dropdown, num_envs_slider, num_episodes_slider


@app.cell
def _(checkpoint_dropdown, mo, num_envs_slider, num_episodes_slider, torch):
    from mixture_of_experts import MixtureOfExpertsPolicy, reset_hidden_on_done, compute_lpc
    from eval_mop import EvalVectorEnv, encode_obs_batch, load_checkpoint, evaluate

    mo.stop(not checkpoint_dropdown.value, mo.md("**Select a checkpoint above to begin.**"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    policy, config = load_checkpoint(checkpoint_dropdown.value, device)
    task_id = config['task_id']

    # Create evaluation environment
    vec_env = EvalVectorEnv(task_id, num_envs_slider.value, device)

    # Run evaluation
    metrics, routing_data = evaluate(
        policy, vec_env, num_episodes_slider.value, device
    )

    mo.md(f"""
    ## Evaluation Results
    - **Task**: {task_id}
    - **Episodes**: {metrics['total_episodes']}
    - **Success Rate**: {metrics['success_rate']:.1%}
    - **Path Ratio**: {metrics['path_ratio']:.2f}
    - **Mean LPC**: {metrics['mean_lpc']:.2f}
    - **Routing Samples**: {len(routing_data)}
    """)
    return routing_data, task_id


@app.cell
def _(defaultdict, np, routing_data):
    # Aggregate routing data by position
    position_routing = defaultdict(list)
    for _pos, _layer_routing, _lpc in routing_data:
        position_routing[_pos].append(_layer_routing)

    # Compute average routing weights per position
    avg_routing_by_pos = {}
    for _pos, _routings in position_routing.items():
        avg_routing_by_pos[_pos] = {
            _layer: np.mean([r[_layer] for r in _routings], axis=0)
            for _layer in _routings[0].keys()
        }

    # Get grid bounds
    all_positions = list(position_routing.keys())
    x_coords = [p[0] for p in all_positions]
    y_coords = [p[1] for p in all_positions]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    grid_width = x_max - x_min + 1
    grid_height = y_max - y_min + 1

    # Get number of layers and experts
    sample_routing = list(avg_routing_by_pos.values())[0]
    num_layers = len(sample_routing)
    layer_names = sorted(sample_routing.keys())
    num_experts_per_layer = [len(sample_routing[ln]) for ln in layer_names]

    (grid_width, grid_height, num_layers, num_experts_per_layer, x_min, y_min)
    return (
        avg_routing_by_pos,
        grid_height,
        grid_width,
        layer_names,
        num_experts_per_layer,
        x_min,
        y_min,
    )


@app.cell
def _(task_id, torch):
    import gymnasium as gym

    # Render a sample environment for reference
    sample_env = gym.make(task_id, render_mode="rgb_array")
    sample_env.reset()
    env_image = sample_env.unwrapped.get_frame(tile_size=32, agent_pov=False, highlight=False)
    sample_env.close()

    env_image_tensor = torch.from_numpy(env_image)
    return (env_image,)


@app.cell
def _(
    avg_routing_by_pos,
    env_image,
    grid_height,
    grid_width,
    layer_names,
    np,
    num_experts_per_layer,
    plt,
    x_min,
    y_min,
):
    # Create visualization with environment render and routing heatmaps
    _num_heatmaps = len(layer_names)
    _fig, _axes = plt.subplots(1, _num_heatmaps + 1, figsize=(5 * (_num_heatmaps + 1), 5))

    # Handle single heatmap case (axes not iterable)
    if _num_heatmaps == 0:
        _axes = [_axes]

    # Plot 1: Environment render
    _axes[0].imshow(env_image)
    _axes[0].set_title("Environment Layout")
    _axes[0].axis('off')

    # Distinct colormaps for each expert - using perceptually uniform maps
    # Each expert gets its own colormap that shows intensity clearly
    _expert_cmaps = [
        plt.cm.Blues,    # Expert 0: blue
        plt.cm.Oranges,  # Expert 1: orange
        plt.cm.Greens,   # Expert 2: green
        plt.cm.Reds,     # Expert 3: red
        plt.cm.Purples,  # Expert 4: purple
        plt.cm.Greys,    # Expert 5: grey
    ]

    # For each layer, create a dominant expert + intensity heatmap
    for _layer_idx, _layer_name in enumerate(layer_names):
        _ax = _axes[_layer_idx + 1]

        # Create grids for dominant expert and confidence
        _dominant_grid = np.full((grid_height, grid_width), -1, dtype=int)
        _confidence_grid = np.full((grid_height, grid_width), np.nan)

        for _pos, _routing in avg_routing_by_pos.items():
            _x, _y = _pos
            _gx, _gy = _x - x_min, _y - y_min

            _weights = _routing[_layer_name]
            _dominant_expert = np.argmax(_weights)
            _confidence = _weights[_dominant_expert]

            _dominant_grid[_gy, _gx] = _dominant_expert
            _confidence_grid[_gy, _gx] = _confidence

        # Create RGB image with intensity mapped through colormaps
        _rgb_image = np.ones((grid_height, grid_width, 3)) * 0.9  # Light gray background

        for _gy in range(grid_height):
            for _gx in range(grid_width):
                _expert_idx = _dominant_grid[_gy, _gx]
                if _expert_idx >= 0:
                    _cmap = _expert_cmaps[_expert_idx % len(_expert_cmaps)]
                    # Map confidence (0.33 to 1.0 typical range) to color intensity (0.3 to 1.0)
                    # This makes low confidence visible but still distinguishable
                    _intensity = 0.3 + 0.7 * _confidence_grid[_gy, _gx]
                    _color = _cmap(_intensity)
                    _rgb_image[_gy, _gx, :] = _color[:3]

        _im = _ax.imshow(_rgb_image, origin='upper')
        _ax.set_title(f"{_layer_name}\n(color=expert, intensity=confidence)")
        _ax.set_xlabel("X")
        _ax.set_ylabel("Y")

        # Add grid lines
        _ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
        _ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
        _ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        _ax.tick_params(which='minor', size=0)

    # Add colorbar for confidence scale (light = low confidence, dark = high confidence)
    _cbar_ax = _fig.add_axes([0.92, 0.15, 0.02, 0.7])
    _sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=plt.Normalize(vmin=0.33, vmax=1.0))
    _cbar = _fig.colorbar(_sm, cax=_cbar_ax)
    _cbar.set_label('Confidence')
    _cbar.set_ticks([0.33, 0.5, 0.67, 0.83, 1.0])
    _cbar.set_ticklabels(['0.33', '0.5', '0.67', '0.83', '1.0'])

    # Add legend for expert colors
    _legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=_expert_cmaps[i](0.7)[:3],
                   markersize=12, label=f'Expert {i}')
        for i in range(max(num_experts_per_layer))
    ]
    _fig.legend(handles=_legend_elements, loc='lower center', ncol=max(num_experts_per_layer),
               bbox_to_anchor=(0.45, -0.02))

    plt.tight_layout(rect=[0, 0.05, 0.9, 1])
    plt.gca()
    return


@app.cell
def _(avg_routing_by_pos, layer_names, plt):
    # Detailed bar chart showing routing distribution at each position
    # Select a few representative positions
    _positions = list(avg_routing_by_pos.keys())
    _num_positions_to_show = min(9, len(_positions))
    _sample_positions = _positions[::max(1, len(_positions) // _num_positions_to_show)][:_num_positions_to_show]

    _fig2, _axes2 = plt.subplots(
        len(_sample_positions), len(layer_names),
        figsize=(4 * len(layer_names), 2 * len(_sample_positions))
    )

    if len(layer_names) == 1:
        _axes2 = _axes2.reshape(-1, 1)
    if len(_sample_positions) == 1:
        _axes2 = _axes2.reshape(1, -1)

    for _pos_idx, _pos in enumerate(_sample_positions):
        _routing = avg_routing_by_pos[_pos]
        for _layer_idx, _layer_name in enumerate(layer_names):
            _ax2 = _axes2[_pos_idx, _layer_idx]
            _weights = _routing[_layer_name]
            _ax2.bar(range(len(_weights)), _weights, color=plt.cm.tab10.colors[:len(_weights)])
            _ax2.set_ylim(0, 1)
            _ax2.set_xlabel("Expert")
            _ax2.set_ylabel("Weight")
            if _pos_idx == 0:
                _ax2.set_title(_layer_name)
            if _layer_idx == 0:
                _ax2.set_ylabel(f"Pos {_pos}\nWeight")

    _fig2.suptitle("Routing Distribution by Position", fontsize=14)
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
