"""
Plotting utilities for routing visualization.

Provides reusable functions for aggregating routing data and creating heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable


# Expert colormaps for consistent visualization across all plots
EXPERT_CMAPS = [
    plt.cm.Blues,    # Expert 0: blue
    plt.cm.Oranges,  # Expert 1: orange
    plt.cm.Greens,   # Expert 2: green
    plt.cm.Reds,     # Expert 3: red
    plt.cm.Purples,  # Expert 4: purple
    plt.cm.Greys,    # Expert 5: grey
]

# Color for unvisited cells
UNVISITED_COLOR = (0.15, 0.15, 0.15)  # Dark gray


def compute_grid_bounds(positions: list[tuple]) -> dict:
    """
    Compute grid bounds from a list of positions.

    Args:
        positions: List of (x, y) tuples

    Returns:
        dict with x_min, x_max, y_min, y_max, grid_width, grid_height
    """
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'grid_width': x_max - x_min + 1,
        'grid_height': y_max - y_min + 1,
    }


def aggregate_routing_by_position(
    routing_data: list,
    filter_fn: Callable = None
) -> tuple[dict, dict, list]:
    """
    Aggregate routing weights and LPC by position.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples
        filter_fn: Optional function (pos, layer_routing, lpc, env_context) -> bool
                   to filter samples before aggregation

    Returns:
        avg_routing_by_pos: dict mapping position -> {layer_name: avg_weights}
        avg_lpc_by_pos: dict mapping position -> avg_lpc
        layer_names: sorted list of layer names
    """
    position_routing = defaultdict(list)
    position_lpc = defaultdict(list)

    for sample in routing_data:
        pos, layer_routing, lpc, env_context = sample

        # Apply filter if provided
        if filter_fn is not None and not filter_fn(pos, layer_routing, lpc, env_context):
            continue

        position_routing[pos].append(layer_routing)
        position_lpc[pos].append(lpc)

    if not position_routing:
        return {}, {}, []

    # Compute averages
    avg_routing_by_pos = {}
    for pos, routings in position_routing.items():
        avg_routing_by_pos[pos] = {
            layer: np.mean([r[layer] for r in routings], axis=0)
            for layer in routings[0].keys()
        }

    avg_lpc_by_pos = {pos: np.mean(lpcs) for pos, lpcs in position_lpc.items()}

    # Get layer names from first sample
    layer_names = sorted(list(position_routing.values())[0][0].keys())

    return avg_routing_by_pos, avg_lpc_by_pos, layer_names


def render_routing_heatmap(
    ax,
    avg_routing_by_pos: dict,
    grid_info: dict,
    layer_name: str,
    expert_cmaps: list = None
):
    """
    Render a routing heatmap for a single layer on the given axes.

    Shows dominant expert (by color) and confidence (by intensity).
    Unvisited cells are rendered as dark gray.

    Args:
        ax: Matplotlib axes to render on
        avg_routing_by_pos: dict mapping position -> {layer_name: avg_weights}
        grid_info: dict from compute_grid_bounds()
        layer_name: Name of the layer to visualize
        expert_cmaps: List of colormaps for each expert (defaults to EXPERT_CMAPS)
    """
    if expert_cmaps is None:
        expert_cmaps = EXPERT_CMAPS

    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    # Create grids for dominant expert and confidence
    dominant_grid = np.full((grid_height, grid_width), -1, dtype=int)
    confidence_grid = np.full((grid_height, grid_width), np.nan)

    for pos, routing in avg_routing_by_pos.items():
        x, y = pos
        gx, gy = x - x_min, y - y_min

        weights = routing[layer_name]
        dominant_expert = np.argmax(weights)
        confidence = weights[dominant_expert]

        dominant_grid[gy, gx] = dominant_expert
        confidence_grid[gy, gx] = confidence

    # Create RGB image - unvisited cells are dark gray
    rgb_image = np.ones((grid_height, grid_width, 3)) * np.array(UNVISITED_COLOR)

    for gy in range(grid_height):
        for gx in range(grid_width):
            expert_idx = dominant_grid[gy, gx]
            if expert_idx >= 0:
                cmap = expert_cmaps[expert_idx % len(expert_cmaps)]
                # Map confidence (0.33 to 1.0 typical range) to color intensity (0.3 to 1.0)
                intensity = 0.3 + 0.7 * confidence_grid[gy, gx]
                color = cmap(intensity)
                rgb_image[gy, gx, :] = color[:3]

    ax.imshow(rgb_image, origin='upper')
    ax.set_title(f"{layer_name}\n(color=expert, intensity=confidence)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)


def render_lpc_heatmap(ax, avg_lpc_by_pos: dict, grid_info: dict):
    """
    Render an LPC heatmap on the given axes.

    Unvisited cells are rendered as dark gray (NaN in colormap).

    Args:
        ax: Matplotlib axes to render on
        avg_lpc_by_pos: dict mapping position -> avg_lpc
        grid_info: dict from compute_grid_bounds()
    """
    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    # Create LPC grid - NaN for unvisited
    lpc_grid = np.full((grid_height, grid_width), np.nan)

    for pos, lpc in avg_lpc_by_pos.items():
        x, y = pos
        gx, gy = x - x_min, y - y_min
        lpc_grid[gy, gx] = lpc

    # Create colormap that shows NaN as dark gray
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=UNVISITED_COLOR)

    im = ax.imshow(lpc_grid, origin='upper', cmap=cmap)
    ax.set_title("Mean LPC by Position")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    return im


def plot_overall_routing(
    routing_data: list,
    env_image: np.ndarray = None,
    filter_fn: Callable = None,
    title_suffix: str = ""
) -> plt.Figure:
    """
    Create a complete routing visualization figure.

    Shows environment layout (if provided), routing heatmaps for each layer,
    and an LPC heatmap. Unvisited cells are rendered as dark gray.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples
        env_image: Optional environment render to show
        filter_fn: Optional function to filter samples before aggregation
        title_suffix: Optional suffix to add to figure title

    Returns:
        matplotlib Figure object
    """
    # Aggregate data
    avg_routing_by_pos, avg_lpc_by_pos, layer_names = aggregate_routing_by_position(
        routing_data, filter_fn
    )

    if not avg_routing_by_pos:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No data after filtering", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Compute grid bounds
    positions = list(avg_routing_by_pos.keys())
    grid_info = compute_grid_bounds(positions)

    # Get number of experts per layer
    sample_routing = list(avg_routing_by_pos.values())[0]
    num_experts_per_layer = [len(sample_routing[ln]) for ln in layer_names]

    # Determine subplot layout
    num_heatmaps = len(layer_names)
    num_plots = num_heatmaps + 1  # +1 for LPC
    if env_image is not None:
        num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Environment render (if provided)
    if env_image is not None:
        axes[plot_idx].imshow(env_image)
        axes[plot_idx].set_title("Environment Layout")
        axes[plot_idx].axis('off')
        plot_idx += 1

    # Routing heatmaps for each layer
    for layer_name in layer_names:
        render_routing_heatmap(axes[plot_idx], avg_routing_by_pos, grid_info, layer_name)
        plot_idx += 1

    # LPC heatmap
    lpc_im = render_lpc_heatmap(axes[plot_idx], avg_lpc_by_pos, grid_info)
    lpc_cbar = fig.colorbar(lpc_im, ax=axes[plot_idx], shrink=0.8)
    lpc_cbar.set_label('LPC')

    # Add legend for expert colors
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=EXPERT_CMAPS[i](0.7)[:3],
                   markersize=12, label=f'Expert {i}')
        for i in range(max(num_experts_per_layer))
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=max(num_experts_per_layer),
               bbox_to_anchor=(0.4, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    return fig


def get_available_analyses(routing_data: list) -> list[str]:
    """
    Determine which analysis types are available based on the routing data.

    Returns a list of available analysis type names.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples

    Returns:
        List of available analysis names (e.g., ['overall', 'by_target_quadrant', 'by_door_position'])
    """
    available = ['overall']

    if not routing_data:
        return available

    # Check first sample's env_context for available fields
    _, _, _, env_context = routing_data[0]

    if not env_context:
        return available

    # Check for target-related fields (balls are often targets in BabyAI)
    if env_context.get('balls') or env_context.get('goals'):
        available.append('by_target_quadrant')

    # Check for doors
    if env_context.get('doors'):
        available.append('by_door_position')

    # Check for keys
    if env_context.get('keys'):
        available.append('by_key_position')

    return available
