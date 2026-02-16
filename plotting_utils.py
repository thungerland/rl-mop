"""
Plotting utilities for routing visualization.

Provides reusable functions for aggregating routing data and creating heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from collections import defaultdict
from typing import Callable


def _make_expert_cmap(light_color, dark_color, name):
    """Create a colormap from a light to dark shade for expert confidence gradients."""
    return mcolors.LinearSegmentedColormap.from_list(name, [light_color, dark_color], N=256)


# Expert colormaps: intensity reflects expert complexity.
# Expert 0 (identity/skip) is mildest, higher experts are progressively more vivid.
# Each colormap provides a gradient from light (low confidence) to dark (high confidence).
EXPERT_CMAPS = [
    _make_expert_cmap('#d4e6f1', '#7fb3d8', 'expert0_light_blue'),   # Expert 0: soft pastel blue
    _make_expert_cmap('#c3a6d6', '#8e44ad', 'expert1_purple'),       # Expert 1: medium purple
    _make_expert_cmap('#f1948a', '#c0392b', 'expert2_red'),          # Expert 2: strong red
    _make_expert_cmap('#f0b27a', '#e67e22', 'expert3_orange'),       # Expert 3: orange
    _make_expert_cmap('#82e0aa', '#27ae60', 'expert4_green'),        # Expert 4: green
    _make_expert_cmap('#aab7b8', '#515a5a', 'expert5_grey'),         # Expert 5: grey
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
    env_mission: str = "",
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
        env_mission: Optional language instruction to show under the environment image
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

    num_experts = max(num_experts_per_layer)

    # GridSpec layout: top row has main plots + narrow LPC colorbar column,
    # bottom row has expert colorbars spanning the full width.
    # The LPC colorbar gets its own column so it doesn't steal from the LPC plot.
    num_cols = num_plots + 1  # +1 for dedicated LPC colorbar column
    fig = plt.figure(figsize=(5 * num_plots, 7))
    gs = gridspec.GridSpec(
        2, num_cols,
        height_ratios=[1, 0.04],
        width_ratios=[1] * num_plots + [0.05],
        hspace=0.08, wspace=0.35,
    )

    # Main plot axes (top row)
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]

    plot_idx = 0

    # Environment render (if provided)
    if env_image is not None:
        axes[plot_idx].imshow(env_image)
        axes[plot_idx].set_title("Environment Layout")
        if env_mission:
            axes[plot_idx].set_xlabel(env_mission, fontsize=9)
            axes[plot_idx].set_xticks([])
            axes[plot_idx].set_yticks([])
        else:
            axes[plot_idx].axis('off')
        plot_idx += 1

    # Routing heatmaps for each layer
    for layer_name in layer_names:
        render_routing_heatmap(axes[plot_idx], avg_routing_by_pos, grid_info, layer_name)
        plot_idx += 1

    # LPC heatmap with its own dedicated colorbar column
    lpc_im = render_lpc_heatmap(axes[plot_idx], avg_lpc_by_pos, grid_info)
    lpc_cbar_ax = fig.add_subplot(gs[0, num_plots])
    lpc_cbar = fig.colorbar(lpc_im, cax=lpc_cbar_ax)
    lpc_cbar.set_label('LPC')

    # Expert colorbars along the bottom row, spanning the main plot columns
    for i in range(num_experts):
        cbar_ax = fig.add_subplot(gs[1, i])
        norm = mcolors.Normalize(vmin=0.3, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap=EXPERT_CMAPS[i], norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cb.set_label(f'Expert {i}', fontsize=9)
        cb.set_ticks([0.3, 0.5, 0.7, 0.9])
        cb.set_ticklabels(['0.3', '0.5', '0.7', '0.9'])
        cb.ax.tick_params(labelsize=7)

    # Hide any unused bottom-row cells
    for i in range(num_experts, num_cols):
        ax_empty = fig.add_subplot(gs[1, i])
        ax_empty.axis('off')

    return fig


def get_available_analyses(routing_data: list) -> list[str]:
    """
    Determine which analysis types are available based on the routing data.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples

    Returns:
        List of available analysis names (e.g., ['overall', 'by_starting_room'])
    """
    available = ['overall']

    if not routing_data:
        return available

    # Check first sample's env_context for available fields
    _, _, _, env_context = routing_data[0]

    if not env_context:
        return available

    # Check for room-based grouping
    if env_context.get('agent_start_room') is not None:
        available.append('by_starting_room')

    return available


def group_routing_data(routing_data: list, group_by: str) -> dict[tuple, list]:
    """
    Group routing data by a field in env_context.

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples
        group_by: Field name in env_context to group by (e.g., 'agent_start_room')

    Returns:
        Dict mapping group_key -> list of samples in that group.
        Keys are tuples for hashability.
    """
    groups = defaultdict(list)
    for sample in routing_data:
        pos, layer_routing, lpc, env_context = sample
        key = env_context.get(group_by)
        if key is not None:
            if isinstance(key, list):
                key = tuple(key)
            elif not isinstance(key, tuple):
                key = (key,)
            groups[key].append(sample)
    return dict(groups)


def room_label(room_top: tuple, room_grid_shape: tuple = None) -> str:
    """
    Convert room top-left position to a human-readable label.

    Args:
        room_top: (top_x, top_y) position of the room's top-left corner
        room_grid_shape: (num_cols, num_rows) if known, used for positional names
    """
    if room_grid_shape is not None:
        num_cols, num_rows = room_grid_shape
        # Estimate room index from top position and grid shape
        # Room positions typically follow a regular pattern
        row_names = {0: 'Top', num_rows - 1: 'Bottom'}
        col_names = {0: 'Left', num_cols - 1: 'Right'}

        # We need to figure out which room index this is
        # Since room_top is pixel coords, we need to find the index
        # For now, just use the raw position
        return f"Room at {room_top}"
    return f"Room at {room_top}"


def plot_grouped_routing(
    routing_data: list,
    group_by: str,
    env_image: np.ndarray = None,
    env_mission: str = "",
    max_groups: int = 9,
) -> plt.Figure:
    """
    Create a multi-row figure with one row of heatmaps per group.

    Each row shows: [env_image | layer_0 | layer_1 | ... | LPC]
    Row labels indicate the group (e.g., which starting room).

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples
        group_by: Field in env_context to group by (e.g., 'agent_start_room')
        env_image: Optional environment render (shown in first column of each row)
        env_mission: Optional mission string
        max_groups: Maximum number of groups to display

    Returns:
        matplotlib Figure object
    """
    groups = group_routing_data(routing_data, group_by)

    if not groups:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, f"No data for grouping by '{group_by}'",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Sort groups by key for consistent ordering
    sorted_keys = sorted(groups.keys())[:max_groups]
    num_groups = len(sorted_keys)

    # Get room_grid_shape from first sample (for room labels)
    _, _, _, first_ctx = routing_data[0]
    room_grid_shape = first_ctx.get('room_grid_shape')

    # Aggregate each group to determine layer names and num_experts
    first_avg, _, layer_names = aggregate_routing_by_position(groups[sorted_keys[0]])
    if not first_avg:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No visited positions in first group",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    sample_routing = list(first_avg.values())[0]
    num_experts = max(len(sample_routing[ln]) for ln in layer_names)

    # Layout: columns = [env_image?] + [layer heatmaps] + [LPC] + [LPC colorbar]
    num_main_cols = len(layer_names) + 1  # layers + LPC
    has_env_image = env_image is not None
    if has_env_image:
        num_main_cols += 1

    num_cols = num_main_cols + 1  # +1 for LPC colorbar column

    # Figure dimensions
    col_width = 4.5
    row_height = 4.5
    fig = plt.figure(figsize=(col_width * num_main_cols, row_height * num_groups + 1.2))

    # GridSpec: num_groups rows for data + 1 row for expert colorbars
    height_ratios = [1] * num_groups + [0.04]
    width_ratios = [1] * num_main_cols + [0.05]

    gs = gridspec.GridSpec(
        num_groups + 1, num_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.35, wspace=0.35,
    )

    # Compute grid bounds from ALL data (so all rows share the same coordinate system)
    all_positions = [sample[0] for sample in routing_data]
    global_grid_info = compute_grid_bounds(all_positions)

    lpc_im = None  # Track for colorbar

    for row_idx, group_key in enumerate(sorted_keys):
        group_data = groups[group_key]
        avg_routing, avg_lpc, _ = aggregate_routing_by_position(group_data)

        if not avg_routing:
            continue

        col_idx = 0

        # Row label
        if group_by == 'agent_start_room':
            label = room_label(group_key, room_grid_shape)
        else:
            label = f"{group_by}={group_key}"

        # Environment image
        if has_env_image:
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(env_image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"{label}\n({len(group_data)} samples)", fontsize=9)
            if row_idx == 0:
                ax.set_title("Environment Layout")
                if env_mission:
                    ax.set_xlabel(env_mission, fontsize=8)
            col_idx += 1

        # Routing heatmaps for each layer
        for layer_name in layer_names:
            ax = fig.add_subplot(gs[row_idx, col_idx])
            render_routing_heatmap(ax, avg_routing, global_grid_info, layer_name)
            if row_idx == 0:
                ax.set_title(f"{layer_name}\n(color=expert, intensity=confidence)")
            else:
                ax.set_title("")
            if not has_env_image and col_idx == 0:
                ax.set_ylabel(f"{label}\n({len(group_data)} samples)", fontsize=9)
            col_idx += 1

        # LPC heatmap
        ax = fig.add_subplot(gs[row_idx, col_idx])
        lpc_im = render_lpc_heatmap(ax, avg_lpc, global_grid_info)
        if row_idx == 0:
            ax.set_title("Mean LPC by Position")
        else:
            ax.set_title("")

    # LPC colorbar in the dedicated column (spanning all data rows)
    if lpc_im is not None:
        lpc_cbar_ax = fig.add_subplot(gs[:num_groups, num_main_cols])
        fig.colorbar(lpc_im, cax=lpc_cbar_ax).set_label('LPC')

    # Expert colorbars along the bottom row
    for i in range(num_experts):
        cbar_ax = fig.add_subplot(gs[num_groups, i])
        norm = mcolors.Normalize(vmin=0.3, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap=EXPERT_CMAPS[i], norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cb.set_label(f'Expert {i}', fontsize=9)
        cb.set_ticks([0.3, 0.5, 0.7, 0.9])
        cb.set_ticklabels(['0.3', '0.5', '0.7', '0.9'])
        cb.ax.tick_params(labelsize=7)

    # Hide unused bottom-row cells
    for i in range(num_experts, num_cols):
        ax_empty = fig.add_subplot(gs[num_groups, i])
        ax_empty.axis('off')

    return fig
