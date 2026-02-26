"""
Plotting utilities for routing visualization.

Provides reusable functions for aggregating routing data and creating heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from collections import defaultdict


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


def pos_to_quadrant(x: int, y: int, room_bounds: tuple) -> str:
    """Map a grid position to a room quadrant label.

    Args:
        x: Grid x coordinate
        y: Grid y coordinate
        room_bounds: (x_min, y_min, x_max, y_max) inclusive interior floor bounds

    Returns:
        One of 'TL', 'TR', 'BL', 'BR'.
        MiniGrid convention: y increases downward, so lower y = Top.
    """
    x_min, y_min, x_max, y_max = room_bounds
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    top = y < mid_y
    left = x < mid_x
    if top and left:
        return "TL"
    elif top:
        return "TR"
    elif left:
        return "BL"
    else:
        return "BR"


def aggregate_routing_by_position(routing_data: list) -> tuple[dict, dict, list]:
    """
    Aggregate routing weights and LPC by position.

    Args:
        routing_data: List of dicts with keys position, layer_routing, lpc, env_context

    Returns:
        avg_routing_by_pos: dict mapping position -> {layer_name: avg_weights}
        avg_lpc_by_pos: dict mapping position -> avg_lpc
        layer_names: sorted list of layer names
    """
    position_routing = defaultdict(list)
    position_lpc = defaultdict(list)

    for sample in routing_data:
        pos = sample['position']
        layer_routing = sample['layer_routing']
        lpc = sample['lpc']

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
) -> plt.Figure:
    """
    Create a complete routing visualization figure.

    Shows environment layout (if provided), routing heatmaps for each layer,
    and an LPC heatmap. Unvisited cells are rendered as dark gray.

    Args:
        routing_data: List of dicts with keys position, layer_routing, lpc, env_context
        env_image: Optional environment render to show
        env_mission: Optional language instruction to show under the environment image

    Returns:
        matplotlib Figure object
    """
    # Aggregate data
    avg_routing_by_pos, avg_lpc_by_pos, layer_names = aggregate_routing_by_position(
        routing_data
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
    env_context = routing_data[0]['env_context']

    if not env_context:
        return available

    # Check for room-based grouping
    if env_context.get('agent_start_room') is not None:
        available.append('by_starting_room')

    # Door-location grouping: only offer if distinct door-position-sets <= 9.
    # Sample up to 200 timesteps — env_context repeats within episodes so a
    # small probe is sufficient to discover the distinct door configurations.
    if env_context.get('doors') is not None:
        distinct_door_sets = set()
        for sample in routing_data[:200]:
            ctx = sample['env_context']
            doors = ctx.get('doors')
            if doors:
                door_pos_key = tuple(sorted((d[0], d[1]) for d in doors))
                distinct_door_sets.add(door_pos_key)
        if 1 <= len(distinct_door_sets) <= 9:
            available.append('by_door_location')

    # Combined door+box-row grouping: only offer if both doors and boxes exist
    # and distinct combined keys <= 16.
    if env_context.get('doors') is not None and env_context.get('boxes') is not None:
        distinct_combined = set()
        for sample in routing_data[:200]:
            ctx = sample['env_context']
            doors = ctx.get('doors')
            boxes = ctx.get('boxes')
            if doors and boxes:
                door_key = tuple(sorted((d[0], d[1]) for d in doors))
                box_row_key = tuple(sorted(b[1] for b in boxes))
                distinct_combined.add((door_key, box_row_key))
        if 1 <= len(distinct_combined) <= 16:
            available.append('by_door_and_box_row')

    # Carrying-phase split: only offer if both carrying=0 and carrying=1 timesteps exist.
    if len(routing_data) > 0 and 'carrying' in routing_data[0]:
        carrying_values = set(sample['carrying'] for sample in routing_data)
        if carrying_values == {0, 1}:
            available.append('by_carrying_phase')

    # Agent+target quadrant grouping: requires agent_start_pos, target_pos, room_bounds.
    # Only offered when there is meaningful variation in agent start quadrant (>= 2 distinct)
    # and the total number of combinations is within reason (<= 16, i.e. 4x4).
    if (env_context.get('agent_start_pos') is not None
            and env_context.get('target_pos') is not None
            and env_context.get('room_bounds') is not None):
        distinct_quad_combos = set()
        distinct_agent_quads = set()
        for sample in routing_data[:200]:
            ctx = sample['env_context']
            agent_pos = ctx.get('agent_start_pos')
            target_pos = ctx.get('target_pos')
            room_bounds = ctx.get('room_bounds')
            if agent_pos is None or target_pos is None or room_bounds is None:
                continue
            aq = pos_to_quadrant(agent_pos[0], agent_pos[1], room_bounds)
            tq = pos_to_quadrant(target_pos[0], target_pos[1], room_bounds)
            distinct_quad_combos.add((aq, tq))
            distinct_agent_quads.add(aq)
        if len(distinct_agent_quads) >= 2 and len(distinct_quad_combos) <= 16:
            available.append('by_agent_and_target_quadrant')

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
        env_context = sample['env_context']

        if group_by == 'carrying_phase':
            key = (sample.get('carrying', 0),)
        elif group_by == 'door_location':
            doors = env_context.get('doors')
            if not doors:
                continue
            # sorted tuple of (x, y) pairs — order-independent, ignores color/state
            key = tuple(sorted((d[0], d[1]) for d in doors))
        elif group_by == 'door_and_box_row':
            doors = env_context.get('doors')
            boxes = env_context.get('boxes')
            if not doors or not boxes:
                continue
            door_key = tuple(sorted((d[0], d[1]) for d in doors))
            box_row_key = tuple(sorted(b[1] for b in boxes))
            key = (door_key, box_row_key)
        elif group_by == 'agent_and_target_quadrant':
            agent_pos = env_context.get('agent_start_pos')
            target_pos = env_context.get('target_pos')
            room_bounds = env_context.get('room_bounds')
            if agent_pos is None or target_pos is None or room_bounds is None:
                continue
            aq = pos_to_quadrant(agent_pos[0], agent_pos[1], room_bounds)
            tq = pos_to_quadrant(target_pos[0], target_pos[1], room_bounds)
            key = (aq, tq)
        else:
            key = env_context.get(group_by)
            if key is None:
                continue
            if isinstance(key, list):
                key = tuple(key)
            elif not isinstance(key, tuple):
                key = (key,)

        groups[key].append(sample)
    return dict(groups)


def _positional_names(n: int, axis: str) -> list[str]:
    """Generate positional names for a grid axis.

    Args:
        n: Number of positions along this axis
        axis: 'row' for Top/Bottom, 'col' for Left/Right
    """
    if axis == 'row':
        ends = ('Top', 'Bottom')
    else:
        ends = ('Left', 'Right')
    if n == 1:
        return ['Center']
    if n == 2:
        return list(ends)
    if n == 3:
        return [ends[0], 'Center', ends[1]]
    return [ends[0]] + [f'{axis.title()} {i}' for i in range(1, n - 1)] + [ends[1]]


def room_labels_for_groups(sorted_keys: list, room_grid_shape: tuple = None) -> dict:
    """Map room top-left keys to human-readable labels like 'Top-Left', 'Center'.

    Args:
        sorted_keys: List of room_top tuples, sorted
        room_grid_shape: (num_cols, num_rows) if known

    Returns:
        Dict mapping room_key -> label string
    """
    if room_grid_shape is None or not sorted_keys:
        return {k: f"Room at {k}" for k in sorted_keys}

    num_cols, num_rows = room_grid_shape
    xs = sorted(set(k[0] for k in sorted_keys))
    ys = sorted(set(k[1] for k in sorted_keys))
    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: j for j, y in enumerate(ys)}

    row_names = _positional_names(num_rows, 'row')
    col_names = _positional_names(num_cols, 'col')

    labels = {}
    for key in sorted_keys:
        col_idx = x_to_col[key[0]]
        row_idx = y_to_row[key[1]]
        r = row_names[row_idx] if row_idx < len(row_names) else f"Row {row_idx}"
        c = col_names[col_idx] if col_idx < len(col_names) else f"Col {col_idx}"
        if r == 'Center' and c == 'Center':
            labels[key] = 'Center'
        else:
            labels[key] = f"{r}-{c}"
    return labels


def door_location_labels_for_groups(sorted_keys: list) -> dict:
    """Map door-location group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a tuple of (x, y) tuples,
                     e.g. ((5, 3),) for a single door.

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {}
    for key in sorted_keys:
        if len(key) == 1:
            labels[key] = f"Door at {key[0][0]},{key[0][1]}"
        else:
            coords = ",".join(f"({x},{y})" for x, y in key)
            labels[key] = f"Doors at {coords}"
    return labels


def door_and_box_row_labels_for_groups(sorted_keys: list) -> dict:
    """Map door+box-row group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a tuple of
                     (door_pos_tuple, box_row_tuple), e.g.
                     (((5, 3),), (2,)) meaning door at (5,3), box on row 2.

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {}
    for key in sorted_keys:
        door_part, box_row_part = key
        if len(door_part) == 1:
            door_str = f"Door {door_part[0][0]},{door_part[0][1]}"
        else:
            door_str = "Doors " + ",".join(f"({x},{y})" for x, y in door_part)
        if len(box_row_part) == 1:
            box_str = f"Box row {box_row_part[0]}"
        else:
            box_str = "Box rows " + ",".join(str(r) for r in box_row_part)
        labels[key] = f"{door_str} / {box_str}"
    return labels


def agent_and_target_quadrant_labels_for_groups(sorted_keys: list) -> dict:
    """Map (agent_quadrant, target_quadrant) group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a 2-tuple of quadrant strings
                     e.g. ('TL', 'BR')

    Returns:
        Dict mapping group_key -> label string like 'Agent: TL / Target: BR'
    """
    return {key: f"Agent: {key[0]} / Target: {key[1]}" for key in sorted_keys}


ACTION_NAMES = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']


def carrying_phase_labels_for_groups(sorted_keys: list) -> dict:
    """Map carrying-phase group keys to human-readable labels.

    Args:
        sorted_keys: List of group keys, each a 1-tuple (carrying,)
                     where carrying is 0 (not carrying) or 1 (carrying object).

    Returns:
        Dict mapping group_key -> label string
    """
    labels = {(0,): "Not carrying", (1,): "Carrying object"}
    return {k: labels.get(k, f"Phase {k[0]}") for k in sorted_keys}


def plot_grouped_routing(
    routing_data: list,
    group_by: str,
    env_image: np.ndarray = None,
    env_mission: str = "",
    room_env_images: dict = None,
    max_groups: int = 9,
) -> plt.Figure:
    """
    Create a multi-row figure with one row of heatmaps per group.

    Each row shows: [env_image | layer_0 | layer_1 | ... | LPC]
    Row labels indicate the group (e.g., which starting room).

    Args:
        routing_data: List of (position, layer_routing, lpc, env_context) tuples
        group_by: Field in env_context to group by (e.g., 'agent_start_room')
        env_image: Optional fallback environment render
        env_mission: Optional fallback mission string
        room_env_images: Optional dict mapping group_key -> (image, mission) per group
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
    first_ctx = routing_data[0]['env_context']
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
    all_positions = [sample['position'] for sample in routing_data]
    global_grid_info = compute_grid_bounds(all_positions)

    lpc_im = None  # Track for colorbar

    # Compute human-readable labels for all groups
    if group_by == 'agent_start_room':
        group_labels = room_labels_for_groups(sorted_keys, room_grid_shape)
    elif group_by == 'door_location':
        group_labels = door_location_labels_for_groups(sorted_keys)
    elif group_by == 'door_and_box_row':
        group_labels = door_and_box_row_labels_for_groups(sorted_keys)
    elif group_by == 'carrying_phase':
        group_labels = carrying_phase_labels_for_groups(sorted_keys)
    elif group_by == 'agent_and_target_quadrant':
        group_labels = agent_and_target_quadrant_labels_for_groups(sorted_keys)
    else:
        group_labels = {k: f"{group_by}={k}" for k in sorted_keys}

    for row_idx, group_key in enumerate(sorted_keys):
        group_data = groups[group_key]
        avg_routing, avg_lpc, _ = aggregate_routing_by_position(group_data)

        if not avg_routing:
            continue

        col_idx = 0
        label = group_labels[group_key]

        # Environment image (per-room if available, otherwise fallback)
        if has_env_image:
            if room_env_images and group_key in room_env_images:
                row_image, row_mission = room_env_images[group_key]
            else:
                row_image, row_mission = env_image, env_mission

            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(row_image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"{label}\n({len(group_data)} samples)", fontsize=9)
            ax.set_xlabel(row_mission, fontsize=8)
            if row_idx == 0:
                ax.set_title("Environment Layout")
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


def plot_action_frequency(routing_data: list, group_by: str = None) -> plt.Figure:
    """
    Bar chart of how often each action is chosen (argmax of logits).

    Args:
        routing_data: List of dicts with keys position, layer_routing, lpc, env_context,
                      carrying, action_logits
        group_by: If 'carrying', shows side-by-side bars for carrying=0 vs carrying=1.
                  If None, shows a single bar per action.

    Returns:
        matplotlib Figure object
    """
    valid = [s for s in routing_data if s.get('action_logits') is not None]
    if not valid:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No action_logits in data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    n_actions = len(ACTION_NAMES)
    x = np.arange(n_actions)

    fig, ax = plt.subplots(figsize=(9, 5))

    if group_by == 'carrying':
        groups = {0: [], 1: []}
        for s in valid:
            groups[s.get('carrying', 0)].append(np.argmax(s['action_logits']))

        width = 0.35
        colors = ['#5b9bd5', '#ed7d31']
        labels = ['Not carrying', 'Carrying object']
        for i, (carrying_val, label, color) in enumerate(zip([0, 1], labels, colors)):
            counts = np.array(groups[carrying_val])
            if len(counts) == 0:
                freqs = np.zeros(n_actions)
            else:
                freqs = np.bincount(counts, minlength=n_actions)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, freqs, width, label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, freqs):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            str(int(val)), ha='center', va='bottom', fontsize=7)
        ax.legend()
    else:
        chosen = np.array([np.argmax(s['action_logits']) for s in valid])
        freqs = np.bincount(chosen, minlength=n_actions)
        bars = ax.bar(x, freqs, color='#5b9bd5', alpha=0.85)
        for bar, val in zip(bars, freqs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(int(val)), ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_xlabel("Action")
    ax.set_ylabel("Timesteps")
    title = "Action Frequency"
    if group_by == 'carrying':
        title += " by Carrying Phase"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_action_entropy_heatmap(
    routing_data: list,
    env_image: np.ndarray = None,
    env_mission: str = "",
) -> plt.Figure:
    """
    Spatial heatmap of mean softmax entropy per grid cell.

    H = -sum(p * log(p + 1e-9)) where p = softmax(logits).
    Shows where the agent is most/least uncertain.

    Args:
        routing_data: List of dicts with keys position, layer_routing, lpc, env_context,
                      carrying, action_logits
        env_image: Optional environment render to show alongside
        env_mission: Optional mission string

    Returns:
        matplotlib Figure object
    """
    valid = [s for s in routing_data if s.get('action_logits') is not None]
    if not valid:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No action_logits in data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Accumulate entropy values per position
    position_entropies = defaultdict(list)
    for s in valid:
        logits = s['action_logits'].astype(np.float64)
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        p = exp_logits / exp_logits.sum()
        entropy = -np.sum(p * np.log(p + 1e-9))
        position_entropies[s['position']].append(entropy)

    avg_entropy_by_pos = {pos: np.mean(vals) for pos, vals in position_entropies.items()}
    positions = list(avg_entropy_by_pos.keys())
    grid_info = compute_grid_bounds(positions)

    grid_width = grid_info['grid_width']
    grid_height = grid_info['grid_height']
    x_min = grid_info['x_min']
    y_min = grid_info['y_min']

    entropy_grid = np.full((grid_height, grid_width), np.nan)
    for pos, val in avg_entropy_by_pos.items():
        gx, gy = pos[0] - x_min, pos[1] - y_min
        entropy_grid[gy, gx] = val

    num_plots = 2 if env_image is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0
    if env_image is not None:
        axes[plot_idx].imshow(env_image)
        axes[plot_idx].set_title("Environment Layout")
        if env_mission:
            axes[plot_idx].set_xlabel(env_mission, fontsize=9)
        axes[plot_idx].set_xticks([])
        axes[plot_idx].set_yticks([])
        plot_idx += 1

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color=UNVISITED_COLOR)
    im = axes[plot_idx].imshow(entropy_grid, origin='upper', cmap=cmap)
    axes[plot_idx].set_title("Mean Action Entropy by Position")
    axes[plot_idx].set_xlabel("X")
    axes[plot_idx].set_ylabel("Y")
    axes[plot_idx].set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    axes[plot_idx].set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    axes[plot_idx].grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    axes[plot_idx].tick_params(which='minor', size=0)
    fig.colorbar(im, ax=axes[plot_idx], label='Entropy (nats)')

    fig.tight_layout()
    return fig

