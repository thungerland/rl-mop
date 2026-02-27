import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import marimo as mo
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from plotting_utils import (
        plot_action_frequency,
        plot_action_entropy_heatmap,
    )

    return (
        Path,
        json,
        mo,
        np,
        plot_action_entropy_heatmap,
        plot_action_frequency,
    )


@app.cell
def _(Path, mo):
    # Cache selector — scans evaluation_cache/ for routing_data.json files.
    # logit_viz always loads from cache, since action_logits only exist in v3 cache files.
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
def _(Path, json, mo, np, task_dropdown):
    # Cache loader — reads routing_data.json for the selected task/trial.
    # Shows a warning if action_logits are missing (old cache, needs re-evaluation).
    mo.stop(not task_dropdown.value, mo.md("**Select a task above to begin.**"))

    cache_path = Path(task_dropdown.value)
    with open(cache_path) as _f:
        _cached = json.load(_f)

    task_id = _cached['task_id']
    trial = _cached['trial']

    _episodes = _cached.get('episodes')
    routing_data = [
        {
            'position': tuple(r['position']),
            'layer_routing': {k: np.array(v) for k, v in r['layer_routing'].items()},
            'lpc': r['lpc'],
            'env_context': _episodes[r['episode']] if _episodes is not None else r.get('env_context', {}),
            'carrying': r.get('carrying', 0),
            'action_logits': np.array(r['action_logits'], dtype=np.float32) if 'action_logits' in r else None,
        }
        for r in _cached['routing_data']
    ]

    _metrics = _cached['metrics']
    _has_logits = routing_data[0].get('action_logits') is not None

    _warning = ""
    if not _has_logits:
        _warning = "\n\n> **Warning:** This cache file has no `action_logits` (old format). Re-run evaluation to populate logit-based plots."

    mo.md(f"""
    ## Evaluation Cache
    - **Task**: {task_id}
    - **Trial**: {trial}
    - **Evaluated at**: {_cached.get('evaluated_at', 'unknown')[:10]}
    - **Episodes**: {_cached.get('num_episodes', '?')}
    - **Success Rate**: {_metrics.get('success_rate', float('nan')):.1%}
    - **Mean LPC**: {_metrics.get('mean_lpc', float('nan')):.2f}
    - **Routing Samples**: {len(routing_data)}
    - **Action logits present**: {'yes' if _has_logits else 'no'}{_warning}
    """)
    return routing_data, task_id, trial


@app.cell
def _(mo, routing_data):
    # Plot type selector — only shows carrying-phase options when both phases exist,
    # and only shows logit-based options when action_logits are present.
    _has_logits = routing_data[0].get('action_logits') is not None
    _carrying_values = set(s.get('carrying', 0) for s in routing_data)
    _has_both_phases = _carrying_values == {0, 1}

    _all_options = {
        'Action frequency': 'action_frequency',
        'Action frequency by carrying phase': 'action_frequency_carrying',
        'Action entropy heatmap': 'entropy_heatmap',
    }

    _carrying_options = {
        'Action frequency by carrying phase',
    }
    _logit_options = {
        'Action frequency',
        'Action frequency by carrying phase',
        'Action entropy heatmap',
    }

    _available = {}
    for _label, _val in _all_options.items():
        if _label in _logit_options and not _has_logits:
            continue
        if _label in _carrying_options and not _has_both_phases:
            continue
        _available[_label] = _val

    plot_type_dropdown = mo.ui.dropdown(
        options=_available,
        label="Plot Type",
        value=list(_available.keys())[0] if _available else None,
    )

    mo.vstack([
        mo.md("## Plot Selection"),
        plot_type_dropdown,
    ])
    return (plot_type_dropdown,)


@app.cell
def _(plot_type_dropdown, task_id):
    # Environment image sampling — only runs for the entropy heatmap plot type,
    # which needs one env render to show alongside the grid.
    import gymnasium as gym

    env_image = None
    env_mission = ""

    if plot_type_dropdown.value == 'entropy_heatmap':
        _sample_env = gym.make(task_id, render_mode="rgb_array")
        _uw = _sample_env.unwrapped
        _obs, _ = _sample_env.reset()
        env_image = _uw.get_frame(tile_size=32, agent_pov=False, highlight=False)
        env_mission = _obs.get("mission", "") if isinstance(_obs, dict) else ""
        _sample_env.close()
    return env_image, env_mission


@app.cell
def _(
    env_image,
    env_mission,
    plot_action_entropy_heatmap,
    plot_action_frequency,
    plot_type_dropdown,
    routing_data,
):
    # Plot generation — routes to the appropriate function based on dropdown selection.
    _ptype = plot_type_dropdown.value

    if _ptype == 'action_frequency':
        fig = plot_action_frequency(routing_data)
    elif _ptype == 'action_frequency_carrying':
        fig = plot_action_frequency(routing_data, group_by='carrying')
    elif _ptype == 'entropy_heatmap':
        fig = plot_action_entropy_heatmap(routing_data, env_image=env_image, env_mission=env_mission)
    else:
        import matplotlib.pyplot as _plt
        fig, _ax = _plt.subplots()
        _ax.text(0.5, 0.5, "Select a plot type above", ha='center', va='center')
        _ax.axis('off')
    return (fig,)


@app.cell
def _(fig, mo):
    # Interactive display
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    # Save button
    save_button = mo.ui.run_button(label="Save Plot")
    save_button
    return (save_button,)


@app.cell
def _(Path, fig, mo, plot_type_dropdown, save_button, task_id, trial):
    # Save handler — writes to plots/<task_id>/trial_<N>/logit_<plot_type>.png
    _save_result = None
    if save_button.value:
        _plot_dir = Path("plots") / task_id / f"trial_{trial}"
        _plot_dir.mkdir(parents=True, exist_ok=True)
        _filename = f"logit_{plot_type_dropdown.value}.png"
        fig.savefig(_plot_dir / _filename, dpi=150, bbox_inches='tight')
        _save_result = f"Saved to {_plot_dir / _filename}"

    mo.md(f"**{_save_result}**") if _save_result else None
    return


if __name__ == "__main__":
    app.run()
