import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers BabyAI envs
    import matplotlib.pyplot as plt
    from minigrid.utils.baby_ai_bot import BabyAIBot
    from PIL import Image
    return BabyAIBot, Image, gym, mo, plt


@app.cell
def _(mo):
    seed_number = mo.ui.number(value=0, label="Seed")
    tile_size_slider = mo.ui.slider(16, 64, step=16, value=32, label="Tile size")
    mo.hstack([seed_number, tile_size_slider])
    return seed_number, tile_size_slider


@app.cell
def _(BabyAIBot, gym, seed_number, tile_size_slider):
    _env = gym.make("BabyAI-GoToObjDoor-v0", max_steps=200)
    _env = _env.unwrapped
    _obs, _ = _env.reset(seed=int(seed_number.value))
    mission = _obs["mission"]
    _bot = BabyAIBot(_env)
    _prev_action = None

    frames = []
    success = False
    while True:
        frames.append(_env.get_frame(tile_size=tile_size_slider.value, agent_pov=False, highlight=False))
        try:
            _action = _bot.replan(_prev_action)
        except Exception:
            break
        _obs, _reward, _terminated, _truncated, _ = _env.step(_action)
        _prev_action = _action
        if _terminated:
            frames.append(_env.get_frame(tile_size=tile_size_slider.value, agent_pov=False, highlight=False))
            success = True
            break
        if _truncated:
            break
    _env.close()
    return frames, mission, success


@app.cell
def _(frames, mission, mo, success):
    frame_slider = mo.ui.slider(0, len(frames) - 1, value=0, label="Step")
    mo.vstack([
        mo.md(f"**Mission: {mission}  |  Steps: {len(frames)}  |  Success: {success}**"),
        frame_slider,
    ])
    return (frame_slider,)


@app.cell
def _(frame_slider, frames, mission, plt):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.imshow(frames[frame_slider.value])
    _ax.axis("off")
    _ax.set_title(f"{mission}\nStep {frame_slider.value}")
    plt.gca()


@app.cell
def _(Image, frames, mission, mo, seed_number):
    save_button = mo.ui.run_button(label="Save GIF")
    mo.vstack([save_button])
    return (save_button,)


@app.cell
def _(Image, frames, mo, save_button, seed_number):
    from pathlib import Path as _Path
    mo.stop(not save_button.value)
    _task_id = "BabyAI-GoToObjDoor-v0"
    _out_dir = _Path("trajectories")
    _out_dir.mkdir(exist_ok=True)
    _path = _out_dir / f"trajectory_{_task_id}_seed{int(seed_number.value)}.gif"
    _pil_frames = [Image.fromarray(f) for f in frames]
    _pil_frames[0].save(
        _path,
        save_all=True,
        append_images=_pil_frames[1:],
        duration=200,
        loop=0,
    )
    mo.md(f"Saved **{len(frames)} frames** to `{_path}`")


@app.cell
def _(frames, plt):
    import math as _math
    _cols = 8
    _rows = _math.ceil(len(frames) / _cols)
    _fig2, _axes = plt.subplots(_rows, _cols, figsize=(_cols * 2, _rows * 2))
    _axes = _axes.flatten() if hasattr(_axes, "flatten") else [_axes]
    for _i, _ax in enumerate(_axes):
        if _i < len(frames):
            _ax.imshow(frames[_i])
            _ax.set_title(str(_i), fontsize=7)
        _ax.axis("off")
    _fig2.tight_layout()
    plt.gca()


if __name__ == "__main__":
    app.run()
