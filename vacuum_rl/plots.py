"""Figures derived from saved trajectories and episode logs."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

INK, MUTED, TEAL, ORANGE = "#243242", "#687381", "#127c7a", "#c17b37"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titleweight": "bold", "figure.facecolor": "white"})


def draw_map(ax, layout, path=None, title=None, color=TEAL):
    grid = np.zeros((layout.rows, layout.columns))
    for r, c in layout.obstacles:
        grid[r, c] = 1
    ax.imshow(grid, cmap=ListedColormap(["#f4f6f6", "#b7bec5"]), vmin=0, vmax=1)
    ax.set_xticks(range(layout.columns))
    ax.set_yticks(range(layout.rows))
    ax.set_xticks(np.arange(-.5, layout.columns, 1), minor=True)
    ax.set_yticks(np.arange(-.5, layout.rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="both", length=0, labelsize=8)
    if path:
        xs, ys = [p[1] for p in path], [p[0] for p in path]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=.9, zorder=3)
        # Direction markers every six moves keep repeated edges readable.
        for i in range(0, len(path)-1, 6):
            x, y = path[i][1], path[i][0]
            dx, dy = path[i+1][1]-x, path[i+1][0]-y
            if dx or dy:
                ax.annotate("", xy=(x+.6*dx, y+.6*dy), xytext=(x, y),
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
    r, c = layout.home
    ax.scatter(c, r, marker="s", s=85, c=INK, edgecolors="white", linewidths=1, zorder=5)
    ax.text(c, r, "H", ha="center", va="center", fontsize=7, weight="bold", color="white", zorder=6)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    if title:
        ax.set_title(title, loc="left", pad=12, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_path(layout, path, output, title="Greedy rollout", color=TEAL):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    draw_map(ax, layout, path, title, color)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def moving_mean(values, window=100):
    a = np.asarray(values, dtype=float)
    totals = np.cumsum(np.r_[0., a])
    indices = np.arange(len(a))
    starts = np.maximum(0, indices + 1 - window)
    return (totals[indices+1] - totals[starts]) / (indices+1-starts)


def save_training(history, output, title):
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.4), sharex=True)
    episodes = [r["episode"] for r in history]
    for ax, field, label in zip(axes, ["reward", "coverage", "success"],
                              ["Episode return", "Target coverage", "Completion rate"]):
        raw = np.array([r[field] for r in history], dtype=float)
        if field == "reward":
            ax.plot(episodes, raw, color=TEAL, alpha=.14, linewidth=.45)
        ax.plot(episodes, moving_mean(raw), color=TEAL, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#e4e8eb", linewidth=.6)
        if field != "reward":
            ax.set_ylim(-.02, 1.02)
    axes[0].set_title(title + " | 100-episode moving mean", loc="left", fontsize=11)
    axes[-1].set_xlabel("Training episode")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_gif(layout, path, output):
    """Draw from the evaluated path; never rerun a policy while rendering."""
    cell, margin, header = 35, 24, 58
    width, height = layout.columns*cell + 2*margin, layout.rows*cell + header+margin
    frames, visited = [], {layout.home}
    for i, pos in enumerate(path):
        visited.add(tuple(pos))
        frame = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((margin, 12), "ROBOTIC VACUUM | GREEDY POLICY REPLAY", fill=INK)
        draw.text((margin, 30), f"Step {i}/{len(path)-1}   Cleaned {len(visited)-1}/{len(layout.free_cells)-1}", fill=MUTED)
        for r in range(layout.rows):
            for c in range(layout.columns):
                p = r, c
                fill = "#b7bec5" if p in layout.obstacles else ("#c6e6df" if p in visited else "#f3eee1")
                x, y = margin+c*cell, header+r*cell
                draw.rectangle((x,y,x+cell-1,y+cell-1), fill=fill, outline="white")
        if i:
            pts=[(margin+p[1]*cell+cell//2, header+p[0]*cell+cell//2) for p in path[:i+1]]
            draw.line(pts, fill=TEAL, width=2)
        hr, hc = layout.home
        draw.text((margin+hc*cell+12, header+hr*cell+10), "H", fill=INK)
        x, y = margin+pos[1]*cell+cell//2, header+pos[0]*cell+cell//2
        draw.ellipse((x-9,y-9,x+9,y+9),fill=INK,outline="white",width=2)
        frames.append(frame)
    durations = [90]*len(frames)
    durations[-1] = 1300
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
