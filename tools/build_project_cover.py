"""Render the portfolio cover from the archived, matched seed-42 rollouts."""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Reinforcement Learning-Based Robotic Vacuum Cleaner Path Planning"
INK = "#263239"
MUTED = "#657078"
GRID = "#dce1e3"
OBSTACLE = "#c4cbcf"
REVISIT = "#f2e3cd"


def load_run(algorithm):
    folder = ROOT / "results" / "matched" / f"{algorithm}_seed42"
    metrics = json.loads((folder / "metrics.json").read_text())
    trajectory = json.loads((folder / "trajectory.json").read_text())
    path = [tuple(p) for p in trajectory["path"]]
    layout = metrics["layout"]
    obstacles = set(map(tuple, layout["obstacles"]))
    home = tuple(layout["home"])
    visited = Counter(path)
    evaluation = metrics["evaluation"]
    assert path[0] == path[-1] == home
    assert len(path) - 1 == evaluation["steps"]
    assert len(visited) - 1 == evaluation["cleaned_cells"] == 71
    assert sum(visited.values()) - len(visited) == evaluation["revisit_moves"]
    assert all(p not in obstacles for p in path)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
               for a, b in zip(path, path[1:]))
    return metrics, path, visited


def draw_path(ax, metrics, path, visits, color):
    layout = metrics["layout"]
    obstacles = set(map(tuple, layout["obstacles"]))
    for r in range(layout["rows"]):
        for c in range(layout["columns"]):
            cell = (r, c)
            fill = OBSTACLE if cell in obstacles else (
                REVISIT if visits[cell] > 1 else "white")
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1,
                                   facecolor=fill, edgecolor=GRID,
                                   linewidth=.5, zorder=0))
    ax.plot([p[1] for p in path], [p[0] for p in path],
            color=color, linewidth=1.7, solid_capstyle="round",
            solid_joinstyle="round", zorder=2)
    # Sparse direction marks retain the legibility of the recorded route.
    for a, b in list(zip(path, path[1:]))[::5]:
        ax.annotate("", xy=(a[1] + .72 * (b[1] - a[1]),
                           a[0] + .72 * (b[0] - a[0])),
                    xytext=(a[1] + .34 * (b[1] - a[1]),
                            a[0] + .34 * (b[0] - a[0])),
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "lw": 1, "mutation_scale": 7}, zorder=3)
    r, c = layout["home"]
    ax.scatter([c], [r], s=110, marker="s", color="white",
               edgecolor=INK, linewidth=.85, zorder=5)
    ax.text(c, r, "H", ha="center", va="center", fontsize=7.4,
            fontweight="bold", color=INK, zorder=6)
    ax.set_xlim(-.5, 11.5)
    ax.set_ylim(11.5, -.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#aeb8bd")
        spine.set_linewidth(.75)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    runs = [load_run(algorithm) for algorithm in ("qlearning", "sarsa")]
    settings = ("episodes", "max_steps", "alpha", "gamma", "epsilon", "seed")
    assert all(runs[0][0]["metadata"][k] == runs[1][0]["metadata"][k]
               for k in settings)
    assert runs[0][0]["layout"] == runs[1][0]["layout"]

    fig = plt.figure(figsize=(12, 9), dpi=150, facecolor="white")
    fig.text(.055, .953, TITLE.replace(" Robotic", "\nRobotic", 1), fontsize=27,
             fontweight="bold", va="top", linespacing=1.15)
    fig.text(.057, .845, "Q-learning and SARSA in a 12 × 12 GridWorld",
             fontsize=11.5, color=MUTED)
    fig.add_artist(Line2D([.055, .945], [.825, .825], color=GRID, lw=.8))

    for x, label, color, run in zip(
            (.07, .555), ("Q-learning", "SARSA"),
            ("#316d83", "#a1683c"), runs):
        metrics, path, visits = run
        evaluation = metrics["evaluation"]
        fig.text(x, .788, label, fontsize=21, fontweight="bold", color=color)
        fig.text(x, .752, f"{evaluation['steps']} steps", fontsize=17.5,
                 fontweight="bold")
        fig.text(x + .161, .753, f"{evaluation['revisit_moves']} revisits",
                 fontsize=12.5, color=MUTED)
        draw_path(fig.add_axes([x, .229, .375, .5]), metrics, path, visits, color)

    legend_y = .182
    for x, fill, label in ((.07, OBSTACLE, "Obstacle"),
                           (.276, REVISIT, "Visited more than once")):
        fig.add_artist(Rectangle((x, legend_y - .006), .0135, .018,
                                 transform=fig.transFigure, facecolor=fill,
                                 edgecolor=GRID, linewidth=.7))
        fig.text(x + .022, legend_y, label, fontsize=10.1, va="center", color=MUTED)
    fig.text(.66, legend_y, "H  Start / finish", fontsize=10.1,
             va="center", color=MUTED)
    fig.add_artist(Line2D([.055, .945], [.14, .14], color=GRID, lw=.8))
    fig.text(.057, .104, "Both clean all 71 targets and return home.",
             fontsize=12.5, fontweight="bold")
    fig.text(.057, .065, "Recorded greedy rollouts · identical training settings · seed 42",
             fontsize=10.4, color=MUTED)
    fig.text(.057, .039, "8,000 training episodes · full ten-seed comparison in the project report",
             fontsize=9.2, color=MUTED)
    destination = ROOT / "docs" / "figures" / "project_cover.png"
    fig.savefig(destination, dpi=150, facecolor="white", metadata={"Title": TITLE})
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
