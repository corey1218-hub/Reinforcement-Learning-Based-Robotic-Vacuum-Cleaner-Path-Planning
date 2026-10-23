"""CLI: train, evaluate, export evidence, or replay an exported trajectory."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from . import __version__
from .environment import Layout, VacuumEnv, bedroom_layout
from .evaluation import evaluate, nearest_target_route
from .learning import preset_config, train


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_run(folder, layout, metadata, history, summary, path, trace, render=False, gif=False):
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "project_version": __version__,
               "python_version": platform.python_version(), "numpy_version": np.__version__,
               "layout": layout.to_dict(), "metadata": metadata, "evaluation": summary}
    payload["layout_sha256"] = hashlib.sha256(json.dumps(layout.to_dict(),sort_keys=True).encode()).hexdigest()
    (folder / "metrics.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    write_csv(folder / "training.csv", history)
    write_csv(folder / "rollout.csv", trace)
    (folder / "trajectory.json").write_text(json.dumps({"path": path,
        "actions": [t["action"] for t in trace]}, indent=2)+"\n", encoding="utf-8")
    if render or gif:
        from .plots import save_path, save_training, save_gif
        label = metadata["algorithm"].replace("qlearning", "Q-learning").replace("sarsa", "SARSA")
        if render:
            save_path(layout, path, folder / "path.png", f"{label} | {summary['steps']} steps")
            if history:
                save_training(history, folder / "training.png", label)
        if gif:
            save_gif(layout, path, folder / "replay.gif")


def replay(folder):
    info = json.loads((folder/"metrics.json").read_text())
    trajectory = json.loads((folder/"trajectory.json").read_text())
    layout = Layout.from_dict(info["layout"])
    summary, path, trace = evaluate(VacuumEnv(layout), max_steps=info["evaluation"]["evaluation_step_cap"],
                                    actions=trajectory["actions"])
    if list(map(list, path)) != trajectory["path"]:
        raise ValueError("Saved trajectory does not match the environment transitions.")
    for key in ["steps", "success", "cleaned_cells", "reward", "wall_collisions", "obstacle_collisions"]:
        if summary[key] != info["evaluation"][key]:
            raise ValueError(f"Replay disagrees with stored metric: {key}")
    from .plots import save_gif
    save_gif(layout, path, folder/"replay.gif")
    print(f"Verified replay: {summary['steps']} steps; success={summary['success']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["portfolio", "matched"], default="portfolio")
    p.add_argument("--algorithm", choices=["qlearning", "sarsa", "both"], default="both")
    p.add_argument("--seeds", nargs="+", type=int, default=[789])
    p.add_argument("--episodes", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--render", action="store_true")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--save-qtable", action="store_true", help="Optional gzip JSONL; may be large.")
    p.add_argument("--replay", type=Path, help="Verify saved trajectory and regenerate its GIF without retraining.")
    args = p.parse_args(argv)
    if args.replay:
        replay(args.replay)
        return
    if args.eval_steps < 1:
        p.error("--eval-steps must be positive")
    layout = bedroom_layout()
    algorithms = ["qlearning", "sarsa"] if args.algorithm == "both" else [args.algorithm]
    rows = []
    for algorithm in algorithms:
        for seed in args.seeds:
            config = preset_config(args.preset, algorithm, seed)
            overrides = {k: v for k, v in {"episodes": args.episodes, "max_steps": args.max_steps}.items() if v is not None}
            config = replace(config, **overrides)
            start = time.perf_counter()
            q, history = train(VacuumEnv(layout), config)
            seconds = time.perf_counter()-start
            summary, path, trace = evaluate(VacuumEnv(layout), q, args.eval_steps)
            metadata = {**config.to_dict(), "preset": args.preset,
                "override_applied": bool(overrides), "training_seconds": round(seconds, 3),
                "stored_states": len(q), "tie_break": "first maximum in E, W, N, S order",
                "epsilon_schedule": "epsilon0 * (episodes - zero_based_episode) / episodes",
                "training_cap_treatment": "truncation; bootstrap retained",
                "terminal_target": "reward only"}
            folder = args.out / f"{algorithm}_seed{seed}"
            save_run(folder, layout, metadata, history, summary, path, trace, args.render, args.gif)
            if args.save_qtable:
                with gzip.open(folder/"qtable.jsonl.gz", "wt", encoding="utf-8") as f:
                    for (position, mask), vals in q.items():
                        f.write(json.dumps([position, str(mask), vals.tolist()])+"\n")
            row = {"algorithm": algorithm, "seed": seed, "preset": args.preset, **summary}
            rows.append(row)
            print(f"{algorithm} seed={seed}: steps={summary['steps']} success={summary['success']} "
                  f"coverage={summary['coverage']:.3f} train={seconds:.1f}s", flush=True)
            del q
    if args.baselines:
        for seed in args.seeds:
            summary, path, trace = evaluate(VacuumEnv(layout), max_steps=args.eval_steps, seed=seed)
            save_run(args.out/f"random_seed{seed}", layout, {"algorithm": "random", "seed": seed},
                     [], summary, path, trace, args.render, False)
            rows.append({"algorithm": "random", "seed": seed, "preset": args.preset, **summary})
        route = nearest_target_route(layout)
        summary, path, trace = evaluate(VacuumEnv(layout), actions=route, max_steps=args.eval_steps)
        save_run(args.out/"nearest_target", layout, {"algorithm": "nearest_target", "seed": None},
                 [], summary, path, trace, args.render, False)
        rows.append({"algorithm": "nearest_target", "seed": "", "preset": args.preset, **summary})
    write_csv(args.out/"summary.csv", rows)


if __name__ == "__main__":
    main()
