"""Evaluation without exploration or mutation of the learned Q-table."""
from collections import deque
from random import Random
import numpy as np
from .environment import MOVES


def evaluate(env, q=None, max_steps=500, seed=789, actions=None):
    if max_steps < 1:
        raise ValueError("Evaluation step cap must be positive.")
    state = env.reset()
    rng, zero = Random(seed), np.zeros(4, dtype=np.float32)
    path = [env.position]
    trace = []
    missing = 0
    for i in range(max_steps):
        if env.done:
            break
        if actions is not None:
            if i >= len(actions):
                break
            action = int(actions[i])
        elif q is None:
            action = rng.randrange(4)
        else:
            missing += state not in q
            action = int(np.argmax(q.get(state, zero)))
        state, reward, done, info = env.step(action)
        path.append(env.position)
        trace.append({"step": i + 1, "action": action, "row": env.position[0],
                      "column": env.position[1], "remaining": env.remaining,
                      "reward": reward, "done": done, **info})
    success = env.done
    summary = {"steps": len(trace), "success": success,
        "coverage": 1. - env.remaining / len(env.targets) if env.targets else 1.,
        "cleaned_cells": len(env.targets) - env.remaining,
        "target_cells": len(env.targets), "remaining_cells": env.remaining,
        "returned_home": env.position == env.layout.home,
        "reward": sum(t["reward"] for t in trace),
        "wall_collisions": sum(t["collision"] == "wall" for t in trace),
        "obstacle_collisions": sum(t["collision"] == "obstacle" for t in trace),
        "revisit_moves": sum(not t["collision"] and not t["cleaned"] for t in trace),
        "unseen_state_actions": missing, "evaluation_step_cap": max_steps,
        "stop_reason": "completed" if success else ("step_cap" if len(trace) == max_steps else "actions_exhausted")}
    return summary, path, trace


def nearest_target_route(layout):
    """Deterministic BFS nearest-unvisited heuristic, then shortest route home.

    This is a classical, full-map baseline, not a learned or optimal policy.
    """
    remaining = set(layout.free_cells) - {layout.home}
    current, actions = layout.home, []
    while remaining or current != layout.home:
        goals = remaining if remaining else {layout.home}
        queue = deque([(current, [])])
        seen = {current}
        while queue:
            pos, route = queue.popleft()
            if pos in goals:
                break
            for action, (dr, dc) in enumerate(MOVES):
                nxt = pos[0] + dr, pos[1] + dc
                if (0 <= nxt[0] < layout.rows and 0 <= nxt[1] < layout.columns
                        and nxt not in layout.obstacles and nxt not in seen):
                    seen.add(nxt)
                    queue.append((nxt, route + [action]))
        else:
            raise ValueError("Unreachable target.")
        for action in route:
            dr, dc = MOVES[action]
            current = current[0] + dr, current[1] + dc
            remaining.discard(current)
            actions.append(action)
    return actions
