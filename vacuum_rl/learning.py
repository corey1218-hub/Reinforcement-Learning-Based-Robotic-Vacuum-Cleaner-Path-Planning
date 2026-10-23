"""Q-learning and SARSA with reproducible, recorded training settings."""
from dataclasses import asdict, dataclass
from random import Random
import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "qlearning"
    episodes: int = 8000
    max_steps: int = 500
    alpha: float = .3
    gamma: float = .95
    epsilon: float = 1.
    seed: int = 789

    def __post_init__(self):
        if self.algorithm not in {"qlearning", "sarsa"}:
            raise ValueError("Algorithm must be qlearning or sarsa.")
        if self.episodes < 1 or self.max_steps < 1:
            raise ValueError("Episode and step counts must be positive.")
        if not (0 < self.alpha <= 1 and 0 <= self.gamma <= 1 and 0 <= self.epsilon <= 1):
            raise ValueError("Invalid learning rate, discount, or exploration rate.")

    def to_dict(self):
        return asdict(self)


def preset_config(preset, algorithm, seed):
    if preset not in {"portfolio", "matched"}:
        raise ValueError("Unknown preset.")
    if preset == "portfolio" and algorithm == "qlearning":
        return TrainConfig(algorithm, 5000, 100, .2, .99, 1., seed)
    return TrainConfig(algorithm=algorithm, seed=seed)


def choose_action(values, epsilon, rng):
    """Preserve the source demos' first-argmax tie rule and Random.choices RNG."""
    weights = [epsilon / 4] * 4
    weights[int(np.argmax(values))] += 1. - epsilon
    return rng.choices(range(4), weights=weights)[0]


def td_target(reward, next_value, gamma, terminated):
    """True terminal transitions must never bootstrap."""
    return reward if terminated else reward + gamma * next_value


def train(env, config):
    rng = Random(config.seed)
    q = {}
    history = []

    def values(state):
        if state not in q:
            q[state] = np.zeros(4, dtype=np.float32)
        return q[state]

    for episode in range(config.episodes):
        state = env.reset()
        epsilon = config.epsilon * (config.episodes - episode) / config.episodes
        total_reward, collisions, steps = 0., 0, 0
        terminated = env.done
        if terminated:
            history.append({"episode": episode + 1, "epsilon": epsilon, "steps": 0,
                "reward": 0., "coverage": 1., "success": True, "truncated": False,
                "collisions": 0})
            continue
        current = values(state)
        action = choose_action(current, epsilon, rng) if config.algorithm == "sarsa" else None
        for steps in range(1, config.max_steps + 1):
            if config.algorithm == "qlearning":
                current = values(state)
                action = choose_action(current, epsilon, rng)
            next_state, reward, terminated, info = env.step(action)
            following = values(next_state)
            if config.algorithm == "sarsa":
                # Sample even at termination to retain the uploaded demo's seeded
                # RNG stream; the value is discarded by the terminal target mask.
                next_action = choose_action(following, epsilon, rng)
                future = following[next_action]
            else:
                future = np.max(following)
            target = td_target(reward, future, config.gamma, terminated)
            current[action] += config.alpha * (target - current[action])
            total_reward += reward
            collisions += bool(info["collision"])
            if terminated:
                break
            state = next_state
            if config.algorithm == "sarsa":
                current, action = following, next_action
        history.append({"episode": episode + 1, "epsilon": epsilon, "steps": steps,
                        "reward": total_reward, "coverage": 1. - env.remaining / len(env.targets),
                        "success": terminated, "truncated": not terminated,
                        "collisions": collisions})
    return q, history
