"""Deterministic coverage environment, independent of plotting and training."""
from collections import deque
from dataclasses import dataclass

State = tuple[int, int]  # flattened position, remaining-target bit mask
MOVES = ((0, 1), (0, -1), (-1, 0), (1, 0))
ACTION_NAMES = ("E", "W", "N", "S")


@dataclass(frozen=True)
class Layout:
    rows: int
    columns: int
    home: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]

    def __post_init__(self):
        if self.rows < 1 or self.columns < 1:
            raise ValueError("Map dimensions must be positive.")
        for r, c in (*self.obstacles, self.home):
            if not (0 <= r < self.rows and 0 <= c < self.columns):
                raise ValueError("Home and obstacles must lie inside the map.")
        if self.home in self.obstacles:
            raise ValueError("Home cannot be an obstacle.")
        if self.reachable_cells() != set(self.free_cells):
            raise ValueError("Every free cell must be reachable from home.")

    @property
    def free_cells(self):
        return tuple((r, c) for r in range(self.rows)
                     for c in range(self.columns) if (r, c) not in self.obstacles)

    def reachable_cells(self):
        seen, queue = {self.home}, deque([self.home])
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES:
                nxt = r + dr, c + dc
                if (0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.columns
                        and nxt not in self.obstacles and nxt not in seen):
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def to_dict(self):
        return {"rows": self.rows, "columns": self.columns,
                "home": list(self.home), "obstacles": sorted(map(list, self.obstacles))}

    @classmethod
    def from_dict(cls, data):
        return cls(data["rows"], data["columns"], tuple(data["home"]),
                   frozenset(map(tuple, data["obstacles"])))


def bedroom_layout():
    """The exact 12 x 12 obstacle coordinates used by the uploaded demos."""
    rectangles = [(0, 2, 0, 2), (0, 6, 2, 7), (10, 12, 3, 7),
                  (0, 4, 10, 12), (6, 12, 10, 12), (0, 2, 7, 10),
                  (7, 9, 7, 9)]
    obstacles = frozenset((r, c) for r0, r1, c0, c1 in rectangles
                          for r in range(r0, r1) for c in range(c0, c1))
    return Layout(12, 12, (2, 0), obstacles)


class VacuumEnv:
    """Clean each free cell once, then return home; one action costs one step.

    Rewards are additive: -1 per attempted move, +3 on first visiting a
    dirty cell, +100 on successful return. Home starts clean.
    """

    def __init__(self, layout=None):
        self.layout = layout if layout is not None else bedroom_layout()
        self.targets = tuple(p for p in self.layout.free_cells if p != self.layout.home)
        self.target_bits = {p: 1 << i for i, p in enumerate(self.targets)}
        self.reset()

    @property
    def state(self) -> State:
        r, c = self.position
        return r * self.layout.columns + c, self.remaining_mask

    @property
    def remaining(self):
        return self.remaining_mask.bit_count()

    @property
    def done(self):
        return self.remaining_mask == 0 and self.position == self.layout.home

    def reset(self):
        self.position = self.layout.home
        self.remaining_mask = (1 << len(self.targets)) - 1
        return self.state

    def step(self, action):
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(4):
            raise ValueError("Action must be an integer in [0, 3].")
        info = {"collision": "", "cleaned": 0, "move_reward": 0.,
                "clean_reward": 0., "finish_reward": 0.}
        if self.done:
            return self.state, 0., True, info
        r, c = self.position
        dr, dc = MOVES[action]
        nxt = r + dr, c + dc
        info["move_reward"] = -1.
        if not (0 <= nxt[0] < self.layout.rows and 0 <= nxt[1] < self.layout.columns):
            info["collision"] = "wall"
        elif nxt in self.layout.obstacles:
            info["collision"] = "obstacle"
        else:
            self.position = nxt
            bit = self.target_bits.get(nxt, 0)
            if self.remaining_mask & bit:
                self.remaining_mask &= ~bit
                info["cleaned"], info["clean_reward"] = 1, 3.
            if self.done:
                info["finish_reward"] = 100.
        reward = info["move_reward"] + info["clean_reward"] + info["finish_reward"]
        return self.state, reward, self.done, info
