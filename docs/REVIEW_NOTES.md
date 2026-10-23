# Review notes

Reviewed on 2026-10-23. Scope: the vacuum-cleaner task in the two supplied ZIPs
and its existing portfolio PDF. Delivery-robot and Expected SARSA variants were
identified but were not evaluated or incorporated into this focused portfolio.

## What was verified

- The two vacuum map definitions have the same 72 obstacles and home at (2, 0).
  All 72 free cells are reachable; 71 start dirty.
- The curated source reruns produce 84 Q-learning steps and 96 SARSA steps.
  Both clean all targets, return home, and have no collision attempts in their
  greedy evaluation. Total rewards are 229 and 217, respectively.
- Both refactored action sequences exactly match those source reruns.
- The original report's 94-step Q-learning result uses different training
  settings. The matched preset reproduces 94 at seed 789. It is not evidence
  that the 84-step curated run is incorrect.
- Original source fingerprints are recorded in `source_manifest.json`.
  Reference action sequences are in `tests/fixtures/reference_runs.json`.
- The portfolio reuses the room illustration embedded in the supplied PDF.
  It is shown beside the code-defined grid for context; no image-based map
  reconstruction is performed. Both update formulas use the same font size,
  canvas dimensions, and PDF scaling.

## Changes made

| Finding | Change | Effect |
|---|---|---|
| Whole 144-entry dirty map stored in each state key | Store remaining targets as an integer bit mask | Same Markov information, fewer Python objects |
| Q-learning and SARSA duplicate substantial setup code | Shared training/configuration and separate environment/evaluation modules | Easier inspection and consistent settings |
| Terminal targets always include a next-state term | Explicit terminal mask | Correct even if terminal values are nonzero; did not alter verified source routes because their terminal values stayed zero |
| Evaluation inserts zero rows into the Q-table for unseen states | Read-only lookup and unseen-state counter | Evaluation cannot silently change the model |
| Settings and plots are hard-coded in separate demos | CLI, metadata JSON, per-episode and per-action CSV | Reproduction commands and output provenance are explicit |
| Plots alone do not preserve episode data | Export metrics and histories | Figures can be regenerated and independently checked |
| Existing prose calls SARSA safer or inherently more conservative | Remove the safety inference | The deterministic simulator provides no physical-robot safety evidence |
| 72 described as a proven shortest route | Label it a counting lower bound and show the reasoning | No unsupported optimality claim |
| Unequal hyperparameters presented beside an algorithm ranking | Separate portfolio preset from matched-setting check | Results answer distinct questions |
| Many screenshots, cached bytecode, and full Q-tables in the raw archive | Omit caches and redundant frame exports from the release | Smaller, focused project package |

SARSA retains the original random draw for its next action even at termination
solely to preserve the seeded RNG stream; the terminal target discards its value.
The first-argmax tie rule, action ordering, float32 Q-values, and linear epsilon
schedule are also retained and documented. These are compatibility choices,
not claims of best exploration design.

## Validation

1. Run `python -m unittest discover -s tests -v` for environment accounting,
   reward balance, reachability, terminal behavior, deterministic learning,
   evaluation immutability, and reference trajectory checks.
2. Rerun the portfolio preset, then execute
   `python tools/verify_reference_runs.py --results runs/portfolio`.
3. Run the matched preset for seeds 1-8, 42, and 789; inspect `summary.csv`.
4. Replay archived trajectories to check stored positions and metrics against
   the environment, independently of the rendering code.

## Limits of the evidence

The matched check uses ten seeds and one fixed deterministic map. There is no
held-out map set, dynamic-obstacle test, noise model, battery model, physical
robot deployment, or proof of a globally shortest coverage route. Success on
the training map should not be described as general navigation performance.
Training completion rates refer to exploratory behavior under the shared
epsilon schedule. They are not greedy-policy evaluation checkpoints. The first
80% rolling window is a descriptive milestone, not guaranteed convergence.
The seed 42 animation is an illustrative pair; all ten outcomes, including a
SARSA win and a tie, are retained. Uniform random action runs are capped at 500 steps; their capped lengths are
not completion lengths. A deterministic nearest-target BFS baseline is provided,
but it is not an exact solver or an exhaustive baseline study.

This reviewed version was prepared with AI assistance after the original group
submission. The changes above and the new comparison must not be represented as
work already completed during the original course project.
