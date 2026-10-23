"""Compare a fresh portfolio run against the uploaded code's seeded actions."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=ROOT/"results/portfolio")
args = parser.parse_args()
refs = json.loads((ROOT/"tests/fixtures/reference_runs.json").read_text())
for ref in refs:
    run = args.results / f"{ref['algorithm']}_seed{ref['seed']}"
    metrics = json.loads((run/"metrics.json").read_text())["evaluation"]
    actions = json.loads((run/"trajectory.json").read_text())["actions"]
    if actions != ref["source_actions"]:
        raise SystemExit(f"Action sequence differs: {ref['algorithm']}")
    for field in ["steps", "success", "reward"]:
        if metrics[field] != ref[field]:
            raise SystemExit(f"Metric differs: {ref['algorithm']} / {field}")
    print(f"PASS {ref['algorithm']}: {len(actions)} actions identical to uploaded source.")
