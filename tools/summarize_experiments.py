"""Aggregate every committed matched run; never select seeds by their outcome."""
import csv
import gzip
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vacuum_rl.environment import Layout, VacuumEnv
from vacuum_rl.evaluation import evaluate
from vacuum_rl.plots import moving_mean

SEEDS = [1,2,3,4,5,6,7,8,42,789]


def load_history(folder):
    path=folder/"training.csv"
    handle=path.open(newline="") if path.exists() else gzip.open(folder/"training.csv.gz","rt",newline="")
    with handle:
        return list(csv.DictReader(handle))


def aggregate():
    out={"seeds":SEEDS,"protocol":{
        "episodes":8000,"training_step_cap":500,"evaluation_step_cap":500,
        "alpha":.3,"gamma":.95,"epsilon_initial":1.,
        "smoothing_window":100,"band":"one sample standard deviation across seeds",
        "milestone":"first full 100-episode window with at least 80% training completions",
        "representative_animation_seed":42},"algorithms":{}}
    summary=[]
    for preset in ["portfolio","matched"]:
        for folder in sorted((ROOT/"results"/preset).iterdir()):
            if not folder.is_dir() or not (folder/"metrics.json").exists():
                continue
            meta=json.loads((folder/"metrics.json").read_text())
            trajectory=json.loads((folder/"trajectory.json").read_text())
            actual,_,_=evaluate(VacuumEnv(Layout.from_dict(meta["layout"])),actions=trajectory["actions"],
                               max_steps=meta["evaluation"]["evaluation_step_cap"])
            for key in ["steps","success","reward","cleaned_cells","wall_collisions","obstacle_collisions"]:
                if actual[key]!=meta["evaluation"][key]:
                    raise ValueError(f"Invalid archived run: {folder.name}/{key}")
            meta["evaluation"]["revisit_moves"]=actual["revisit_moves"]
            (folder/"metrics.json").write_text(json.dumps(meta,indent=2)+"\n")
            if preset=="matched":
                summary.append({"algorithm":meta["metadata"]["algorithm"],"seed":meta["metadata"].get("seed"),
                                "preset":preset,**meta["evaluation"]})
    for algorithm in ["qlearning","sarsa"]:
        steps,revisits,curves,returns,thresholds=[],[],[],[],[]
        for seed in SEEDS:
            folder=ROOT/"results/matched"/f"{algorithm}_seed{seed}"
            data=json.loads((folder/"metrics.json").read_text())
            config=data["metadata"]
            for k,expected in [("episodes",8000),("max_steps",500),("alpha",.3),("gamma",.95)]:
                if config[k]!=expected:
                    raise ValueError("Mismatched experiment settings.")
            history=load_history(folder)
            if len(history)!=8000:
                raise ValueError("Incomplete history.")
            success=moving_mean([r["success"]=="True" for r in history],100)
            ret=moving_mean([float(r["reward"]) for r in history],100)
            threshold=np.flatnonzero(success[99:]>=.8)
            thresholds.append(int(threshold[0]+100) if len(threshold) else None)
            curves.append(success); returns.append(ret)
            steps.append(data["evaluation"]["steps"] if data["evaluation"]["success"] else None)
            revisits.append(data["evaluation"]["revisit_moves"])
        completed=[x for x in steps if x is not None]
        reached=[x for x in thresholds if x is not None]
        out["algorithms"][algorithm]={"steps_by_seed":steps,"completed_runs":len(completed),
            "step_mean":float(np.mean(completed)),"step_sd":float(np.std(completed,ddof=1)),
            "revisit_moves_by_seed":revisits,"revisit_mean":float(np.mean(revisits)),
            "training_success_at_6000_by_seed":[float(c[5999]) for c in curves],
            "training_success_at_6000_mean":float(np.mean([c[5999] for c in curves])),
            "first_80_percent_window_by_seed":thresholds,
            "first_80_percent_window_mean":float(np.mean(reached)),
            "runs_reaching_threshold":len(reached)}
        rows=[]
        for i in range(8000):
            values=[c[i] for c in curves]; rewards=[r[i] for r in returns]
            rows.append({"episode":i+1,"success_mean":float(np.mean(values)),
                         "success_sd":float(np.std(values,ddof=1)),
                         "return_mean":float(np.mean(rewards)),"return_sd":float(np.std(rewards,ddof=1))})
        with (ROOT/"results/matched"/f"{algorithm}_aggregate.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    q,s=out["algorithms"]["qlearning"],out["algorithms"]["sarsa"]
    out["comparison"]={"completion_step_reduction_percent":100*(s["step_mean"]-q["step_mean"])/s["step_mean"],
        "revisit_move_reduction_percent":100*(s["revisit_mean"]-q["revisit_mean"])/s["revisit_mean"],
        "training_success_gap_at_6000_percentage_points":100*(q["training_success_at_6000_mean"]-s["training_success_at_6000_mean"])}
    (ROOT/"results/matched/aggregate.json").write_text(json.dumps(out,indent=2)+"\n")
    with (ROOT/"results/matched/summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    aggregate()
