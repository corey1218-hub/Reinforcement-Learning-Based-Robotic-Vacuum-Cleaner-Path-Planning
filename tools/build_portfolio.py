"""Build the English portfolio and evidence figures from included run records."""
import csv
import gzip
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vacuum_rl.environment import bedroom_layout
from vacuum_rl.plots import draw_map, moving_mean, INK, MUTED, TEAL, ORANGE
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FIG = ROOT/"docs/figures"
FIG.mkdir(parents=True, exist_ok=True)
LAYOUT = bedroom_layout()
TITLE = "Reinforcement Learning-Based Robotic Vacuum Cleaner Path Planning"
REPO = "https://github.com/corey1218-hub/rl-robotic-vacuum-cleaner"
AGG = json.loads((ROOT/"results/matched/aggregate.json").read_text())
SEEDS = AGG["seeds"]


def load_run(preset, algorithm, seed=789):
    p = ROOT/"results"/preset/f"{algorithm}_seed{seed}"
    return json.loads((p/"metrics.json").read_text()), json.loads((p/"trajectory.json").read_text())


def load_history(folder):
    plain = folder/"training.csv"
    handle = plain.open(newline="") if plain.exists() else gzip.open(folder/"training.csv.gz", "rt", newline="")
    with handle:
        return list(csv.DictReader(handle))


def evidence_figures():
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    draw_map(ax, LAYOUT)
    fig.tight_layout()
    fig.savefig(FIG/"environment.png", dpi=210)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for ax, algorithm, label, color in zip(axes, ["qlearning", "sarsa"], ["Q-learning", "SARSA"], [TEAL, ORANGE]):
        data, trajectory = load_run("matched", algorithm,42)
        draw_map(ax, LAYOUT, trajectory["path"], f"{label} | {data['evaluation']['steps']} steps", color)
        from collections import Counter
        visits=Counter(map(tuple,trajectory["path"]))
        for (r,col),count in visits.items():
            if count>1:
                ax.scatter(col,r,s=95,color="#efc69a",alpha=.8,zorder=2)
                if (r,col)!=LAYOUT.home:
                    ax.text(col,r,str(count),ha="center",va="center",fontsize=6.5,color=INK,zorder=7)
    fig.tight_layout(w_pad=2)
    fig.savefig(FIG/"matched_paths_seed42.png", dpi=210)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.35))
    for algorithm, label, color in [("qlearning", "Q-learning", TEAL), ("sarsa", "SARSA", ORANGE)]:
        history = load_history(ROOT/"results/matched"/f"{algorithm}_seed42")
        episodes = np.arange(1, len(history)+1)
        returns = [float(r["reward"]) for r in history]
        successes = [r["success"] == "True" for r in history]
        axes[0].plot(episodes, moving_mean(returns), label=label, color=color, linewidth=1.8)
        axes[1].plot(episodes, moving_mean(successes), label=label, color=color, linewidth=1.8)
    for ax in axes:
        ax.grid(axis="y", color="#e4e8eb", linewidth=.7)
        ax.set_xlabel("Training episode")
        ax.ticklabel_format(axis="x", style="plain")
    axes[0].set_ylabel("Undiscounted episode return")
    axes[1].set_ylabel("Training completion rate")
    axes[1].set_ylim(-.02, 1.02)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(w_pad=2)
    fig.savefig(FIG/"matched_training_seed42.png", dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    for y, algorithm, color in [(2, "qlearning", TEAL), (1, "sarsa", ORANGE)]:
        values = [load_run("matched", algorithm, s)[0]["evaluation"]["steps"] for s in SEEDS]
        ax.plot([min(values), max(values)], [y,y], color=color, alpha=.45, lw=2)
        for dy, value, seed in zip(np.linspace(.22,-.22,len(SEEDS)), values, SEEDS):
            ax.scatter(value, y+dy, color=color, s=45, zorder=4)
        mean, sd = np.mean(values), np.std(values, ddof=1)
        ax.errorbar(mean, y-.34, xerr=sd, fmt="s", color=color, ms=5, capsize=3)
    ax.scatter(102, 0, color=MUTED, marker="D", s=45)
    ax.axvline(72, linestyle="--", color="#a1a9b0", linewidth=1)
    ax.text(72.6, 2.39, "72-step counting lower bound", color=MUTED, fontsize=9)
    ax.set_yticks([2,1,0], ["Q-learning", "SARSA", "Nearest-target BFS"])
    ax.set_xlim(70, 111)
    ax.set_ylim(-.55,2.65)
    ax.set_xlabel("Greedy completion length (steps)")
    ax.grid(axis="x", color="#e4e8eb", linewidth=.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG/"matched_comparison.png", dpi=210)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,3.9))
    for algorithm,label,color in [("qlearning","Q-learning",TEAL),("sarsa","SARSA",ORANGE)]:
        with (ROOT/"results/matched"/f"{algorithm}_aggregate.csv").open() as f:
            records=list(csv.DictReader(f))
        episodes=np.array([int(r["episode"]) for r in records])
        for ax,field in zip(axes,["success","return"]):
            mean=np.array([float(r[field+"_mean"]) for r in records])
            sd=np.array([float(r[field+"_sd"]) for r in records])
            lo,hi=mean-sd,mean+sd
            if field=="success":lo,hi=np.clip(lo,0,1),np.clip(hi,0,1)
            ax.fill_between(episodes,lo,hi,color=color,alpha=.15,linewidth=0)
            ax.plot(episodes,mean,color=color,label=label,lw=2)
        val=AGG["algorithms"][algorithm]["training_success_at_6000_mean"]
        axes[0].scatter(6000,val,c=color,s=28,zorder=5)
        axes[0].annotate(f"{val:.1%}",xy=(6000,val),xytext=(4100,val+.10),
                         color=color,fontsize=10,weight="bold",arrowprops={"arrowstyle":"-","color":color})
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.grid(axis="y",color="#e4e8eb",lw=.6)
        ax.axvline(6000,color="#a8afb5",lw=.8,linestyle="--")
    axes[0].set_ylim(-.03,1.03)
    axes[0].set_ylabel("Training completion rate")
    axes[1].set_ylabel("Undiscounted episode return")
    axes[0].legend(frameon=False,loc="upper left",fontsize=9)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIG/"learning_comparison.png",dpi=220)
    plt.close(fig)

    formulas = {
        "q_update": r"$Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\,\max_{a'}Q(s',a')-Q(s,a)]$",
        "sarsa_update": r"$Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\,Q(s',a')-Q(s,a)]$",
    }
    for name, expression in formulas.items():
        # Fixed canvas and PDF scale preserve the same physical type size.
        # Tight cropping changes the aspect ratio when a formula has a subscript.
        fig = plt.figure(figsize=(7.1, .75))
        fig.text(.025, .5, expression, fontsize=17, va="center", color=INK)
        fig.savefig(FIG/(name+".png"), dpi=240)
        plt.close(fig)


def build_pdf():
    fontdir = Path("/usr/share/fonts/truetype/dejavu")
    if (fontdir/"DejaVuSans.ttf").exists():
        for name, filename in [("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf"), ("Mono", "DejaVuSansMono.ttf")]:
            pdfmetrics.registerFont(TTFont(name,str(fontdir/filename)))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")
    else:
        for name, base in [("Body","Helvetica"),("Bold","Helvetica-Bold"),("Mono","Courier")]:
            pdfmetrics.registerFont(pdfmetrics.Font(name,base,"WinAnsiEncoding"))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")

    W,H=A4
    M=43
    CW=W-2*M
    out=ROOT/"docs/portfolio.pdf"
    c=canvas.Canvas(str(out), pagesize=A4)
    c.setTitle(TITLE)
    c.setAuthor("Corey Chiang")
    c.setSubject("Group 9 course project: reproducible coverage path planning and implementation review")

    def rule(y, x=M, width=CW, color="#d8dde1"):
        c.setStrokeColor(colors.HexColor(color)); c.setLineWidth(.6)
        c.line(x,H-y,x+width,H-y)

    def text(s,x,y,width,size=9.5,leading=14,color=INK,font="Body",limit=None):
        style=ParagraphStyle("p",fontName=font,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=0)
        p=Paragraph(s,style)
        _,h=p.wrap(width,H)
        if limit is not None and h>limit:
            raise ValueError(f"Text overflow: {h}>{limit}: {s[:60]}")
        if y+h>H-45 and y<H-40:
            raise ValueError(f"Page overflow at {y+h}: {s[:60]}")
        p.drawOn(c,x,H-y-h)
        return h

    def image(name,x,y,width,height):
        c.drawImage(str(FIG/name),x,H-y-height,width,height,preserveAspectRatio=True,anchor="c",mask="auto")

    def start(n,tag,title=None,subtitle=None):
        text("COREY CHIANG  /  ENGINEERING PROJECT",M,27,350,7.8,10,MUTED)
        text(tag.upper(),W-M-130,27,130,7.8,10,MUTED)
        rule(48)
        if title:
            text(title,M,74,CW,23,29,font="Bold")
        if subtitle:
            text(subtitle,M,112,CW,10,15,MUTED,limit=45)
        rule(H-40)
        text(f"{n:02d} / 05",W-M-42,H-30,42,7,9,MUTED)

    def heading(s,y):
        text(s,M,y,CW,11.5,16,font="Bold")

    def table(headers,rows,y,widths,align=None,rowh=28,size=9):
        x=M
        rule(y,color=INK)
        for i,h in enumerate(headers):
            text(h,x+5,y+7,widths[i]-10,size-1,11,MUTED,font="Bold",limit=rowh-8)
            x+=widths[i]
        y+=rowh
        rule(y)
        for row in rows:
            x=M
            for i,v in enumerate(row):
                text(str(v),x+5,y+7,widths[i]-10,size,12,limit=rowh-8)
                x+=widths[i]
            y+=rowh
            rule(y)
        return y

    # Page 1: original room illustration beside the code-defined simulation map.
    start(1,"Project overview")
    text(TITLE.replace(" Robotic", "<br/>Robotic", 1),M,76,CW,22.5,32,font="Bold",limit=64)
    text("Coverage planning with Q-learning and SARSA",M,151,CW,12,17,MUTED)
    text("Corey Chiang  |  Electrical Engineering, National Chung Hsing University",M,183,CW,9,13)
    text("A Python simulation of a room-cleaning task: visit every reachable dirty cell, "
         "avoid furniture, and return to the charging position. The project compares two "
         "tabular temporal-difference methods on a fixed bedroom map and records how their "
         "learned policies complete the task.",M,217,CW,10,15,limit=65)
    text("Room illustration",M,293,240,10,14,font="Bold")
    text("Simulation map",M+267,293,242,10,14,font="Bold")
    image("room_illustration.jpg",M,316,241,241)
    image("environment.png",M+253,311,260,250)
    text("Figure 1. Original portfolio room illustration (left) and the code-defined simulation "
         "map (right). Gray cells are blocked; H marks the start and required finish. "
         "Obstacle coordinates are specified in code.",M,576,CW,8.5,12,MUTED,limit=38)
    for x,big,small in [(M,"12 x 12","72 reachable free cells"),
                        (M+177,"71 targets","Home is initially clean"),
                        (M+354,"4 actions","East, west, north, south")]:
        text(big,x,628,155,14,19,font="Bold")
        text(small,x,652,155,8.5,12,MUTED)
    rule(676)
    heading("What this project demonstrates",691)
    text("Environment modeling, reward accounting, tabular control, reproducible experiments, "
         "and verification of complete trajectories. The reviewed package separates training, "
         "evaluation, and visualization so each reported result can be traced to settings and logs.",
         M,715,CW,9.5,14,limit=45)
    text("Across 10 matched seeds, Q-learning averages <b>6.4% fewer completion steps</b> and "
         "<b>22.5% fewer revisit moves</b>. The clearest difference is in training completion rate (page 4).",
         M,764,CW,8.7,12,limit=26)
    c.showPage()

    # Page 2: actual implementation and mathematical contract.
    start(2,"Method","Model and learning loop","The state includes both position and cleaning history.")
    heading("State and termination",158)
    text("Each state is stored as <b>(position_index, remaining_target_mask)</b>. A bit marks "
         "each dirty target; visiting that target clears its bit. This preserves the original "
         "dirty-map information while making the dictionary keys smaller. An episode succeeds "
         "only when the mask is zero and the robot is back at home.",M,184,CW,9.5,14,limit=65)
    table(["Transition event","Reward components","Net reward"],[
        ["Wall or furniture collision","-1 move cost","-1"],
        ["Move to an already clean cell","-1 move cost","-1"],
        ["First visit to a dirty target","-1 move + 3 clean","+2"],
        ["Return home after cleaning all targets","-1 move + 100 finish","+99"],
    ],265,[263,151,95],rowh=29,size=8.8)
    heading("Action-value updates",438)
    text("Q-learning: bootstrap from the largest next-state value.",M,465,CW,9,13)
    formula_height = CW * .75 / 7.1
    image("q_update.png",M,484,CW,formula_height)
    text("SARSA: bootstrap from the next epsilon-greedy action actually sampled.",M,550,CW,9,13)
    image("sarsa_update.png",M,569,CW,formula_height)
    text("At a true terminal transition, both targets reduce to the immediate reward. "
         "A training step cap is an external truncation, so it is logged separately and does "
         "not remove the bootstrap term.",M,640,CW,9.5,14,limit=50)
    heading("Choices retained for source reproducibility",706)
    text("Float32 Q-values, Python's seeded random generator, and linear epsilon decay are retained. "
         "Ties choose the first maximum in E/W/N/S order, as in the source. This rule creates a "
         "directional bias; it is a compatibility choice. Greedy evaluation uses read-only table lookups.",
         M,732,CW,9.5,14,limit=55)
    c.showPage()

    # Page 3: one transparent illustration under the same settings.
    start(3,"Policy comparison","How the policies differ","Matched settings, seed 42. All 10 seeds are included in the next page's aggregate results.")
    table(["Method","Episodes","Train cap","Alpha","Gamma","Steps"],[
        ["Q-learning","8,000","500","0.3","0.95","94"],
        ["SARSA","8,000","500","0.3","0.95","104"],
    ],159,[133,82,86,65,65,78],rowh=29,size=9)
    image("matched_paths_seed42.png",M-7,267,CW+14,243)
    text("Figure 2. Greedy evaluation trajectories. Each has 100% target coverage, returns home, "
         "and records zero collisions. Tan markers identify revisited cells; numbers show visit "
         "counts. Q-learning has 23 revisit moves and SARSA has 33, including return home.",M,516,CW,8.5,12,MUTED,limit=38)
    image("matched_training_seed42.png",M-10,567,CW+20,179)
    text("Figure 3. Seed 42 training histories with identical settings, shown as trailing "
         "100-episode means. The included comparison GIF replays both evaluated routes on the "
         "same step clock. This single seed illustrates behavior; it is not the aggregate result.",M,751,CW,8.3,11.5,MUTED,limit=35)
    c.showPage()

    # Page 4: modest, fully enumerated evidence rather than a broad claim.
    start(4,"New experiment","Matched-setting comparison","8,000 episodes; training cap 500; alpha 0.3; gamma 0.95; identical map and schedule.")
    image("learning_comparison.png",M-10,155,CW+20,222)
    text("Figure 4. Mean of 100-episode rolling statistics across 10 seeds; shading is one sample "
         "standard deviation. At episode 6,000, mean training completion is 58.5% for Q-learning "
         "and 1.0% for SARSA. This measures the exploratory training policy, not a greedy checkpoint.",
         M,386,CW,8.5,12,MUTED,limit=42)
    table(["Method","Final steps<br/>mean +/- SD","Revisit<br/>mean","First 80% window<br/>mean episode","Greedy<br/>success"],[
        ["Q-learning","93.0 +/- 2.5","22.0","6,502","10 / 10"],
        ["SARSA","99.4 +/- 6.7","28.4","7,497","10 / 10"],
    ],454,[105,124,75,133,72],rowh=32,size=8.8)
    text("Milestone: first full 100-episode window with at least 80% completed training episodes; "
         "all runs reach it. Revisit moves include the required return home. Seeds: 1-8, 42, 789.",
         M,562,CW,8.5,12,MUTED,limit=30)
    heading("What the difference means",614)
    text("The largest separation is in learning progress, with a 57.5-percentage-point completion "
         "gap at episode 6,000. Final greedy routes differ more modestly: Q-learning averages "
         "6.4% fewer steps and 22.5% fewer revisits. It wins eight paired seeds, ties one, and loses one.",
         M,640,CW,9.5,14,limit=65)
    text("Both learners finish in all 10 final evaluations. A nearest-target BFS baseline takes "
         "102 steps. These results concern one known, deterministic room; they do not establish "
         "general navigation performance or physical safety.",M,707,CW,9.2,13.5,limit=52)
    c.showPage()

    # Page 5: engineering deliverables, commands, and honest scope.
    start(5,"Engineering handoff","Reproduce and inspect","The figures and metrics in this report are backed by the included run records.")
    table(["Component","Responsibility"],[
        ["environment.py","Map validation, compact state, transitions, reward accounting"],
        ["learning.py","Configuration, exploration, Q-learning and SARSA updates"],
        ["evaluation.py","Read-only rollouts, coverage checks, classical baseline"],
        ["plots.py","Paths, learning curves, and trajectory-based GIF replay"],
        ["results/ and tests/","Recorded experiments, reference trajectories, behavioral tests"],
    ],160,[143,366],rowh=30,size=8.7)
    heading("Run from the project root",369)
    c.setFillColor(colors.HexColor("#f4f6f7")); c.rect(M,H-404-126,CW,126,fill=1,stroke=0)
    commands=["python -m pip install -r requirements.txt",
              "python -m unittest discover -s tests -v",
              "python -m vacuum_rl --preset portfolio --out runs/portfolio --render --gif",
              "python tools/verify_reference_runs.py --results runs/portfolio",
              "python -m vacuum_rl --preset matched --seeds 1 2 3 4 5 6 7 8 42 789",
              "python -m vacuum_rl --replay results/portfolio/qlearning_seed789"]
    for i,line in enumerate(commands):
        text(line,M+12,416+i*17,CW-24,7.7,12,font="Mono",limit=14)
    text("Recorded environment: Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8, Pillow 12.3.0. "
         "Each run includes configuration and version metadata, episode history, a step-by-step "
         "rollout, and a replayable trajectory.",M,546,CW,9,13,limit=45)
    heading("Validation and limits",609)
    text("The package passes 14 automated tests. The uploaded portfolio's 84-step Q-learning "
         "and 96-step SARSA runs are reproduced action by action under their original, unequal "
         "settings. The original report's 94-step Q-learning run matches the shared preset at seed 789.",
         M,634,CW,9.4,14,limit=50)
    text("The 72-step reference is a counting lower bound (71 new targets plus one move home), "
         "not a proved optimum. No sensors, motion noise, battery use, or dynamic obstacles are "
         "modeled. Compact keys reduce overhead; the tabular state space remains exponential.",
         M,688,CW,9.4,14,limit=50)
    text("Based on the vacuum task from the four-member Group 9 course project. Portfolio prepared "
         "by Corey Chiang. The 2026-10-23 review, refactor, and matched-setting experiments are "
         "subsequent additions, documented separately from the original submission.",M,742,CW,7.9,11.5,MUTED,limit=35)
    text(f'<link href="{REPO}" color="#127c7a">Repository: github.com/corey1218-hub/rl-robotic-vacuum-cleaner</link>',
         M,783,CW,7.8,10,limit=11)
    c.showPage()
    c.save()
    print(f"Created {out}")


if __name__ == "__main__":
    evidence_figures()
    build_pdf()
