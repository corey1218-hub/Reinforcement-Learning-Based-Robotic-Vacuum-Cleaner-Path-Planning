"""Create a synchronized, equal-settings comparison from seed 42 trajectories."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vacuum_rl.environment import bedroom_layout


def comparison():
    layout=bedroom_layout()
    runs=[]
    for algorithm,label,color in [("qlearning","Q-LEARNING","#127c7a"),("sarsa","SARSA","#c17b37")]:
        folder=ROOT/"results/matched"/f"{algorithm}_seed42"
        path=[tuple(p) for p in json.loads((folder/"trajectory.json").read_text())["path"]]
        with (folder/"rollout.csv").open() as f:
            trace=list(csv.DictReader(f))
        runs.append((label,color,path,trace))
    cell=29
    side=cell*12
    pad,gap,header=25,38,118
    width=2*side+2*pad+gap
    height=header+side+88
    fontdir=Path("/usr/share/fonts/truetype/dejavu")
    def font(size,bold=False):
        p=fontdir/("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()
    frames=[]
    longest=max(len(r[2]) for r in runs)
    for step in range(longest):
        frame=Image.new("RGB",(width,height),"white")
        d=ImageDraw.Draw(frame)
        d.text((pad,13),"SAME ROOM. SAME TRAINING SETTINGS.",font=font(18,True),fill="#243242")
        d.text((pad,41),"Seed 42 | 8,000 episodes | alpha 0.3 | gamma 0.95 | synchronized step clock",font=font(12),fill="#687381")
        for idx,(label,color,path,trace) in enumerate(runs):
            x0=pad+idx*(side+gap)
            t=min(step,len(path)-1)
            visited=set(path[:t+1])
            repeats=sum(not row['collision'] and int(row['cleaned'])==0 for row in trace[:t])
            d.text((x0,73),label,font=font(18,True),fill=color)
            status=f"{t} steps | {repeats} revisits"
            if t==len(path)-1:
                status+=" | FINISHED"
            d.text((x0,98),status,font=font(11),fill="#243242")
            counts=Counter(path[:t+1])
            for r in range(12):
                for c in range(12):
                    p=(r,c)
                    fill="#b7bec5" if p in layout.obstacles else ("#c6e6df" if p in visited else "#f3eee1")
                    if p not in layout.obstacles and counts[p]>1:
                        fill="#f0c89e"
                    x,y=x0+c*cell,header+r*cell
                    d.rectangle((x,y,x+cell-1,y+cell-1),fill=fill,outline="white")
            if t:
                points=[(x0+p[1]*cell+cell//2,header+p[0]*cell+cell//2) for p in path[:t+1]]
                d.line(points,fill=color,width=2)
            r,c=path[t];x,y=x0+c*cell+cell//2,header+r*cell+cell//2
            d.ellipse((x-8,y-8,x+8,y+8),fill=color,outline="white",width=2)
            clean=len(visited)-1
            y=header+side+14
            d.text((x0,y),f"Targets cleaned: {clean}/71",font=font(12),fill="#243242")
            d.rectangle((x0,y+23,x0+side,y+30),fill="#e7ebed")
            d.rectangle((x0,y+23,x0+side*clean/71,y+30),fill=color)
        d.text((pad,height-25),"Green: first visit | Tan: revisit | Revisits include the required return home",font=font(11),fill="#687381")
        frames.append(frame)
    out=ROOT/"docs/figures"
    out.mkdir(parents=True,exist_ok=True)
    durations=[100]*len(frames);durations[-1]=2200
    frames[0].save(out/"comparison_seed42.gif",save_all=True,append_images=frames[1:],duration=durations,loop=0)
    frames[-1].save(out/"comparison_seed42.png")
    print(f"Created synchronized comparison, {len(frames)-1} action steps.")


if __name__=="__main__":
    comparison()
