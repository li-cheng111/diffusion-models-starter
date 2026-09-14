"""Plot the Project 4 FM NFE-FID curve against the Project 2 DDIM baseline."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def rows(path):
    payload=json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    return payload.get("results", [])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fm", default="results/fm_nfe.json")
    ap.add_argument("--ddpm", default="../project2_samplers/runs/benchmark_ddim.json")
    ap.add_argument("--output", default="results/nfe_fid_curve.png")
    args=ap.parse_args()
    fm=[r for r in rows(args.fm) if float(r.get("cfg_scale", 0)) == 0]
    dd=[r for r in rows(args.ddpm) if r.get("sampler", "ddim") == "ddim"]
    fig, ax=plt.subplots(figsize=(7.2,5.0))
    if fm:
        fm=sorted(fm,key=lambda r:r["nfe"])
        ax.plot([r["nfe"] for r in fm],[r["fid"] for r in fm],"o-",label="FM + Euler")
    if dd:
        dd=sorted(dd,key=lambda r:r["nfe"])
        ax.plot([r["nfe"] for r in dd],[r["fid"] for r in dd],"s-",label="DDPM + DDIM")
    ax.set_xscale("log"); ax.set_xlabel("NFE (log scale)"); ax.set_ylabel("FID (lower is better)")
    ax.set_title("CIFAR-10 NFE-FID: Flow Matching vs DDPM")
    ax.grid(True,which="both",alpha=0.25); ax.legend(); fig.tight_layout()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=180); plt.close(fig)
    print(out)

if __name__ == "__main__":
    main()
