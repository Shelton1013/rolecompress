# -*- coding: utf-8 -*-
"""Aggregate 05_eval summaries into the two headline figures:
  (1) accuracy vs mean visual tokens (Pareto) per policy,
  (2) synergy-subset accuracy vs budget (the crossover: rolecompress >> remo at low budget).
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default="pareto.png")
    args = ap.parse_args()

    summaries = []
    for p in glob.glob(os.path.join(args.results_dir, "*.summary.json")):
        summaries.append(json.load(open(p)))
    if not summaries:
        print("no summaries found"); return

    by_policy = {}
    for s in summaries:
        by_policy.setdefault(s["policy"], []).append(s)
    for pol in by_policy:
        by_policy[pol].sort(key=lambda x: x["mean_visual_tokens"])

    # dump a machine-readable table too
    table = {pol: [{"vtok": s["mean_visual_tokens"], "acc": s["accuracy"],
                    "syn_acc": s.get("synergy_subset_accuracy")} for s in rows]
             for pol, rows in by_policy.items()}
    json.dump(table, open(args.out + ".json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for pol, rows in by_policy.items():
            ax1.plot([r["mean_visual_tokens"] for r in rows], [r["accuracy"] for r in rows], marker="o", label=pol)
            syn = [(r["mean_visual_tokens"], r.get("synergy_subset_accuracy")) for r in rows if r.get("synergy_subset_accuracy") is not None]
            if syn:
                ax2.plot([x for x, _ in syn], [y for _, y in syn], marker="s", label=pol)
        ax1.set_xlabel("mean visual tokens"); ax1.set_ylabel("accuracy"); ax1.set_title("Accuracy–budget Pareto"); ax1.legend()
        ax2.set_xlabel("mean visual tokens"); ax2.set_ylabel("synergy-subset accuracy"); ax2.set_title("Synergy subset (crossover)"); ax2.legend()
        plt.tight_layout(); plt.savefig(args.out, dpi=140)
        print(f"wrote {args.out}")
    except Exception as e:
        print(f"[plot skipped: {e}] table at {args.out}.json")


if __name__ == "__main__":
    main()
