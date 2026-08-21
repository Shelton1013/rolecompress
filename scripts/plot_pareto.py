# -*- coding: utf-8 -*-
"""Aggregate 05_eval summaries into the two headline figures + a table:
  (1) accuracy vs mean visual tokens (Pareto) per policy,
  (2) synergy-subset accuracy vs budget (the crossover).

Handles DATA-PARALLEL shards automatically: files named `<name>_s<i>.jsonl.summary.json`
are merged (n-weighted) into a single point `<name>`, so a policy run across 8 GPUs shows
up as one curve.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict


def base_key(summary_path: str) -> str:
    """Filename -> group key, with any `_s<i>` shard suffix stripped."""
    b = os.path.basename(summary_path)
    if b.endswith(".jsonl.summary.json"):
        b = b[: -len(".jsonl.summary.json")]
    elif b.endswith(".summary.json"):
        b = b[: -len(".summary.json")]
    return re.sub(r"_s\d+$", "", b)


def _wavg(summaries, key, weight="n"):
    num = sum(s.get(key, 0) * s.get(weight, 0) for s in summaries if s.get(key) is not None)
    den = sum(s.get(weight, 0) for s in summaries if s.get(key) is not None)
    return (num / den) if den else None


def merge_shards(summaries):
    """n-weighted merge of shard summaries into one point."""
    n = sum(s.get("n", 0) for s in summaries)
    syn_n = sum(s.get("synergy_subset_n", 0) for s in summaries)
    syn_acc = None
    if syn_n:
        syn_acc = sum(s.get("synergy_subset_accuracy", 0) * s.get("synergy_subset_n", 0)
                      for s in summaries if s.get("synergy_subset_accuracy") is not None) / syn_n
    return {
        "policy": summaries[0].get("policy"),
        "n": n,
        "accuracy": _wavg(summaries, "accuracy"),
        "mean_visual_tokens": _wavg(summaries, "mean_visual_tokens"),
        "mean_flops_proxy": _wavg(summaries, "mean_flops_proxy"),
        "synergy_subset_n": syn_n,
        "synergy_subset_accuracy": syn_acc,
        "n_high": summaries[0].get("n_high"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default="pareto.png")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "*.summary.json"))
    if not files:
        print(f"no *.summary.json in {args.results_dir}"); return

    groups = defaultdict(list)
    for f in files:
        try:
            groups[base_key(f)].append(json.load(open(f)))
        except Exception as e:
            print(f"[skip] {f}: {e}")

    points = [merge_shards(v) for v in groups.values() if v and v[0].get("policy")]
    by_policy = defaultdict(list)
    for p in points:
        by_policy[p["policy"]].append(p)
    for pol in by_policy:
        by_policy[pol].sort(key=lambda x: (x["mean_visual_tokens"] or 0))

    # machine-readable table
    table = {pol: [{"vis_tok": round(p["mean_visual_tokens"] or 0, 1),
                    "acc": round(p["accuracy"] or 0, 4),
                    "syn_acc": (round(p["synergy_subset_accuracy"], 4) if p.get("synergy_subset_accuracy") is not None else None),
                    "n": p["n"], "syn_n": p.get("synergy_subset_n", 0)}
                   for p in rows] for pol, rows in by_policy.items()}
    json.dump(table, open(args.out + ".json", "w"), indent=2)
    print(json.dumps(table, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
        for pol, rows in sorted(by_policy.items()):
            xs = [r["mean_visual_tokens"] for r in rows]
            ax1.plot(xs, [r["accuracy"] for r in rows], marker="o", label=pol)
            syn = [(r["mean_visual_tokens"], r["synergy_subset_accuracy"]) for r in rows
                   if r.get("synergy_subset_accuracy") is not None]
            if syn:
                ax2.plot([x for x, _ in syn], [y for _, y in syn], marker="s", label=pol)
        ax1.set_xlabel("mean visual tokens"); ax1.set_ylabel("accuracy")
        ax1.set_title("Accuracy–budget Pareto"); ax1.legend(fontsize=8)
        ax2.set_xlabel("mean visual tokens"); ax2.set_ylabel("synergy-subset accuracy")
        ax2.set_title("Synergy subset (crossover)"); ax2.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(args.out, dpi=140)
        print(f"wrote {args.out}")
    except Exception as e:
        print(f"[plot skipped: {e}] table at {args.out}.json")


if __name__ == "__main__":
    main()
