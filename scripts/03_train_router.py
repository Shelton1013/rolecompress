# -*- coding: utf-8 -*-
"""Stage 3: train the RoleRouter on cached features + self-supervised soft role targets.

Small model, trains in minutes-hours on 1 GPU. Reports role accuracy vs the hard labels
and (if provided) agreement with a human-labeled subset.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from rolecompress.data import RouterFeatureDataset, collate_router, read_jsonl
from rolecompress.router import RoleRouter, RouterConfig, router_loss
from rolecompress.roles import NUM_ROLES, Role


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_role_correct = {r.name: [0, 0] for r in Role}
    with torch.no_grad():
        for b in loader:
            logits = model(b["vis"].to(device), b["txt"].to(device), b["scal"].to(device), b["mask"].to(device))
            pred = logits.argmax(-1)
            m = b["mask"].to(device)
            gt = b["hard"].to(device)
            ok = (pred == gt) & m
            correct += ok.sum().item(); total += m.sum().item()
            for r in Role:
                sel = (gt == int(r)) & m
                per_role_correct[r.name][0] += ((pred == gt) & sel).sum().item()
                per_role_correct[r.name][1] += sel.sum().item()
    acc = correct / max(1, total)
    per_role = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in per_role_correct.items()}
    model.train()
    return acc, per_role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--feats_dir", required=True)
    ap.add_argument("--val_labels", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d_visual", type=int, default=1280)
    ap.add_argument("--d_text", type=int, default=3584)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl_weight", type=float, default=1.0)
    ap.add_argument("--ce_weight", type=float, default=0.3)
    ap.add_argument("--synergy_class_weight", type=float, default=3.0,
                    help="upweight the rare SYNERGISTIC class in the aux CE")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RouterFeatureDataset(args.labels, args.feats_dir)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate_router, num_workers=4)
    val_dl = None
    if args.val_labels:
        val_ds = RouterFeatureDataset(args.val_labels, args.feats_dir)
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, collate_fn=collate_router)

    cfg = RouterConfig(d_visual=args.d_visual, d_text=args.d_text)
    model = RoleRouter(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(dl)))

    cw = torch.ones(NUM_ROLES, device=device)
    cw[int(Role.SYNERGISTIC)] = args.synergy_class_weight

    best = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in dl:
            logits = model(b["vis"].to(device), b["txt"].to(device), b["scal"].to(device), b["mask"].to(device))
            loss = router_loss(logits, b["soft"].to(device), b["mask"].to(device), b["hard"].to(device),
                               args.kl_weight, args.ce_weight, cw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        msg = f"epoch {ep}: loss {tot/max(1,len(dl)):.4f}"
        if val_dl is not None:
            acc, per_role = evaluate(model, val_dl, device)
            msg += f" | val_role_acc {acc:.3f} | " + " ".join(f"{k[:3]}={v:.2f}" for k, v in per_role.items())
            if acc > best:
                best = acc
                torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, args.out)
        print(msg)
    if val_dl is None:
        torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, args.out)
    print(f"saved router to {args.out} (best val role acc {best:.3f})")


if __name__ == "__main__":
    main()
