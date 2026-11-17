"""Offline analysis of risk logs written by train_rlpd.py (<checkpoint_path>/risk_log/*.pkl).

This is the cheap "kill test" for the attention-budget idea:
  1. Oracle bound: fraction of policy-controlled steps that lie within H steps before a bad event
     (failure / timeout / human takeover). A perfect predictor needs at least this much attention.
     If it is >> your budget, the idea has little headroom on this task.
  2. Predictiveness of each risk score (AUROC, AUPRC) for "bad event within H steps".
  3. Recall at a fixed attention budget: fraction of bad events preceded by an alert with at least
     `min_lead` steps of lead time (enough for the human to react), when alerting on the top-b steps.
  4. Training curves: success rate and attention fraction per bin of env steps.

    python scripts/analyze_risk_log.py <checkpoint_path> [--H 10] [--budget 0.2] [--min_lead 10]
"""
import argparse
import glob
import os
import pickle as pkl

import numpy as np


def load(path):
    files = sorted(glob.glob(os.path.join(path, "risk_log", "risk_*.pkl")),
                   key=lambda f: int(os.path.basename(f)[5:-4]))
    rows = []
    for f in files:
        with open(f, "rb") as fh:
            rows.extend(pkl.load(fh))
    if not rows:
        raise SystemExit(f"no risk logs under {path}/risk_log")
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def auroc(score, y):
    ok = ~np.isnan(score)
    score, y = score[ok], y[ok].astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort") + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(score, y):
    ok = ~np.isnan(score)
    score, y = score[ok], y[ok].astype(bool)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-score)
    tp = np.cumsum(y[order])
    precision = tp / np.arange(1, len(tp) + 1)
    return float((precision * y[order]).sum() / y.sum())


def bad_events(L):
    """Indices of bad events: start of a human takeover, or an unsuccessful episode end."""
    interv = L["intervened"].astype(bool)
    takeover_start = interv & ~np.concatenate([[False], interv[:-1]])
    episode_fail = L["done"].astype(bool) & ~L["succeed"].astype(bool)
    return np.where(takeover_start | episode_fail)[0]


def labels_within(n, events, H, done):
    """y[t] = 1 if a bad event happens in (t, t+H] within the same episode."""
    y = np.zeros(n, dtype=bool)
    ep_id = np.concatenate([[0], np.cumsum(done[:-1])])
    for e in events:
        lo = max(0, e - H)
        idx = np.arange(lo, e)
        y[idx[ep_id[idx] == ep_id[e]]] = True
    return y


def replay(score, thr, react, attend):
    """Same FSM as SimHumanIntervention in alert mode. Returns (attention mask, watching mask, alert start)."""
    n = len(score)
    attn, watch = np.zeros(n, bool), np.zeros(n, bool)
    alert_t = np.full(n, -1)
    state, timer, t_alert = "away", 0, -1
    for t in range(n):
        if score[t] > thr:
            if state == "away":
                state, timer, t_alert = "react", react, t
            elif state == "watch":
                timer = max(timer, attend)
        if state == "react":
            attn[t] = True
            timer -= 1
            if timer <= 0:
                state, timer = "watch", attend
        elif state == "watch":
            attn[t] = watch[t] = True
            alert_t[t] = t_alert
            timer -= 1
            if timer <= 0:
                state = "away"
    return attn, watch, alert_t


def replay_at_budget(score, events, budget, react, attend):
    lo, hi = np.nanmin(score[np.isfinite(score)]), np.nanmax(score[np.isfinite(score)])
    for _ in range(40):  # bisection on the threshold so that attention ~= budget
        mid = (lo + hi) / 2
        attn, _, _ = replay(score, mid, react, attend)
        lo, hi = (mid, hi) if attn.mean() > budget else (lo, mid)
    attn, watch, alert_t = replay(score, hi, react, attend)
    hits = [e for e in events if watch[e]]
    leads = [e - alert_t[e] for e in hits]
    return len(hits) / max(1, len(events)), float(attn.mean()), leads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--H", type=int, default=10, help="prediction horizon in steps (10 = 1 s)")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--react", type=int, default=20, help="human reaction/switch steps per alert")
    ap.add_argument("--attend", type=int, default=30, help="steps the human watches after reacting")
    ap.add_argument("--bin", type=int, default=2000)
    args = ap.parse_args()

    L = load(args.path)
    n = len(L["step"])
    done = L["done"].astype(bool)
    policy_ctrl = ~L["intervened"].astype(bool)
    events = bad_events(L)
    y = labels_within(n, events, args.H, done)

    print(f"steps logged: {n}  episodes: {done.sum()}  bad events: {len(events)}  "
          f"(takeovers {int((L['intervened'].astype(bool) & ~np.concatenate([[False], L['intervened'][:-1].astype(bool)])).sum())})")
    print(f"\n[1] oracle attention lower bound (H={args.H}): "
          f"{y[policy_ctrl].mean():.3f} of policy-controlled steps   (budget = {args.budget})")

    scores = {
        "q_std (critic ensemble std)": L["q_std"],
        "neg_q (low value)": -L["q_mean"],
        "random": np.random.default_rng(0).random(n),
    }
    if "risk" in L and not np.all(np.isnan(L["risk"])):
        scores["risk used online"] = L["risk"]

    print(f"\n[2] predicting 'bad event within {args.H} steps' on policy-controlled steps")
    print(f"    {'score':32s} {'AUROC':>7s} {'AUPRC':>7s}   base rate {y[policy_ctrl].mean():.3f}")
    for name, s in scores.items():
        print(f"    {name:32s} {auroc(s[policy_ctrl], y[policy_ctrl]):7.3f} {auprc(s[policy_ctrl], y[policy_ctrl]):7.3f}")

    print(f"\n[3] budget-matched replay of the human model (reaction {args.react} + attend {args.attend} steps "
          f"per alert, threshold tuned to {args.budget:.0%} attention): event caught if the human is already "
          f"watching (past reaction) when it happens")
    for name, s in scores.items():
        if np.all(np.isnan(s)):
            continue
        caught, att, leads = replay_at_budget(np.nan_to_num(s, nan=-np.inf), events, args.budget,
                                              args.react, args.attend)
        lead = f"median lead {np.median(leads):.0f} steps" if leads else ""
        print(f"    {name:32s} recall {caught:.3f}   attention {att:.3f}   {lead}")

    print(f"\n[4] training curve (bins of {args.bin} env steps)")
    print(f"    {'steps':>8s} {'episodes':>8s} {'success':>8s} {'attention':>9s} {'takeover':>8s} {'alerts':>6s}")
    for b0 in range(0, n, args.bin):
        sl = slice(b0, min(n, b0 + args.bin))
        eps = done[sl].sum()
        sr = L["succeed"][sl][done[sl]].mean() if eps else np.nan
        print(f"    {int(L['step'][sl][-1]):8d} {eps:8d} {sr:8.2f} {L['attention'][sl].mean():9.3f} "
              f"{L['intervened'][sl].mean():8.3f} {int(L['alert'][sl].sum()):6d}")
    print(f"\n    cumulative attention: {L['attention'].sum()} steps = {L['attention'].sum() / 36000:.2f} h at 10 Hz")


if __name__ == "__main__":
    main()
