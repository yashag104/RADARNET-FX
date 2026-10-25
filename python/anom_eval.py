#!/usr/bin/env python3
"""
anom_eval.py — one evaluation routine for every anomaly-detection method
=========================================================================
The CNN (FP32), the bit-exact integer hardware model and the classical
baselines are all scored here, so every table and curve in the paper is
computed the same way.

report(name, y_true, y_pred, meta, out_dir, scores=None) writes
<out_dir>/<name>_metrics.json and returns the metrics dict:
  classification (y_pred given): accuracy, macro-F1, per-class P/R/F1, confusion,
      argmax detection P_d / P_fa, accuracy per 2-dB JSR bin and per SNR
  detection (scores given, higher = more anomalous): ROC, AUC and the
      Neyman-Pearson operating points P_d at P_fa = 1 %, 5 %, 10 %, plus
      P_d vs JSR at P_fa = 5 % — the standard way detectors are compared
"""
import json
import os

import numpy as np

CLASSES = ["Normal", "Barrage", "Tone", "Sweep", "Pulsed", "Replay", "CoChannel"]
JSR_BINS = np.arange(-10, 21, 2)


def _prf(conf):
    tp = np.diag(conf).astype(float)
    p = tp / np.maximum(conf.sum(0), 1)
    r = tp / np.maximum(conf.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return p, r, f


PFA_POINTS = [0.01, 0.05, 0.10]


def detection_curves(y_true, scores, meta):
    """ROC / AUC / P_d at fixed P_fa from a continuous anomaly score."""
    anom = y_true != 0
    s_n, s_a = scores[~anom], scores[anom]
    order = np.argsort(np.concatenate([s_n, s_a]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[len(s_n):].sum() - len(s_a) * (len(s_a) + 1) / 2) / (len(s_a) * len(s_n))
    thr = {p: np.quantile(s_n, 1 - p) for p in PFA_POINTS}
    pd_at = {f"{100 * p:g}%": float((s_a > thr[p]).mean()) for p in PFA_POINTS}
    jsr = meta["jsr_db"]
    pd5 = []
    for lo in JSR_BINS[:-1]:
        sel = anom & (jsr >= lo) & (jsr < lo + 2)
        pd5.append(float((scores[sel] > thr[0.05]).mean()) if sel.any() else None)
    q = np.quantile(s_n, np.linspace(0, 1, 101))
    roc = dict(p_fa=[float((s_n > t).mean()) for t in q], p_d=[float((s_a > t).mean()) for t in q])
    return dict(auc=float(auc), p_d_at_p_fa=pd_at, p_d_vs_jsr_at_pfa5=pd5, roc=roc)


def report(name, y_true, y_pred, meta, out_dir, classes=CLASSES, quiet=False, scores=None):
    y_true = np.asarray(y_true)
    m = dict(method=name, n=int(len(y_true)), jsr_bins_db=JSR_BINS[:-1].tolist())
    if scores is not None:
        m["detection_np"] = detection_curves(y_true, np.asarray(scores, dtype=np.float64), meta)
    if y_pred is None:
        _save(m, name, out_dir, quiet)
        return m
    y_pred = np.asarray(y_pred)
    k = len(classes)
    conf = np.zeros((k, k), dtype=np.int64)
    np.add.at(conf, (y_true, y_pred), 1)
    p, r, f = _prf(conf)
    acc = float((y_true == y_pred).mean())

    anom_t, anom_p = y_true != 0, y_pred != 0
    pd = float((anom_p & anom_t).sum() / max(anom_t.sum(), 1))
    pfa = float((anom_p & ~anom_t).sum() / max((~anom_t).sum(), 1))

    jsr = meta["jsr_db"]
    acc_jsr, pd_jsr = [], []
    for lo in JSR_BINS[:-1]:
        s = anom_t & (jsr >= lo) & (jsr < lo + 2)
        acc_jsr.append(float((y_true[s] == y_pred[s]).mean()) if s.any() else None)
        pd_jsr.append(float(anom_p[s].mean()) if s.any() else None)
    snr = meta["snr_db"]
    snr_levels = sorted(set(np.asarray(snr).tolist()))
    acc_snr = [float((y_true[snr == v] == y_pred[snr == v]).mean()) for v in snr_levels]

    m.update(accuracy=acc, macro_f1=float(f.mean()),
             per_class={c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]))
                        for i, c in enumerate(classes)},
             confusion=conf.tolist(), detection=dict(p_d=pd, p_fa=pfa),
             accuracy_vs_jsr=acc_jsr, p_d_vs_jsr=pd_jsr,
             snr_db=snr_levels, accuracy_vs_snr=acc_snr)
    if not quiet:
        print(f"── {name} ──  acc {100 * acc:.2f}%  macro-F1 {100 * f.mean():.2f}%  "
              f"(argmax: P_d {100 * pd:.2f}%  P_fa {100 * pfa:.2f}%)")
        print("   per-class F1: " + "  ".join(f"{c} {100 * f[i]:.1f}" for i, c in enumerate(classes)))
    _save(m, name, out_dir, quiet)
    return m


def _save(m, name, out_dir, quiet):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    if not quiet and "detection_np" in m:
        d = m["detection_np"]
        print(f"── {name} ──  AUC {d['auc']:.4f}   P_d @ P_fa " + "  ".join(
            f"{k}: {100 * v:.2f}%" for k, v in d["p_d_at_p_fa"].items()))
        print("   P_d vs JSR (P_fa 5%): " + "  ".join(
            f"{lo:+d}:{'-' if v is None else f'{100 * v:.0f}'}"
            for lo, v in zip(m["jsr_bins_db"], d["p_d_vs_jsr_at_pfa5"])))


def anomaly_score(logits):
    """Detection score from class logits: strongest anomaly logit minus Normal."""
    logits = np.asarray(logits, dtype=np.float64)
    return logits[:, 1:].max(axis=1) - logits[:, 0]


def load_meta(data_dir, split):
    d = np.load(os.path.join(data_dir, f"meta_{split}.npz"))
    return {k: d[k] for k in d.files}


def _get(m, key):
    """key may be 'a.b' to reach into nested dicts."""
    for part in key.split("."):
        m = m.get(part) if isinstance(m, dict) else None
    return m


def plot_curves(metric_files, out_png, key="detection_np.p_d_vs_jsr_at_pfa5",
                ylabel="Detection probability at P_fa = 5%", labels=None):
    """Overlay one curve per method (FP32 CNN, INT8 HW, baselines) vs JSR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for path in metric_files:
        with open(path) as fh:
            m = json.load(fh)
        vals = _get(m, key)
        if vals is None:
            continue
        x = np.array(m["jsr_bins_db"]) + 1
        y = np.array([np.nan if v is None else v for v in vals])
        ax.plot(x, 100 * y, "o-", ms=3, label=(labels or {}).get(m["method"], m["method"]))
    ax.set_xlabel("JSR (dB)")
    ax.set_ylabel(ylabel + " (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_roc(metric_files, out_png, labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for path in metric_files:
        with open(path) as fh:
            m = json.load(fh)
        d = m.get("detection_np")
        if d is None:
            continue
        ax.plot(100 * np.array(d["roc"]["p_fa"]), 100 * np.array(d["roc"]["p_d"]),
                label=f"{(labels or {}).get(m['method'], m['method'])} (AUC {d['auc']:.3f})")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 100)
    ax.set_xlabel("P_fa (%)")
    ax.set_ylabel("P_d (%)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
