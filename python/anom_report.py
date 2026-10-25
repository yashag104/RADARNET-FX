#!/usr/bin/env python3
"""
anom_report.py — summary table, figures and RTL test vectors for Option A
==========================================================================
Reads <ck-dir>/*_metrics.json written by train_anomaly.py, baselines.py and
`int_quant.py eval --report-dir`, then writes:
  summary.md            comparison table (accuracy, macro-F1, AUC, P_d at P_fa 1 % / 5 %)
  pd_vs_jsr.png         P_d at P_fa = 5 % vs JSR, all methods
  roc.png               ROC curves, all methods
  acc_vs_jsr.png        anomaly-type accuracy vs JSR, classifiers
  tb_input_vectors.hex  first N test frames as INT8 {I,Q} words for tb_rn_top.v
  tb_true_labels.hex    their labels

Usage:
    python anom_report.py --data-dir ../data_anom --ck-dir ../checkpoints_anom
"""
import argparse
import json
import os

import numpy as np

from anom_eval import plot_curves, plot_roc

ORDER = ["cnn_fp32", "int8_hw", "features_gbm", "energy_detector"]
LABEL = {"cnn_fp32": "RadarNet-1D (FP32)", "int8_hw": "RadarNet-1D INT8 (hardware, bit-exact)",
         "features_gbm": "Expert features + GBM", "energy_detector": "Energy detector"}


def pct(v):
    return "–" if v is None else f"{100 * v:.2f}"


def low_jsr_pd(m, hi=-4):
    d = m.get("detection_np")
    if d is None:
        return None
    v = [p for lo, p in zip(m["jsr_bins_db"], d["p_d_vs_jsr_at_pfa5"]) if lo < hi and p is not None]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("..", "data_anom"))
    ap.add_argument("--ck-dir", default=os.path.join("..", "checkpoints_anom"))
    ap.add_argument("--tb-vectors", type=int, default=50)
    args = ap.parse_args()

    found = [k for k in ORDER if os.path.exists(os.path.join(args.ck_dir, f"{k}_metrics.json"))]
    ms = {k: json.load(open(os.path.join(args.ck_dir, f"{k}_metrics.json"))) for k in found}

    rows = ["| Method | Accuracy (%) | Macro-F1 (%) | AUC | P_d @ P_fa 1% | P_d @ P_fa 5% "
            "| P_d @ P_fa 5%, JSR < −4 dB |",
            "|---|---|---|---|---|---|---|"]
    for k in found:
        m = ms[k]
        d = m.get("detection_np", {})
        pd_at = d.get("p_d_at_p_fa", {})
        rows.append(f"| {LABEL[k]} | {pct(m.get('accuracy'))} | {pct(m.get('macro_f1'))} | "
                    f"{d.get('auc', float('nan')):.4f} | {pct(pd_at.get('1%'))} | "
                    f"{pct(pd_at.get('5%'))} | {pct(low_jsr_pd(m))} |")
    if "cnn_fp32" in ms:
        rows += ["", "Per-class F1 (%) — " + ", ".join(
            f"{c}: {100 * v['f1']:.1f}" for c, v in ms["cnn_fp32"]["per_class"].items())]
        snr = ms["cnn_fp32"]["snr_db"]
        acc = ms["cnn_fp32"]["accuracy_vs_snr"]
        rows += ["", "CNN accuracy vs SNR (%) — " + ", ".join(
            f"{s:g} dB: {100 * a:.1f}" for s, a in zip(snr, acc))]
    table = "\n".join(rows)
    with open(os.path.join(args.ck_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    files = [os.path.join(args.ck_dir, f"{k}_metrics.json") for k in found]
    plot_curves(files, os.path.join(args.ck_dir, "pd_vs_jsr.png"), labels=LABEL)
    plot_roc(files, os.path.join(args.ck_dir, "roc.png"), labels=LABEL)
    plot_curves([p for p, k in zip(files, found) if k != "energy_detector"],
                os.path.join(args.ck_dir, "acc_vs_jsr.png"),
                key="accuracy_vs_jsr", ylabel="Anomaly-type accuracy", labels=LABEL)

    X = np.load(os.path.join(args.data_dir, "X_test.npy"))[:args.tb_vectors]
    y = np.load(os.path.join(args.data_dir, "y_test.npy"))[:args.tb_vectors]
    q = np.clip(np.round(X * 127.0), -128, 127).astype(np.int64) & 0xFF
    with open(os.path.join(args.ck_dir, "tb_input_vectors.hex"), "w") as f:
        for v in q:
            for t in range(v.shape[0]):
                f.write(f"{v[t, 0]:02X}{v[t, 1]:02X}\n")
    with open(os.path.join(args.ck_dir, "tb_true_labels.hex"), "w") as f:
        f.write("".join(f"{int(c):02X}\n" for c in y))
    print(f"figures + {len(y)} RTL test vectors → {args.ck_dir}")


if __name__ == "__main__":
    main()
