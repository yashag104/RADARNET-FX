#!/usr/bin/env python3
"""
baselines.py — classical detectors on the same anomaly dataset
===============================================================
  energy    : energy detector. Score = window power of the same AGC-normalised
              input the CNN sees (the AGC fixes the long-term level, so a power
              rise is visible). Scored with ROC / P_d at fixed P_fa like every
              other method. Detection only — it cannot tell anomaly types apart.
  features  : expert-feature classifier. 49 hand-crafted features (spectral
              flatness/peaks/occupancy, envelope statistics, instantaneous
              frequency, autocorrelation lags 1-16, higher-order cumulants,
              short-time power variation) + gradient-boosted trees.

Both write <out-dir>/<name>_metrics.json in the anom_eval format.

Usage:
    python baselines.py --data-dir ../data_anom --out-dir ../checkpoints_anom
"""
import argparse
import json
import os
import time

import numpy as np

from anom_eval import load_meta, report


def load(data_dir, split):
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    return X[..., 0] + 1j * X[..., 1], y


# ── Energy detector ──────────────────────────────────────────────────────────
def energy_detector(data_dir, out_dir):
    x_te, y_te = load(data_dir, "test")
    power_db = 10 * np.log10(np.mean(np.abs(x_te) ** 2, axis=1) + 1e-12)
    report("energy_detector", y_te, None, load_meta(data_dir, "test"), out_dir, scores=power_db)


# ── Expert features ──────────────────────────────────────────────────────────
def _kurt(v, axis=1):
    v = v - v.mean(axis, keepdims=True)
    return (v ** 4).mean(axis) / ((v ** 2).mean(axis) ** 2 + 1e-12)


def features(x):
    n, L = x.shape
    x = x / np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True) + 1e-12)
    F = []
    # spectrum
    psd = np.abs(np.fft.fft(x, axis=1)) ** 2
    psd = psd / psd.sum(1, keepdims=True)
    F.append(np.exp(np.mean(np.log(psd + 1e-12), 1)) / psd.mean(1))           # flatness
    F.append(psd.max(1) / psd.mean(1))                                          # peak/mean
    srt = -np.sort(-psd, axis=1)
    F.append(srt[:, :3].sum(1))                                                 # top-3 line power
    F.append((psd > 10 * np.median(psd, 1, keepdims=True)).sum(1))             # strong bins
    F.append(-(psd * np.log(psd + 1e-12)).sum(1))                               # entropy
    F.append((np.cumsum(srt, 1) < 0.9).sum(1) / L)                              # occupied BW
    f = np.fft.fftfreq(L)[None, :]
    cen = (psd * f).sum(1)
    F.append(cen)
    F.append(np.sqrt((psd * (f - cen[:, None]) ** 2).sum(1)))                   # spread
    # envelope
    a = np.abs(x)
    F.append(a.std(1) / (a.mean(1) + 1e-12))
    F.append(_kurt(a))
    F.append((a ** 2).max(1))                                                   # PAPR
    F.append((a < 0.2 * a.mean(1, keepdims=True)).mean(1))                      # gated fraction
    st = (a ** 2).reshape(n, L // 16, 16).mean(2)
    F.append(st.std(1) / (st.mean(1) + 1e-12))                                  # short-time power var
    F.append(st.max(1) / (st.min(1) + 1e-6))
    # instantaneous frequency
    inst_f = np.angle(x[:, 1:] * np.conj(x[:, :-1])) / (2 * np.pi)
    F.append(inst_f.std(1))
    F.append(_kurt(inst_f))
    F.append(np.abs(np.diff(inst_f, axis=1)).mean(1))
    seg = inst_f[:, :L - 1 - (L - 1) % 8].reshape(n, -1, 8).mean(2)
    F.append(np.abs(np.diff(seg, axis=1)).mean(1))                              # chirp regularity
    # autocorrelation lags 1..16 (replay / delayed copies)
    for lag in range(1, 17):
        F.append(np.abs((x[:, lag:] * np.conj(x[:, :-lag])).mean(1)))
    # envelope periodicity (pulsed): max normalised autocorr of |x|^2, lags 8..64
    e = a ** 2 - (a ** 2).mean(1, keepdims=True)
    ac = np.stack([(e[:, l:] * e[:, :-l]).mean(1) for l in range(8, 65, 4)], 1)
    F.append(ac.max(1) / ((e ** 2).mean(1) + 1e-12))
    # higher-order cumulants (modulation statistics)
    c20 = (x ** 2).mean(1)
    c21 = (a ** 2).mean(1)
    c40 = (x ** 4).mean(1) - 3 * c20 ** 2
    c42 = (a ** 4).mean(1) - np.abs(c20) ** 2 - 2 * c21 ** 2
    F += [np.abs(c20), np.abs(c40), np.real(c42), np.abs(c40) / (c21 ** 2 + 1e-12)]
    # coarse 8-band power profile
    bands = np.fft.fftshift(psd, axes=1).reshape(n, 8, L // 8).sum(2)
    F += [bands[:, i] for i in range(8)]
    F.append(bands.std(1))
    return np.stack(F, 1).astype(np.float32)


def feature_classifier(data_dir, out_dir, classes):
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    x_tr, y_tr = load(data_dir, "train")
    x_te, y_te = load(data_dir, "test")
    f_tr, f_te = features(x_tr), features(x_te)
    print(f"   {f_tr.shape[1]} features, extraction {time.time() - t0:.0f}s")
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1,
                                         early_stopping=True, random_state=0)
    clf.fit(f_tr, y_tr)
    proba = clf.predict_proba(f_te)
    report("features_gbm", y_te, proba.argmax(1), load_meta(data_dir, "test"), out_dir, classes,
           scores=1.0 - proba[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("..", "data_anom"))
    ap.add_argument("--out-dir", default=os.path.join("..", "checkpoints_anom"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.data_dir, "dataset_info.json")) as f:
        classes = json.load(f)["classes"]
    energy_detector(args.data_dir, args.out_dir)
    feature_classifier(args.data_dir, args.out_dir, classes)


if __name__ == "__main__":
    main()
