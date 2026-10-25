#!/usr/bin/env python3
"""
anomaly_dataset.py — RF anomaly dataset built on RadioML 2018.01A
==================================================================
Legitimate traffic is taken from RadioML 2018.01A (24 modulations, real channel
impairments). Anomalies are injected on top of it at a controlled
jammer-to-signal ratio (JSR), measured against the *signal* power of each frame
(derived from its labelled SNR, not from its noisy total power).

Classes
-------
  0 Normal     clean RadioML frame
  1 Barrage    band-limited Gaussian noise, 30–100 % of the band
  2 Tone       1–3 CW tones (spot jamming)
  3 Sweep      linear-FM chirp, sawtooth, 20–90 % of the band
  4 Pulsed     noise or tone gated on/off, duty 10–50 %  (JSR = on-time power)
  5 Replay     delayed (4–64 samples), Doppler-shifted copy of the same frame
               (meaconing / replay spoofing; the re-transmission delay exceeds
               the short multipath spread already present in RadioML)
  6 CoChannel  independent RadioML frame (SNR >= 20 dB) of another modulation,
               frequency-offset

Leakage control: RadioML frames are split into train/val/test *before*
injection, so a base frame (and any co-channel interferer) never appears in
two splits.

Output (in --out-dir):
  X_{split}.npy   (N, 128, 2) float32, AGC-normalised: scaled by the frame's long-term
                  average power so its RMS is AGC_RMS (−12 dBFS), clipped to [−1, 1].
                  Absolute level changes caused by an anomaly stay visible to every
                  method, as in a receiver with a slow AGC.
  y_{split}.npy   (N,) int64 class
  meta_{split}.npz  jsr_db, snr_db, mod, pow_ratio_db (window power relative to the
                    frame's long-term average power, for the energy-detector baseline)

Usage:
  python anomaly_dataset.py --hdf5 GOLD_XYZ_OSC.0001_1024.hdf5 --out-dir ../data_anom
"""
import argparse
import json
import os

import numpy as np

CLASSES = ["Normal", "Barrage", "Tone", "Sweep", "Pulsed", "Replay", "CoChannel"]
DATASET_VERSION = 2   # bump whenever the generated data changes (Colab cache key)
NUM_MODS, NUM_SNRS, FRAME_LEN = 24, 26, 1024     # RadioML 2018.01A layout
SNR_LEVELS = np.arange(-20, 32, 2)
L = 128              # model input length (unchanged from the paper / hardware)
MIN_DELAY = 4        # replay delay range (samples); segments keep MAX_DELAY of history
MAX_DELAY = 64
AGC_RMS = 0.25       # long-term RMS after AGC = −12 dBFS → 12 dB headroom before clipping
SEG = L + MAX_DELAY


# ─────────────────────────────────────────────────────────────────────────────
# Base frames
# ─────────────────────────────────────────────────────────────────────────────
def read_base_frames(h5_path, snr_min, frames_per_block, seed):
    """
    Read `frames_per_block` frames from every (modulation, SNR >= snr_min) block.
    Returns complex segments of length SEG (random window inside the frame),
    the signal power of each frame, its SNR, its modulation, and a block-wise
    70/15/15 split assignment.
    """
    import h5py
    rng = np.random.default_rng(seed)
    segs, p_sig, p_frame, snrs, mods, split = [], [], [], [], [], []
    with h5py.File(h5_path, "r") as f:
        X, Y, Z = f["X"], f["Y"], f["Z"]
        assert X.shape[0] % (NUM_MODS * NUM_SNRS) == 0, X.shape
        fpb = X.shape[0] // (NUM_MODS * NUM_SNRS)          # 4096 in the real file
        frames_per_block = min(frames_per_block, fpb)
        for m in range(NUM_MODS):
            for si, snr in enumerate(SNR_LEVELS):
                if snr < snr_min:
                    continue
                start = (m * NUM_SNRS + si) * fpb
                # Verify the canonical block layout before trusting it
                zb = Z[start:start + 2, 0]
                yb = np.argmax(Y[start:start + 2], axis=1)
                assert (zb == snr).all() and (yb == m).all(), \
                    f"unexpected RadioML layout at block mod={m} snr={snr}"
                xb = X[start:start + frames_per_block]            # (k, 1024, 2)
                c = xb[..., 0] + 1j * xb[..., 1]
                p_tot = np.mean(np.abs(c) ** 2, axis=1)
                lin = 10.0 ** (snr / 10.0)
                p_sig.append(p_tot * lin / (1.0 + lin))
                p_frame.append(p_tot)
                w0 = rng.integers(MAX_DELAY, FRAME_LEN - L + 1, size=len(c))
                idx = w0[:, None] - MAX_DELAY + np.arange(SEG)[None, :]
                segs.append(np.take_along_axis(c, idx, axis=1).astype(np.complex64))
                snrs.append(np.full(len(c), snr))
                mods.append(np.full(len(c), m))
                u = rng.permutation(len(c))
                sp = np.zeros(len(c), dtype=np.int8)
                sp[u[int(0.70 * len(c)):int(0.85 * len(c))]] = 1
                sp[u[int(0.85 * len(c)):]] = 2
                split.append(sp)
            print(f"    modulation {m + 1}/{NUM_MODS} read", end="\r")
    print()
    return (np.concatenate(segs), np.concatenate(p_sig).astype(np.float64),
            np.concatenate(p_frame).astype(np.float64),
            np.concatenate(snrs), np.concatenate(mods), np.concatenate(split))


# ─────────────────────────────────────────────────────────────────────────────
# Interference generators — all return unit-power (N, L) complex arrays
# ─────────────────────────────────────────────────────────────────────────────
def _unit_power(j):
    return j / np.sqrt(np.mean(np.abs(j) ** 2, axis=1, keepdims=True) + 1e-12)


def gen_barrage(n, rng):
    """Band-limited complex Gaussian noise; filtered on 2L samples, centre cropped."""
    bw = rng.uniform(0.3, 1.0, n)
    fc = rng.uniform(-(1 - bw) / 2, (1 - bw) / 2)
    w = (rng.standard_normal((n, 2 * L)) + 1j * rng.standard_normal((n, 2 * L)))
    f = np.fft.fftfreq(2 * L)[None, :]
    mask = np.abs(f - fc[:, None]) <= bw[:, None] / 2
    j = np.fft.ifft(np.fft.fft(w, axis=1) * mask, axis=1)[:, L // 2: L // 2 + L]
    return _unit_power(j), dict(bw=bw, fc=fc)


def gen_tone(n, rng):
    t = np.arange(L)[None, :]
    k = rng.integers(1, 4, n)
    j = np.zeros((n, L), dtype=np.complex128)
    for i in range(3):
        on = (k > i)[:, None]
        f = rng.uniform(-0.45, 0.45, n)[:, None]
        a = rng.uniform(0.5, 1.0, n)[:, None]
        ph = rng.uniform(0, 2 * np.pi, n)[:, None]
        j += on * a * np.exp(1j * (2 * np.pi * f * t + ph))
    return _unit_power(j), dict(n_tones=k)


def gen_sweep(n, rng):
    t = np.arange(L)[None, :] + rng.uniform(0, 512, n)[:, None]   # random phase in sweep
    bw = rng.uniform(0.2, 0.9, n)[:, None]
    fc = rng.uniform(-(0.95 - bw) / 2, (0.95 - bw) / 2)
    T = rng.uniform(16, 256, n)[:, None]
    frac = (t % T) / T                                             # sawtooth 0..1
    finst = fc - bw / 2 + bw * frac
    phase = 2 * np.pi * np.cumsum(finst, axis=1)
    j = np.exp(1j * (phase + rng.uniform(0, 2 * np.pi, n)[:, None]))
    return _unit_power(j), dict(bw=bw[:, 0], period=T[:, 0])


def gen_pulsed(n, rng):
    """Gated noise or tone; unit power while ON, so JSR is the peak (on-time) JSR."""
    use_tone = rng.random(n) < 0.5
    base = np.where(use_tone[:, None], gen_tone(n, rng)[0], gen_barrage(n, rng)[0])
    period = rng.integers(16, 65, n)[:, None]
    duty = rng.uniform(0.1, 0.5, n)[:, None]
    off = rng.integers(0, 64, n)[:, None]
    gate = (((np.arange(L)[None, :] + off) % period) < np.maximum(1, duty * period))
    return base * gate, dict(duty=duty[:, 0], period=period[:, 0], tone=use_tone)


def gen_replay(segs, rng):
    """Delayed (MIN_DELAY..MAX_DELAY), Doppler-shifted, phase-rotated copy of the segment."""
    n = len(segs)
    d = rng.integers(MIN_DELAY, MAX_DELAY + 1, n)
    idx = (MAX_DELAY - d)[:, None] + np.arange(L)[None, :]
    rep = np.take_along_axis(segs, idx, axis=1)
    fd = rng.uniform(-0.02, 0.02, n)[:, None]
    ph = rng.uniform(0, 2 * np.pi, n)[:, None]
    rep = rep * np.exp(1j * (2 * np.pi * fd * np.arange(L)[None, :] + ph))
    return rep, dict(delay=d, doppler=fd[:, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────
def make_split(pool, n_per_class, jsr_range, rng):
    segs, p_sig, p_frame, snrs, mods = pool
    nb = len(segs)
    X, y, jsr_out, snr_out, mod_out, pr_out = [], [], [], [], [], []
    for c in range(len(CLASSES)):
        b = rng.integers(0, nb, n_per_class)
        s = segs[b, MAX_DELAY:].astype(np.complex128)              # (n, L) legit window
        ps = p_sig[b]
        jsr = rng.uniform(*jsr_range, n_per_class)
        amp = np.sqrt(10.0 ** (jsr / 10.0) * ps)[:, None]
        if c == 0:
            interf = np.zeros_like(s)
            jsr = np.full(n_per_class, np.nan)
        elif c == 1:
            interf = amp * gen_barrage(n_per_class, rng)[0]
        elif c == 2:
            interf = amp * gen_tone(n_per_class, rng)[0]
        elif c == 3:
            interf = amp * gen_sweep(n_per_class, rng)[0]
        elif c == 4:
            interf = amp * gen_pulsed(n_per_class, rng)[0]
        elif c == 5:
            # replica of the same frame: scale so its signal part has JSR·P_s
            rep = gen_replay(segs[b].astype(np.complex128), rng)[0]
            interf = np.sqrt(10.0 ** (jsr / 10.0))[:, None] * rep
        else:
            # independent high-SNR frame (its own noise negligible) of a
            # *different* modulation, from the same split
            hi = np.flatnonzero(snrs >= 20)
            hi = hi if len(hi) else np.arange(nb)
            o = hi[rng.integers(0, len(hi), n_per_class)]
            for _ in range(10):
                same = mods[o] == mods[b]
                if not same.any():
                    break
                o[same] = hi[rng.integers(0, len(hi), same.sum())]
            other = segs[o, MAX_DELAY:].astype(np.complex128)
            g = np.sqrt(10.0 ** (jsr / 10.0) * ps / p_sig[o])[:, None]
            fo = rng.uniform(-0.25, 0.25, n_per_class)[:, None]
            interf = g * other * np.exp(1j * 2 * np.pi * fo * np.arange(L)[None, :])
        x = s + interf
        # window power relative to the frame's long-term average power — what a
        # receiver with a calibrated noise/signal floor sees (energy detector)
        pr = 10 * np.log10(np.mean(np.abs(x) ** 2, axis=1) / p_frame[b])
        x = x * (AGC_RMS / np.sqrt(p_frame[b]))[:, None]                 # slow AGC
        X.append(x); y.append(np.full(n_per_class, c)); jsr_out.append(jsr)
        snr_out.append(snrs[b]); mod_out.append(mods[b]); pr_out.append(pr)
    X = np.concatenate(X)
    perm = rng.permutation(len(X))
    X = np.stack([X.real, X.imag], axis=2)[perm]
    X = np.clip(X, -1.0, 1.0)                                      # ADC full scale
    meta = dict(jsr_db=np.concatenate(jsr_out)[perm], snr_db=np.concatenate(snr_out)[perm],
                mod=np.concatenate(mod_out)[perm], pow_ratio_db=np.concatenate(pr_out)[perm])
    return X.astype(np.float32), np.concatenate(y)[perm].astype(np.int64), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out-dir", default=os.path.join("..", "data_anom"))
    ap.add_argument("--snr-min", type=int, default=0,
                    help="base-frame SNR floor (0 dB, as in the 4-class paper split)")
    ap.add_argument("--frames-per-block", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=30000, help="per class")
    ap.add_argument("--n-eval", type=int, default=6000, help="per class, val and test")
    ap.add_argument("--jsr-min", type=float, default=-10.0)
    ap.add_argument("--jsr-max", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/3] Reading base frames (SNR >= {args.snr_min} dB, "
          f"{args.frames_per_block} per modulation/SNR block) …")
    segs, p_sig, p_frame, snrs, mods, split = read_base_frames(
        args.hdf5, args.snr_min, args.frames_per_block, args.seed)
    print(f"    {len(segs):,} frames  (train {np.sum(split == 0):,} / "
          f"val {np.sum(split == 1):,} / test {np.sum(split == 2):,})")

    print("[2/3] Injecting anomalies …")
    rng = np.random.default_rng(args.seed + 1)
    for si, (name, n) in enumerate([("train", args.n_train), ("val", args.n_eval),
                                    ("test", args.n_eval)]):
        sel = split == si
        pool = (segs[sel], p_sig[sel], p_frame[sel], snrs[sel], mods[sel])
        X, y, meta = make_split(pool, n, (args.jsr_min, args.jsr_max), rng)
        np.save(os.path.join(args.out_dir, f"X_{name}.npy"), X)
        np.save(os.path.join(args.out_dir, f"y_{name}.npy"), y)
        np.savez(os.path.join(args.out_dir, f"meta_{name}.npz"), **meta)
        print(f"    {name:5s}: X {X.shape}  classes {np.bincount(y).tolist()}")

    with open(os.path.join(args.out_dir, "dataset_info.json"), "w") as f:
        json.dump(dict(version=DATASET_VERSION, classes=CLASSES, window=L,
                       snr_min=args.snr_min, agc_rms=AGC_RMS,
                       replay_delay=[MIN_DELAY, MAX_DELAY],
                       jsr_range_db=[args.jsr_min, args.jsr_max],
                       frames_per_block=args.frames_per_block, seed=args.seed,
                       n_train_per_class=args.n_train, n_eval_per_class=args.n_eval,
                       source="RadioML 2018.01A (GOLD_XYZ_OSC.0001_1024.hdf5)"), f, indent=2)
    print(f"[3/3] Done → {args.out_dir}")


if __name__ == "__main__":
    main()
