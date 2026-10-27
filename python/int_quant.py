#!/usr/bin/env python3
"""
int_quant.py — Bit-exact integer-only RadarNet-1D (golden model for rtl_v2)
============================================================================
Unlike quantize.py (which only rounds the *weights* to INT8 and keeps FP32
activations), this script builds the network exactly as the hardware computes
it, using integer arithmetic only:

    weights     : INT8, symmetric, per-output-channel scale
    activations : INT8, symmetric, per-tensor scale (calibrated)
    bias        : INT32 at scale  s_w[oc] * s_x
    requant     : y = sat8( relu?( ((acc + b) * M[oc] + (skip * Ms << SK) + 2^(SH-1)) >> SH ) )
    GAP         : g = sat8( (sum_t x + 16) >> 5 )
    FC          : logits = acc + b   (INT32, per-tensor weight scale) → argmax

Sub-commands
------------
  calib   : calibrate activation scales and save the integer model (.npz)
  eval    : accuracy of FP32 vs integer model on a test split
  export  : write ROM images + layer table for rtl_v2 with W×W PE lanes,
            plus golden logits for the RTL testbench
  export-c: write the same integer model as a C header (ARM baseline, sw/)

Usage:
    python int_quant.py calib  --checkpoint ../checkpoints/radarnet_best.pth \
                               --data-dir ../data --out ../checkpoints/radarnet_intq.npz
    python int_quant.py eval   --intq ../checkpoints/radarnet_intq.npz \
                               --checkpoint ../checkpoints/radarnet_best.pth --data-dir ../data
    python int_quant.py export --intq ../checkpoints/radarnet_intq.npz --lanes 8 \
                               --tb-inputs ../tb_vectors/input_vectors.hex --out-dir ../rtl_v2/mem
"""
import argparse
import os

import numpy as np

IN_SCALE = 1.0 / 127.0      # input I/Q is round(x * 127), x in [-1, 1]
M_BITS   = 15               # requant multipliers live in [2^14, 2^15)

# Layer order is the hardware execution order. src/dst/skip are buffer ids
# (0, 1, 2); the input frame is loaded into buffer 2.
#  name        torch prefix          K  S  P  relu  skip-of   in-act   out-act
LAYERS = [
    ("L1",     "conv_in.0",          7, 1, 3, 1,    None,     "in",    "a1"),
    ("L2",     "conv_down.0",        3, 2, 1, 1,    None,     "a1",    "a2"),
    ("RB1C1",  "block1.conv1",       3, 1, 1, 1,    None,     "a2",    "b1h"),
    ("RB1C2",  "block1.conv2",       3, 1, 1, 1,    "a2",     "b1h",   "b1o"),
    ("RB2P",   "block2.shortcut.0",  1, 2, 0, 0,    None,     "b1o",   "p"),
    ("RB2C1",  "block2.conv1",       3, 2, 1, 1,    None,     "b1o",   "b2h"),
    ("RB2C2",  "block2.conv2",       3, 1, 1, 1,    "p",      "b2h",   "b2o"),
]
# Buffer allocation (src, dst, skip) — see rtl_v2/README.md for the liveness chart
BUFS = {"L1": (2, 0, 0), "L2": (0, 1, 0), "RB1C1": (1, 0, 0), "RB1C2": (0, 2, 1),
        "RB2P": (2, 1, 0), "RB2C1": (2, 0, 0), "RB2C2": (0, 2, 1), "FC": (0, 0, 0)}
GAP_SRC, GAP_DST = 2, 0


# ─────────────────────────────────────────────────────────────────────────────
# Float reference (BN-folded) — used for calibration only
# ─────────────────────────────────────────────────────────────────────────────
def load_folded(checkpoint):
    import torch
    from model import RadarNet1D
    from quantize import fold_all_bn
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RadarNet1D(in_channels=2, num_classes=sd["fc.weight"].shape[0])
    model.load_state_dict(sd)
    model.eval()
    folded = fold_all_bn(model).eval()
    params = {k: v.detach().numpy().astype(np.float64)
              for k, v in folded.state_dict().items()}
    return model, params


def conv1d_f(x, w, b, s, p):
    """x (N,C,L) float, w (O,C,K) → (N,O,Lout)"""
    n, c, l = x.shape
    o, _, k = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
    lout = (l + 2 * p - k) // s + 1
    y = np.zeros((n, o, lout))
    for kk in range(k):
        xs = xp[:, :, kk: kk + s * (lout - 1) + 1: s]
        y += np.einsum("oc,ncl->nol", w[:, :, kk], xs)
    return y + b[None, :, None]


def float_activations(params, x):
    """Forward pass of the folded float model, returning every activation."""
    acts = {"in": x}
    for name, pre, k, s, p, relu, skip, a_in, a_out in LAYERS:
        y = conv1d_f(acts[a_in], params[pre + ".weight"], params[pre + ".bias"], s, p)
        if skip is not None:
            y = y + acts[skip]
        acts[a_out] = np.maximum(y, 0) if relu else y
    acts["gap"] = acts["b2o"].mean(axis=2)
    acts["logits"] = acts["gap"] @ params["fc.weight"].T + params["fc.bias"]
    return acts


# ─────────────────────────────────────────────────────────────────────────────
# Integer model
# ─────────────────────────────────────────────────────────────────────────────
def pick_shift(values):
    """Largest SH such that round(max(values) * 2^SH) < 2^M_BITS."""
    vmax = max(values)
    sh = int(np.floor(np.log2((2 ** M_BITS - 1) / vmax)))
    while np.round(vmax * 2.0 ** sh) >= 2 ** M_BITS:
        sh -= 1
    assert 8 <= sh <= 40, f"unreasonable shift {sh}"
    return sh


def build_int_model(params, act_scale):
    q = {"act_scale": act_scale}
    for name, pre, k, s, p, relu, skip, a_in, a_out in LAYERS:
        w = params[pre + ".weight"]
        b = params[pre + ".bias"]
        s_w = np.abs(w).reshape(w.shape[0], -1).max(axis=1) / 127.0
        s_w = np.where(s_w == 0, 1.0, s_w)
        wq = np.clip(np.round(w / s_w[:, None, None]), -127, 127).astype(np.int64)
        s_acc = s_w * act_scale[a_in]
        bq = np.round(b / s_acc).astype(np.int64)
        real_m = s_acc / act_scale[a_out]
        real_ms = act_scale[skip] / act_scale[a_out] if skip else 0.0
        # Main path and skip path get independent precision: the skip product
        # is pre-shifted left by SK so both multipliers use the full M_BITS.
        sh = pick_shift(list(real_m))
        if skip:
            # Ms << SK ≈ real_ms · 2^SH must fit the RTL's 24-bit skip constant
            sh = min(sh, int(np.floor(np.log2((2 ** 23 - 1) / real_ms))))
            assert np.max(real_m) * 2.0 ** sh >= 2 ** 12, "main-path precision too low"
        sk = sh - pick_shift([real_ms]) if skip else 0
        sk = max(sk, 0)
        ms = int(np.round(real_ms * 2.0 ** (sh - sk)))
        assert ms < 2 ** M_BITS
        assert (ms << sk) < 2 ** 23, "Ms << SK must fit the 24-bit RTL constant"
        q[name] = dict(w=wq, b=bq, M=np.round(real_m * 2.0 ** sh).astype(np.int64),
                       Ms=ms, SK=sk, SH=sh)
    w = params["fc.weight"]
    s_fc = np.abs(w).max() / 127.0
    q["FC"] = dict(w=np.clip(np.round(w / s_fc), -127, 127).astype(np.int64),
                   b=np.round(params["fc.bias"] / (s_fc * act_scale["b2o"])).astype(np.int64))
    return q


def sat8(v):
    return np.clip(v, -128, 127)


def rshift_round(v, sh):
    return (v + (1 << (sh - 1))) >> sh          # numpy >> on int64 is arithmetic


def conv1d_i(x, w, s, p):
    n, c, l = x.shape
    o, _, k = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
    lout = (l + 2 * p - k) // s + 1
    acc = np.zeros((n, o, lout), dtype=np.int64)
    for kk in range(k):
        xs = xp[:, :, kk: kk + s * (lout - 1) + 1: s]
        acc += np.einsum("oc,ncl->nol", w[:, :, kk], xs)
    return acc


def int_forward(q, xq):
    """xq: (N, 2, 128) int  →  dict of integer activations and int32 logits."""
    acts = {"in": xq.astype(np.int64)}
    for name, pre, k, s, p, relu, skip, a_in, a_out in LAYERS:
        L = q[name]
        acc = conv1d_i(acts[a_in], L["w"], s, p) + L["b"][None, :, None]
        v = acc * L["M"][None, :, None]
        if skip is not None:
            v = v + ((acts[skip] * L["Ms"]) << L["SK"])
        y = rshift_round(v, L["SH"])
        if relu:
            y = np.maximum(y, 0)
        acts[a_out] = sat8(y)
    acts["gap"] = sat8((acts["b2o"].sum(axis=2) + 16) >> 5)
    acts["logits"] = acts["gap"] @ q["FC"]["w"].T + q["FC"]["b"]
    return acts


def quantize_input(x):
    """x (N, 2, 128) float in [-1, 1] → int8 exactly as generate_tb_vectors.py."""
    return np.clip(np.round(x * 127.0), -128, 127).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────
def save_intq(path, q):
    flat = {}
    for k, v in q["act_scale"].items():
        flat[f"scale.{k}"] = np.array(v)
    for name in [l[0] for l in LAYERS] + ["FC"]:
        for f, v in q[name].items():
            flat[f"{name}.{f}"] = np.array(v)
    np.savez(path, **flat)


def load_intq(path):
    d = np.load(path)
    q = {"act_scale": {}}
    for key in d.files:
        a, f = key.split(".", 1)
        if a == "scale":
            q["act_scale"][f] = float(d[key])
        else:
            v = d[key]
            q.setdefault(a, {})[f] = int(v) if v.ndim == 0 else v.astype(np.int64)
    return q


def read_tb_inputs(path, n_samples=128):
    """input_vectors.hex: one {I[7:0],Q[7:0]} word per line → (N, 2, 128) int."""
    words = [int(l, 16) for l in open(path) if l.strip()]
    arr = np.array(words, dtype=np.int64).reshape(-1, n_samples)
    i = (arr >> 8) & 0xFF
    qq = arr & 0xFF
    to_s = lambda v: np.where(v >= 128, v - 256, v)
    return np.stack([to_s(i), to_s(qq)], axis=1)


def load_split(data_dir, split):
    x = np.load(os.path.join(data_dir, f"X_{split}.npy"))     # (N, 128, 2)
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    return np.transpose(x, (0, 2, 1)).astype(np.float64), y


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────
def cmd_calib(args):
    _, params = load_folded(args.checkpoint)
    if args.tb_inputs:
        print("WARNING: calibrating on testbench vectors — development only. "
              "Use --data-dir (training split) for any reported number.")
        x = read_tb_inputs(args.tb_inputs) / 127.0
    else:
        x, _ = load_split(args.data_dir, "train")
        rng = np.random.default_rng(0)
        x = x[rng.choice(len(x), min(args.num_calib, len(x)), replace=False)]
    # Calibrate on the quantised input grid so scales see what the HW sees
    x = quantize_input(x) / 127.0
    acts = {}
    for i in range(0, len(x), 512):
        a = float_activations(params, x[i:i + 512])
        for k, v in a.items():
            acts.setdefault(k, []).append(np.abs(v).reshape(-1))
    act_scale = {"in": IN_SCALE}
    for k in ["a1", "a2", "b1h", "b1o", "p", "b2h", "b2o"]:
        v = np.concatenate(acts[k])
        act_scale[k] = max(np.percentile(v, args.percentile), 1e-6) / 127.0
        print(f"  {k:4s}: max={v.max():8.4f}  p{args.percentile}={act_scale[k]*127:8.4f}")
    q = build_int_model(params, act_scale)
    for name in [l[0] for l in LAYERS]:
        print(f"  {name:6s} SH={q[name]['SH']:2d}  M∈[{q[name]['M'].min()},"
              f"{q[name]['M'].max()}]  Ms={q[name]['Ms']}<<{q[name]['SK']}  |b|max={np.abs(q[name]['b']).max()}")
    save_intq(args.out, q)
    print(f"Integer model → {args.out}")


def cmd_eval(args):
    q = load_intq(args.intq)
    model, params = load_folded(args.checkpoint)
    if args.tb_inputs:
        xq = read_tb_inputs(args.tb_inputs)
        y = np.array([int(l, 16) for l in open(args.tb_labels) if l.strip()]) \
            if args.tb_labels else None
    else:
        x, y = load_split(args.data_dir, "test")
        xq = None
    n_ok_f = n_ok_i = n_agree = n = 0
    k = len(q["FC"]["b"])
    conf = np.zeros((k, k), dtype=np.int64)
    all_pi, all_lg = [], []
    bs = 1024
    total = len(xq) if xq is not None else len(x)
    for i in range(0, total, bs):
        if xq is not None:
            xb_q = xq[i:i + bs]
            xb_f = xb_q / 127.0
        else:
            xb_f = x[i:i + bs]
            xb_q = quantize_input(xb_f)
        pf = float_activations(params, xb_f)["logits"].argmax(1)
        lg = int_forward(q, xb_q)["logits"]
        pi = lg.argmax(1)
        all_lg.append(lg)
        n_agree += (pf == pi).sum()
        all_pi.append(pi)
        if y is not None:
            yb = y[i:i + bs]
            n_ok_f += (pf == yb).sum()
            n_ok_i += (pi == yb).sum()
            for t, p_ in zip(yb, pi):
                conf[t, p_] += 1
        n += len(pi)
    print(f"Samples              : {n}")
    if y is not None:
        print(f"FP32 (folded) acc    : {100 * n_ok_f / n:.2f}%")
        print(f"Integer-only acc     : {100 * n_ok_i / n:.2f}%   "
              f"(Δ = {100 * (n_ok_i - n_ok_f) / n:+.2f} pp)")
        print("Confusion (rows = true):")
        print(conf)
    print(f"FP32 / INT agreement : {100 * n_agree / n:.2f}%")
    meta_path = os.path.join(args.data_dir, "meta_test.npz")
    if args.report_dir and xq is None and os.path.exists(meta_path):
        from anom_eval import anomaly_score, load_meta, report
        import json
        with open(os.path.join(args.data_dir, "dataset_info.json")) as f:
            classes = json.load(f)["classes"]
        report("int8_hw", y, np.concatenate(all_pi), load_meta(args.data_dir, "test"),
               args.report_dir, classes, scores=anomaly_score(np.concatenate(all_lg)))


def to_hex(v, bits):
    return format(int(v) & ((1 << bits) - 1), f"0{(bits + 3) // 4}X")


def cmd_export(args):
    q = load_intq(args.intq)
    W = args.lanes
    os.makedirs(args.out_dir, exist_ok=True)
    cdiv = lambda a, b: -(-a // b)

    shapes = {"L1": (2, 128), "L2": (16, 128), "RB1C1": (32, 64), "RB1C2": (32, 64),
              "RB2P": (32, 64), "RB2C1": (32, 64), "RB2C2": (64, 32), "FC": (64, 1)}
    geom = {l[0]: l[2:5] for l in LAYERS}
    geom["FC"] = (1, 1, 0)

    wrows, prows, table = [], [], []
    for name in [l[0] for l in LAYERS] + ["FC"]:
        L = q[name]
        w = L["w"] if name != "FC" else L["w"][:, :, None]
        oc, ic, k = w.shape
        s_, p_ = geom[name][1], geom[name][2]
        cin, lin = shapes[name]
        assert cin == ic
        lout = (lin + 2 * p_ - k) // s_ + 1
        nog, nig = cdiv(oc, W), cdiv(ic, W)
        w_base, p_base = len(wrows), len(prows)
        for og in range(nog):
            for ig in range(nig):
                for kk in range(k):
                    row = 0
                    for o in range(W):
                        for i in range(W):
                            co, ci = og * W + o, ig * W + i
                            v = w[co, ci, kk] if co < oc and ci < ic else 0
                            row |= (int(v) & 0xFF) << (8 * (o * W + i))
                    wrows.append(row)
        for og in range(nog):
            row = 0
            for o in range(W):
                co = og * W + o
                b = int(L["b"][co]) if co < oc else 0
                m = int(L["M"][co]) if (co < oc and name != "FC") else 0
                row |= ((b & 0xFFFFFFFF) | ((m & 0xFFFF) << 32)) << (48 * o)
            prows.append(row)
        src, dst, skp = BUFS[name]
        layer = next((l for l in LAYERS if l[0] == name), None)
        table.append(dict(name=name, nig=nig, nog=nog, lin=lin, lout=lout, k=k, s=s_, p=p_,
                          relu=int(layer[5]) if layer else 0,
                          res=int(layer is not None and layer[6] is not None),
                          raw=int(name == "FC"), src=src, dst=dst, skp=skp,
                          wb=w_base, pb=p_base,
                          sh=L.get("SH", 1), ms=L.get("Ms", 0), sk=L.get("SK", 0)))

    with open(os.path.join(args.out_dir, "weights.mem"), "w") as f:
        for r in wrows:
            f.write(to_hex(r, 8 * W * W) + "\n")
    with open(os.path.join(args.out_dir, "params.mem"), "w") as f:
        for r in prows:
            f.write(to_hex(r, 48 * W) + "\n")

    # Layer table as a Verilog include (one case item per layer)
    with open(os.path.join(args.out_dir, "layer_table.vh"), "w") as f:
        f.write(f"// Auto-generated by int_quant.py export --lanes {W}. Do not edit.\n")
        f.write("// Guarded so tools that parse headers on their own see an empty file;\n")
        f.write("// rn_top.v defines RN_LAYER_TABLE right before including it.\n")
        f.write("`ifdef RN_LAYER_TABLE\n")
        f.write(f"localparam NUM_LAYERS = {len(table)};\n")
        f.write(f"localparam W_DEPTH    = {len(wrows)};\n")
        f.write(f"localparam P_DEPTH    = {len(prows)};\n")
        f.write(f"localparam EXPORT_LANES = {W};\n")
        f.write(f"localparam EXPORT_CLASSES = {len(q['FC']['b'])};\n")
        f.write("always @(*) begin\n    case (layer)\n")
        for i, t in enumerate(table):
            f.write(f"    {i}: begin // {t['name']}\n"
                    f"        d_nig={t['nig']}; d_nog={t['nog']}; d_lin={t['lin']}; "
                    f"d_lout={t['lout']}; d_k={t['k']}; d_s={t['s']}; d_p={t['p']};\n"
                    f"        d_relu={t['relu']}; d_res={t['res']}; d_raw={t['raw']}; "
                    f"d_src={t['src']}; d_dst={t['dst']}; d_skp={t['skp']};\n"
                    f"        d_wb={t['wb']}; d_pb={t['pb']}; d_sh={t['sh']}; d_ms={t['ms']}; d_sk={t['sk']};\n"
                    f"    end\n")
        f.write("    default: begin\n"
                "        d_nig=1; d_nog=1; d_lin=1; d_lout=1; d_k=1; d_s=1; d_p=0;\n"
                "        d_relu=0; d_res=0; d_raw=0; d_src=0; d_dst=0; d_skp=0;\n"
                "        d_wb=0; d_pb=0; d_sh=1; d_ms=0; d_sk=0;\n"
                "    end\n    endcase\nend\n`endif\n")

    mac_cycles = sum(t["nog"] * t["lout"] * t["nig"] * t["k"] for t in table)
    print(f"W={W}: weight ROM {len(wrows)} x {8 * W * W} b, param ROM {len(prows)} x {48 * W} b,"
          f" MAC-issue cycles = {mac_cycles}")

    if args.tb_inputs:
        xq = read_tb_inputs(args.tb_inputs)
        a = int_forward(q, xq)
        with open(os.path.join(args.out_dir, "golden_logits.hex"), "w") as f:
            for lg in a["logits"]:
                f.write(" ".join(to_hex(v, 32) for v in lg) + "\n")
        with open(os.path.join(args.out_dir, "golden_class.hex"), "w") as f:
            for c in a["logits"].argmax(1):
                f.write(f"{int(c):02X}\n")
        print(f"Golden outputs for {len(xq)} vectors → {args.out_dir}")


def cmd_export_c(args):
    """C header with the integer model, for the ARM (software) baseline in sw/."""
    q = load_intq(args.intq)
    arr = lambda name, ctype, v: (f"static const {ctype} {name}[{v.size}] = {{"
                                  + ",".join(str(int(x)) for x in v.reshape(-1)) + "};\n")
    with open(args.out, "w") as f:
        f.write("/* Auto-generated by int_quant.py export-c. Do not edit. */\n")
        f.write("#ifndef RN_MODEL_H\n#define RN_MODEL_H\n#include <stdint.h>\n\n")
        f.write(f"#define RN_NUM_CLASSES {len(q['FC']['b'])}\n\n")
        for name, pre, k, s_, p_, relu, skip, a_in, a_out in LAYERS:
            L = q[name]
            oc, ic, _ = L["w"].shape
            f.write(f"/* {name}: {ic}->{oc}, k={k}, s={s_}, p={p_}, relu={relu}, "
                    f"skip={skip} */\n")
            f.write(arr(f"{name}_w", "int8_t", L["w"]))
            f.write(arr(f"{name}_b", "int32_t", L["b"]))
            f.write(arr(f"{name}_m", "int32_t", L["M"]))
            f.write(f"#define {name}_SH {L['SH']}\n#define {name}_MS {L['Ms']}\n"
                    f"#define {name}_SK {L.get('SK', 0)}\n\n")
        f.write(arr("FC_w", "int8_t", q["FC"]["w"]))
        f.write(arr("FC_b", "int32_t", q["FC"]["b"]))
        f.write("\n#endif\n")
    print(f"C model header → {args.out}")
    if args.tb_inputs:
        xq = read_tb_inputs(args.tb_inputs)
        lg = int_forward(q, xq)["logits"]
        words = [int(l, 16) for l in open(args.tb_inputs) if l.strip()]
        vec = os.path.join(os.path.dirname(args.out), "rn_vectors.h")
        with open(vec, "w") as f:
            f.write("/* Auto-generated by int_quant.py export-c --tb-inputs. Do not edit. */\n")
            f.write("#ifndef RN_VECTORS_H\n#define RN_VECTORS_H\n#include <stdint.h>\n")
            f.write(f"#define RN_NUM_VECTORS {len(xq)}\n")
            f.write("/* channel-major INT8 input per vector: I[0..127], Q[0..127] */\n")
            f.write(arr("rn_vec_in", "int8_t", xq.reshape(len(xq), -1)))
            f.write("/* the same frames as 16-bit {I,Q} words, as written to the accelerator */\n")
            f.write(arr("rn_vec_iq", "uint16_t", np.array(words)))
            f.write(arr("rn_vec_logits", "int32_t", lg))
            if args.tb_labels:
                lab = [int(l, 16) for l in open(args.tb_labels) if l.strip()][:len(xq)]
                f.write("#define RN_HAVE_LABELS 1\n")
                f.write(arr("rn_vec_label", "uint8_t", np.array(lab)))
            f.write("#endif\n")
        print(f"{len(xq)} test vectors + golden logits → {vec}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--data-dir", default=os.path.join("..", "data"))
    c.add_argument("--tb-inputs", help="dev only: calibrate on testbench vectors")
    c.add_argument("--num-calib", type=int, default=4096)
    c.add_argument("--percentile", type=float, default=99.99)
    c.add_argument("--out", default=os.path.join("..", "checkpoints", "radarnet_intq.npz"))
    e = sub.add_parser("eval")
    e.add_argument("--intq", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--data-dir", default=os.path.join("..", "data"))
    e.add_argument("--tb-inputs")
    e.add_argument("--tb-labels")
    e.add_argument("--report-dir", help="anomaly dataset: write int8_hw_metrics.json here")
    x = sub.add_parser("export")
    x.add_argument("--intq", required=True)
    x.add_argument("--lanes", type=int, default=8)
    x.add_argument("--tb-inputs")
    x.add_argument("--out-dir", default=os.path.join("..", "rtl_v2", "mem"))
    c = sub.add_parser("export-c")
    c.add_argument("--intq", required=True)
    c.add_argument("--out", default=os.path.join("..", "sw", "rn_model.h"))
    c.add_argument("--tb-inputs", help="also write rn_vectors.h with these vectors")
    c.add_argument("--tb-labels", help="true labels of --tb-inputs (printed by rn_bench.c)")
    args = ap.parse_args()
    {"calib": cmd_calib, "eval": cmd_eval, "export": cmd_export,
     "export-c": cmd_export_c}[args.cmd](args)


if __name__ == "__main__":
    main()
