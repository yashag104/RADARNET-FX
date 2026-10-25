# ── Step 2 (run in Colab with:  %run radarnet_step2.py ) ─────────────────────
# Option A: RF anomaly dataset on RadioML 2018.01A + training + baselines +
# integer (hardware) model, end to end.
#
#   1-2. unpack project files, mount Google Drive
#   3.   download RadioML 2018.01A (skipped if the dataset is cached in Drive)
#   4.   generate the anomaly dataset (7 classes, JSR -10..+20 dB, base SNR >= 0 dB, AGC)
#   5.   train RadarNet-1D on the GPU
#   6.   classical baselines (energy detector, expert features + GBM)
#   7.   integer hardware model: calibrate + evaluate
#   8.   summary table, figures, RTL test vectors
#   9.   save everything to Drive/RADARNET-FX-results/step2

ANOM_FILES = SPLIT_FILES + ["meta_train.npz", "meta_val.npz", "meta_test.npz",
                            "dataset_info.json"]
DATASET_VERSION = 2   # must match anomaly_dataset.DATASET_VERSION


def cache_is_current(cache_dir):
    import json
    info = os.path.join(cache_dir, "dataset_info.json")
    if not os.path.exists(info):
        return False
    with open(info) as f:
        return json.load(f).get("version") == DATASET_VERSION


def main():
    t0 = time.time()
    unpack_and_mount("step2")

    import torch
    if not torch.cuda.is_available() and not os.environ.get("RADARNET_ALLOW_CPU"):
        raise SystemExit(
            "\n*** No GPU in this session. Training needs one.\n"
            "*** Runtime → Change runtime type → choose 'T4 GPU' → Save,\n"
            "*** then upload radarnet_step2.py again and run  %run radarnet_step2.py")
    say(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU test mode)'}")

    data_cache = os.path.join(DRIVE_ROOT, "data_anom")
    data_dir = os.path.join(WORK, "data_anom")
    if cache_is_current(data_cache) and cached(data_dir, data_cache, ANOM_FILES):
        banner(3, "anomaly dataset already in Drive — copied it (download skipped)")
    else:
        if os.path.exists(data_cache):
            say("  the anomaly dataset in Drive is from an older version — regenerating it")
            shutil.rmtree(data_cache)
        h5 = get_radioml(3)
        banner(4, "generating the anomaly dataset (~5 min)")
        run(f'python anomaly_dataset.py --hdf5 "{h5}" --out-dir ../data_anom '
            + os.environ.get("RADARNET_DATASET_ARGS", ""))
        say("  saving the dataset to Drive so later runs skip this …")
        shutil.copytree(data_dir, data_cache, dirs_exist_ok=True)

    ck = os.path.join(WORK, "checkpoints_anom")
    banner(5, "training RadarNet-1D (about 20-40 min on a T4 GPU)")
    run("python train_anomaly.py --data-dir ../data_anom --out-dir ../checkpoints_anom "
        + os.environ.get("RADARNET_TRAIN_ARGS", ""))

    banner(6, "classical baselines (about 5-10 min)")
    run("python baselines.py --data-dir ../data_anom --out-dir ../checkpoints_anom")

    banner(7, "integer hardware model: calibrate + evaluate on the test set")
    intq = os.path.join(ck, "radarnet_anom_intq.npz")
    pth = os.path.join(ck, "radarnet_anom_best.pth")
    run(f"python int_quant.py calib --checkpoint {pth} --data-dir ../data_anom --out {intq}")
    run(f"python int_quant.py eval --intq {intq} --checkpoint {pth} "
        f"--data-dir ../data_anom --report-dir {ck}")

    banner(8, "summary table, figures, RTL test vectors")
    run("python anom_report.py --data-dir ../data_anom --ck-dir ../checkpoints_anom")

    banner(9, "saving results to Google Drive")
    shutil.copytree(ck, OUT, dirs_exist_ok=True)
    for f in sorted(os.listdir(ck)):
        say(f"  {OUT}/{f}")
    finish("step2", t0, [
        "Send Claude: the table printed in STEP 8 (or the file summary.md).",
        "Then download the whole folder Drive/RADARNET-FX-results/step2",
        "into RADARNET-FX/checkpoints_anom/ on your laptop."])


entry(main, "step2")
