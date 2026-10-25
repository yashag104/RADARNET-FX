# ── Step 1 (run in Colab with:  %run radarnet_step1.py ) ─────────────────────
# Rebuilds the paper's dataset split from RadioML 2018.01A and measures the
# accuracy of the bit-exact integer model the v2 hardware implements.
#
#   1-2. unpack project files, mount Google Drive
#   3.   download RadioML 2018.01A (~18 GB) from the public Kaggle mirror
#   4.   stream it through preprocess.py → data/*.npy (identical split to the paper)
#   5-6. calibrate the integer model, evaluate FP32 vs integer on the full test set
#   7.   save radarnet_intq.npz + step1_log.txt to Drive/RADARNET-FX-results/step1


def main():
    t0 = time.time()
    unpack_and_mount("step1")
    data_dir = os.path.join(WORK, "data")
    data_cache = os.path.join(DRIVE_ROOT, "data")
    if cached(data_dir, data_cache, SPLIT_FILES):
        banner(3, "dataset split already in Drive — copied it (download skipped)")
    else:
        h5 = get_radioml(3)
        banner(4, "rebuilding the paper's train/val/test split")
        run(f'python preprocess.py --hdf5 "{h5}" --out-dir ../data')
        say("  saving the split to Drive so later runs skip the download …")
        shutil.copytree(data_dir, data_cache, dirs_exist_ok=True)

    banner(5, "integer model: calibrate on training data")
    intq = os.path.join(WORK, "checkpoints", "radarnet_intq.npz")
    run(f"python int_quant.py calib --checkpoint ../checkpoints/radarnet_best.pth "
        f"--data-dir ../data --out {intq}")

    banner(6, "accuracy on the full test set (FP32 vs integer hardware model)")
    run(f"python int_quant.py eval --intq {intq} "
        f"--checkpoint ../checkpoints/radarnet_best.pth --data-dir ../data")
    say(f"  (paper / metrics.json FP32 test accuracy: {EXPECTED_FP32_ACC:.2f}% — "
        f"the 'FP32 (folded) acc' line above should match it)")

    banner(7, "saving results to Google Drive")
    shutil.copy(intq, OUT)
    say(f"  {OUT}/radarnet_intq.npz")
    finish("step1", t0, [
        "Send Claude: the lines from STEP 6 (or the file step1_log.txt),",
        "and download radarnet_intq.npz from Drive/RADARNET-FX-results/step1",
        "into RADARNET-FX/checkpoints/ on your laptop."])


entry(main, "step1")
