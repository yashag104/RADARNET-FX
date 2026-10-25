# ── Shared helpers for the one-file Colab step scripts (bundled by build.py) ──
import base64
import os
import shutil
import subprocess
import sys
import time
import zipfile

BASE = os.environ.get("RADARNET_BASE", "/content")      # override only for local testing
WORK = os.path.join(BASE, "radarnet")
PY = os.path.join(WORK, "python")
RAW = os.path.join(BASE, "raw")
DRIVE_ROOT = os.path.join(BASE, "drive/MyDrive/RADARNET-FX-results")
KAGGLE_URL = os.environ.get("RADARNET_URL",  # override only for local testing
                            "https://www.kaggle.com/api/v1/datasets/download/pinxau1000/radioml2018")
H5_NAME = "GOLD_XYZ_OSC.0001_1024.hdf5"
SPLIT_FILES = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]

LOG = []
OUT = None          # set by each step to Drive/RADARNET-FX-results/<step>


def say(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def banner(n, msg):
    say("")
    say("=" * 70)
    say(f" STEP {n}: {msg}")
    say("=" * 70)


def run(cmd, cwd=PY):
    """Run a command, stream its output, keep it for the log, stop on failure."""
    say(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        if "\r" not in line:
            LOG.append(line)
    if p.wait() != 0:
        raise SystemExit(f"\n*** Command failed: {cmd}\n*** Copy everything above and send it to Claude.")


def save_log(step):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{step}_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


def unpack_and_mount(step):
    """Write the embedded project files and mount Google Drive."""
    global OUT
    banner(1, "unpacking project files")
    for rel, b64 in EMBEDDED_FILES.items():
        path = os.path.join(WORK, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        say(f"  wrote {rel}")
    banner(2, "connecting Google Drive (approve the pop-up)")
    from google.colab import drive
    drive.mount(os.path.join(BASE, "drive"))
    OUT = os.path.join(DRIVE_ROOT, step)
    os.makedirs(OUT, exist_ok=True)
    say(f"  results folder: {OUT}")


def get_radioml(n):
    """Download + extract RadioML 2018.01A once per Colab session; returns the .hdf5 path."""
    h5 = os.path.join(RAW, H5_NAME)
    if os.path.exists(h5):
        banner(n, "RadioML 2018.01A already downloaded in this session")
        return h5
    banner(n, "downloading RadioML 2018.01A (~18 GB, usually 5-15 min)")
    free_gb = shutil.disk_usage(BASE).free / 1e9
    say(f"  free disk: {free_gb:.0f} GB")
    if free_gb < 45:
        raise SystemExit("*** Not enough disk space (need ~45 GB). "
                         "Runtime → Disconnect and delete runtime, then try again.")
    os.makedirs(RAW, exist_ok=True)
    zpath = os.path.join(RAW, "archive.zip")
    run(f'curl -L --fail --retry 5 -C - -o "{zpath}" "{KAGGLE_URL}"', cwd=RAW)
    say("  extracting the .hdf5 file (a few minutes) …")
    with zipfile.ZipFile(zpath) as z, z.open(H5_NAME) as src, open(h5, "wb") as dst:
        shutil.copyfileobj(src, dst, length=64 << 20)
    os.remove(zpath)
    say(f"  dataset ready: {os.path.getsize(h5) / 1e9:.1f} GB")
    return h5


def cached(src_dir, cache_dir, files):
    """Copy files from a Drive cache if all are present; True on success."""
    if all(os.path.exists(os.path.join(cache_dir, f)) for f in files):
        shutil.copytree(cache_dir, src_dir, dirs_exist_ok=True)
        return True
    return False


def finish(step, t0, lines):
    save_log(step)
    say("")
    say("#" * 70)
    say(f" ALL DONE in {(time.time() - t0) / 60:.0f} min.")
    for line in lines:
        say(" " + line)
    say("#" * 70)
    save_log(step)


def entry(main, step):
    try:
        main()
    except SystemExit as e:
        print(e)
        try:
            save_log(step)
        except Exception:
            pass
