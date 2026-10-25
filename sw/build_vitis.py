#!/usr/bin/env python3
"""
build_vitis.py — build the bare-metal benchmark (rn_bench.elf) for the ZedBoard
================================================================================
    vitis -s build_vitis.py <rn_zed.xsa> <workspace_dir> [cpu]

cpu: ps7_cortexa9_0 (ZedBoard, default) or psu_cortexa53_0 (ZCU104).

Creates a standalone Cortex-A9 platform from the XSA, an application component
with rn_bench.c + rn_infer.c and the generated headers, builds it at -O3 and
prints the ELF path.
"""
import os
import shutil
import stat
import sys

import vitis

XSA = os.path.abspath(sys.argv[1])
WS = os.path.abspath(sys.argv[2])
CPU = sys.argv[3] if len(sys.argv) > 3 else "ps7_cortexa9_0"
CPU_FLAGS = {"ps7_cortexa9_0": "-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard",
             "psu_cortexa53_0": "-mcpu=cortex-a53"}[CPU]
SW = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["rn_bench.c", "rn_infer.c", "rn_infer.h", "rn_model.h", "rn_vectors.h"]

def _force_remove(func, path, _):
    os.chmod(path, stat.S_IWRITE)             # Windows: generated files are read-only
    func(path)


if os.path.isdir(WS):
    shutil.rmtree(WS, onerror=_force_remove)
os.makedirs(WS)
client = vitis.create_client()
try:
    client.set_workspace(path=WS)

    plat = client.create_platform_component(name="rn_plat", hw_design=XSA, os="standalone",
                                            cpu=CPU, domain_name="standalone_app")
    plat.build()
    xpfm = client.find_platform_in_repos("rn_plat")

    app = None
    for template in ("empty_application", "hello_world"):
        try:
            app = client.create_app_component(name="rn_bench", platform=xpfm,
                                              domain="standalone_app", template=template)
            print(f"app created from template '{template}'")
            break
        except Exception as e:                      # template name differs between releases
            print(f"template '{template}' not usable: {e}")
    if app is None:
        raise SystemExit("could not create the application component")

    src_dir = os.path.join(WS, "rn_bench", "src")
    for f in ("helloworld.c",):                     # drop the template's own main()
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            os.remove(p)
    app.import_files(from_loc=SW, files=SOURCES, dest_dir_in_cmp="src")
    for key, val in (("USER_COMPILE_OPTIMIZATION_LEVEL", "-O3"),
                     ("USER_COMPILE_OTHER_FLAGS", CPU_FLAGS)):
        try:
            app.set_app_config(key=key, values=val)
        except Exception as e:
            print(f"config {key} not set: {e}")
    app.build()

    elf = None
    for root, _, files in os.walk(os.path.join(WS, "rn_bench")):
        if "rn_bench.elf" in files:
            elf = os.path.join(root, "rn_bench.elf")
    print(f"ELF_PATH={elf}")
finally:
    vitis.dispose()                            # never leave a server holding the workspace
