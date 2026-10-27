# 2026-10-26T17:34:35.682950600
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis")

platform = client.create_platform_component(name = "board_test_2610",hw_design = "$COMPONENT_LOCATION/../../vivado/rn_bd_wrapper.xsa",os = "standalone",cpu = "ps7_cortexa9_0",domain_name = "standalone_ps7_cortexa9_0",compiler = "gcc")

platform = client.get_component(name="board_test_2610")
status = platform.build()

comp = client.create_app_component(name="testing2610",platform = "$COMPONENT_LOCATION/../board_test_2610/export/board_test_2610/board_test_2610.xpfm",domain = "standalone_ps7_cortexa9_0")

comp = client.get_component(name="testing2610")
status = comp.import_files(from_loc="$COMPONENT_LOCATION/../../../sw", files=["rn_infer.h", "rn_vectors.h"], dest_dir_in_cmp = "src", is_skip_copy_sources = False)

status = platform.build()

comp = client.get_component(name="testing2610")
comp.build()

comp = client.get_component(name="testing2610")
status = comp.import_files(from_loc="$COMPONENT_LOCATION/../../../sw", files=["rn_model.h"], dest_dir_in_cmp = "src", is_skip_copy_sources = False)

status = platform.build()

comp = client.get_component(name="testing2610")
comp.build()

status = platform.build()

comp.build()

comp = client.get_component(name="testing2610")
status = comp.import_files(from_loc="$COMPONENT_LOCATION/../../../sw", files=["rn_bench.c"], dest_dir_in_cmp = "src", is_skip_copy_sources = False)

comp.set_app_config(key = "USER_COMPILE_OPTIMIZATION_LEVEL", values = ["-O3"])

status = platform.build()

comp = client.get_component(name="testing2610")
comp.build()

status = platform.build()

comp.build()

status = platform.build()

comp.build()

status = platform.build()

comp.build()

comp = client.get_component(name="testing2610")
status = comp.import_files(from_loc="$COMPONENT_LOCATION/../../../sw", files=["rn_bench.c", "rn_vectors.h"], dest_dir_in_cmp = "src", is_skip_copy_sources = False)

status = platform.build()

comp = client.get_component(name="testing2610")
comp.build()

vitis.dispose()

