transcript off
onbreak {quit -force}
onerror {quit -force}
transcript on

asim +access +r +m+tb_radarnet  -L xil_defaultlib -L xpm -L blk_mem_gen_v8_4_12 -L unisims_ver -L unimacro_ver -L secureip -O5 xil_defaultlib.tb_radarnet xil_defaultlib.glbl

do {tb_radarnet.udo}

run 1000ns

endsim

quit -force
