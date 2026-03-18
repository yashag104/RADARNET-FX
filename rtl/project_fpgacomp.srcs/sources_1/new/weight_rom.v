`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 16.03.2026 16:44:57
// Design Name: 
// Module Name: weight_rom
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

/*
 * weight_rom.v - BRAM-Based Weight ROM for All Layers
 * =====================================================
 * Single ROM module initialised from .hex files via $readmemh.
 * Each layer has a contiguous address range (see address_map.txt).
 *
 * The ROM is single-port, synchronous read with 1-cycle latency.
 * Total depth is set by TOTAL_DEPTH parameter to accommodate all
 * ~29K INT8 weight values.
 *
 * For synthesis, Vivado will infer BRAM blocks automatically.
 */
module weight_rom #(
    parameter DATA_WIDTH  = 8,
    parameter TOTAL_DEPTH = 32768
)(
    input  wire                         clk,
    input  wire [$clog2(TOTAL_DEPTH)-1:0] addr,
    output wire [DATA_WIDTH-1:0]        data_out
);

    // Instantiate Single Port ROM BRAM IP
    blk_mem_gen_0 u_bram (
        .clka(clk),
        .ena(1'b1),
        .addra(addr),
        .douta(data_out)
    );

endmodule
