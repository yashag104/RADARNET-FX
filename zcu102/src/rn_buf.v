`timescale 1ns / 1ps
/*
 * rn_buf.v — Feature-map buffer (simple dual-port, synchronous read)
 * Word = W channels × 8 bit at one time position.
 * Address layout: (channel_group * L) + t
 * Read latency is 2 cycles (BRAM output register enabled).
 */
module rn_buf #(
    parameter W     = 8,
    parameter DEPTH = 256,
    parameter AW    = $clog2(DEPTH)
)(
    input  wire           clk,
    input  wire           we,
    input  wire [AW-1:0]  waddr,
    input  wire [8*W-1:0] wdata,
    input  wire [AW-1:0]  raddr,
    output reg  [8*W-1:0] rdata
);
    (* ram_style = "block" *)
    reg [8*W-1:0] mem [0:DEPTH-1];
    reg [8*W-1:0] rd_q;

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rd_q  <= mem[raddr];
        rdata <= rd_q;
    end
endmodule
