`timescale 1ns / 1ps
/*
 * input_buffer.v — Stores 128 I/Q pairs, provides byte-addressable read.
 * Address layout: 0..127 = I channel, 128..255 = Q channel.
 */
module input_buffer #(
    parameter NUM_SAMPLES = 128,
    parameter DATA_WIDTH  = 8
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        wr_en,
    input  wire        clear,
    input  wire [2*DATA_WIDTH-1:0] din,  // {I[7:0], Q[7:0]}
    output reg         full,
    // Address-based read port for downstream
    input  wire [15:0] rd_addr,
    output wire [DATA_WIDTH-1:0] rd_data
);
    // Storage: 128 x 16-bit (I+Q pairs)
    reg [2*DATA_WIDTH-1:0] buffer [0:NUM_SAMPLES-1];
    reg [$clog2(NUM_SAMPLES):0] wr_ptr;

    // Async read: addr[7]=0 → I (upper byte), addr[7]=1 → Q (lower byte)
    wire [$clog2(NUM_SAMPLES)-1:0] sample_idx = rd_addr[$clog2(NUM_SAMPLES)-1:0];
    wire ch_sel = rd_addr[$clog2(NUM_SAMPLES)]; // channel select
    assign rd_data = ch_sel ? buffer[sample_idx][DATA_WIDTH-1:0]
                            : buffer[sample_idx][2*DATA_WIDTH-1:DATA_WIDTH];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0; full <= 0;
        end else if (clear) begin
            wr_ptr <= 0; full <= 0;
        end else if (wr_en && !full) begin
            buffer[wr_ptr] <= din;
            if (wr_ptr == NUM_SAMPLES - 1)
                full <= 1'b1;
            wr_ptr <= wr_ptr + 1;
        end
    end
endmodule