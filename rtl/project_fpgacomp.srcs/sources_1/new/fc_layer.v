`timescale 1ns / 1ps
/*
 * fc_layer.v — Fully Connected Layer (64→4, INT8, BRAM-friendly)
 * Address-based input, unified ROM port. Output kept as register (small).
 */
module fc_layer #(
    parameter IN_FEATURES  = 64,
    parameter OUT_FEATURES = 4,
    parameter DATA_WIDTH   = 8
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    // Input read port (address -> upstream GAP memory)
    output reg  [15:0] fin_addr,
    input  wire [DATA_WIDTH-1:0] fin_data,
    // Unified ROM: bias @ 0..OF-1, weights @ OF..OF+IF*OF-1
    output reg  [15:0] rom_addr,
    input  wire [DATA_WIDTH-1:0] rom_data,
    // Output logits (small, kept as register)
    output reg  [OUT_FEATURES*DATA_WIDTH-1:0] fc_out,
    output reg         out_valid
);
    localparam S_IDLE=0, S_BADDR=1, S_BREAD=2, S_ADDR=3, S_MAC=4, S_STORE=5, S_DONE=6;
    reg [2:0] state;
    reg [$clog2(OUT_FEATURES+1)-1:0] neuron;
    reg [$clog2(IN_FEATURES+1)-1:0]  feat;
    reg signed [23:0] acc;
    reg signed [DATA_WIDTH-1:0] bias_reg;

    wire signed [DATA_WIDTH-1:0] w_val = $signed(rom_data);
    wire signed [DATA_WIDTH-1:0] x_val = $signed(fin_data);
    wire signed [15:0] product = w_val * x_val;

    function [DATA_WIDTH-1:0] sat8;
        input signed [23:0] v;
        if (v > 127) sat8 = 8'sd127;
        else if (v < -128) sat8 = -8'sd128;
        else sat8 = v[DATA_WIDTH-1:0];
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=0; out_valid<=0;
            neuron<=0; feat<=0; acc<=0; bias_reg<=0;
            rom_addr<=0; fin_addr<=0; fc_out<=0;
        end else begin
            case (state)
                S_IDLE: begin
                    done<=0; out_valid<=0;
                    if (start) begin
                        neuron<=0; feat<=0; acc<=0;
                        rom_addr<=16'd0; // bias[0]
                        state<=S_BADDR;
                    end
                end
                S_BADDR: state <= S_BREAD;
                S_BREAD: begin
                    bias_reg <= $signed(rom_data);
                    state <= S_ADDR;
                end
                S_ADDR: begin
                    fin_addr <= feat;
                    rom_addr <= OUT_FEATURES + neuron * IN_FEATURES + feat;
                    state <= S_MAC;
                end
                S_MAC: begin
                    acc <= acc + {{8{product[15]}}, product};
                    if (feat < IN_FEATURES-1) begin
                        feat <= feat + 1;
                        state <= S_ADDR;
                    end else begin
                        feat <= 0;
                        state <= S_STORE;
                    end
                end
                S_STORE: begin
                    fc_out[neuron*DATA_WIDTH +: DATA_WIDTH] <=
                        sat8(acc + {{16{bias_reg[7]}}, bias_reg});
                    acc <= 0;
                    if (neuron < OUT_FEATURES-1) begin
                        neuron <= neuron + 1;
                        rom_addr <= neuron + 16'd1; // bias[next]
                        state <= S_BADDR;
                    end else
                        state <= S_DONE;
                end
                S_DONE: begin done<=1; out_valid<=1; state<=S_IDLE; end
            endcase
        end
    end
endmodule