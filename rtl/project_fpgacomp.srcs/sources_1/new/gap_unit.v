`timescale 1ns / 1ps
/*
 * gap_unit.v — Global Average Pooling (INT8, BRAM-friendly)
 * Address-based input. Output register (small: 64 bytes).
 */
module gap_unit #(
    parameter NUM_CHANNELS = 64,
    parameter TIME_STEPS   = 32,
    parameter DATA_WIDTH   = 8,
    parameter SHIFT_BITS   = 5
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    // Input read port
    output reg  [15:0] fin_addr,
    input  wire [DATA_WIDTH-1:0] fin_data,
    // Output (small, kept as register)
    output reg  [NUM_CHANNELS*DATA_WIDTH-1:0] gap_out,
    output reg         out_valid
);
    localparam S_IDLE=0, S_ADDR=1, S_ACC=2, S_DIV=3, S_DONE=4;
    reg [2:0] state;
    reg [$clog2(NUM_CHANNELS+1)-1:0] ch;
    reg [$clog2(TIME_STEPS+1)-1:0]   ts;
    reg signed [15:0] acc;

    function [DATA_WIDTH-1:0] sat8;
        input signed [15:0] v;
        if (v > 127) sat8 = 8'sd127;
        else if (v < -128) sat8 = -8'sd128;
        else sat8 = v[DATA_WIDTH-1:0];
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=0; out_valid<=0;
            ch<=0; ts<=0; acc<=0; fin_addr<=0;
            gap_out<= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done<=0; out_valid<=0;
                    if (start) begin
                        ch<=0; ts<=0; acc<=0;
                        state<=S_ADDR;
                    end
                end
                S_ADDR: begin
                    fin_addr <= ch * TIME_STEPS + ts;
                    state <= S_ACC;
                end
                S_ACC: begin
                    acc <= acc + {{8{fin_data[DATA_WIDTH-1]}}, fin_data};
                    if (ts < TIME_STEPS-1) begin
                        ts <= ts + 1;
                        state <= S_ADDR;
                    end else begin
                        state <= S_DIV;
                    end
                end
                S_DIV: begin
                    gap_out[ch*DATA_WIDTH +: DATA_WIDTH] <= sat8(acc >>> SHIFT_BITS);
                    acc <= 0; ts <= 0;
                    if (ch < NUM_CHANNELS-1) begin
                        ch <= ch + 1;
                        state <= S_ADDR;
                    end else
                        state <= S_DONE;
                end
                S_DONE: begin done<=1; out_valid<=1; state<=S_IDLE; end
            endcase
        end
    end
endmodule