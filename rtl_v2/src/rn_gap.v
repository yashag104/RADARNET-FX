`timescale 1ns / 1ps
/*
 * rn_gap.v — Lane-parallel Global Average Pooling
 * ================================================
 * Reads NCH×T activations as W-channel words, sums each lane over T and
 * writes g = sat8((sum + T/2) >>> log2(T)) — a rounding divide by T=32
 * with no divider. Output word cg goes to address cg (layout for L=1).
 */
module rn_gap #(
    parameter W   = 8,
    parameter NCH = 64,
    parameter T   = 32,
    parameter AW  = 8
)(
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start,
    output reg            done,
    output reg  [AW-1:0]  raddr,
    input  wire [8*W-1:0] rdata,
    output reg            we,
    output reg  [AW-1:0]  waddr,
    output reg  [8*W-1:0] wdata
);
    localparam NCG = (NCH + W - 1) / W;
    localparam TB  = $clog2(T);
    localparam SW  = 8 + TB + 1;

    reg        run;
    reg [7:0]  cg, t;
    reg        i_v, i_first, i_last;  // issue stage (aligned with raddr)
    reg [7:0]  i_cg;
    reg        d_v, d_first, d_last;  // data stage (aligned with rdata)
    reg [7:0]  d_cg;
    reg signed [SW-1:0] sum [0:W-1];

    function [7:0] sat8;
        input signed [SW-1:0] v;
        sat8 = (v > 127) ? 8'sd127 : (v < -128) ? -8'sd128 : v[7:0];
    endfunction

    integer o;
    reg [8*W-1:0]       rq;               // registered BRAM data
    reg                 e_v, e_first, e_last;  // 2nd BRAM latency cycle
    reg [7:0]           e_cg;
    reg                 q_v, q_first, q_last;
    reg [7:0]           q_cg;
    reg                 w_v;              // sum complete → saturate & write
    reg [7:0]           w_cg;
    reg signed [SW-1:0] avg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run <= 0; done <= 0; cg <= 0; t <= 0;
            i_v <= 0; d_v <= 0; e_v <= 0; q_v <= 0; w_v <= 0; we <= 0;
        end else begin
            done <= 0;
            we   <= 0;
            // Issue
            i_v <= run;
            if (start) begin
                run <= 1; cg <= 0; t <= 0;
            end else if (run) begin
                if (t < T - 1) t <= t + 1;
                else begin
                    t <= 0;
                    if (cg < NCG - 1) cg <= cg + 1;
                    else run <= 0;
                end
            end
            d_v <= i_v;           // BRAM latency (2 cycles)
            e_v <= d_v;
            q_v <= e_v;           // data register
            w_v <= q_v & q_last;  // accumulate done
            if (w_v) begin
                we <= 1;
                if (w_cg == NCG - 1) done <= 1;
            end
        end
    end

    // Datapath (no reset needed)
    always @(posedge clk) begin
        raddr   <= cg * T + t;
        i_first <= (t == 0);
        i_last  <= (t == T - 1);
        i_cg    <= cg;
        d_first <= i_first; d_last <= i_last; d_cg <= i_cg;
        e_first <= d_first; e_last <= d_last; e_cg <= d_cg;
        rq      <= rdata;
        q_first <= e_first; q_last <= e_last; q_cg <= e_cg;
        if (q_v)
            for (o = 0; o < W; o = o + 1)
                sum[o] <= (q_first ? 0 : sum[o]) + $signed(rq[8*o +: 8]);
        w_cg <= q_cg;
        waddr <= w_cg;
        for (o = 0; o < W; o = o + 1) begin
            avg = (sum[o] + (T / 2)) >>> TB;
            wdata[8*o +: 8] <= sat8(avg);
        end
    end
endmodule
