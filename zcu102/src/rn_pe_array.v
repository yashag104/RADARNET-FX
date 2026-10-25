`timescale 1ns / 1ps
/*
 * rn_pe_array.v — W×W INT8 processing-element array
 * ===================================================
 * One activation word (W input channels at the same time position) is
 * broadcast to W output-channel lanes. Each lane multiplies it with its own
 * W weights and reduces the W products through a fully registered binary
 * adder tree, so W*W MACs are issued per clock.
 *
 *   sum[o] = Σ_i  w[o][i] * x[i]
 *
 * Latency: 1 (multiply) + log2(W) (adder tree) cycles. W must be a power of 2.
 * USE_DSP=1 maps the W*W multipliers to DSP48 slices (operand registers are
 * absorbed into the DSP); USE_DSP=0 builds them from LUTs.
 */
module rn_pe_array #(
    parameter W       = 8,
    parameter USE_DSP = 1,
    parameter LV      = $clog2(W),
    parameter SUMW = 16 + LV
)(
    input  wire                  clk,
    input  wire [8*W-1:0]        x_word,   // lane i = input channel i
    input  wire [8*W*W-1:0]      w_row,    // byte (o*W + i) = w[o][i]
    output wire [SUMW*W-1:0]     sum_out   // lane o = output channel o
);
    genvar o, l, j;
    generate
        for (o = 0; o < W; o = o + 1) begin : pe
            for (l = 0; l <= LV; l = l + 1) begin : lvl
                for (j = 0; j < (W >> l); j = j + 1) begin : nd
                    reg signed [SUMW-1:0] s;
                    if (l == 0 && USE_DSP) begin : mul
                        wire signed [7:0] xv = x_word[8*j +: 8];
                        wire signed [7:0] wv = w_row[8*(o*W + j) +: 8];
                        (* use_dsp = "yes" *) wire signed [15:0] pv = xv * wv;
                        always @(posedge clk) s <= pv;
                    end else if (l == 0) begin : mul_lut
                        wire signed [7:0] xv = x_word[8*j +: 8];
                        wire signed [7:0] wv = w_row[8*(o*W + j) +: 8];
                        (* use_dsp = "no" *) wire signed [15:0] pv = xv * wv;
                        always @(posedge clk) s <= pv;
                    end else begin : add
                        always @(posedge clk)
                            s <= lvl[l-1].nd[2*j].s + lvl[l-1].nd[2*j+1].s;
                    end
                end
            end
            assign sum_out[SUMW*o +: SUMW] = lvl[LV].nd[0].s;
        end
    endgenerate
endmodule
