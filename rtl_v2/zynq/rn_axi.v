`timescale 1ns / 1ps
/*
 * rn_axi.v — AXI4-Lite slave around rn_top for the Zynq PS
 * =========================================================
 * Register map (32-bit, byte offsets):
 *   0x00 CTRL    W: bit0 = 1 → start an inference (clears DONE)
 *   0x04 STATUS  R: bit0 busy, bit1 done (sticky until next start)
 *   0x08 IQ      W: {I[15:8], Q[7:0]} — one sample per write, 128 per frame
 *   0x0C CLASS   R: predicted class
 *   0x10 CYCLES  R: accelerator cycles, start → result (incl. 128-sample load)
 *   0x14 INFO    R: {16'h524E ("RN"), NUM_CLASSES[7:0], W[7:0]}
 *   0x20 + 4c    R: INT32 logit of class c
 *
 * dbg_* ports expose the accelerator's status for an ILA / VIO in the block
 * design. They are observation-only; leave them unconnected if unused.
 */
module rn_axi #(
    parameter W            = 8,
    parameter NUM_CLASSES  = 4,
    parameter PE_DSP       = (W * W <= 128),   // DSP48 PE multipliers (see rn_top)
    parameter WEIGHTS_FILE = "weights.mem",
    parameter PARAMS_FILE  = "params.mem"
)(
    input  wire        s_axi_aclk,
    input  wire        s_axi_aresetn,
    input  wire [7:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [7:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,
    // debug / observation (ILA, VIO)
    output wire        dbg_start,
    output wire        dbg_iq_valid,
    output wire        dbg_busy,
    output wire        dbg_done,
    output wire        dbg_result_valid,
    output wire        dbg_anomaly_flag,
    output wire [7:0]  dbg_class_id,
    output wire [31:0] dbg_cycle_count
);
    localparam CW = (NUM_CLASSES > 2) ? $clog2(NUM_CLASSES) : 1;
    wire clk = s_axi_aclk;

    reg         start, iq_valid, done_sticky;
    reg  [15:0] iq_data;
    wire        busy, done, result_valid, anomaly_flag;
    wire [CW-1:0] class_id;
    wire [31:0] cycle_count;
    wire [32*NUM_CLASSES-1:0] logits;

    rn_top #(.W(W), .NUM_CLASSES(NUM_CLASSES), .PE_DSP(PE_DSP),
             .WEIGHTS_FILE(WEIGHTS_FILE), .PARAMS_FILE(PARAMS_FILE)) u_rn (
        .clk(clk), .rst_n(s_axi_aresetn), .start(start), .iq_valid(iq_valid),
        .iq_data(iq_data), .busy(busy), .done(done), .result_valid(result_valid),
        .class_id(class_id), .anomaly_flag(anomaly_flag), .cycle_count(cycle_count),
        .logits(logits));

    assign dbg_start        = start;
    assign dbg_iq_valid     = iq_valid;
    assign dbg_busy         = busy;
    assign dbg_done         = done;
    assign dbg_result_valid = result_valid;
    assign dbg_anomaly_flag = anomaly_flag;
    assign dbg_class_id     = {{(8-CW){1'b0}}, class_id};
    assign dbg_cycle_count  = cycle_count;

    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    // ── Write channel ───────────────────────────────────────────────
    // Accept AW and W together; the register write and BVALID follow in the
    // next cycle, i.e. strictly after both handshakes (AXI4 dependency rule).
    reg  [7:0]  wa;
    reg  [31:0] wd;
    wire wr_accept = s_axi_awvalid && s_axi_wvalid && !s_axi_awready && !s_axi_bvalid;
    always @(posedge clk) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 0; s_axi_wready <= 0; s_axi_bvalid <= 0;
            start <= 0; iq_valid <= 0; iq_data <= 0; done_sticky <= 0; wa <= 0; wd <= 0;
        end else begin
            start    <= 0;
            iq_valid <= 0;
            s_axi_awready <= wr_accept;
            s_axi_wready  <= wr_accept;
            if (wr_accept) begin
                wa <= s_axi_awaddr;
                wd <= s_axi_wdata;
            end
            if (s_axi_awready) begin                    // handshake completed last edge
                case (wa[7:2])
                    6'h00: if (wd[0]) begin start <= 1; done_sticky <= 0; end
                    6'h02: begin iq_valid <= 1; iq_data <= wd[15:0]; end
                    default: ;
                endcase
                s_axi_bvalid <= 1;
            end else if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 0;
            if (done) done_sticky <= 1;
        end
    end

    // ── Read channel (RVALID one cycle after the AR handshake) ──────
    reg  [7:0]  ra;
    integer c;
    reg [31:0] rd;
    always @(*) begin
        rd = 32'd0;
        case (ra[7:2])
            6'h01: rd = {30'd0, done_sticky, busy};
            6'h03: rd = class_id;
            6'h04: rd = cycle_count;
            6'h05: rd = {16'h524E, NUM_CLASSES[7:0], W[7:0]};
            default:
                for (c = 0; c < NUM_CLASSES; c = c + 1)
                    if (ra[7:2] == 6'h08 + c) rd = logits[32*c +: 32];
        endcase
    end
    always @(posedge clk) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 0; s_axi_rvalid <= 0; s_axi_rdata <= 0; ra <= 0;
        end else begin
            s_axi_arready <= s_axi_arvalid && !s_axi_arready && !s_axi_rvalid;
            if (s_axi_arvalid && !s_axi_arready && !s_axi_rvalid)
                ra <= s_axi_araddr;
            if (s_axi_arready) begin
                s_axi_rdata  <= rd;
                s_axi_rvalid <= 1;
            end else if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 0;
        end
    end
endmodule
