`timescale 1ns / 1ps
/*
 * radarnet_top.v — Top-level FSM (BRAM-friendly, address-based I/O)
 * All feature maps stored in sub-module distributed RAMs.
 * Single weight ROM shared via address mux with base offsets.
 * Skip-add + ReLU done element-by-element in the FSM.
 */
module radarnet_top #(
    parameter DATA_WIDTH  = 8,
    parameter NUM_SAMPLES = 128,
    parameter NUM_CLASSES = 4
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg         busy,
    input  wire        iq_valid,
    input  wire [15:0] iq_data,
    output wire [1:0]  class_id,
    output wire        anomaly_flag,
    output reg         result_valid,
    output reg  [2:0]  fsm_state,
    output reg  [31:0] cycle_count
);
    // FSM states
    localparam S_IDLE=0, S_LOAD=1, S_CONV_IN=2, S_RES_BLK1=3,
               S_RES_BLK2=4, S_GAP=5, S_FC_OUT=6, S_DONE=7;

    // ROM base offsets
    localparam L1_BASE      = 16'd0;
    localparam L2_BASE      = 16'd240;
    localparam RB1_C1_BASE  = 16'd1808;
    localparam RB1_C2_BASE  = 16'd4912;
    localparam RB2_C1_BASE  = 16'd8016;
    localparam RB2_C2_BASE  = 16'd14224;
    localparam RB2_PROJ_BASE= 16'd26576;
    localparam FC_BASE      = 16'd28688;

    reg [2:0]  state;
    reg [2:0]  sub_phase;
    reg        ib_clear;

    // Start signals
    reg l1_start, l2_start, gap_start, fc_start;
    reg rb1_c1_start, rb1_c2_start;
    reg rb2_c1_start, rb2_c2_start, rb2_proj_start;

    // ── Weight ROM ──────────────────────────────────────────────────
    reg  [15:0] wrom_addr;
    wire [DATA_WIDTH-1:0] wrom_data;
    weight_rom #(.DATA_WIDTH(DATA_WIDTH),.TOTAL_DEPTH(32768))
    u_weight_rom (.clk(clk),.addr(wrom_addr[14:0]),.data_out(wrom_data));

    // ── Input Buffer ────────────────────────────────────────────────
    wire ib_full;
    wire [15:0] l1_fin_addr;
    wire [DATA_WIDTH-1:0] ib_rd_data;
    input_buffer #(.NUM_SAMPLES(NUM_SAMPLES),.DATA_WIDTH(DATA_WIDTH))
    u_ib (.clk(clk),.rst_n(rst_n),
          .wr_en(iq_valid && (state==S_LOAD)),.clear(ib_clear),.din(iq_data),
          .full(ib_full),.rd_addr(l1_fin_addr),.rd_data(ib_rd_data));

    // ── L1: Conv1D(2→16, k=7, s=1, pad=3) + ReLU ──────────────────
    wire l1_done; wire [15:0] l1_rom_addr;
    wire [DATA_WIDTH-1:0] l1_fout_data;
    wire [15:0] l2_fin_addr;
    conv1d_engine #(.IN_CHANNELS(2),.OUT_CHANNELS(16),.IN_LENGTH(128),
        .KERNEL_SIZE(7),.STRIDE(1),.PAD(3),.BIAS_EN(1),.RELU_EN(1))
    u_l1 (.clk(clk),.rst_n(rst_n),.start(l1_start),.done(l1_done),
          .fin_addr(l1_fin_addr),.fin_data(ib_rd_data),
          .rom_addr(l1_rom_addr),.rom_data(wrom_data),
          .fout_rd_addr(l2_fin_addr),.fout_rd_data(l1_fout_data),
          .out_valid());

    // ── L2: Conv1D(16→32, k=3, s=2, pad=1) + ReLU ──────────────────
    wire l2_done; wire [15:0] l2_rom_addr;
    wire [DATA_WIDTH-1:0] l2_fout_data;
    // L2 output is read by both RB1_C1 (main path) and RB1 skip-add
    wire [15:0] rb1c1_fin_addr;
    wire [15:0] rb1_skip_addr;
    // Mux: during skip-add, top-level reads L2 output; otherwise RB1_C1 reads it
    reg         rb1_skip_reading;
    wire [15:0] l2_rd_addr = rb1_skip_reading ? rb1_skip_addr : rb1c1_fin_addr;

    conv1d_engine #(.IN_CHANNELS(16),.OUT_CHANNELS(32),.IN_LENGTH(128),
        .KERNEL_SIZE(3),.STRIDE(2),.PAD(1),.BIAS_EN(1),.RELU_EN(1))
    u_l2 (.clk(clk),.rst_n(rst_n),.start(l2_start),.done(l2_done),
          .fin_addr(l2_fin_addr),.fin_data(l1_fout_data),
          .rom_addr(l2_rom_addr),.rom_data(wrom_data),
          .fout_rd_addr(l2_rd_addr),.fout_rd_data(l2_fout_data),
          .out_valid());

    // ── RB1 Conv1: Conv1D(32→32, k=3, s=1, pad=1) + ReLU ───────────
    wire rb1c1_done; wire [15:0] rb1c1_rom_addr;
    wire [DATA_WIDTH-1:0] rb1c1_fout_data;
    wire [15:0] rb1c2_fin_addr;
    conv1d_engine #(.IN_CHANNELS(32),.OUT_CHANNELS(32),.IN_LENGTH(64),
        .KERNEL_SIZE(3),.STRIDE(1),.PAD(1),.BIAS_EN(1),.RELU_EN(1))
    u_rb1c1 (.clk(clk),.rst_n(rst_n),.start(rb1_c1_start),.done(rb1c1_done),
             .fin_addr(rb1c1_fin_addr),.fin_data(l2_fout_data),
             .rom_addr(rb1c1_rom_addr),.rom_data(wrom_data),
             .fout_rd_addr(rb1c2_fin_addr),.fout_rd_data(rb1c1_fout_data),
             .out_valid());

    // ── RB1 Conv2: Conv1D(32→32, k=3, s=1, pad=1) NO ReLU ──────────
    wire rb1c2_done; wire [15:0] rb1c2_rom_addr;
    wire [DATA_WIDTH-1:0] rb1c2_fout_data;
    // RB1 skip-add reads both rb1c2 output and l2 output
    wire [15:0] rb1_main_addr;
    reg  rb1_main_reading;
    wire [15:0] rb1c2_rd_addr = rb1_main_reading ? rb1_main_addr : rb2c1_fin_addr;
    conv1d_engine #(.IN_CHANNELS(32),.OUT_CHANNELS(32),.IN_LENGTH(64),
        .KERNEL_SIZE(3),.STRIDE(1),.PAD(1),.BIAS_EN(1),.RELU_EN(0))
    u_rb1c2 (.clk(clk),.rst_n(rst_n),.start(rb1_c2_start),.done(rb1c2_done),
             .fin_addr(rb1c2_fin_addr),.fin_data(rb1c1_fout_data),
             .rom_addr(rb1c2_rom_addr),.rom_data(wrom_data),
             .fout_rd_addr(rb1c2_rd_addr),.fout_rd_data(rb1c2_fout_data),
             .out_valid());

    // ── RB1 skip-add result stored in distributed RAM ───────────────
    (* ram_style = "distributed" *)
    reg [DATA_WIDTH-1:0] rb1_out_mem [0:32*64-1];
    reg        rb1_we;
    reg [15:0] rb1_wr_addr;
    reg [DATA_WIDTH-1:0] rb1_wr_data;
    always @(posedge clk) if (rb1_we) rb1_out_mem[rb1_wr_addr] <= rb1_wr_data;
    // Read port for downstream
    wire [15:0] rb2c1_fin_addr;
    wire [15:0] rb2proj_fin_addr;
    reg  rb2_proj_reading;
    wire [15:0] rb1_out_rd_addr = rb2_proj_reading ? rb2proj_fin_addr : rb2c1_fin_addr;
    wire [DATA_WIDTH-1:0] rb1_out_rd_data = rb1_out_mem[rb1_out_rd_addr];

    // ── RB2 Conv1: Conv1D(32→64, k=3, s=2, pad=1) + ReLU ───────────
    wire rb2c1_done; wire [15:0] rb2c1_rom_addr;
    wire [DATA_WIDTH-1:0] rb2c1_fout_data;
    wire [15:0] rb2c2_fin_addr;
    conv1d_engine #(.IN_CHANNELS(32),.OUT_CHANNELS(64),.IN_LENGTH(64),
        .KERNEL_SIZE(3),.STRIDE(2),.PAD(1),.BIAS_EN(1),.RELU_EN(1))
    u_rb2c1 (.clk(clk),.rst_n(rst_n),.start(rb2_c1_start),.done(rb2c1_done),
             .fin_addr(rb2c1_fin_addr),.fin_data(rb1_out_rd_data),
             .rom_addr(rb2c1_rom_addr),.rom_data(wrom_data),
             .fout_rd_addr(rb2c2_fin_addr),.fout_rd_data(rb2c1_fout_data),
             .out_valid());

    // ── RB2 Conv2: Conv1D(64→64, k=3, s=1, pad=1) NO ReLU ──────────
    wire rb2c2_done; wire [15:0] rb2c2_rom_addr;
    wire [DATA_WIDTH-1:0] rb2c2_fout_data;
    wire [15:0] rb2_main_addr;
    reg  rb2_main_reading;
    wire [15:0] rb2c2_rd_addr = rb2_main_reading ? rb2_main_addr : gap_fin_addr;
    conv1d_engine #(.IN_CHANNELS(64),.OUT_CHANNELS(64),.IN_LENGTH(32),
        .KERNEL_SIZE(3),.STRIDE(1),.PAD(1),.BIAS_EN(1),.RELU_EN(0))
    u_rb2c2 (.clk(clk),.rst_n(rst_n),.start(rb2_c2_start),.done(rb2c2_done),
             .fin_addr(rb2c2_fin_addr),.fin_data(rb2c1_fout_data),
             .rom_addr(rb2c2_rom_addr),.rom_data(wrom_data),
             .fout_rd_addr(rb2c2_rd_addr),.fout_rd_data(rb2c2_fout_data),
             .out_valid());

    // ── RB2 Projection: proj_conv(32→64, s=2) ──────────────────────
    wire rb2proj_done; wire [15:0] rb2proj_rom_addr;
    wire [DATA_WIDTH-1:0] rb2proj_fout_data;
    wire [15:0] rb2_skip_addr;
    proj_conv #(.IN_CHANNELS(32),.OUT_CHANNELS(64),.IN_LENGTH(64),.STRIDE(2))
    u_rb2proj (.clk(clk),.rst_n(rst_n),.start(rb2_proj_start),.done(rb2proj_done),
               .fin_addr(rb2proj_fin_addr),.fin_data(rb1_out_rd_data),
               .rom_addr(rb2proj_rom_addr),.rom_data(wrom_data),
               .fout_rd_addr(rb2_skip_addr),.fout_rd_data(rb2proj_fout_data),
               .out_valid());

    // ── RB2 skip-add result stored in distributed RAM ───────────────
    (* ram_style = "distributed" *)
    reg [DATA_WIDTH-1:0] rb2_out_mem [0:64*32-1];
    reg        rb2_we;
    reg [15:0] rb2_wr_addr;
    reg [DATA_WIDTH-1:0] rb2_wr_data;
    always @(posedge clk) if (rb2_we) rb2_out_mem[rb2_wr_addr] <= rb2_wr_data;
    wire [15:0] gap_fin_addr;
    wire [DATA_WIDTH-1:0] rb2_out_rd_data = rb2_out_mem[gap_fin_addr];

    // ── GAP ─────────────────────────────────────────────────────────
    wire gap_done;
    wire [64*DATA_WIDTH-1:0] gap_out;
    gap_unit #(.NUM_CHANNELS(64),.TIME_STEPS(32),.DATA_WIDTH(DATA_WIDTH),.SHIFT_BITS(5))
    u_gap (.clk(clk),.rst_n(rst_n),.start(gap_start),.done(gap_done),
           .fin_addr(gap_fin_addr),.fin_data(rb2_out_rd_data),
           .gap_out(gap_out),.out_valid());

    // ── FC: read GAP output via address ─────────────────────────────
    wire fc_done; wire [15:0] fc_rom_addr;
    wire [15:0] fc_fin_addr;
    wire [DATA_WIDTH-1:0] fc_fin_data = gap_out[fc_fin_addr*DATA_WIDTH +: DATA_WIDTH];
    wire [4*DATA_WIDTH-1:0] fc_out_bus;
    fc_layer #(.IN_FEATURES(64),.OUT_FEATURES(4),.DATA_WIDTH(DATA_WIDTH))
    u_fc (.clk(clk),.rst_n(rst_n),.start(fc_start),.done(fc_done),
          .fin_addr(fc_fin_addr),.fin_data(fc_fin_data),
          .rom_addr(fc_rom_addr),.rom_data(wrom_data),
          .fc_out(fc_out_bus),.out_valid());

    // ── Argmax ──────────────────────────────────────────────────────
    reg [4*DATA_WIDTH-1:0] feat_fc;
    argmax_out u_argmax (
        .logit_0($signed(feat_fc[0*DATA_WIDTH+:DATA_WIDTH])),
        .logit_1($signed(feat_fc[1*DATA_WIDTH+:DATA_WIDTH])),
        .logit_2($signed(feat_fc[2*DATA_WIDTH+:DATA_WIDTH])),
        .logit_3($signed(feat_fc[3*DATA_WIDTH+:DATA_WIDTH])),
        .valid_in(result_valid),.class_id(class_id),
        .anomaly_flag(anomaly_flag),.valid_out());

    // ── ROM Address Mux (combinational) ─────────────────────────────
    always @(*) begin
        case (state)
            S_CONV_IN:
                if (sub_phase <= 3'd1) wrom_addr = l1_rom_addr + L1_BASE;
                else                   wrom_addr = l2_rom_addr + L2_BASE;
            S_RES_BLK1:
                if (sub_phase <= 3'd2) wrom_addr = rb1c1_rom_addr + RB1_C1_BASE;
                else if (sub_phase <= 3'd4) wrom_addr = rb1c2_rom_addr + RB1_C2_BASE;
                else wrom_addr = 16'd0;
            S_RES_BLK2:
                case (sub_phase)
                    3'd0,3'd1,3'd2: wrom_addr = rb2c1_rom_addr + RB2_C1_BASE;
                    3'd3,3'd4: wrom_addr = rb2c2_rom_addr + RB2_C2_BASE;
                    default: wrom_addr = 16'd0;
                endcase
            S_FC_OUT: wrom_addr = fc_rom_addr + FC_BASE;
            default:  wrom_addr = 16'd0;
        endcase
    end

    // Note: RB2 proj runs in parallel with RB2 conv but they share the ROM.
    // To avoid ROM conflicts, proj runs BEFORE conv1 (sequentially).

    // ── Skip-add helpers ────────────────────────────────────────────
    reg [15:0] add_idx;
    // For RB1: we read rb1c2 output and l2 output sequentially
    reg [DATA_WIDTH-1:0] main_byte, skip_byte;
    assign rb1_main_addr = add_idx;
    assign rb1_skip_addr = add_idx;
    assign rb2_main_addr = add_idx;
    assign rb2_skip_addr = add_idx;

    wire signed [DATA_WIDTH:0] add_sum_rb1 =
        {main_byte[DATA_WIDTH-1],main_byte} + {skip_byte[DATA_WIDTH-1],skip_byte};
    wire [DATA_WIDTH-1:0] add_sat_rb1 =
        (add_sum_rb1 > $signed(9'd127)) ? 8'sd127 :
        (add_sum_rb1 < -$signed(9'd128)) ? -8'sd128 :
        add_sum_rb1[DATA_WIDTH-1:0];
    wire [DATA_WIDTH-1:0] add_relu_rb1 =
        add_sat_rb1[DATA_WIDTH-1] ? {DATA_WIDTH{1'b0}} : add_sat_rb1;

    // ── FSM ─────────────────────────────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=0; busy<=0; result_valid<=0;
            fsm_state<=0; cycle_count<=0;
            ib_clear<=0; sub_phase<=0; add_idx<=0;
            l1_start<=0; l2_start<=0;
            rb1_c1_start<=0; rb1_c2_start<=0;
            rb2_c1_start<=0; rb2_c2_start<=0; rb2_proj_start<=0;
            gap_start<=0; fc_start<=0;
            rb1_skip_reading<=0; rb1_main_reading<=0;
            rb2_proj_reading<=0; rb2_main_reading<=0;
            feat_fc<=0; main_byte<=0; skip_byte<=0;
            rb1_we<=0; rb1_wr_addr<=0; rb1_wr_data<=0;
            rb2_we<=0; rb2_wr_addr<=0; rb2_wr_data<=0;
        end else begin
            fsm_state <= state;
            // Deassert starts and write enables
            l1_start<=0; l2_start<=0;
            rb1_c1_start<=0; rb1_c2_start<=0;
            rb2_c1_start<=0; rb2_c2_start<=0; rb2_proj_start<=0;
            rb1_we<=0; rb2_we<=0;
            gap_start<=0; fc_start<=0;

            case (state)
                S_IDLE: begin
                    done<=0; result_valid<=0;
                    if (start || iq_valid) begin
                        state<=S_LOAD; busy<=1;
                        ib_clear<=1; cycle_count<=0;
                    end
                end

                S_LOAD: begin
                    ib_clear<=0;
                    cycle_count<=cycle_count+1;
                    if (ib_full) begin state<=S_CONV_IN; sub_phase<=0; end
                end

                // L1 then L2
                S_CONV_IN: begin
                    cycle_count<=cycle_count+1;
                    case (sub_phase)
                        0: begin l1_start<=1; sub_phase<=1; end
                        1: if (l1_done) sub_phase<=2;
                        2: begin l2_start<=1; sub_phase<=3; end
                        3: if (l2_done) begin state<=S_RES_BLK1; sub_phase<=0; end
                        default: ;
                    endcase
                end

                // RB1: conv1 → conv2 → skip-add+ReLU
                S_RES_BLK1: begin
                    cycle_count<=cycle_count+1;
                    rb1_skip_reading <= 0;
                    rb1_main_reading <= 0;
                    case (sub_phase)
                        0: begin rb1_c1_start<=1; sub_phase<=1; end
                        1: if (rb1c1_done) sub_phase<=2;
                        2: begin rb1_c2_start<=1; sub_phase<=3; end
                        3: if (rb1c2_done) begin
                               add_idx<=0; sub_phase<=4;
                               rb1_skip_reading<=1; rb1_main_reading<=1;
                           end
                        4: begin // Read main(rb1c2) and skip(l2) at add_idx
                               rb1_skip_reading<=1; rb1_main_reading<=1;
                               main_byte <= rb1c2_fout_data;
                               skip_byte <= l2_fout_data;
                               sub_phase<=5;
                           end
                        5: begin // Store result via write port
                               rb1_we <= 1'b1;
                               rb1_wr_addr <= add_idx;
                               rb1_wr_data <= add_relu_rb1;
                               if (add_idx < 32*64-1) begin
                                   add_idx<=add_idx+1; sub_phase<=4;
                               end else begin
                                   rb1_skip_reading<=0; rb1_main_reading<=0;
                                   state<=S_RES_BLK2; sub_phase<=0;
                               end
                           end
                        default: ;
                    endcase
                end

                // RB2: proj first, then conv1 → conv2 → skip-add+ReLU
                S_RES_BLK2: begin
                    cycle_count<=cycle_count+1;
                    rb2_proj_reading<=0; rb2_main_reading<=0;
                    case (sub_phase)
                        0: begin rb2_proj_start<=1; rb2_proj_reading<=1; sub_phase<=1; end
                        1: begin
                               rb2_proj_reading<=1;
                               if (rb2proj_done) begin
                                   rb2_proj_reading<=0;
                                   rb2_c1_start<=1; sub_phase<=2;
                               end
                           end
                        2: if (rb2c1_done) begin rb2_c2_start<=1; sub_phase<=3; end
                        3: if (rb2c2_done) begin
                               add_idx<=0; sub_phase<=4;
                               rb2_main_reading<=1;
                           end
                        4: begin // Read main(rb2c2) and skip(proj)
                               rb2_main_reading<=1;
                               main_byte <= rb2c2_fout_data;
                               skip_byte <= rb2proj_fout_data;
                               sub_phase<=5;
                           end
                        5: begin
                               rb2_we <= 1'b1;
                               rb2_wr_addr <= add_idx;
                               rb2_wr_data <= add_relu_rb1;
                               if (add_idx < 64*32-1) begin
                                   add_idx<=add_idx+1; sub_phase<=4;
                               end else begin
                                   rb2_main_reading<=0;
                                   state<=S_GAP; sub_phase<=0;
                               end
                           end
                        default: ;
                    endcase
                end

                S_GAP: begin
                    cycle_count<=cycle_count+1;
                    case (sub_phase)
                        0: begin gap_start<=1; sub_phase<=1; end
                        1: if (gap_done) begin state<=S_FC_OUT; sub_phase<=0; end
                        default: ;
                    endcase
                end

                S_FC_OUT: begin
                    cycle_count<=cycle_count+1;
                    case (sub_phase)
                        0: begin fc_start<=1; sub_phase<=1; end
                        1: if (fc_done) begin
                               feat_fc<=fc_out_bus;
                               state<=S_DONE;
                           end
                        default: ;
                    endcase
                end

                S_DONE: begin
                    done<=1; busy<=0; result_valid<=1;
                    state<=S_IDLE; sub_phase<=0;
                end
            endcase
        end
    end
endmodule
