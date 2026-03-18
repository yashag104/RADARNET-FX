`timescale 1ns / 1ps
module conv1d_engine #(
    parameter IN_CHANNELS  = 2,
    parameter OUT_CHANNELS = 16,
    parameter IN_LENGTH    = 128,
    parameter KERNEL_SIZE  = 7,
    parameter STRIDE       = 1,
    parameter PAD          = 3,
    parameter BIAS_EN      = 1,
    parameter RELU_EN      = 0
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [15:0] fin_addr,
    input  wire [ 7:0] fin_data,
    output reg  [15:0] rom_addr,
    input  wire [ 7:0] rom_data,
    input  wire [15:0] fout_rd_addr,
    output wire [ 7:0] fout_rd_data,
    output reg         out_valid
);
    localparam OUT_LENGTH    = (IN_LENGTH + 2*PAD - KERNEL_SIZE) / STRIDE + 1;

    // ── Output RAM (separate write block for clean inference) ────────
    (* ram_style = "distributed" *)
    reg [7:0] fout_mem [0:OUT_CHANNELS*OUT_LENGTH-1];
    assign fout_rd_data = fout_mem[fout_rd_addr]; // async read

    reg        fout_we;
    reg [15:0] fout_wr_addr;
    reg [ 7:0] fout_wr_data;

    always @(posedge clk)
        if (fout_we) fout_mem[fout_wr_addr] <= fout_wr_data;

    // ── FSM ─────────────────────────────────────────────────────────
    localparam S_IDLE=0, S_BADDR=1, S_BREAD=2, S_ADDR=3, S_MAC=4, S_STORE=5, S_DONE=6;
    reg [2:0] state;
    reg [$clog2(OUT_CHANNELS+1)-1:0] oc;
    reg [$clog2(OUT_LENGTH+1)-1:0]   op;
    reg [$clog2(IN_CHANNELS+1)-1:0]  ic;
    reg [$clog2(KERNEL_SIZE+1)-1:0]  kk;
    reg signed [23:0] acc;
    reg signed [ 7:0] bias_reg;
    reg                ib_reg;

    wire signed [15:0] in_pos = $signed({1'b0,op}) * STRIDE + $signed({1'b0,kk}) - PAD;
    wire in_bounds = (in_pos >= 0) && (in_pos < IN_LENGTH);
    wire signed [7:0] w_val = $signed(rom_data);
    wire signed [7:0] x_val = ib_reg ? $signed(fin_data) : 8'sd0;
    wire signed [15:0] product = w_val * x_val;

    function [7:0] sat8; input signed [23:0] v;
        if (v > 127) sat8 = 8'sd127;
        else if (v < -128) sat8 = -8'sd128;
        else sat8 = v[7:0];
    endfunction

    // Compute store value
    wire signed [23:0] final_val = BIAS_EN ? (acc + {{16{bias_reg[7]}},bias_reg}) : acc;
    wire [7:0] sat_val = sat8(final_val);
    wire [7:0] store_val = (RELU_EN && sat_val[7]) ? 8'd0 : sat_val;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=0; out_valid<=0;
            oc<=0; op<=0; ic<=0; kk<=0;
            acc<=0; bias_reg<=0; ib_reg<=0;
            rom_addr<=0; fin_addr<=0;
            fout_we<=0; fout_wr_addr<=0; fout_wr_data<=0;
        end else begin
            fout_we <= 1'b0; // default: no write
            case (state)
                S_IDLE: begin
                    done<=0; out_valid<=0;
                    if (start) begin
                        oc<=0; op<=0; ic<=0; kk<=0; acc<=0;
                        if (BIAS_EN) begin rom_addr<=16'd0; state<=S_BADDR; end
                        else state<=S_ADDR;
                    end
                end
                S_BADDR: state<=S_BREAD;
                S_BREAD: begin bias_reg<=$signed(rom_data); state<=S_ADDR; end
                S_ADDR: begin
                    fin_addr <= in_bounds ? (ic*IN_LENGTH + in_pos[15:0]) : 16'd0;
                    rom_addr <= OUT_CHANNELS + oc*IN_CHANNELS*KERNEL_SIZE + ic*KERNEL_SIZE + kk;
                    ib_reg <= in_bounds;
                    state <= S_MAC;
                end
                S_MAC: begin
                    acc <= acc + {{8{product[15]}}, product};
                    if (kk < KERNEL_SIZE-1) begin kk<=kk+1; state<=S_ADDR; end
                    else if (ic < IN_CHANNELS-1) begin kk<=0; ic<=ic+1; state<=S_ADDR; end
                    else begin kk<=0; ic<=0; state<=S_STORE; end
                end
                S_STORE: begin
                    fout_we <= 1'b1;
                    fout_wr_addr <= oc * OUT_LENGTH + op;
                    fout_wr_data <= store_val;
                    acc <= 24'd0;
                    if (op < OUT_LENGTH-1) begin op<=op+1; state<=S_ADDR; end
                    else begin
                        op<=0;
                        if (oc < OUT_CHANNELS-1) begin
                            oc<=oc+1;
                            if (BIAS_EN) begin rom_addr<=oc+16'd1; state<=S_BADDR; end
                            else state<=S_ADDR;
                        end else state<=S_DONE;
                    end
                end
                S_DONE: begin done<=1; out_valid<=1; state<=S_IDLE; end
            endcase
        end
    end
endmodule