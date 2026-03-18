`timescale 1ns / 1ps
module proj_conv #(
    parameter IN_CHANNELS  = 32,
    parameter OUT_CHANNELS = 64,
    parameter IN_LENGTH    = 64,
    parameter STRIDE       = 2,
    parameter DATA_WIDTH   = 8
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [15:0] fin_addr,
    input  wire [DATA_WIDTH-1:0] fin_data,
    output reg  [15:0] rom_addr,
    input  wire [DATA_WIDTH-1:0] rom_data,
    input  wire [15:0] fout_rd_addr,
    output wire [DATA_WIDTH-1:0] fout_rd_data,
    output reg         out_valid
);
    localparam OUT_LENGTH = IN_LENGTH / STRIDE;

    // ── Output RAM (separate write block) ───────────────────────────
    (* ram_style = "distributed" *)
    reg [DATA_WIDTH-1:0] fout_mem [0:OUT_CHANNELS*OUT_LENGTH-1];
    assign fout_rd_data = fout_mem[fout_rd_addr];

    reg        fout_we;
    reg [15:0] fout_wr_addr;
    reg [DATA_WIDTH-1:0] fout_wr_data;
    always @(posedge clk)
        if (fout_we) fout_mem[fout_wr_addr] <= fout_wr_data;

    // ── FSM ─────────────────────────────────────────────────────────
    localparam S_IDLE=0, S_BADDR=1, S_BREAD=2, S_ADDR=3, S_MAC=4, S_STORE=5, S_DONE=6;
    reg [2:0] state;
    reg [$clog2(OUT_CHANNELS+1)-1:0] oc;
    reg [$clog2(OUT_LENGTH+1)-1:0]   ot;
    reg [$clog2(IN_CHANNELS+1)-1:0]  ic;
    reg signed [23:0] acc;
    reg signed [7:0] bias_reg;

    wire signed [DATA_WIDTH-1:0] w_val = $signed(rom_data);
    wire signed [DATA_WIDTH-1:0] x_val = $signed(fin_data);
    wire signed [15:0] product = w_val * x_val;

    function [DATA_WIDTH-1:0] sat8; input signed [23:0] v;
        if (v>127) sat8=8'sd127; else if (v< -128) sat8=-8'sd128; else sat8=v[DATA_WIDTH-1:0];
    endfunction

    wire signed [23:0] final_val = acc + {{16{bias_reg[7]}},bias_reg};
    wire [DATA_WIDTH-1:0] store_val = sat8(final_val);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=0; out_valid<=0;
            oc<=0; ot<=0; ic<=0; acc<=0; bias_reg<=0;
            rom_addr<=0; fin_addr<=0;
            fout_we<=0; fout_wr_addr<=0; fout_wr_data<=0;
        end else begin
            fout_we <= 1'b0;
            case (state)
                S_IDLE: begin done<=0; out_valid<=0;
                    if (start) begin oc<=0;ot<=0;ic<=0;acc<=0; rom_addr<=0; state<=S_BADDR; end
                end
                S_BADDR: state<=S_BREAD;
                S_BREAD: begin bias_reg<=$signed(rom_data); state<=S_ADDR; end
                S_ADDR: begin
                    fin_addr <= ic * IN_LENGTH + ot * STRIDE;
                    rom_addr <= OUT_CHANNELS + oc * IN_CHANNELS + ic;
                    state <= S_MAC;
                end
                S_MAC: begin
                    acc <= acc + {{8{product[15]}}, product};
                    if (ic < IN_CHANNELS-1) begin ic<=ic+1; state<=S_ADDR; end
                    else begin ic<=0; state<=S_STORE; end
                end
                S_STORE: begin
                    fout_we <= 1'b1;
                    fout_wr_addr <= oc*OUT_LENGTH+ot;
                    fout_wr_data <= store_val;
                    acc <= 0;
                    if (ot < OUT_LENGTH-1) begin ot<=ot+1; state<=S_ADDR; end
                    else begin ot<=0;
                        if (oc < OUT_CHANNELS-1) begin oc<=oc+1; rom_addr<=oc+16'd1; state<=S_BADDR; end
                        else state<=S_DONE;
                    end
                end
                S_DONE: begin done<=1; out_valid<=1; state<=S_IDLE; end
            endcase
        end
    end
endmodule