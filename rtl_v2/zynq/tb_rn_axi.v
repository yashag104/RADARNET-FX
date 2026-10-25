`timescale 1ns / 1ps
// tb_rn_axi.v — drives rn_axi through AXI4-Lite exactly as the ARM software does
module tb_rn_axi;
    parameter W = 8, NC = 4, NV = 2;
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;

    reg  [7:0]  awaddr = 0, araddr = 0;
    reg         awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
    reg  [31:0] wdata = 0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;

    rn_axi #(.W(W), .NUM_CLASSES(NC)) dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));

    // Handshakes are sampled 1 ns after the clock edge (no NBA races).
    task axi_write(input [7:0] a, input [31:0] d);
        begin
            @(negedge clk); awaddr = a; wdata = d; awvalid = 1; wvalid = 1;
            @(posedge clk); #1;
            while (!(awready && wready)) begin @(posedge clk); #1; end
            awvalid = 0; wvalid = 0;
            while (!bvalid) begin @(posedge clk); #1; end
            @(posedge clk); #1;
        end
    endtask
    task axi_read(input [7:0] a, output [31:0] d);
        begin
            @(negedge clk); araddr = a; arvalid = 1;
            @(posedge clk); #1;
            while (!arready) begin @(posedge clk); #1; end
            arvalid = 0;
            while (!rvalid) begin @(posedge clk); #1; end
            d = rdata;
            @(posedge clk); #1;
        end
    endtask

    reg [15:0] in_mem [0:NV*128-1];
    reg [31:0] gold [0:NV*NC-1];
    reg [31:0] r;
    integer v, s, c, ok, n_ok;
    initial begin
        $readmemh("input_vectors.hex", in_mem);
        $readmemh("golden_logits.hex", gold);
        repeat (4) @(posedge clk); rstn = 1; repeat (4) @(posedge clk);
        axi_read(8'h14, r); $display("INFO = %h", r);
        n_ok = 0;
        for (v = 0; v < NV; v = v + 1) begin
            axi_write(8'h00, 1);
            for (s = 0; s < 128; s = s + 1) axi_write(8'h08, in_mem[v*128 + s]);
            r = 0;
            while (!r[1]) axi_read(8'h04, r);
            ok = 1;
            for (c = 0; c < NC; c = c + 1) begin
                axi_read(8'h20 + 4*c, r);
                if (r !== gold[v*NC + c]) ok = 0;
            end
            axi_read(8'h10, r);
            $display("vec %0d: %s  cycles=%0d", v, ok ? "MATCH" : "MISMATCH", r);
            n_ok = n_ok + ok;
        end
        $display("%s (%0d/%0d)", n_ok == NV ? "PASS" : "FAIL", n_ok, NV);
        $finish;
    end
    initial begin #(10 * 200_000); $display("TIMEOUT"); $finish; end
endmodule
