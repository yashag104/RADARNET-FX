`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 16.03.2026 18:04:31
// Design Name: 
// Module Name: top_test_wrapper
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module top_test_wrapper (
    input clk
);

    // VIO to FPGA signals
    wire rst_n, start, iq_valid;
    wire [15:0] iq_data;

    // FPGA to VIO signals
    wire done, busy, result_valid, anomaly_flag;
    wire [1:0] class_id;

    // Internal probes for ILA
    wire [2:0] state_probe;
    wire [31:0] cycle_count_probe;

    // Instantiate VIO
    vio_0 u_vio (
        .clk(clk),
        .probe_in0(done),
        .probe_in1(busy),
        .probe_in2(result_valid),
        .probe_in3(class_id),
        .probe_in4(anomaly_flag),
        .probe_out0(rst_n),
        .probe_out1(start),
        .probe_out2(iq_valid),
        .probe_out3(iq_data)
    );

    // Instantiate RadarNet Top Level
    radarnet_top u_radarnet (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .iq_valid(iq_valid),
        .iq_data(iq_data),
        .class_id(class_id),
        .anomaly_flag(anomaly_flag),
        .result_valid(result_valid),
        .fsm_state(state_probe),
        .cycle_count(cycle_count_probe)
    );

    // Instantiate ILA
    ila_0 u_ila (
        .clk(clk),
        .probe0(state_probe),
        .probe1(cycle_count_probe),
        .probe2(15'b0), // Wire up wrom signals if promoted to top level
        .probe3(8'b0)
    );

endmodule
