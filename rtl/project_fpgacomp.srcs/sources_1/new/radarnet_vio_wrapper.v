`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 16.03.2026 16:53:41
// Design Name: 
// Module Name: radarnet_vio_wrapper
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

module radarnet_vio_wrapper (
    input wire clk // Map this to your board's physical clock
);

    // ── VIO output wires (directly driven by VIO probe_out) ─────────
    wire        vio_reset;       // probe_out0:  1 bit
    wire        vio_start;       // probe_out1:  1 bit
    wire        vio_iq_valid;    // probe_out2:  1 bit
    wire [15:0] vio_iq_data;     // probe_out3: 16 bits  {I[7:0], Q[7:0]}

    // ── radarnet_top output wires ───────────────────────────────────
    wire        done;
    wire        busy;
    wire [1:0]  class_id;
    wire        anomaly_flag;
    wire        result_valid;
    wire [2:0]  fsm_state;
    wire [31:0] cycle_count;

    // ── VIO Instance ────────────────────────────────────────────────
    //  probe_in  widths:  1, 1, 1, 3, 1
    //  probe_out widths:  1, 1, 1, 16
    vio_0 your_vio (
        .clk(clk),
        .probe_in0(done),            //  1 bit
        .probe_in1(busy),            //  1 bit
        .probe_in2(anomaly_flag),    //  1 bit
        .probe_in3(fsm_state),       //  3 bits
        .probe_in4(result_valid),    //  1 bit
        .probe_out0(vio_reset),      //  1 bit
        .probe_out1(vio_start),      //  1 bit
        .probe_out2(vio_iq_valid),   //  1 bit
        .probe_out3(vio_iq_data)     // 16 bits
    );

    // ── ILA Instance ────────────────────────────────────────────────
    //  probe widths:  3, 32, 15, 8
    ila_0 your_ila (
        .clk(clk),
        .probe0(fsm_state),          //  3 bits  - FSM state
        .probe1(cycle_count),        // 32 bits  - latency counter
        .probe2(vio_iq_data[14:0]),  // 15 bits  - input data snapshot
        .probe3({2'b0, class_id, anomaly_flag, result_valid, busy, done})
                                     //  8 bits  - status bundle
    );

    // ── RadarNet Top Instance ───────────────────────────────────────
    radarnet_top radar_inst (
        .clk(clk),
        .rst_n(~vio_reset),          // VIO gives active-high; module uses active-low
        .start(vio_start),
        .done(done),
        .busy(busy),
        .iq_valid(vio_iq_valid),     // Controlled from VIO probe_out2
        .iq_data(vio_iq_data),       // Full 16-bit I/Q from VIO probe_out3
        .class_id(class_id),
        .anomaly_flag(anomaly_flag),
        .result_valid(result_valid),
        .fsm_state(fsm_state),
        .cycle_count(cycle_count)
    );
endmodule