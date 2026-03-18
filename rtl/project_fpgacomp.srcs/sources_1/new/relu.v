`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 16.03.2026 16:44:08
// Design Name: 
// Module Name: relu
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

/*
 * relu.v - ReLU Activation for Signed INT8
 * ==========================================
 * Combinational ReLU: output = (input >= 0) ? input : 0
 * For signed 8-bit: check MSB (sign bit).
 *
 * Ports:
 *   data_in  [7:0]  - signed INT8 input
 *   data_out [7:0]  - ReLU output (unsigned, clamped to 0)
 */

module relu (
    input  wire signed [7:0] data_in,
    output wire signed [7:0] data_out
);

    // If MSB = 1 (negative), output 0; otherwise pass through
    assign data_out = (data_in[7]) ? 8'sd0 : data_in;

endmodule
