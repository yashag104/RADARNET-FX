`timescale 1ns / 1ps
/*
 * tb_rn_top.v — Bit-exact verification of rn_top against int_quant.py
 * ====================================================================
 * For every test vector the four INT32 logits and the class must equal the
 * golden integer model exactly. Latency is measured by the DUT's own cycle
 * counter (start → result_valid, including the 128-sample load).
 *
 * Files (from `int_quant.py export --tb-inputs ...`), in the run directory:
 *   input_vectors.hex, golden_logits.hex, golden_class.hex, true_labels.hex,
 *   weights.mem, params.mem, layer_table.vh (compile-time include)
 */
module tb_rn_top;
    parameter W           = 8;
    parameter NC          = 4;      // number of classes
    parameter NUM_VECTORS = 50;
    parameter CLK_PERIOD  = 10;

    reg clk = 0, rst_n = 0, start = 0, iq_valid = 0;
    reg [15:0] iq_data = 0;
    wire busy, done, result_valid, anomaly_flag;
    wire [((NC > 2) ? $clog2(NC) : 1)-1:0] class_id;
    wire [31:0] cycle_count;
    wire [32*NC-1:0] logits;

    always #(CLK_PERIOD / 2) clk = ~clk;

    rn_top #(.W(W), .NUM_CLASSES(NC)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .iq_valid(iq_valid), .iq_data(iq_data),
        .busy(busy), .done(done), .result_valid(result_valid), .class_id(class_id),
        .anomaly_flag(anomaly_flag), .cycle_count(cycle_count), .logits(logits));

    reg [15:0]  in_mem   [0:NUM_VECTORS*128-1];
    reg [32*NC-1:0] gold_lg [0:NUM_VECTORS-1];
    reg [7:0]   gold_cls [0:NUM_VECTORS-1];
    reg [7:0]   true_cls [0:NUM_VECTORS-1];
    reg [31:0]  lg_words [0:NUM_VECTORS*NC-1];

    integer v, s, j, n_exact, n_cls, n_true;
    reg ok;

    // ── Waveform helpers: add these to the wave window ───────────────
    //   wv_vector       index of the vector being processed
    //   wv_expected     class the golden integer model predicts for it
    //   wv_true_label   dataset label of that vector
    //   wv_hw_correct   1 at result_valid if the hardware class == wv_expected
    //   wv_label_match  1 at result_valid if the hardware class == wv_true_label
    //   n_cls / n_true  running counts of the two (n_exact: all logits identical)
    wire [31:0] wv_vector      = v;
    wire [7:0]  wv_expected    = gold_cls[v];
    wire [7:0]  wv_true_label  = true_cls[v];
    wire        wv_hw_correct  = result_valid && (class_id == wv_expected);
    wire        wv_label_match = result_valid && (class_id == wv_true_label);
    initial begin
`ifdef RN_VIVADO_SIM
        // Vivado copies .mem files into the xsim run directory
        $readmemh("input_vectors.mem", in_mem);
        $readmemh("golden_logits.mem", lg_words);
        $readmemh("golden_class.mem",  gold_cls);
        $readmemh("true_labels.mem",   true_cls);
`else
        $readmemh("input_vectors.hex", in_mem);
        $readmemh("golden_logits.hex", lg_words);
        $readmemh("golden_class.hex",  gold_cls);
        $readmemh("true_labels.hex",   true_cls);
`endif
        for (v = 0; v < NUM_VECTORS; v = v + 1)
            for (j = 0; j < NC; j = j + 1)
                gold_lg[v][32*j +: 32] = lg_words[NC*v + j];

        n_exact = 0; n_cls = 0; n_true = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        for (v = 0; v < NUM_VECTORS; v = v + 1) begin
            @(negedge clk) start = 1;
            @(negedge clk) start = 0;
            for (s = 0; s < 128; s = s + 1) begin
                iq_valid = 1; iq_data = in_mem[v*128 + s];
                @(negedge clk);
            end
            iq_valid = 0;
            @(posedge result_valid);
            @(negedge clk);
            ok = (logits === gold_lg[v]);
            n_exact = n_exact + ok;
            n_cls   = n_cls + (class_id == gold_cls[v]);
            n_true  = n_true + (class_id == true_cls[v]);
            if (!ok || v < 3) begin
                $display("vec %0d: %s cls=%0d gold=%0d cycles=%0d", v, ok ? "MATCH" : "MISMATCH",
                         class_id, gold_cls[v], cycle_count);
                for (j = 0; j < NC; j = j + 1)
                    $display("    logit%0d  rtl=%0d  gold=%0d", j,
                             $signed(logits[32*j +: 32]), $signed(gold_lg[v][32*j +: 32]));
            end
        end
        $display("==================================================");
        $display(" W=%0d  vectors=%0d", W, NUM_VECTORS);
        $display(" bit-exact logits : %0d / %0d", n_exact, NUM_VECTORS);
        $display(" class == golden  : %0d / %0d", n_cls, NUM_VECTORS);
        $display(" class == true    : %0d / %0d", n_true, NUM_VECTORS);
        $display(" latency (cycles) : %0d  (start -> result_valid, incl. 128-cycle load)",
                 cycle_count);
        $display("==================================================");
        if (n_exact == NUM_VECTORS) $display("PASS"); else $display("FAIL");
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 400_000 * NUM_VECTORS);
        $display("TIMEOUT"); $finish;
    end
endmodule
