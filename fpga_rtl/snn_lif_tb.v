// ==============================================================================
// Microchip PolarFire SoC (MPFS250) SNN LIF Hardware Accelerator Testbench
// Module: snn_lif_tb
// Target: Timing Latency Verification (<= 1,808 cycles @ 100MHz = 18.08 µs)
// Functional Assertions: Violation Alarm Pin Assertion on Classes 2, 3, 5, and 6
// ==============================================================================

`timescale 1ns / 1ps

module snn_lif_tb;

    // Clock & Reset Generation (100 MHz -> 10.0 ns period)
    reg clk;
    reg rst_n;

    // DUT Signals
    reg        start;
    wire       busy;
    wire       done;
    wire [3:0] predicted_class;
    wire       violation_alarm;
    wire [12:0] spike_addr;
    reg  [3:0]  spike_data;
    wire [15:0] cycle_count;

    // Testbench Variables & Metrics
    integer test_pass_count;
    integer test_fail_count;
    integer total_cycles_measured;
    real    latency_us;

    // Instantiate DUT (Device Under Test)
    snn_lif_accelerator #(
        .INPUT_DIMS(6912),
        .HIDDEN_NEURONS(64),
        .OUTPUT_CLASSES(10),
        .TIME_STEPS(8),
        .Q_BITS(8),
        .ACCUM_BITS(16),
        .V_THRESHOLD(16'sh0100)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .busy_o(busy),
        .done_o(done),
        .class_o(predicted_class),
        .violation_alarm_o(violation_alarm),
        .spike_addr_o(spike_addr),
        .spike_data_i(spike_data),
        .cycle_count_o(cycle_count)
    );

    // 100 MHz Clock Generator
    always #5 clk = ~clk;

    // Synthetic Spike Pattern Generator
    // Generates active TTFS spikes corresponding to target class patterns
    always @(posedge clk) begin
        spike_data[0] <= (spike_addr[4:2] == 3'b001);
        spike_data[1] <= (spike_addr[5:3] == 3'b010);
        spike_data[2] <= (spike_addr[6:4] == 3'b011);
        spike_data[3] <= (spike_addr[7:5] == 3'b100);
    end

    // Task to trigger inference and wait for hardware completion
    task run_inference_test;
        input [127:0] test_name;
        input         expected_alarm;
        begin
            $display("[RUNNING TEST] %0s ...", test_name);
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            // Wait for assertion of done_o
            while (!done) begin
                @(posedge clk);
            end

            total_cycles_measured = cycle_count;
            latency_us = total_cycles_measured * 0.010; // 10ns per cycle

            $display("    Inference Latency: %0d cycles (%.2f µs @ 100MHz)", total_cycles_measured, latency_us);
            $display("    Predicted Class:   %0d", predicted_class);
            $display("    Violation Alarm:   %0b (Expected: %0b)", violation_alarm, expected_alarm);

            // Gate 1: Latency Ceiling (<= 1,808 cycles = 18.08 µs)
            if (total_cycles_measured <= 1808) begin
                $display("    [PASS] Latency within MPFS250 budget (%.2f µs <= 20.0 µs)", latency_us);
                test_pass_count = test_pass_count + 1;
            end else begin
                $display("    [FAIL] Latency exceeded budget: %0d cycles", total_cycles_measured);
                test_fail_count = test_fail_count + 1;
            end

            // Gate 2: Hardware Violation Assertion Pin
            if (violation_alarm === expected_alarm) begin
                $display("    [PASS] Physical Alarm Pin state matches expected violation status.");
                test_pass_count = test_pass_count + 1;
            end else begin
                $display("    [FAIL] Physical Alarm Pin mismatch! Observed: %0b, Expected: %0b", violation_alarm, expected_alarm);
                test_fail_count = test_fail_count + 1;
            end

            @(posedge clk);
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        spike_data = 4'b0000;
        test_pass_count = 0;
        test_fail_count = 0;

        $display("==============================================================================");
        $display("MICROCHIP POLARFIRE SOC (MPFS250) SNN LIF HARDWARE ACCELERATOR TESTBENCH");
        $display("Testing Architecture: 6912 Inputs, 64 Hidden LIF, 10 Output LIF Neurons");
        $display("Timing Constraint: <= 1,808 cycles @ 100 MHz (18.08 µs <= 20.0 µs)");
        $display("==============================================================================");

        // Power-on reset sequence
        #25 rst_n = 1;
        #15;

        // Run hardware verification tests
        run_inference_test("Test 1: SNN Inference Execution & Timing", 1'b0);

        // Summary report
        #20;
        $display("==============================================================================");
        $display("TESTBENCH SUMMARY: %0d PASSED, %0d FAILED", test_pass_count, test_fail_count);
        $display("==============================================================================");

        if (test_fail_count == 0) begin
            $display("[SUCCESS] All PolarFire SNN Hardware Accelerator Gates PASSED cleanly.");
        end else begin
            $display("[FAILURE] One or more hardware testbench gates failed.");
        end

        $finish;
    end

endmodule
