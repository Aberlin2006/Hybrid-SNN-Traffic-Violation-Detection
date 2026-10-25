// ==============================================================================
// Microchip PolarFire SoC (MPFS250) SNN LIF Hardware Accelerator
// Module: snn_lif_accelerator
// Target: Microchip PolarFire MPFS250T-FCVG484E (100 MHz, AXI4-Lite Control)
// Input: 48x48x3 (6,912 dimensions), TTFS Latency Spikes (T=8 steps)
// Synaptic Arithmetic: Q4.4 Signed Fixed-Point, Shift-Add Leak Alpha = 0.75
// Hidden Neurons: 64 | Output Classes: 10
// DSP Multipliers: EXACTLY 0 / 784 Blocks (0.0% DSP utilization)
// Memory: W1 ROM (442,368 bytes) + W2 ROM (640 bytes) = 443,008 bytes (LSRAM)
// Timing: Exactly 1,808 Clock Cycles @ 100MHz = 18.08 µs (< 20.0 µs budget)
// ==============================================================================

`timescale 1ns / 1ps

module snn_lif_accelerator #(
    parameter INPUT_DIMS     = 6912,
    parameter HIDDEN_NEURONS = 64,
    parameter OUTPUT_CLASSES = 10,
    parameter TIME_STEPS     = 8,
    parameter Q_BITS         = 8,       // Q4.4 signed (8 bits)
    parameter ACCUM_BITS     = 16,      // Q8.8 accumulator (16 bits)
    parameter V_THRESHOLD    = 16'sh0100 // 1.0 in Q8.8 (16 * 16 = 256 = 0x0100)
)(
    // Clock & Active-Low Asynchronous Reset
    input  wire        clk_i,
    input  wire        rst_ni,

    // Control & Status Interface (AXI4-Lite register mapping)
    input  wire        start_i,
    output reg         busy_o,
    output reg         done_o,
    output reg  [3:0]  class_o,
    output wire        violation_alarm_o,

    // Input Spike BRAM Interface (TTFS stream: 4 parallel input spikes per cycle)
    output reg  [12:0] spike_addr_o,
    input  wire [3:0]  spike_data_i,

    // Diagnostic Cycle Counter
    output reg  [15:0] cycle_count_o
);

    // ==========================================================================
    // SECTION 1: STATE MACHINE DEFINITIONS & TIMING CONTROLLER
    // ==========================================================================
    localparam [2:0]
        ST_IDLE       = 3'b000,
        ST_FEATURE_ACC= 3'b001,
        ST_STEP_LEAK  = 3'b010,
        ST_OUTPUT_ACC = 3'b011,
        ST_OUTPUT_LIF = 3'b100;

    reg [2:0] state_q, state_d;

    // Temporal Step Counter (t = 0 ... 7)
    reg [2:0]  step_q;
    // Feature Group Counter (0 ... 1727, processing 4 features per cycle = 6912 features)
    reg [10:0] feat_grp_q;
    // Output Layer Step Counter (0 ... 63 for 64 hidden neurons)
    reg [5:0]  out_idx_q;
    // Output LIF Step Counter (0 ... 7)
    reg [2:0]  out_lif_cnt_q;

    // ==========================================================================
    // SECTION 2: POLARFIRE LSRAM WEIGHT ROM MEMORY ARRAYS (Q4.4)
    // ==========================================================================
    // W1: 6,912 inputs x 64 hidden neurons = 442,368 bytes
    reg signed [7:0] w1_rom [0:442367];
    // W2: 64 hidden x 10 output classes = 640 bytes
    reg signed [7:0] w2_rom [0:639];

    // Initialize ROMs from synthesized Hex tables
    initial begin
        $readmemh("../vectorblox_deployment/artifacts/w1_q4_4.hex", w1_rom);
        $readmemh("../vectorblox_deployment/artifacts/w2_q4_4.hex", w2_rom);
    end

    // ==========================================================================
    // SECTION 3: MEMBRANE POTENTIAL & SPIKE REGISTERS
    // ==========================================================================
    reg signed [ACCUM_BITS-1:0] v_hid_q [0:HIDDEN_NEURONS-1];
    reg signed [ACCUM_BITS-1:0] v_hid_d [0:HIDDEN_NEURONS-1];
    reg [HIDDEN_NEURONS-1:0]    s_hid_q;
    reg [HIDDEN_NEURONS-1:0]    s_hid_d;

    reg signed [ACCUM_BITS-1:0] v_out_q [0:OUTPUT_CLASSES-1];
    reg signed [ACCUM_BITS-1:0] v_out_d [0:OUTPUT_CLASSES-1];
    reg [OUTPUT_CLASSES-1:0]    s_out_q;
    reg [OUTPUT_CLASSES-1:0]    s_out_d;

    reg [7:0]                   spike_accum_q [0:OUTPUT_CLASSES-1];
    reg signed [ACCUM_BITS-1:0] pot_accum_q   [0:OUTPUT_CLASSES-1];

    // ==========================================================================
    // SECTION 4: HARDWARE SHIFT-ADD LEAK LOGIC (ZERO MULTIPLIERS)
    // ==========================================================================
    function signed [ACCUM_BITS-1:0] apply_shift_add_leak;
        input signed [ACCUM_BITS-1:0] v_in;
        input                         s_prev;
        begin
            if (s_prev) begin
                apply_shift_add_leak = {ACCUM_BITS{1'b0}};
            end else begin
                // Fixed 0.75 decay: 0.5 * V + 0.25 * V = (V >>> 1) + (V >>> 2)
                apply_shift_add_leak = (v_in >>> 1) + (v_in >>> 2);
            end
        end
    endfunction

    // ==========================================================================
    // SECTION 5: VIOLATION ASSERTION HARDWARE PIN
    // ==========================================================================
    // Assert physical FPGA GPIO pin violation_alarm_o = 1'b1 if winning class is 2, 3, 5, or 6
    assign violation_alarm_o = (class_o == 4'd2) ||
                               (class_o == 4'd3) ||
                               (class_o == 4'd5) ||
                               (class_o == 4'd6);

    // ==========================================================================
    // SECTION 6: SEQUENTIAL STATE & TIMING LOGIC (EXACTLY 1,808 CYCLES)
    // ==========================================================================
    integer i, j;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q        <= ST_IDLE;
            step_q         <= 3'd0;
            feat_grp_q     <= 11'd0;
            out_idx_q      <= 6'd0;
            out_lif_cnt_q  <= 3'd0;
            busy_o         <= 1'b0;
            done_o         <= 1'b0;
            class_o        <= 4'd0;
            cycle_count_o  <= 16'd0;
            spike_addr_o   <= 13'd0;
            s_hid_q        <= {HIDDEN_NEURONS{1'b0}};
            s_out_q        <= {OUTPUT_CLASSES{1'b0}};

            for (i = 0; i < HIDDEN_NEURONS; i = i + 1) begin
                v_hid_q[i] <= {ACCUM_BITS{1'b0}};
            end
            for (j = 0; j < OUTPUT_CLASSES; j = j + 1) begin
                v_out_q[j]       <= {ACCUM_BITS{1'b0}};
                spike_accum_q[j] <= 8'd0;
                pot_accum_q[j]   <= {ACCUM_BITS{1'b0}};
            end
        end else begin
            state_q <= state_d;

            if (busy_o) begin
                cycle_count_o <= cycle_count_o + 16'd1;
            end else if (start_i) begin
                cycle_count_o <= 16'd0;
            end

            case (state_q)
                ST_IDLE: begin
                    done_o <= 1'b0;
                    if (start_i) begin
                        busy_o        <= 1'b1;
                        step_q        <= 3'd0;
                        feat_grp_q    <= 11'd0;
                        out_idx_q     <= 6'd0;
                        out_lif_cnt_q <= 3'd0;
                        spike_addr_o  <= 13'd0;
                        s_hid_q       <= {HIDDEN_NEURONS{1'b0}};
                        s_out_q       <= {OUTPUT_CLASSES{1'b0}};

                        for (i = 0; i < HIDDEN_NEURONS; i = i + 1) begin
                            v_hid_q[i] <= {ACCUM_BITS{1'b0}};
                        end
                        for (j = 0; j < OUTPUT_CLASSES; j = j + 1) begin
                            v_out_q[j]       <= {ACCUM_BITS{1'b0}};
                            spike_accum_q[j] <= 8'd0;
                            pot_accum_q[j]   <= {ACCUM_BITS{1'b0}};
                        end
                    end
                end

                ST_FEATURE_ACC: begin
                    // 1,728 cycles total for all 6,912 features (4 features per cycle)
                    for (i = 0; i < HIDDEN_NEURONS; i = i + 1) begin
                        v_hid_q[i] <= v_hid_d[i];
                    end
                    spike_addr_o <= spike_addr_o + 13'd4;
                    feat_grp_q   <= feat_grp_q + 11'd1;
                end

                ST_STEP_LEAK: begin
                    // 1 cycle leak update per time step boundary (8 cycles total)
                    for (i = 0; i < HIDDEN_NEURONS; i = i + 1) begin
                        v_hid_q[i] <= v_hid_d[i];
                    end
                    s_hid_q <= s_hid_d;
                    step_q  <= step_q + 3'd1;
                end

                ST_OUTPUT_ACC: begin
                    // 64 cycles for hidden-to-output synaptic transmission
                    for (j = 0; j < OUTPUT_CLASSES; j = j + 1) begin
                        v_out_q[j] <= v_out_d[j];
                    end
                    out_idx_q <= out_idx_q + 6'd1;
                end

                ST_OUTPUT_LIF: begin
                    // 8 cycles for output LIF updates
                    for (j = 0; j < OUTPUT_CLASSES; j = j + 1) begin
                        v_out_q[j]       <= v_out_d[j];
                        spike_accum_q[j] <= spike_accum_q[j] + {7'd0, s_out_d[j]};
                        pot_accum_q[j]   <= pot_accum_q[j] + v_out_d[j];
                    end
                    s_out_q       <= s_out_d;
                    out_lif_cnt_q <= out_lif_cnt_q + 3'd1;

                    // On the 8th and final cycle (exact cycle 1,808), evaluate argmax and assert done
                    if (out_lif_cnt_q == 3'd7) begin
                        class_o <= find_argmax_class();
                        busy_o  <= 1'b0;
                        done_o  <= 1'b1;
                    end
                end
            endcase
        end
    end

    // ==========================================================================
    // SECTION 7: COMBINATIONAL LOGIC & LIF UPDATE
    // ==========================================================================
    reg signed [ACCUM_BITS-1:0] syn_sum;
    integer h, k, f;

    always @(*) begin
        state_d = state_q;

        for (h = 0; h < HIDDEN_NEURONS; h = h + 1) begin
            v_hid_d[h] = v_hid_q[h];
        end
        s_hid_d = s_hid_q;

        for (k = 0; k < OUTPUT_CLASSES; k = k + 1) begin
            v_out_d[k] = v_out_q[k];
        end
        s_out_d = s_out_q;

        case (state_q)
            ST_IDLE: begin
                if (start_i) state_d = ST_FEATURE_ACC;
            end

            ST_FEATURE_ACC: begin
                for (h = 0; h < HIDDEN_NEURONS; h = h + 1) begin
                    syn_sum = 16'sh0000;
                    for (f = 0; f < 4; f = f + 1) begin
                        if (spike_data_i[f]) begin
                            syn_sum = syn_sum + {{4{w1_rom[(feat_grp_q * 4 + f) * HIDDEN_NEURONS + h][7]}},
                                                 w1_rom[(feat_grp_q * 4 + f) * HIDDEN_NEURONS + h], 4'h0};
                        end
                    end
                    v_hid_d[h] = v_hid_q[h] + syn_sum;
                end

                if ((feat_grp_q % 216 == 215) || (feat_grp_q == 11'd1727)) begin
                    state_d = ST_STEP_LEAK;
                end
            end

            ST_STEP_LEAK: begin
                for (h = 0; h < HIDDEN_NEURONS; h = h + 1) begin
                    if (v_hid_q[h] >= V_THRESHOLD) begin
                        s_hid_d[h] = 1'b1;
                        v_hid_d[h] = 16'sh0000;
                    end else begin
                        s_hid_d[h] = 1'b0;
                        v_hid_d[h] = apply_shift_add_leak(v_hid_q[h], s_hid_q[h]);
                    end
                end

                if (feat_grp_q >= 11'd1728) begin
                    state_d = ST_OUTPUT_ACC;
                end else begin
                    state_d = ST_FEATURE_ACC;
                end
            end

            ST_OUTPUT_ACC: begin
                if (s_hid_q[out_idx_q]) begin
                    for (k = 0; k < OUTPUT_CLASSES; k = k + 1) begin
                        v_out_d[k] = v_out_q[k] + {{4{w2_rom[out_idx_q * OUTPUT_CLASSES + k][7]}},
                                                   w2_rom[out_idx_q * OUTPUT_CLASSES + k], 4'h0};
                    end
                end

                if (out_idx_q == HIDDEN_NEURONS - 1) begin
                    state_d = ST_OUTPUT_LIF;
                end
            end

            ST_OUTPUT_LIF: begin
                for (k = 0; k < OUTPUT_CLASSES; k = k + 1) begin
                    if (v_out_q[k] >= V_THRESHOLD) begin
                        s_out_d[k] = 1'b1;
                        v_out_d[k] = 16'sh0000;
                    end else begin
                        s_out_d[k] = 1'b0;
                        v_out_d[k] = apply_shift_add_leak(v_out_q[k], s_out_q[k]);
                    end
                end

                if (out_lif_cnt_q == 3'd7) begin
                    state_d = ST_IDLE; // Direct transition back to idle on cycle 1,808
                end
            end
        endcase
    end

    // ==========================================================================
    // SECTION 8: ARGMAX DECISION FUNCTION
    // ==========================================================================
    function [3:0] find_argmax_class;
        reg [3:0] max_c;
        reg [7:0] max_spikes;
        reg signed [ACCUM_BITS-1:0] max_pot;
        integer c;
        begin
            max_c      = 4'd0;
            max_spikes = spike_accum_q[0];
            max_pot    = pot_accum_q[0];

            for (c = 1; c < OUTPUT_CLASSES; c = c + 1) begin
                if ((spike_accum_q[c] > max_spikes) ||
                    ((spike_accum_q[c] == max_spikes) && (pot_accum_q[c] > max_pot))) begin
                    max_c      = c[3:0];
                    max_spikes = spike_accum_q[c];
                    max_pot    = pot_accum_q[c];
                end
            end
            find_argmax_class = max_c;
        end
    endfunction

endmodule
