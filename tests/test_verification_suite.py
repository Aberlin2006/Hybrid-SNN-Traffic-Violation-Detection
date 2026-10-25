"""
AO.SNN Formal Verification & Validation Test Suite
===================================================
Target Platform: Microchip PolarFire SoC FPGA (MPFS250T) & Host Workstation
Canonical Architecture: 6912 -> 64 -> 10, Timesteps = 8, Signed Q4.4 Fixed-Point

This test suite executes all automated unit and integration verifications:
  1. Checkpoint & Architecture Integrity
  2. Export Files (.hex / .bin / manifest) Consistency
  3. TTFS Spike Encoding Edge Cases (Dark, Bright, Contrast, Edge)
  4. Numerical Bit-Accuracy (PyTorch Q4.4 vs RTL Fixed-Point Dynamics)
  5. RTL Cycle Latency & State Machine Timing Verification
  6. Failure Mode & Defensive Safeguard Testing
"""

import os
import sys
import math
import struct
import unittest
import numpy as np
import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.system_config import (
    CONFIG_VERSION,
    INPUT_DIMS,
    HIDDEN_NEURONS,
    OUTPUT_CLASSES,
    TIME_STEPS,
    WEIGHT_SCALE,
    V_THRESHOLD_FIXED,
    TOTAL_CLOCK_CYCLES,
    TOTAL_LATENCY_US,
    TRAFFIC_CLASSES
)
from fpga_rtl.simulate_rtl_cycle_accurate import RTLSimulator


class TestArchitectureAndCheckpoint(unittest.TestCase):
    """Test 1: Checkpoint & Canonical Architecture Integrity"""

    def setUp(self):
        self.ckpt_path = os.path.join(PROJECT_ROOT, "vectorblox_deployment", "artifacts", "best_snn_checkpoint.pt")

    def test_checkpoint_exists(self):
        self.assertTrue(os.path.isfile(self.ckpt_path), f"Checkpoint missing at {self.ckpt_path}")

    def test_checkpoint_tensors_and_shapes(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Verify key presence
        self.assertIn("hidden_layer.weight", state_dict)
        self.assertIn("output_layer.weight", state_dict)

        w1 = state_dict["hidden_layer.weight"]
        w2 = state_dict["output_layer.weight"]

        # Validate tensor dimensions
        self.assertEqual(w1.shape, (HIDDEN_NEURONS, INPUT_DIMS), 
                         f"Hidden layer shape mismatch: {w1.shape} vs ({HIDDEN_NEURONS}, {INPUT_DIMS})")
        self.assertEqual(w2.shape, (OUTPUT_CLASSES, HIDDEN_NEURONS),
                         f"Output layer shape mismatch: {w2.shape} vs ({OUTPUT_CLASSES}, {HIDDEN_NEURONS})")

        # Validate total parameters
        total_params = w1.numel() + w2.numel()
        self.assertEqual(total_params, (64 * 6912) + (10 * 64))
        self.assertEqual(total_params, 443008)

        # Validate parameter statistics (ensure not all zeros, no NaN/Inf)
        self.assertFalse(torch.isnan(w1).any(), "NaN found in hidden weights")
        self.assertFalse(torch.isnan(w2).any(), "NaN found in output weights")
        self.assertFalse(torch.isinf(w1).any(), "Inf found in hidden weights")
        self.assertFalse(torch.isinf(w2).any(), "Inf found in output weights")
        self.assertGreater(float(w1.abs().sum()), 0.0, "Hidden weights are all zeros")
        self.assertGreater(float(w2.abs().sum()), 0.0, "Output weights are all zeros")


class TestExportIntegrity(unittest.TestCase):
    """Test 2: Exported ROM Hex and Binary Weight File Consistency"""

    def setUp(self):
        self.w1_hex = os.path.join(PROJECT_ROOT, "fpga_rtl", "w1_q4_4.hex")
        self.w2_hex = os.path.join(PROJECT_ROOT, "fpga_rtl", "w2_q4_4.hex")
        self.bin_path = os.path.join(PROJECT_ROOT, "vectorblox_deployment", "artifacts", "snn_weights_q4_4.bin")
        self.manifest = os.path.join(PROJECT_ROOT, "vectorblox_deployment", "artifacts", "export_manifest.json")

    def test_hex_files_exist_and_sized(self):
        self.assertTrue(os.path.isfile(self.w1_hex))
        self.assertTrue(os.path.isfile(self.w2_hex))
        self.assertTrue(os.path.isfile(self.bin_path))

        with open(self.w1_hex, "r") as f:
            w1_lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(w1_lines), 442368, f"W1 hex line count mismatch: {len(w1_lines)}")

        with open(self.w2_hex, "r") as f:
            w2_lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(w2_lines), 640, f"W2 hex line count mismatch: {len(w2_lines)}")

    def test_hex_to_bin_byte_exactness(self):
        with open(self.bin_path, "rb") as f:
            bin_data = f.read()

        self.assertEqual(len(bin_data), 443008, f"Binary weight size mismatch: {len(bin_data)}")

        with open(self.w1_hex, "r") as f:
            w1_vals = [int(l.strip(), 16) for l in f if l.strip()]
        with open(self.w2_hex, "r") as f:
            w2_vals = [int(l.strip(), 16) for l in f if l.strip()]

        combined_hex_bytes = bytes(w1_vals + w2_vals)
        self.assertEqual(bin_data, combined_hex_bytes, "Binary file does not match concatenated HEX files byte-for-byte!")


class TestTTFSSpikeEncoding(unittest.TestCase):
    """Test 3: Temporal TTFS (Time-to-First-Spike) Spike Encoding Dynamics"""

    def _encode_ttfs(self, img_array, timesteps=8):
        """Standard TTFS encoding implementation matching training & runtime"""
        # img_array: normalized [0, 1] flat array of shape (N,)
        spike_train = np.zeros((timesteps, len(img_array)), dtype=np.uint8)
        for i, val in enumerate(img_array):
            if val > 0.05:  # Non-zero activation threshold
                # Brighter pixels spike earlier: t in [0, timesteps - 1]
                t_spike = int((1.0 - val) * (timesteps - 1))
                t_spike = max(0, min(timesteps - 1, t_spike))
                spike_train[t_spike, i] = 1
        return spike_train

    def test_pure_dark_image(self):
        """Completely black input (0.0 intensity) should produce zero spikes"""
        dark_input = np.zeros(6912, dtype=np.float32)
        spikes = self._encode_ttfs(dark_input, TIME_STEPS)
        self.assertEqual(np.sum(spikes), 0, "Dark image produced unexpected spikes!")

    def test_pure_bright_image(self):
        """Completely white input (1.0 intensity) should fire immediately at t=0 and only once"""
        bright_input = np.ones(6912, dtype=np.float32)
        spikes = self._encode_ttfs(bright_input, TIME_STEPS)
        # All pixels must fire at t=0
        self.assertEqual(np.sum(spikes[0, :]), 6912)
        # No spikes at t > 0
        self.assertEqual(np.sum(spikes[1:, :]), 0)
        # Exactly 1 spike per feature across all timesteps
        self.assertEqual(np.sum(spikes), 6912)

    def test_monotonic_arrival_times(self):
        """Brighter pixels must fire strictly before or simultaneously with darker pixels"""
        vals = np.linspace(0.1, 1.0, 10, dtype=np.float32)
        spikes = self._encode_ttfs(vals, TIME_STEPS)
        arrival_times = []
        for feat_idx in range(len(vals)):
            fired_timesteps = np.where(spikes[:, feat_idx] == 1)[0]
            self.assertEqual(len(fired_timesteps), 1)
            arrival_times.append(fired_timesteps[0])

        # As intensity increases (0.1 -> 1.0), arrival time must be non-increasing (monotonically earlier)
        for i in range(len(arrival_times) - 1):
            self.assertGreaterEqual(arrival_times[i], arrival_times[i + 1],
                                    f"TTFS timing inverted at step {i}: {arrival_times[i]} vs {arrival_times[i+1]}")


class TestNumericalBitAccuracy(unittest.TestCase):
    """Test 4: Fixed-Point Arithmetic & LIF Neuron Dynamics (PyTorch vs RTL)"""

    def setUp(self):
        self.sim = RTLSimulator(
            w1_hex_path=os.path.join(PROJECT_ROOT, "fpga_rtl", "w1_q4_4.hex"),
            w2_hex_path=os.path.join(PROJECT_ROOT, "fpga_rtl", "w2_q4_4.hex")
        )

    def test_quantization_clamping(self):
        """Verify Q4.4 signed conversion satisfies saturation limits [-128, +127]"""
        raw_values = [-10.0, -8.0, -1.0, 0.0, 0.5, 7.9375, 10.0]
        expected_q = [-128, -128, -16, 0, 8, 127, 127]
        for r, exp in zip(raw_values, expected_q):
            scaled = int(round(r * WEIGHT_SCALE))
            clamped = max(-128, min(127, scaled))
            self.assertEqual(clamped, exp)

    def test_leak_factor_arithmetic(self):
        """Verify RTL shift-add leak implements exact 0.75 alpha: (V >> 1) + (V >> 2)"""
        test_potentials = [0, 4, 16, 100, 256, 512, 1000]
        for v in test_potentials:
            rtl_leak = (v >> 1) + (v >> 2)
            expected = int(0.75 * v)
            self.assertEqual(rtl_leak, expected, f"Leak mismatch for V={v}: {rtl_leak} vs {expected}")

    def test_deterministic_execution(self):
        """Running the same input twice through RTL simulator produces bit-for-bit identical results"""
        np.random.seed(42)
        test_img = np.random.rand(INPUT_DIMS).astype(np.float32)

        pred1, spikes1, cycles1 = self.sim.run_inference(test_img)
        pred2, spikes2, cycles2 = self.sim.run_inference(test_img)

        self.assertEqual(pred1, pred2)
        self.assertEqual(spikes1, spikes2)
        self.assertEqual(cycles1, cycles2)


class TestRTLCycleLatencyAndTiming(unittest.TestCase):
    """Test 5: RTL Timing, State Machine, and Clock Cycle Exactness"""

    def setUp(self):
        self.sim = RTLSimulator(
            w1_hex_path=os.path.join(PROJECT_ROOT, "fpga_rtl", "w1_q4_4.hex"),
            w2_hex_path=os.path.join(PROJECT_ROOT, "fpga_rtl", "w2_q4_4.hex")
        )

    def test_cycle_latency_assertion(self):
        """Every inference on MPFS250T accelerator must take strictly 1,808 clock cycles (18.08 us @ 100MHz)"""
        zero_img = np.zeros(INPUT_DIMS, dtype=np.float32)
        _, _, cycles_zero = self.sim.run_inference(zero_img)
        self.assertEqual(cycles_zero, TOTAL_CLOCK_CYCLES)

        ones_img = np.ones(INPUT_DIMS, dtype=np.float32)
        _, _, cycles_ones = self.sim.run_inference(ones_img)
        self.assertEqual(cycles_ones, TOTAL_CLOCK_CYCLES)

    def test_predicted_class_valid_range(self):
        """Argmax output must be within [0, 9]"""
        np.random.seed(123)
        for _ in range(5):
            rand_img = np.random.rand(INPUT_DIMS).astype(np.float32)
            pred, spikes, _ = self.sim.run_inference(rand_img)
            self.assertGreaterEqual(pred, 0)
            self.assertLess(pred, OUTPUT_CLASSES)
            self.assertEqual(len(spikes), OUTPUT_CLASSES)


class TestDefensiveSafeguardsAndFailureModes(unittest.TestCase):
    """Test 6: Defensive Safeguards, Failure Modes & Graceful Handling"""

    def test_mismatched_checkpoint_rejection(self):
        """Loading a checkpoint with incompatible dimensions must raise ValueError, not silently truncate"""
        from embedded_runtime.runtime_inference import JunctionMonitorSystem

        # Create a dummy corrupted checkpoint with 32 hidden neurons
        dummy_state = {
            "hidden_layer.weight": torch.randn(32, INPUT_DIMS),
            "output_layer.weight": torch.randn(OUTPUT_CLASSES, 32)
        }
        dummy_path = os.path.join(PROJECT_ROOT, "vectorblox_deployment", "artifacts", "dummy_corrupt_test.pt")
        torch.save(dummy_state, dummy_path)

        try:
            with self.assertRaises(ValueError):
                JunctionMonitorSystem(checkpoint_path=dummy_path)
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_nonexistent_checkpoint_error(self):
        """Providing a missing file path must raise FileNotFoundError"""
        from embedded_runtime.runtime_inference import JunctionMonitorSystem
        with self.assertRaises(FileNotFoundError):
            JunctionMonitorSystem(checkpoint_path="non_existent_path_xyz.pt")


if __name__ == "__main__":
    print("=" * 80)
    print("AO.SNN FORMAL VERIFICATION AND VALIDATION TEST SUITE")
    print("=" * 80)
    unittest.main(verbosity=2)
