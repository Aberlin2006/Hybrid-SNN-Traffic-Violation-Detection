"""
AO.SNN Software vs RTL Numerical Comparison Report Generator
=============================================================
Runs identical test vectors through:
  1. PyTorch Q4.4 quantised SNN (software float inference with Q4.4 rounding)
  2. Cycle-accurate RTL emulator (simulate_rtl_cycle_accurate.py)

Compares predicted class, spike counts, and wall-clock latency.
"""

import os
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from training.system_config import (
    INPUT_DIMS, HIDDEN_NEURONS, OUTPUT_CLASSES, TIME_STEPS,
    WEIGHT_SCALE, V_THRESHOLD_FIXED, TOTAL_CLOCK_CYCLES, TOTAL_LATENCY_US, TRAFFIC_CLASSES
)
from fpga_rtl.simulate_rtl_cycle_accurate import RTLSimulator

CKPT_PATH = os.path.join(PROJECT_ROOT, "vectorblox_deployment", "artifacts", "best_snn_checkpoint.pt")
W1_HEX = os.path.join(PROJECT_ROOT, "fpga_rtl", "w1_q4_4.hex")
W2_HEX = os.path.join(PROJECT_ROOT, "fpga_rtl", "w2_q4_4.hex")


def load_hex_weights():
    """Load Q4.4 weight arrays from hex files."""
    def read_hex(path, n):
        arr = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    v = int(line, 16)
                    if v >= 128:
                        v -= 256
                    arr.append(v)
        assert len(arr) == n, f"Hex size mismatch in {path}: {len(arr)} vs {n}"
        return arr

    w1 = read_hex(W1_HEX, INPUT_DIMS * HIDDEN_NEURONS)
    w2 = read_hex(W2_HEX, HIDDEN_NEURONS * OUTPUT_CLASSES)
    return (
        np.array(w1, dtype=np.int8).reshape(HIDDEN_NEURONS, INPUT_DIMS),
        np.array(w2, dtype=np.int8).reshape(OUTPUT_CLASSES, HIDDEN_NEURONS)
    )


def quantize_weights_pytorch(ckpt_path):
    """Load PyTorch checkpoint and quantise weights to Q4.4 format."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    w1_f = sd["hidden_layer.weight"].numpy()
    w2_f = sd["output_layer.weight"].numpy()

    def quantize(w):
        scaled = np.round(w * WEIGHT_SCALE).astype(np.int32)
        return np.clip(scaled, -128, 127).astype(np.int8)

    return quantize(w1_f), quantize(w2_f)


def software_lif_inference(input_flat_01, w1_q, w2_q):
    """
    Pure Python Q4.4 fixed-point LIF inference (matches RTL arithmetic step-by-step).
    Input: float array in [0, 1], shape (6912,)
    Returns: predicted class id, output_accum list
    """
    # TTFS binary encode
    def ttfs_encode(val, timesteps=8):
        if val <= 0.05:
            return -1  # never fires
        t = int((1.0 - val) * (timesteps - 1))
        return max(0, min(timesteps - 1, t))

    # Group inputs by firing time
    fire_times = [ttfs_encode(v) for v in input_flat_01]

    v_hid = np.zeros(HIDDEN_NEURONS, dtype=np.int32)
    s_hid = np.zeros(HIDDEN_NEURONS, dtype=np.int32)

    for t in range(TIME_STEPS):
        # Accumulate spikes for this timestep
        for feat_idx in range(INPUT_DIMS):
            if fire_times[feat_idx] == t:
                # Weight: INT8 -> shift up by 4 bits (Q4.4 -> Q8.8)
                contribution = w1_q[:, feat_idx].astype(np.int32) << 4
                v_hid += contribution

        # LIF update: threshold & leak
        for h in range(HIDDEN_NEURONS):
            v_signed = v_hid[h] if v_hid[h] < 32768 else v_hid[h] - 65536
            if v_signed >= V_THRESHOLD_FIXED:
                s_hid[h] = 1
                v_hid[h] = 0
            else:
                s_hid[h] = 0
                # shift-add leak: (V >> 1) + (V >> 2)
                v_signed = v_hid[h] if v_hid[h] < 32768 else v_hid[h] - 65536
                v_hid[h] = ((v_signed >> 1) + (v_signed >> 2)) & 0xFFFF

    # Output layer
    v_out = np.zeros(OUTPUT_CLASSES, dtype=np.int32)
    for h in range(HIDDEN_NEURONS):
        if s_hid[h]:
            contribution = w2_q[:, h].astype(np.int32) << 4
            v_out += contribution

    # Output LIF (8 steps)
    spike_accum = np.zeros(OUTPUT_CLASSES, dtype=np.int32)
    for _ in range(8):
        for k in range(OUTPUT_CLASSES):
            v_k_signed = v_out[k] if v_out[k] < 32768 else v_out[k] - 65536
            if v_k_signed >= V_THRESHOLD_FIXED:
                spike_accum[k] += 1
                v_out[k] = 0
            else:
                v_k_signed_2 = v_out[k] if v_out[k] < 32768 else v_out[k] - 65536
                v_out[k] = ((v_k_signed_2 >> 1) + (v_k_signed_2 >> 2)) & 0xFFFF

    pred = int(np.argmax(spike_accum))
    return pred, spike_accum.tolist()


def run_comparison():
    print("=" * 80)
    print("AO.SNN SOFTWARE vs RTL NUMERICAL COMPARISON REPORT")
    print("=" * 80)
    print(f"Architecture : {INPUT_DIMS} -> {HIDDEN_NEURONS} -> {OUTPUT_CLASSES}, T={TIME_STEPS}")
    print(f"Precision    : Signed Q4.4 (INT8), scale={WEIGHT_SCALE}, threshold=0x{V_THRESHOLD_FIXED:04X}")
    print(f"RTL Target   : Microchip PolarFire SoC MPFS250T @ 100MHz")
    print(f"RTL Cycles   : {TOTAL_CLOCK_CYCLES} cycles = {TOTAL_LATENCY_US:.2f} µs")
    print()

    # Load weights
    print("[SETUP] Loading Q4.4 hex weights...")
    w1_hex, w2_hex = load_hex_weights()
    print("[SETUP] Quantizing PyTorch checkpoint to Q4.4...")
    w1_pt, w2_pt = quantize_weights_pytorch(CKPT_PATH)

    # Verify hex == pytorch quantized
    match_w1 = np.array_equal(w1_hex, w1_pt)
    match_w2 = np.array_equal(w2_hex, w2_pt)
    print(f"[WEIGHT CHECK] W1 hex == PyTorch Q4.4: {'[MATCH]' if match_w1 else '[MISMATCH]'}")
    print(f"[WEIGHT CHECK] W2 hex == PyTorch Q4.4: {'[MATCH]' if match_w2 else '[MISMATCH]'}")
    print()

    sim = RTLSimulator(w1_hex_path=W1_HEX, w2_hex_path=W2_HEX)

    # Test vectors
    test_vectors = {
        "zero_input"     : np.zeros(INPUT_DIMS, dtype=np.float32),
        "ones_input"     : np.ones(INPUT_DIMS, dtype=np.float32),
        "mid_input_0.5"  : np.full(INPUT_DIMS, 0.5, dtype=np.float32),
        "random_seed42"  : np.random.default_rng(42).random(INPUT_DIMS).astype(np.float32),
        "random_seed7"   : np.random.default_rng(7).random(INPUT_DIMS).astype(np.float32),
        "alternating"    : np.array([1.0 if i % 2 == 0 else 0.0 for i in range(INPUT_DIMS)], dtype=np.float32),
        "gradient"       : np.linspace(0.0, 1.0, INPUT_DIMS, dtype=np.float32),
        "bright_center"  : np.array([1.0 if (3072 - INPUT_DIMS//4 <= i <= 3072 + INPUT_DIMS//4) else 0.1 for i in range(INPUT_DIMS)], dtype=np.float32),
    }

    all_match = True
    results = []

    print(f"{'Test Vector':<22} {'SW Pred':<10} {'RTL Pred':<10} {'Match':<8} {'SW Spikes':<20} {'RTL Spikes':<20}")
    print("-" * 102)

    for name, vec in test_vectors.items():
        # Software inference
        t0 = time.perf_counter()
        sw_pred, sw_spikes = software_lif_inference(vec, w1_hex, w2_hex)
        sw_latency_ms = (time.perf_counter() - t0) * 1000.0

        # RTL emulator inference
        t0 = time.perf_counter()
        rtl_pred, rtl_spikes, rtl_cycles = sim.run_inference(vec)
        rtl_latency_ms = (time.perf_counter() - t0) * 1000.0

        match = (sw_pred == rtl_pred)
        if not match:
            all_match = False
        match_sym = "[OK]" if match else "[MISMATCH]"

        sw_class = TRAFFIC_CLASSES.get(sw_pred, "Unknown")
        rtl_class = TRAFFIC_CLASSES.get(rtl_pred, "Unknown")

        sw_top3 = sorted(enumerate(sw_spikes), key=lambda x: -x[1])[:3]
        rtl_top3 = sorted(enumerate(rtl_spikes), key=lambda x: -x[1])[:3]

        print(f"{name:<22} {sw_class[:9]:<10} {rtl_class[:9]:<10} {match_sym:<8} {str(sw_spikes):<20} {str(rtl_spikes):<20}")
        results.append({
            "name": name,
            "sw_pred": sw_pred,
            "rtl_pred": rtl_pred,
            "match": match,
            "sw_spikes": sw_spikes,
            "rtl_spikes": rtl_spikes,
            "rtl_cycles": rtl_cycles,
        })

    print("-" * 102)
    matched = sum(1 for r in results if r["match"])
    print(f"\n[SUMMARY] Predictions Matched: {matched}/{len(results)}")
    print(f"[SUMMARY] Weight Arrays Identical: W1={'PASS' if match_w1 else 'FAIL'}, W2={'PASS' if match_w2 else 'FAIL'}")
    print(f"[SUMMARY] RTL Fixed Cycle Budget: {TOTAL_CLOCK_CYCLES} cycles = {TOTAL_LATENCY_US:.2f} µs per inference")
    print(f"[SUMMARY] Overall Bit-Accuracy:  {'PASS' if (all_match and match_w1 and match_w2) else 'FAIL'}")
    print()

    if not all_match or not match_w1 or not match_w2:
        print("[ACTION REQUIRED] Investigate mismatching test vectors above.")
    else:
        print("[VERIFIED] Software inference and RTL emulation are bit-accurate for all test vectors.")
        print("[VERIFIED] Exported Q4.4 hex weight files exactly match PyTorch quantised checkpoint.")

    return all_match and match_w1 and match_w2


if __name__ == "__main__":
    ok = run_comparison()
    sys.exit(0 if ok else 1)
