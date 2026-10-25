# Hybrid-SNN Architecture

## Inference data flow

1. Capture a camera frame or load an image.
2. YOLOv8n detects candidate objects.
3. The host pipeline selects a region of interest and crops it.
4. The crop is resized to 48 × 48 RGB and preprocessed using the project's CLAHE/DoG pipeline.
5. TTFS encoding converts pixel features into spike events over the configured time window.
6. The dense SNN applies learned weights and LIF membrane dynamics.
7. The output layer selects one of ten classes.
8. The configured decision mapping marks classes 2, 3, 5, and 6 as violations.

## Hardware boundary

The software runtime includes host-side Python/PyTorch inference. The `fpga_rtl/` directory contains a hardware-oriented Verilog implementation and related simulation assets. These are separate execution paths unless the project integrates them through a verified interface. A functional AXI connection, FPGA programming, and on-board SNN inference must be demonstrated independently.

## Reproducibility checklist

- Record checkpoint checksum and source revision.
- Record the validation dataset and split.
- Run software tests and save complete logs.
- Simulate RTL with known input vectors.
- Compare software and RTL outputs under identical quantization and reset assumptions.
- Save Libero synthesis, timing, and power reports.
- Record board model, programmed image, and end-to-end latency measurement method.
