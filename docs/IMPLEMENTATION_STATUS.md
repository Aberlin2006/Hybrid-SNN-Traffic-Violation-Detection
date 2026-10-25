# Implementation and Evidence Status

This note separates included artifacts from claims that require additional evidence.

## Present in this repository
- Python training/model/preprocessing/runtime scripts.
- A saved SNN checkpoint and exported SNN weight/model artifacts.
- Verilog accelerator and testbench source files.
- A Python cycle-oriented RTL emulator and verification scripts.

## Must be verified before making hardware claims
- Successful synthesis and place-and-route in Libero for the stated target device.
- Exact resource utilization from the generated implementation report.
- Clock timing and the measured inference cycle count.
- Power report assumptions or physical power measurement.
- Successful programming and inference on the physical board.
- Software/RTL equivalence using matching weights, quantization, dimensions, and test vectors.

## Results
The presentation figures are reported results, not independently reproduced in the process that assembled this repository. In project documents, latency values differ; use the value from the final matching hardware report rather than mixing values from different drafts.
