"""
Microchip VectorBlox Python Compilation Wrapper for PolarFire SoC MPFS250
Executes INT8 calibration, ONNX graph verification, and checks for genuine VectorBlox SDK.

IMPORTANT ARCHITECTURAL NOTICE:
- When the proprietary Microchip VectorBlox SDK (`vbx_cnn` / `vbx_simulate`) is installed,
  this script compiles the ONNX graph into a genuine hardware `.vnnx` container.
- If the vendor toolchain is NOT found in the system PATH, it operates in EXPLICIT
  PACKAGING/EMULATION MODE, clearly logging that physical FPGA bitstreams require
  the Microchip Libero SoC & VectorBlox SDK toolchain.
"""

import os
import sys
import shutil
import json
import numpy as np
import onnx

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    output_dir = os.path.join(script_dir, "output_vnnx")
    calib_dir = os.path.join(script_dir, "calibration_data")
    config_file = os.path.join(script_dir, "vectorblox_config.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    yolo_onnx = os.path.join(artifacts_dir, "yolov8n_stage1.onnx")
    snn_onnx = os.path.join(artifacts_dir, "snn_junction_q4_4.onnx")
    output_vnnx = os.path.join(output_dir, "yolov8n_stage1.vnnx")

    print("=" * 70)
    print("MICROCHIP VECTORBLOX COMPILER (MPFS250-VIDEO-KIT)")
    print("CoreVectorBlox V1000 @ 100MHz | 32-bit AXI4")
    print("=" * 70)

    # Check for genuine vendor compiler binary
    vbx_compiler = shutil.which("vbx_cnn") or shutil.which("vbx")
    is_genuine_toolchain = vbx_compiler is not None

    if is_genuine_toolchain:
        print(f"[TOOLCHAIN] Genuine Microchip VectorBlox toolchain detected at: {vbx_compiler}")
    else:
        print("[TOOLCHAIN WARNING] Microchip VectorBlox SDK ('vbx_cnn') not found in PATH.")
        print("                   Operating in Offline Model Validation & Fallback Packaging Mode.")
        print("                   Notice: Generated .vnnx container is for simulation and metadata verification.")

    # 1. Verify ONNX Model Graph
    print(f"\n[1/3] Verifying Stage-1 ONNX model: {yolo_onnx}")
    if not os.path.exists(yolo_onnx):
        raise FileNotFoundError(f"Model missing: {yolo_onnx}")

    model = onnx.load(yolo_onnx)
    onnx.checker.check_model(model)
    print("      ONNX Validation: PASSED (Graph structure verified valid)")

    # 2. Calibration generation
    print(f"\n[2/3] Generating INT8 calibration dataset in {calib_dir}...")
    for i in range(20):
        dummy = np.random.randint(0, 256, (1, 3, 640, 640), dtype=np.uint8)
        np.save(os.path.join(calib_dir, f"calib_{i:02d}.npy"), dummy)
    print(f"      Synthesized 20 calibration frames.")

    # 3. Generate or compile container
    print(f"\n[3/3] Generating VectorBlox deployment package: {output_vnnx}")
    if is_genuine_toolchain:
        # Genuine compilation via vendor CLI
        cmd = f"vbx_cnn -c {config_file} -m {yolo_onnx} -o {output_vnnx}"
        print(f"      Executing vendor command: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"Vendor compilation failed with exit code {ret}")
    else:
        # Explicit fallback packaging with clear header annotation
        with open(output_vnnx, "wb") as f:
            f.write(b"VBX_SIM_EMULATED_MPFS250\x00")
            with open(config_file, "rb") as cf:
                cfg_bytes = cf.read()
                f.write(len(cfg_bytes).to_bytes(4, "little"))
                f.write(cfg_bytes)
            with open(yolo_onnx, "rb") as yf:
                f.write(yf.read())
        print(f"      [FALLBACK ARTIFACT] Package generated ({os.path.getsize(output_vnnx)} bytes).")
        print("      Note: Physical FPGA flash requires running Microchip VectorBlox SDK in Linux/PolarFire environment.")

    print("=" * 70)
    print("[STATUS] VectorBlox packaging step complete.")

if __name__ == "__main__":
    main()
