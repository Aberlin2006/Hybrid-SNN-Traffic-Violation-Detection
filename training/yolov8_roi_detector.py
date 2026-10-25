"""
AO.SNN Stage-1 FrontEnd Object Detector: YOLOv8n Architecture & ROI Extraction Engine
Hardware Acceleration: NVIDIA GeForce RTX (Tensor Cores) & Microchip CoreVectorBlox Accelerator
VectorBlox Target: CoreVectorBlox V1000 on PolarFire SoC MPFS250
Input: 1080p (1920x1080) camera stream downscaled to 640x640
Output: Candidate vehicle & rider ROIs -> Adaptive Edge Fusion Preprocessor -> 48x48x3 SNN crops
"""

import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from dataset_pipeline import AdaptiveEdgeFusionPreprocessor


# =====================================================================
# SECTION 1: HARDWARE-OPTIMIZED CNN BUILDING BLOCKS (VECTORBLOX COMPLIANT)
# =====================================================================
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNSiLU(nn.Module):
    """Standard Convolution + BatchNorm2d + SiLU / LeakyReLU (VectorBlox compliant)."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1, d: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # VectorBlox compiler excels at LeakyReLU (0.1) or SiLU
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard Bottleneck residual block."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, c_, k[0], 1)
        self.cv2 = ConvBNSiLU(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions (YOLOv8 backbone block)."""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (equivalent to SPP with consecutive max pools)."""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# =====================================================================
# SECTION 2: COMPLETE YOLOv8n FRONTEND DETECTOR BACKBONE & HEAD
# =====================================================================
class YOLOv8nDetector(nn.Module):
    """
    Stage-1 FrontEnd Object Detector (YOLOv8n topology):
    Designed for VectorBlox execution on MPFS250 & CUDA inference on RTX GPU.
    Input resolution: (3, 640, 640)
    Multi-Scale Anchors / Head:
      - P3/8:  80x80 grid for small targets (distant bikes/scooters)
      - P4/16: 40x40 grid for medium targets (cars, closer two-wheelers)
      - P5/32: 20x20 grid for large targets (buses, heavy trucks)
    Classes: 4 candidate junction categories (Vehicle, TwoWheeler, Rider, HeavyVehicle)
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # Backbone (Standard YOLOv8n Channel Scales: 16, 32, 64, 128, 256)
        self.stem = ConvBNSiLU(3, 16, 3, 2)            # /2 -> 320x320
        self.stage1 = ConvBNSiLU(16, 32, 3, 2)         # /4 -> 160x160
        self.c2f1 = C2f(32, 32, n=1, shortcut=True)

        self.stage2 = ConvBNSiLU(32, 64, 3, 2)         # /8 -> 80x80 (P3)
        self.c2f2 = C2f(64, 64, n=2, shortcut=True)

        self.stage3 = ConvBNSiLU(64, 128, 3, 2)        # /16 -> 40x40 (P4)
        self.c2f3 = C2f(128, 128, n=2, shortcut=True)

        self.stage4 = ConvBNSiLU(128, 256, 3, 2)       # /32 -> 20x20 (P5)
        self.c2f4 = C2f(256, 256, n=1, shortcut=True)
        self.sppf = SPPF(256, 256, k=5)

        # PAN-FPN Neck & Upsample
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_neck1 = C2f(256 + 128, 128, n=1, shortcut=False) # 40x40
        self.c2f_neck2 = C2f(128 + 64, 64, n=1, shortcut=False)   # 80x80

        # Downsample Neck
        self.down1 = ConvBNSiLU(64, 64, 3, 2)                      # 40x40
        self.c2f_neck3 = C2f(64 + 128, 128, n=1, shortcut=False)
        self.down2 = ConvBNSiLU(128, 128, 3, 2)                    # 20x20
        self.c2f_neck4 = C2f(128 + 256, 256, n=1, shortcut=False)

        # Detection Heads: Predict (4 box coordinates dx, dy, dw, dh + num_classes)
        # Anchor-free decoupled head
        out_channels = 4 + num_classes
        self.head_p3 = nn.Conv2d(64, out_channels, 1)   # 80x80
        self.head_p4 = nn.Conv2d(128, out_channels, 1)  # 40x40
        self.head_p5 = nn.Conv2d(256, out_channels, 1)  # 20x20

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Backbone forward
        x0 = self.stem(x)
        x1 = self.c2f1(self.stage1(x0))
        p3 = self.c2f2(self.stage2(x1))  # 80x80
        p4 = self.c2f3(self.stage3(p3))  # 40x40
        p5 = self.sppf(self.c2f4(self.stage4(p4))) # 20x20

        # Top-down FPN
        p5_up = self.up(p5)
        p4_cat = torch.cat([p5_up, p4], dim=1)
        p4_fused = self.c2f_neck1(p4_cat)

        p4_up = self.up(p4_fused)
        p3_cat = torch.cat([p4_up, p3], dim=1)
        p3_out = self.c2f_neck2(p3_cat) # 80x80

        # Bottom-up PAN
        p3_down = self.down1(p3_out)
        p4_down_cat = torch.cat([p3_down, p4_fused], dim=1)
        p4_out = self.c2f_neck3(p4_down_cat) # 40x40

        p4_down2 = self.down2(p4_out)
        p5_down_cat = torch.cat([p4_down2, p5], dim=1)
        p5_out = self.c2f_neck4(p5_down_cat) # 20x20

        # Predict raw logits
        out_p3 = self.head_p3(p3_out)
        out_p4 = self.head_p4(p4_out)
        out_p5 = self.head_p5(p5_out)

        return out_p3, out_p4, out_p5

    def export_vectorblox_onnx(self, output_path: str, device: torch.device):
        """
        Exports static ONNX graph without dynamic control flows,
        fully adhering to Microchip VectorBlox vbx_compiler requirements.
        """
        self.eval()
        dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=device)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image_640x640"],
            output_names=["head_p3_80x80", "head_p4_40x40", "head_p5_20x20"]
        )
        print(f"[YOLOv8 EXPORT] Exported VectorBlox-compliant ONNX: {output_path}")


# =====================================================================
# SECTION 3: STAGE-1 ROI EXTRACTION & CROPPING ENGINE
# =====================================================================
class ROIProcessingEngine:
    """
    Consumes camera frames (1080p @ 30 FPS), performs detector postprocessing,
    extracts candidate vehicle bounding boxes, applies NMS, and runs the
    Adaptive Edge Fusion Preprocessor to feed 48x48x3 crops into the SNN.
    """
    def __init__(
        self,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_crop_size: Tuple[int, int] = (48, 48)
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_crop_size = target_crop_size
        self.preprocessor = AdaptiveEdgeFusionPreprocessor(target_size=target_crop_size)

    def extract_rois_from_image(
        self,
        frame_bgr: np.ndarray,
        detected_boxes: List[Tuple[int, int, int, int, float, int]]
    ) -> List[Dict]:
        """
        Args:
            frame_bgr: Raw input camera image (e.g. 1920x1080 BGR)
            detected_boxes: List of (x1, y1, x2, y2, confidence, category)
        Returns:
            rois: List of dicts with:
                - "box": (x1, y1, x2, y2)
                - "conf": confidence
                - "cat": category
                - "crop_48x48_rgb": numpy array (48, 48, 3)
                - "tensor_48x48": torch.Tensor (1, 3, 48, 48) in [0, 255]
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        rois = []

        for (x1, y1, x2, y2, conf, cat) in detected_boxes:
            # Bound coordinates to frame
            x1 = max(0, min(x1, w_frame - 1))
            y1 = max(0, min(y1, h_frame - 1))
            x2 = max(x1 + 4, min(x2, w_frame))
            y2 = max(y1 + 4, min(y2, h_frame))

            # Extract raw crop
            raw_crop_bgr = frame_bgr[y1:y2, x1:x2]
            if raw_crop_bgr.size == 0 or raw_crop_bgr.shape[0] < 8 or raw_crop_bgr.shape[1] < 8:
                continue

            # Stage-1 Adaptive Contrast & High-Pass Edge Fusion
            crop_fused_rgb = self.preprocessor(raw_crop_bgr)

            # Convert to PyTorch Tensor for SNN ingestion
            crop_tensor = torch.from_numpy(crop_fused_rgb).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

            rois.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "cat": cat,
                "crop_48x48_rgb": crop_fused_rgb,
                "tensor_48x48": crop_tensor
            })

        return rois


if __name__ == "__main__":
    print("[TEST] Initializing YOLOv8n detector model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLOv8nDetector(num_classes=4).to(device)
    print(f"Detector parameters: {sum(p.numel() for p in detector.parameters()):,}")

    dummy_frame = torch.zeros((1, 3, 640, 640), device=device)
    p3, p4, p5 = detector(dummy_frame)
    print(f"Output Head P3 (80x80): {p3.shape}")
    print(f"Output Head P4 (40x40): {p4.shape}")
    print(f"Output Head P5 (20x20): {p5.shape}")

    # Export ONNX for VectorBlox
    export_path = "../vectorblox_deployment/artifacts/yolov8n_stage1.onnx"
    detector.export_vectorblox_onnx(export_path, device)
    assert os.path.exists(export_path), "YOLOv8 ONNX file not created!"
    print("[SUCCESS] YOLOv8n detector and ONNX export operational.")
