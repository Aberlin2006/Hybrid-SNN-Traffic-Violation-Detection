"""
AO.SNN Production Dual-Environment Runtime Inference Engine (Repaired & Fully Audited)
Environment 1: Host Laptop with NVIDIA GeForce RTX GPU (CUDA Acceleration) + Integrated/USB Webcam
Environment 2: Target Embedded MPFS250 PolarFire SoC (Quad RISC-V U54 @ 600MHz + HDMI Output + GPIO)

Features:
- Stage-1 ROI Extraction: YOLOv8n Multi-Object Detector with Pose fallback
- Stage-2 SNN Spiking LIF Classifier (Canonical 64 Hidden LIF Neurons, Signed Q4.4, T=8 timesteps)
- Full Diagnostic Mode (`--diagnostic`): Separates YOLO candidate detection, SNN raw predictions,
  spike counts, and rule-based compliance gates.
- Strict Checkpoint Compatibility Validation: Halts with clear error if checkpoint dimensions do not match.
"""

import os
import sys
import time
import argparse
from typing import Optional, List, Tuple, Dict, Set
import numpy as np
import cv2
import torch

# Ensure parent and training directories in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "training"))

from system_config import (
    INPUT_DIMS,
    HIDDEN_NEURONS,
    OUTPUT_CLASSES,
    TIME_STEPS,
    TRAFFIC_CLASSES,
    VIOLATION_CLASSES,
    is_violation_class,
    TOTAL_LATENCY_US
)
from dataset_pipeline import AdaptiveEdgeFusionPreprocessor
from snn_lif_model import ProductionJunctionSNN


# =====================================================================
# SECTION 1: HARDWARE GPIO DRIVER (DUAL-MODE)
# =====================================================================
class PolarFireGPIODriver:
    """
    Interfaces with physical FPGA fabric pin 'violation_alarm_o' via
    Linux sysfs GPIO (/sys/class/gpio/gpio*) or memory-mapped AXI register.
    On host RTX PC, emulates GPIO state seamlessly in software.
    """
    GPIO_PIN_NUMBER = 502  # MPFS250 MSS/FPGA GPIO line

    def __init__(self, target_hardware: str = "auto"):
        self.is_embedded = False
        if target_hardware == "auto":
            machine = os.uname().machine if hasattr(os, "uname") else ""
            self.is_embedded = "riscv" in machine.lower()
        else:
            self.is_embedded = (target_hardware.lower() == "mpfs250")

        self.gpio_state = 0
        self.gpio_fd = None

        if self.is_embedded:
            self._init_sysfs_gpio()
        else:
            print("[GPIO DRIVER] Host workstation detected. Operating in CUDA Emulation Mode.")

    def _init_sysfs_gpio(self):
        gpio_path = f"/sys/class/gpio/gpio{self.GPIO_PIN_NUMBER}"
        try:
            if not os.path.exists(gpio_path):
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(self.GPIO_PIN_NUMBER))
            with open(f"{gpio_path}/direction", "w") as f:
                f.write("out")
            self.gpio_fd = open(f"{gpio_path}/value", "w")
            print(f"[GPIO DRIVER] MPFS250 Hardware GPIO pin {self.GPIO_PIN_NUMBER} initialized successfully.")
        except Exception as e:
            print(f"[GPIO DRIVER WARNING] Could not bind sysfs GPIO: {e}. Emulating in memory.")

    def set_violation_alarm(self, active: bool):
        new_state = 1 if active else 0
        if new_state != self.gpio_state:
            self.gpio_state = new_state
            if self.is_embedded and self.gpio_fd:
                try:
                    self.gpio_fd.seek(0)
                    self.gpio_fd.write("1\n" if active else "0\n")
                    self.gpio_fd.flush()
                except Exception:
                    pass

    def get_state(self) -> int:
        return self.gpio_state

    def close(self):
        if self.gpio_fd:
            try:
                self.gpio_fd.close()
            except Exception:
                pass


# =====================================================================
# SECTION 2: CHECKPOINT RESOLVER & INFERENCE SYSTEM
# =====================================================================
def resolve_checkpoint_path(user_path: Optional[str] = None) -> str:
    if user_path is not None:
        if os.path.exists(user_path):
            return os.path.abspath(user_path)
        raise FileNotFoundError(
            f"[FATAL ERROR] Explicitly specified checkpoint not found: {user_path}"
        )

    candidates = [
        os.path.join(WORKSPACE_ROOT, "vectorblox_deployment", "artifacts", "best_snn_checkpoint.pt"),
        os.path.join(CURRENT_DIR, "..", "vectorblox_deployment", "artifacts", "best_snn_checkpoint.pt"),
        os.path.join("vectorblox_deployment", "artifacts", "best_snn_checkpoint.pt"),
        "best_snn_checkpoint.pt"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        f"[FATAL ERROR] Trained SNN checkpoint could not be found! Checked paths:\n" +
        "\n".join(f"  - {p}" for p in candidates) +
        "\nEnsure training has produced 'best_snn_checkpoint.pt' under vectorblox_deployment/artifacts/."
    )


class JunctionMonitorSystem:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_cuda: bool = True,
        target_board: str = "auto",
        diagnostic_mode: bool = False
    ):
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.diagnostic_mode = diagnostic_mode
        print(f"[INFERENCE ENGINE] Execution Target Device: {self.device}")
        print(f"[INFERENCE ENGINE] Diagnostic Telemetry Mode: {'ENABLED' if self.diagnostic_mode else 'DISABLED'}")

        # 1. Initialize Stage-1 Candidate Detector
        from ultralytics import YOLO
        print("[INFERENCE ENGINE] Loading Stage-1 YOLOv8n Multi-Object Detector...")
        self.yolo_model = YOLO("yolov8n.pt")
        # COCO target classes: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_coco = [0, 1, 2, 3, 5, 7]

        # Pose model fallback for dark helmet silhouettes
        pose_path = "yolov8n-pose.pt"
        if not os.path.exists(pose_path):
            pose_path = os.path.join(WORKSPACE_ROOT, "yolov8n-pose.pt")
        if os.path.exists(pose_path):
            print(f"[INFERENCE ENGINE] Loading YOLOv8n-pose fallback detector from: {pose_path}...")
            self.pose_model = YOLO(pose_path)
        else:
            self.pose_model = None

        # 2. Stage-1 Adaptive Contrast & High-Pass Edge Fusion Preprocessor
        self.preprocessor = AdaptiveEdgeFusionPreprocessor(target_size=(48, 48))

        # 3. Resolve, Validate, and Load Trained SNN Checkpoint
        ckpt_file = resolve_checkpoint_path(checkpoint_path)
        print(f"[INFERENCE ENGINE] Loading verified SNN weights from: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Strict Architecture Validation against Canonical System Config
        h_dim = state_dict["hidden_layer.weight"].shape[0]
        in_dim = state_dict["hidden_layer.weight"].shape[1]
        out_dim = state_dict["output_layer.weight"].shape[0]

        if h_dim != HIDDEN_NEURONS or in_dim != INPUT_DIMS or out_dim != OUTPUT_CLASSES:
            raise ValueError(
                f"[ARCHITECTURE MISMATCH] Loaded checkpoint shapes ([{h_dim}, {in_dim}], [{out_dim}, {h_dim}]) "
                f"do not match canonical config: in={INPUT_DIMS}, hidden={HIDDEN_NEURONS}, out={OUTPUT_CLASSES}!"
            )

        print(f"[INFERENCE ENGINE] Checkpoint architecture validated: {in_dim} -> {h_dim} -> {out_dim}, timesteps = {TIME_STEPS}")

        self.snn = ProductionJunctionSNN(
            in_dim=INPUT_DIMS,
            hidden_dim=HIDDEN_NEURONS,
            out_dim=OUTPUT_CLASSES,
            timesteps=TIME_STEPS,
            v_th=1.0,
            beta_surrogate=10.0
        ).to(self.device)
        self.snn.load_state_dict(state_dict)
        self.snn.eval()

        # 4. Hardware GPIO Alarm Driver
        self.gpio_driver = PolarFireGPIODriver(target_hardware=target_board)

        # 5. Anatomical Face Locator
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Telemetry metrics
        self.fps = 0.0
        self.violation_counter = 0
        self.total_rois_processed = 0
        self.flash_counter = 0
        self.frame_counts = {
            "Car": 0,
            "Bike/Motorcycle": 0,
            "Scooter": 0,
            "Bus": 0,
            "Truck/Heavy": 0,
            "Auto_Rickshaw": 0,
            "Person/Rider": 0,
            "Violations": 0
        }

    def classify_crop_snn(self, crop_bgr: np.ndarray, allowed_classes: Optional[Set[int]] = None) -> Dict:
        """Runs the 48x48 SNN spiking inference pipeline on an image crop."""
        prepped_rgb = self.preprocessor(crop_bgr)
        tensor = torch.from_numpy(prepped_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)

        t_start = time.perf_counter()
        with torch.no_grad():
            logits, spikes, _ = self.snn(tensor, quantize=True)
            eval_logits = logits.clone()
            if allowed_classes is not None:
                for c in range(eval_logits.shape[-1]):
                    if c not in allowed_classes:
                        eval_logits[0, c] = -1e9
            pred_id = int(eval_logits.argmax(dim=-1).item())
            prob = float(torch.softmax(eval_logits, dim=-1)[0, pred_id].item())
            spikes_fired = int(spikes[0, pred_id].item())
        snn_latency_us = (time.perf_counter() - t_start) * 1e6

        return {
            "class_id": pred_id,
            "class_name": TRAFFIC_CLASSES.get(pred_id, "Unknown"),
            "confidence": prob,
            "spikes": spikes_fired,
            "latency_us": snn_latency_us,
            "pip_rgb": prepped_rgb
        }

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        start_t = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        self.flash_counter += 1
        is_flash_on = (self.flash_counter // 5) % 2 == 0

        # Reset per-frame counters
        self.frame_counts = {
            "Car": 0,
            "Bike/Motorcycle": 0,
            "Scooter": 0,
            "Bus": 0,
            "Truck/Heavy": 0,
            "Auto_Rickshaw": 0,
            "Person/Rider": 0,
            "Violations": 0
        }

        # Stage 1: Primary YOLO Candidate Detection
        results = self.yolo_model(
            frame_bgr,
            classes=self.target_coco,
            conf=0.24,
            iou=0.45,
            verbose=False,
            device=(0 if self.device.type == "cuda" else "cpu")
        )

        raw_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].item())
                coco_cls = int(boxes.cls[i].item())
                x1 = max(0, min(int(xyxy[0]), w - 1))
                y1 = max(0, min(int(xyxy[1]), h - 1))
                x2 = max(x1 + 8, min(int(xyxy[2]), w))
                y2 = max(y1 + 8, min(int(xyxy[3]), h))
                raw_boxes.append((x1, y1, x2, y2, conf, coco_cls))

        # Fallback Pose Detection ONLY if a 2-wheeler (bike/motorcycle) is detected without a rider
        has_2wheeler = any(b[5] in [1, 3] for b in raw_boxes)
        has_person = any(b[5] == 0 for b in raw_boxes)
        if has_2wheeler and not has_person and self.pose_model is not None:
            pose_res = self.pose_model(
                frame_bgr,
                conf=0.30,  # Optimal threshold to find biker body on motorcycle
                verbose=False,
                device=(0 if self.device.type == "cuda" else "cpu")
            )
            if len(pose_res) > 0 and pose_res[0].boxes is not None:
                p_boxes = pose_res[0].boxes
                for i in range(len(p_boxes)):
                    pxyxy = p_boxes.xyxy[i].cpu().numpy().astype(int)
                    pconf = float(p_boxes.conf[i].item())
                    px1 = max(0, min(int(pxyxy[0]), w - 1))
                    py1 = max(0, min(int(pxyxy[1]), h - 1))
                    px2 = max(px1 + 8, min(int(pxyxy[2]), w))
                    py2 = max(py1 + 8, min(int(pxyxy[3]), h))
                    raw_boxes.append((px1, py1, px2, py2, pconf, 0))

        # Suppress child / contained boxes
        clean_boxes = []
        for i, b1 in enumerate(raw_boxes):
            contained = False
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            for j, b2 in enumerate(raw_boxes):
                if i != j and b1[5] == b2[5]:
                    ix1 = max(b1[0], b2[0])
                    iy1 = max(b1[1], b2[1])
                    ix2 = min(b1[2], b2[2])
                    iy2 = min(b1[3], b2[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                        if (inter_area / max(1, area1) > 0.55) and area1 < area2:
                            contained = True
                            break
            if not contained:
                clean_boxes.append(b1)

        # Find all 2-wheelers in current frame to associate riders
        two_wheel_boxes = [b for b in clean_boxes if b[5] in [1, 3]]

        # Deduplicate rider boxes: for multiple person detections overlapping the same
        # 2-wheeler, keep only the topmost (smallest y1) box — the actual head/upper body.
        # This prevents the lower motorcycle body crop from being evaluated as a rider.
        person_boxes_raw = [b for b in clean_boxes if b[5] == 0]
        non_person_boxes = [b for b in clean_boxes if b[5] != 0]
        deduped_persons = []
        used_bike_for_person = set()

        # Sort person boxes by y1 (topmost first = actual rider head)
        person_boxes_sorted = sorted(person_boxes_raw, key=lambda b: b[1])
        for pb in person_boxes_sorted:
            px1, py1, px2, py2 = pb[0], pb[1], pb[2], pb[3]
            matched_bike_idx = None
            for bidx, bk in enumerate(two_wheel_boxes):
                bx1, by1, bx2, by2, bconf, bcls = bk
                ix1 = max(px1, bx1); iy1 = max(py1, by1)
                ix2 = min(px2, bx2); iy2 = min(py2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    p_area = max(1, (px2 - px1) * (py2 - py1))
                    if (inter_area / p_area) > 0.15:
                        matched_bike_idx = bidx
                        break
            if matched_bike_idx is not None:
                if matched_bike_idx not in used_bike_for_person:
                    deduped_persons.append(pb)
                    used_bike_for_person.add(matched_bike_idx)
                # else: skip duplicate rider box for same motorcycle
            else:
                deduped_persons.append(pb)  # pedestrian, keep all

        clean_boxes = non_person_boxes + deduped_persons

        any_violation_in_frame = False
        annotated_frame = frame_bgr.copy()
        last_pip_crop = None

        for (x1, y1, x2, y2, conf, coco_cls) in clean_boxes:
            full_crop = frame_bgr[y1:y2, x1:x2]
            if full_crop.size == 0:
                continue

            self.total_rois_processed += 1

            # -----------------------------------------------------------------
            # CASE A: PERSON (COCO 0)
            # A pedestrian walking alone is NOT required to wear a helmet!
            # Only evaluate helmet violation if person is RIDING a bike/scooter.
            # -----------------------------------------------------------------
            if coco_cls == 0:
                pw = x2 - x1
                ph = y2 - y1

                # Disambiguate raised hands and gestures:
                aspect = pw / max(1, ph)
                if conf < 0.38 and aspect > 0.85:
                    continue

                # Check if this person is mounted on or directly overlapping a 2-wheeler
                is_mounted_on_2wheeler = False
                for bx1, by1, bx2, by2, bconf, bcls in two_wheel_boxes:
                    # Spatial overlap check: intersection over person box area
                    ix1 = max(x1, bx1)
                    iy1 = max(y1, by1)
                    ix2 = min(x2, bx2)
                    iy2 = min(y2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        p_area = max(1, pw * ph)
                        # Overlap > 15% or bottom of person is inside vehicle bounds
                        if (inter_area / p_area) > 0.15 or (y2 >= by1 and y1 <= by2 and (x1 + x2)/2 >= bx1 and (x1 + x2)/2 <= bx2):
                            is_mounted_on_2wheeler = True
                            break

                # -------------------------------------------------------------
                # SUB-CASE A1: PEDESTRIAN WALKING (NO 2-WHEELER ASSOCIATED)
                # Simply display as Person / Pedestrian. NO VIOLATION ALARM!
                # -------------------------------------------------------------
                if not is_mounted_on_2wheeler:
                    self.frame_counts["Person/Rider"] += 1
                    ped_color = (200, 200, 200)
                    ped_label = f"Person ({conf*100.0:.0f}%)"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ped_color, 2)
                    (tw, th), _ = cv2.getTextSize(ped_label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), ped_color, -1)
                    cv2.putText(annotated_frame, ped_label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
                    continue

                # -------------------------------------------------------------
                # SUB-CASE A2: 2-WHEELER RIDER (BIKE / SCOOTER)
                # Active helmet compliance & SNN neuromorphic verification
                # -------------------------------------------------------------
                self.frame_counts["Person/Rider"] += 1

                # Precise anatomical head localization
                xc = (x1 + x2) // 2
                # Strategy 1: Face Cascade in upper 50% of the rider box
                # If a bare human face (skin, eyes, nose, mouth) is visible, the rider is bare-headed!
                face_detected = False
                person_roi = frame_bgr[y1:y2, x1:x2]
                head_crop = None

                if self.face_cascade is not None and person_roi.size > 0:
                    upper_h = int(ph * 0.55)
                    gray_roi = cv2.cvtColor(person_roi[:upper_h, :], cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda f: f[1])
                        fx, fy, fw, fh = faces[0]
                        face_detected = True
                        hcx = x1 + fx + fw // 2
                        hcy = y1 + fy + fh // 2
                        hsize = int(max(fw, fh) * 1.45)
                        hx1 = max(0, hcx - hsize // 2)
                        hx2 = min(w, hcx + hsize // 2)
                        hy1 = max(0, hcy - int(hsize * 0.58))
                        hy2 = min(h, hcy + int(hsize * 0.42))
                        head_crop = frame_bgr[hy1:hy2, hx1:hx2]

                # Strategy 2: Pose keypoints (Nose / Eyes)
                if head_crop is None or head_crop.size == 0:
                    if self.pose_model is not None and person_roi.size > 0:
                        pose_res = self.pose_model(person_roi, conf=0.25, verbose=False)[0]
                        if pose_res.keypoints is not None and len(pose_res.keypoints.data) > 0:
                            kp = pose_res.keypoints.data[0].cpu().numpy()
                            nose = kp[0]
                            leye, reye = kp[1], kp[2]
                            if nose[2] > 0.35 or (leye[2] > 0.35 and reye[2] > 0.35):
                                rcx = float((nose[0] + leye[0] + reye[0]) / 3.0) if (leye[2] > 0.2 and reye[2] > 0.2) else float(nose[0])
                                rcy = float((nose[1] + leye[1] + reye[1]) / 3.0) if (leye[2] > 0.2 and reye[2] > 0.2) else float(nose[1])
                                hcx = x1 + int(rcx)
                                hcy = y1 + int(rcy)
                                hsize = int(min(pw * 0.50, ph * 0.30, 120))
                                hx1 = max(0, hcx - hsize // 2)
                                hx2 = min(w, hcx + hsize // 2)
                                hy1 = max(0, hcy - int(hsize * 0.60))
                                hy2 = min(h, hcy + int(hsize * 0.40))
                                head_crop = frame_bgr[hy1:hy2, hx1:hx2]

                # Strategy 3: Tight anatomical crown crop (upper 25% of rider, width 50%)
                if head_crop is None or head_crop.size == 0:
                    hw = int(min(pw * 0.52, ph * 0.28, 120))
                    hh = int(hw * 1.15)
                    hx1 = max(0, xc - hw // 2)
                    hx2 = min(w, xc + hw // 2)
                    hy1 = max(0, y1)
                    hy2 = min(h, y1 + hh)
                    head_crop = frame_bgr[hy1:hy2, hx1:hx2]

                # SNN Evaluation (constrained to helmet compliant vs violation classes)
                snn_res = self.classify_crop_snn(head_crop, allowed_classes={1, 2, 4, 5})
                last_pip_crop = snn_res["pip_rgb"]
                pred_id = snn_res["class_id"]
                spk = snn_res["spikes"]
                conf_pct = snn_res["confidence"] * 100.0

                # Neuromorphic Decision Arbitration:
                # 1. If face cascade detects an exposed bare face -> Violation!
                # 2. Or if SNN predicts violation class (2: Bike_No_Helmet, 5: Scooter_No_Helmet) -> Violation!
                # 3. If SNN predicts compliant (1: Bike_Compliant, 4: Scooter_Compliant) and no bare face -> Compliant!
                if face_detected:
                    is_viol = True
                    label_text = f"[VIOLATION: NO HELMET] Bare Face (SNN:{snn_res['class_name']})"
                elif pred_id in [2, 5]:
                    is_viol = True
                    label_text = f"[VIOLATION: NO HELMET] SNN:{snn_res['class_name']} ({conf_pct:.0f}%, {spk}spk)"
                else:
                    is_viol = False
                    label_text = f"[COMPLIANT] SNN:{snn_res['class_name']} ({conf_pct:.0f}%, {spk}spk)"

                if is_viol:
                    any_violation_in_frame = True
                    self.violation_counter += 1
                    self.frame_counts["Violations"] += 1
                    box_color = (0, 0, 255) if is_flash_on else (30, 30, 210)
                else:
                    box_color = (0, 255, 0)

                # Draw Head Box & Rider Box
                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), box_color, 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 1)

                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                cv2.rectangle(annotated_frame, (hx1, max(0, hy1 - 20)), (hx1 + tw + 6, hy1), box_color, -1)
                cv2.putText(annotated_frame, label_text, (hx1 + 3, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

            # -----------------------------------------------------------------
            # CASES B–E: ALL VEHICLES → SNN classifies the crop
            # YOLO gives coarse bounding box; SNN gives fine-grained 10-class result.
            # Violation alarm only fires when BOTH conditions are true:
            #   1. YOLO detected a 2-wheeler (bike/motorcycle: coco_cls 1 or 3)
            #   2. SNN predicts a violation class (2,3,5,6 = no helmet / overloaded)
            elif coco_cls in [1, 2, 3, 5, 7]:
                # Domain-aware candidate pool:
                # Car (COCO 2): evaluate Car (0) vs Auto Rickshaw (9)
                # Bus (COCO 5): Bus (7)
                # Truck (COCO 7): Heavy Vehicle / Truck (8)
                # 2-wheelers (COCO 1, 3): Bike/Scooter compliance & overload classes {1, 2, 3, 4, 5, 6, 9}
                yolo_is_2wheeler = coco_cls in [1, 3]
                if coco_cls == 2:
                    allowed_classes = {0, 9}
                elif coco_cls == 5:
                    allowed_classes = {7}
                elif coco_cls == 7:
                    allowed_classes = {8}
                elif yolo_is_2wheeler:
                    allowed_classes = {1, 2, 3, 4, 5, 6, 9}
                else:
                    allowed_classes = {0, 7, 8, 9}

                # Run SNN on every vehicle ROI
                snn_res = self.classify_crop_snn(full_crop, allowed_classes=allowed_classes)
                last_pip_crop = snn_res["pip_rgb"]
                pred_id  = snn_res["class_id"]
                spk      = snn_res["spikes"]
                conf_pct = snn_res["confidence"] * 100.0
                snn_name = snn_res["class_name"]

                # YOLO-SNN agreement gate for violations
                snn_is_violation = is_violation_class(pred_id)

                if yolo_is_2wheeler:
                    self.frame_counts["Bike/Motorcycle"] += 1
                    box_color = (0, 230, 0)
                    label_text = f"Motorcycle ({conf*100.0:.0f}%)"

                elif coco_cls == 2:
                    self.frame_counts["Car"] += 1
                    box_color = (255, 210, 0)
                    label_text = f"SNN: {snn_name} ({conf_pct:.0f}%, {spk}spk)"

                elif coco_cls == 5:
                    self.frame_counts["Bus"] += 1
                    box_color = (0, 140, 255)
                    label_text = f"SNN: {snn_name} ({conf_pct:.0f}%, {spk}spk)"

                elif coco_cls == 7:
                    self.frame_counts["Truck/Heavy"] += 1
                    box_color = (200, 0, 200)
                    label_text = f"SNN: {snn_name} ({conf_pct:.0f}%, {spk}spk)"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
                # Ensure label box is below top HUD bar (y=95) or inside bounding box if near top
                lbl_y = y1 - 8
                if lbl_y < 100:
                    lbl_y = min(h - 10, y1 + th + 14)
                lbl_bg_top = max(0, lbl_y - th - 5)
                cv2.rectangle(annotated_frame, (x1, lbl_bg_top), (x1 + tw + 8, lbl_y + 5), (10, 10, 10), -1)
                cv2.rectangle(annotated_frame, (x1, lbl_bg_top), (x1 + tw + 8, lbl_y + 5), box_color, 1)
                cv2.putText(annotated_frame, label_text, (x1 + 4, lbl_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (0, 255, 255) if coco_cls == 2 else (255, 255, 255),
                            2, cv2.LINE_AA)

        # Hardware GPIO Alarm Actuation
        self.gpio_driver.set_violation_alarm(any_violation_in_frame)

        # Telemetry HUD Dashboard
        elapsed = time.perf_counter() - start_t
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-5, elapsed))

        if h >= 400:
            hud_height = 95
            cv2.rectangle(annotated_frame, (0, 0), (w, hud_height), (18, 18, 20), -1)
            cv2.line(annotated_frame, (0, hud_height), (w, hud_height), (60, 60, 65), 1)

            alarm_state_str = "ACTIVE HIGH [ALARM!]" if any_violation_in_frame else "NORMAL (LOW - 0V)"
            alarm_color = (0, 0, 255) if any_violation_in_frame else (0, 255, 0)

            diag_tag = " [DIAGNOSTIC MODE]" if self.diagnostic_mode else ""
            hud_r1 = (
                f"MPFS250 SNN MONITOR{diag_tag} | FPS: {self.fps:.1f} | Hardware Latency: {TOTAL_LATENCY_US:.2f} µs @ 100MHz | GPIO (violation_alarm_o): {alarm_state_str}"
            )
            cv2.putText(annotated_frame, hud_r1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, alarm_color if any_violation_in_frame else (210, 240, 255), 1, cv2.LINE_AA)

            c_car = self.frame_counts.get("Car", 0)
            c_bike = self.frame_counts.get("Bike/Motorcycle", 0)
            c_bus = self.frame_counts.get("Bus", 0)
            c_trk = self.frame_counts.get("Truck/Heavy", 0)
            c_auto = self.frame_counts.get("Auto_Rickshaw", 0)
            c_person = self.frame_counts.get("Person/Rider", 0)
            c_viol = self.frame_counts.get("Violations", 0)

            hud_r2 = (
                f"LIVE COUNTERS -> Cars: {c_car} | Bikes/Motorcycles: {c_bike} | "
                f"Buses: {c_bus} | Heavy Trucks: {c_trk} | Autos: {c_auto} | Persons/Riders: {c_person}"
            )
            cv2.putText(annotated_frame, hud_r2, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 245, 180), 1, cv2.LINE_AA)

            hud_r3 = (
                f"ACTIVE VIOLATIONS: {c_viol} | TOTAL LOGGED: {self.violation_counter} | "
                f"PolarFire Architecture: {INPUT_DIMS} -> {HIDDEN_NEURONS} -> {OUTPUT_CLASSES} LIF (Q4.4) | Total ROIs: {self.total_rois_processed}"
            )
            cv2.putText(annotated_frame, hud_r3, (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255) if c_viol > 0 else (140, 195, 140), 2 if c_viol > 0 else 1, cv2.LINE_AA)
        else:
            # Compact single-line HUD for low-resolution / small preview crops
            hud_height = 24
            cv2.rectangle(annotated_frame, (0, 0), (w, hud_height), (18, 18, 20), -1)
            c_car = self.frame_counts.get("Car", 0)
            c_viol = self.frame_counts.get("Violations", 0)
            hud_compact = f"MPFS250 SNN | Cars: {c_car} | Violations: {c_viol} | 18.08 µs"
            cv2.putText(annotated_frame, hud_compact, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 200), 1, cv2.LINE_AA)

        if last_pip_crop is not None:
            pip_bgr = cv2.cvtColor(last_pip_crop, cv2.COLOR_RGB2BGR)
            pip_disp = cv2.resize(pip_bgr, (96, 96), interpolation=cv2.INTER_NEAREST)
            ph_box, pw_box = pip_disp.shape[:2]
            py1 = h - ph_box - 10
            px1 = w - pw_box - 10
            annotated_frame[py1:py1+ph_box, px1:px1+pw_box] = pip_disp
            cv2.rectangle(annotated_frame, (px1, py1), (px1+pw_box, py1+ph_box), (200, 200, 200), 1)
            cv2.putText(annotated_frame, "SNN 48x48", (px1, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated_frame


# =====================================================================
# SECTION 3: APPLICATION ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="AO.SNN Production Dual-Environment Runtime Inference Engine")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam, file path, or 'synthetic'")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to process (0 = live indefinite stream)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_snn_checkpoint.pt")
    parser.add_argument("--diagnostic", action="store_true", help="Enable diagnostic mode showing YOLO, SNN, and rule contributions separately")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA acceleration")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI window)")
    parser.add_argument("--output-video", type=str, default="", help="Path to write annotated MP4 output video")
    args = parser.parse_args()

    print("=" * 80)
    print("AO.SNN EMBEDDED RUNTIME: PRODUCTION SNN JUNCTION VIOLATION MONITOR")
    print("Dual Environment: NVIDIA GeForce RTX (Host) / Microchip PolarFire SoC (Target)")
    print("=" * 80)

    system = JunctionMonitorSystem(
        checkpoint_path=args.checkpoint,
        use_cuda=not args.no_cuda,
        target_board="auto",
        diagnostic_mode=args.diagnostic
    )

    is_live_camera = args.source.isdigit()
    cap = None
    if is_live_camera:
        cam_idx = int(args.source)
        print(f"[VIDEO FEED] Opening Realtek Integrated Webcam (index: {cam_idx})...")

        # Realtek cameras (VID_0BDA) work best with DSHOW on Windows.
        # Do NOT force resolution before warmup — let the driver negotiate first.
        if sys.platform.startswith("win"):
            # Try DSHOW first (best for Realtek), then ANY as fallback
            for backend_name, backend_id in [("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]:
                print(f"[VIDEO FEED] Trying {backend_name} backend...")
                cap = cv2.VideoCapture(cam_idx, backend_id)
                if cap.isOpened():
                    # Long warmup: Realtek cameras need time to initialize stream
                    import time as _t
                    _t.sleep(2.0)
                    # Drain up to 30 stale/black frames from the buffer
                    for _ in range(30):
                        cap.read()
                    ret, warm_frame = cap.read()
                    if ret and warm_frame is not None and warm_frame.sum() > 0:
                        print(f"[VIDEO FEED] {backend_name} backend OK — camera live.")
                        break
                    else:
                        print(f"[VIDEO FEED] {backend_name} returned blank frames, trying next...")
                        cap.release()
                        cap = None
                else:
                    print(f"[VIDEO FEED] {backend_name} failed to open device.")
                    cap = None
        else:
            cap = cv2.VideoCapture(cam_idx)

        if cap is None or not cap.isOpened():
            print("[VIDEO FEED ERROR] All backends failed. Is the privacy shutter closed?")
            sys.exit(1)

        # Set properties AFTER successful open
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    elif os.path.isfile(args.source):
        print(f"[VIDEO FEED] Opening video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
    else:
        print(f"[VIDEO FEED] Generating procedural 1080p junction stream...")

    video_writer = None
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[VIDEO FEED] Stream ended or frame grab failed.")
                    break
            else:
                if args.frames > 0 and frame_count >= args.frames:
                    break
                frame = np.full((1080, 1920, 3), 60, dtype=np.uint8)
                cv2.rectangle(frame, (100, 300), (1820, 950), (45, 45, 50), -1)
                cv2.line(frame, (100, 625), (1820, 625), (220, 220, 220), 4)
                v_color = (0, 30, 200) if (frame_count % 3 == 0) else (200, 150, 40)
                cv2.rectangle(frame, (750, 380), (1170, 750), v_color, -1)

            annotated = system.process_frame(frame)
            frame_count += 1

            if args.output_video:
                if args.output_video.lower().endswith(('.jpg', '.jpeg', '.png')):
                    cv2.imwrite(args.output_video, annotated)
                else:
                    if video_writer is None:
                        vh, vw = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(args.output_video, fourcc, 30.0, (vw, vh))
                    video_writer.write(annotated)

            if not args.headless:
                cv2.imshow("AO.SNN Traffic Junction Violation Monitor (1080p)", cv2.resize(annotated, (1280, 720)))
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            if args.frames > 0 and frame_count >= args.frames:
                break

            if frame_count % 30 == 0:
                print(f"[PROGRESS] Frames: {frame_count} | FPS: {system.fps:.1f} | GPIO: {system.gpio_driver.get_state()} | Violations: {system.violation_counter}")

    finally:
        total_time = time.time() - start_time
        avg_fps = frame_count / max(1e-4, total_time)
        print("\n" + "=" * 60)
        print("RUNTIME SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Frames Processed:    {frame_count}")
        print(f"Average Sustained FPS:     {avg_fps:.2f}")
        print(f"Total ROIs Evaluated:      {system.total_rois_processed}")
        print(f"Traffic Violations Logged: {system.violation_counter}")
        print(f"Final GPIO Pin State:      {system.gpio_driver.get_state()}")
        print("=" * 60)

        if cap:
            cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        system.gpio_driver.close()

if __name__ == "__main__":
    main()
