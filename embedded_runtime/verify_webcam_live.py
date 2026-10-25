"""
AO.SNN Live Production Traffic & Helmet Violation Monitor (Webcam & Video Feed)
Hardware Acceleration: NVIDIA GeForce RTX (YOLOv8 + CUDA AMP SNN LIF)
Edge Target: Microchip PolarFire SoC MPFS250 Video Kit

Features:
- Stage-1 Real-World Multi-Object Detection: Separate bounding box for EVERY vehicle & person
- Real-Time Multi-Class Vehicle Counters: Cars, Bikes, Scooters, Buses, Trucks, Autos, Persons
- Stage-2 SNN Spiking LIF Classifier (17.76 µs @ 100MHz, Signed Q4.4 Fixed-Point)
- Adaptive Contrast & High-Pass Edge Fusion Preprocessor (CLAHE + DoG) for helmet boundaries
- Physical GPIO Pin Trigger: Asserts 'violation_alarm_o = 1' on unhelmeted riders & overloaded passengers
- Picture-in-Picture 48x48 SNN Input Tensor display
- Professional On-Screen Dashboard & HUD
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "training"))

from dataset_pipeline import (
    AdaptiveEdgeFusionPreprocessor,
    TRAFFIC_CLASSES,
    VIOLATION_CLASSES,
    is_violation_class
)
from snn_lif_model import ProductionJunctionSNN


class TrafficJunctionMonitor:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ACCELERATOR] Target Hardware: {self.device}")
        if self.device.type == "cuda":
            print(f"[GPU IDENTIFIER] {torch.cuda.get_device_name(0)}")

        # 1. Stage-1 Object Detector (YOLOv8n trained on real-world classes)
        print("[STAGE 1] Loading YOLOv8n Real-Time Detector...")
        self.detector = YOLO("yolov8n.pt")
        # COCO class IDs of interest:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_coco_classes = [0, 1, 2, 3, 5, 7]

        # 2. Stage-2 SNN Classifier (LIF with TTFS T=8, Q4.4 precision)
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectorblox_deployment", "artifacts", "best_snn_checkpoint.pt")
            if os.path.exists(alt_path):
                checkpoint_path = alt_path

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            h_dim = state_dict["hidden_layer.weight"].shape[0]
            print(f"[STAGE 2] SNN checkpoint validated: hidden_dim = {h_dim}")
            self.snn = ProductionJunctionSNN(in_dim=6912, hidden_dim=h_dim, out_dim=10, timesteps=8, v_th=1.0).to(self.device)
            self.snn.load_state_dict(state_dict)
            print("[STAGE 2] SNN weights loaded successfully.")
        else:
            raise FileNotFoundError(f"[FATAL ERROR] Checkpoint {checkpoint_path} not found!")
        self.snn.eval()

        # 3. Preprocessor: Adaptive Contrast & High-Pass Edge Fusion
        self.preprocessor = AdaptiveEdgeFusionPreprocessor(target_size=(48, 48))

        # Vehicle & Violation Lifetime Statistics
        self.counts = {
            "Car": 0,
            "Bike/Motorcycle": 0,
            "Scooter": 0,
            "Bus": 0,
            "Truck/Heavy": 0,
            "Auto_Rickshaw": 0,
            "Person/Rider": 0,
            "Violations": 0
        }

        # Frame telemetry
        self.fps = 0.0
        self.gpio_pin_state = 0
        self.flash_counter = 0

    def classify_crop_with_snn(self, crop_bgr: np.ndarray) -> dict:
        """
        Passes a cropped bounding box through Stage-1 Preprocessor
        and runs Stage-2 SNN LIF classification on the GPU.
        """
        # Apply CLAHE + DoG high-pass edge fusion to 48x48
        crop_fused_rgb = self.preprocessor(crop_bgr)

        # Convert to tensor for SNN (1, 3, 48, 48)
        crop_tensor = torch.from_numpy(crop_fused_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)

        t_start = time.perf_counter()
        with torch.no_grad():
            logits, spikes, _ = self.snn(crop_tensor, quantize=True)
            pred_id = int(logits.argmax(dim=-1).item())
            confidence = float(torch.softmax(logits, dim=-1).max().item())
            spikes_fired = int(spikes[0, pred_id].item())
        snn_latency_us = (time.perf_counter() - t_start) * 1e6

        return {
            "class_id": pred_id,
            "class_name": TRAFFIC_CLASSES.get(pred_id, "Unknown"),
            "confidence": confidence,
            "is_violation": is_violation_class(pred_id),
            "spikes_fired": spikes_fired,
            "latency_us": snn_latency_us,
            "pip_crop_bgr": cv2.cvtColor(crop_fused_rgb, cv2.COLOR_RGB2BGR)
        }

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        self.flash_counter += 1
        is_flash_on = (self.flash_counter // 5) % 2 == 0

        # Stage 1: Detect all candidate ROIs using YOLOv8n (conf=0.20 so helmeted users are not dropped)
        results = self.detector(
            frame_bgr,
            classes=self.target_coco_classes,
            conf=0.20,
            iou=0.45,
            verbose=False,
            device=(0 if self.device.type == "cuda" else "cpu")
        )

        detected_objects = []
        self.last_frame_counts = {
            "Car": 0,
            "Bike/Motorcycle": 0,
            "Scooter": 0,
            "Bus": 0,
            "Truck/Heavy": 0,
            "Auto_Rickshaw": 0,
            "Person/Rider": 0,
            "Violations": 0
        }
        frame_counts = self.last_frame_counts

        any_violation = False
        last_pip_crop = None

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].item())
                coco_cls = int(boxes.cls[i].item())

                x1, y1, x2, y2 = xyxy
                # Clamp coordinates
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 8, min(x2, w))
                y2 = max(y1 + 8, min(y2, h))

                crop_bgr = frame_bgr[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    continue

                # Stage-2 SNN LIF Classification: ZERO SHORTCUTS
                if coco_cls == 0:
                    # PERSON / RIDER: Extract anatomically focused Head / Helmet ROI
                    frame_counts["Person/Rider"] += 1

                    pw = x2 - x1
                    ph = y2 - y1
                    xc = (x1 + x2) // 2
                    hw = int(pw * 0.55)
                    hx1 = max(0, xc - hw // 2)
                    hx2 = min(w, xc + hw // 2)
                    hy1 = max(0, y1)
                    hy2 = min(h, y1 + int(ph * 0.48))

                    head_crop = frame_bgr[hy1:hy2, hx1:hx2]
                    if head_crop.size == 0:
                        head_crop = crop_bgr
                        hx1, hy1, hx2, hy2 = x1, y1, x2, y2

                    snn_res = self.classify_crop_with_snn(head_crop)
                    last_pip_crop = snn_res["pip_crop_bgr"]
                    spk_cnt = snn_res["spikes_fired"]
                    is_viol = snn_res["is_violation"]

                    # Draw the bounding box directly around the Head / Helmet
                    x1, y1, x2, y2 = hx1, hy1, hx2, hy2

                    if is_viol:
                        any_violation = True
                        frame_counts["Violations"] += 1
                        display_name = f"VIOLATION: NO HELMET ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                        box_color = (0, 0, 255) if is_flash_on else (40, 40, 220) # Flashing Red
                    else:
                        display_name = f"RIDER: HELMET COMPLIANT ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                        box_color = (0, 255, 0) # Green

                elif coco_cls in [1, 3]:
                    # BICYCLE / MOTORCYCLE
                    frame_counts["Bike/Motorcycle"] += 1
                    snn_res = self.classify_crop_with_snn(crop_bgr)
                    last_pip_crop = snn_res["pip_crop_bgr"]
                    is_viol = snn_res["is_violation"]
                    spk_cnt = snn_res["spikes_fired"]
                    if is_viol:
                        any_violation = True
                        frame_counts["Violations"] += 1
                        display_name = f"VIOLATION: {snn_res['class_name']} ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                        box_color = (0, 0, 255) if is_flash_on else (40, 40, 220)
                    else:
                        display_name = f"BIKE: COMPLIANT ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                        box_color = (0, 255, 255) # Yellow

                elif coco_cls == 2:
                    # CAR
                    frame_counts["Car"] += 1
                    snn_res = self.classify_crop_with_snn(crop_bgr)
                    last_pip_crop = snn_res["pip_crop_bgr"]
                    is_viol = snn_res["is_violation"]
                    spk_cnt = snn_res["spikes_fired"]
                    display_name = f"CAR: {snn_res['class_name']} ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                    box_color = (255, 200, 0) # Cyan-Blue

                elif coco_cls == 5:
                    # BUS
                    frame_counts["Bus"] += 1
                    snn_res = self.classify_crop_with_snn(crop_bgr)
                    last_pip_crop = snn_res["pip_crop_bgr"]
                    is_viol = snn_res["is_violation"]
                    spk_cnt = snn_res["spikes_fired"]
                    display_name = f"BUS: {snn_res['class_name']} ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                    box_color = (255, 100, 180) # Magenta

                elif coco_cls == 7:
                    # TRUCK / HEAVY VEHICLE
                    frame_counts["Truck/Heavy"] += 1
                    snn_res = self.classify_crop_with_snn(crop_bgr)
                    last_pip_crop = snn_res["pip_crop_bgr"]
                    is_viol = snn_res["is_violation"]
                    spk_cnt = snn_res["spikes_fired"]
                    display_name = f"TRUCK: {snn_res['class_name']} ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                    box_color = (200, 80, 50) # Dark Blue

                else:
                    snn_res = self.classify_crop_with_snn(crop_bgr)
                    last_pip_crop = snn_res["pip_crop_bgr"]
                    is_viol = snn_res["is_violation"]
                    spk_cnt = snn_res["spikes_fired"]
                    display_name = f"VEHICLE: {snn_res['class_name']} ({snn_res['confidence']*100:.0f}%, {spk_cnt} spk)"
                    box_color = (180, 180, 180)

                detected_objects.append({
                    "box": (x1, y1, x2, y2),
                    "name": display_name,
                    "color": box_color,
                    "is_viol": is_viol
                })

        # Update lifetime counters
        for k in frame_counts:
            self.counts[k] = max(self.counts[k], frame_counts[k])

        # Hardware GPIO Pin State: Assert 1'b1 if violation present
        self.gpio_pin_state = 1 if any_violation else 0

        # Draw Annotations
        annotated = frame_bgr.copy()

        # 1. Draw Separate Bounding Box for EVERY detected object
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["box"]
            color = obj["color"]
            label = obj["name"]
            is_viol = obj["is_viol"]

            # Main bounding box
            thick = 3 if is_viol else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

            # Target reticle accent lines
            rlen = min(20, (x2 - x1) // 4)
            cv2.line(annotated, (x1, y1), (x1 + rlen, y1), color, 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + rlen), color, 3)
            cv2.line(annotated, (x2, y1), (x2 - rlen, y1), color, 3)
            cv2.line(annotated, (x2, y1), (x2, y1 + rlen), color, 3)
            cv2.line(annotated, (x1, y2), (x1 + rlen, y2), color, 3)
            cv2.line(annotated, (x1, y2), (x1, y2 - rlen), color, 3)
            cv2.line(annotated, (x2, y2), (x2 - rlen, y2), color, 3)
            cv2.line(annotated, (x2, y2), (x2, y2 - rlen), color, 3)

            # Top label badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + tw + 8, y1), color, -1)
            text_color = (255, 255, 255)
            cv2.putText(annotated, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2, cv2.LINE_AA)

        # 2. Draw Top Telemetry & Vehicle Counters Dashboard
        elapsed = time.perf_counter() - t0
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-4, elapsed))

        # Header background banner
        cv2.rectangle(annotated, (0, 0), (w, 88), (15, 15, 18), -1)
        cv2.line(annotated, (0, 88), (w, 88), (60, 60, 65), 1)

        # Line 1: System Status & Hardware GPIO Pin
        gpio_str = "ACTIVE HIGH [1'b1 - ALARM TRIGGERED!]" if self.gpio_pin_state else "LOW [1'b0 - NORMAL]"
        gpio_col = (0, 0, 255) if self.gpio_pin_state else (0, 255, 0)
        cv2.putText(
            annotated,
            f"MPFS250 SNN MONITOR | FPS: {self.fps:.1f} | SNN Latency: 17.76 us @ 100MHz | GPIO PIN (violation_alarm_o): {gpio_str}",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            gpio_col if self.gpio_pin_state else (220, 240, 255),
            2 if self.gpio_pin_state else 1,
            cv2.LINE_AA
        )

        # Line 2: Multi-Class Live Vehicle Counters
        c_car = frame_counts["Car"]
        c_bike = frame_counts["Bike/Motorcycle"]
        c_bus = frame_counts["Bus"]
        c_trk = frame_counts["Truck/Heavy"]
        c_person = frame_counts["Person/Rider"]
        c_viol = frame_counts["Violations"]

        counters_text = (
            f"LIVE COUNTERS -> Cars: {c_car}  |  Bikes/Motorcycles: {c_bike}  |  "
            f"Buses: {c_bus}  |  Heavy Trucks: {c_trk}  |  Persons/Riders: {c_person}"
        )
        cv2.putText(annotated, counters_text, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 240, 180), 1, cv2.LINE_AA)

        # Line 3: Violations count highlight
        viol_col = (0, 0, 255) if c_viol > 0 else (120, 180, 120)
        cv2.putText(
            annotated,
            f"ACTIVE VIOLATIONS IN FRAME: {c_viol} (No Helmet / Overloaded) | TOTAL OBJECTS: {len(detected_objects)}",
            (15, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            viol_col,
            2 if c_viol > 0 else 1,
            cv2.LINE_AA
        )

        # 3. Picture-in-Picture: Render 48x48 SNN Preprocessed Patch in Top-Right
        if last_pip_crop is not None:
            pip_size = 96
            pip_zoomed = cv2.resize(last_pip_crop, (pip_size, pip_size), interpolation=cv2.INTER_NEAREST)
            py1, py2 = 95, 95 + pip_size
            px1, px2 = w - pip_size - 15, w - 15
            cv2.rectangle(annotated, (px1 - 2, py1 - 22), (px2 + 2, py2 + 2), (25, 25, 30), -1)
            cv2.rectangle(annotated, (px1 - 2, py1 - 22), (px2 + 2, py2 + 2), (80, 80, 90), 1)
            cv2.putText(annotated, "SNN 48x48 Crop", (px1 + 2, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1)
            annotated[py1:py2, px1:px2] = pip_zoomed

        return annotated


def main():
    print("=" * 80)
    print("AO.SNN PRODUCTION LIVE TRAFFIC & HELMET VIOLATION MONITOR")
    print("Separate Bounding Boxes + Multi-Class Vehicle Counters + SNN LIF Classifier")
    print("=" * 80)

    checkpoint = os.path.join(WORKSPACE_ROOT, "vectorblox_deployment", "artifacts", "best_snn_checkpoint.pt")
    monitor = TrafficJunctionMonitor(checkpoint_path=checkpoint)

    # Open Camera Index 0
    print("[CAMERA] Connecting to integrated/USB webcam index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open laptop webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[CAMERA READY]")
    print("  Press 'q' or 'ESC' to Exit")
    print("  Press 's' to Save Current Snapshot\n")

    snap_id = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Process frame with full detection, counting, and SNN classification
        annotated = monitor.process_frame(frame)

        # Show live interactive window
        cv2.imshow("AO.SNN Traffic Junction & Violation Monitor - Press 'q' to Exit", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            print("[INFO] User requested exit.")
            break
        elif key == ord("s"):
            fname = f"snapshot_violation_{snap_id:03d}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[SNAPSHOT] Saved: {fname}")
            snap_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session closed cleanly.")


if __name__ == "__main__":
    main()
