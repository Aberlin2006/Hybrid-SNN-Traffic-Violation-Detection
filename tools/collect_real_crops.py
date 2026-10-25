"""
AO.SNN Real-World Crop Collector
=================================
Uses YOLOv8n to detect vehicles from a video/image/webcam source
and saves real 48x48 crops to user_crops/ with correct filename prefixes:

    car_XXXX.jpg    → class 0 Car
    bus_XXXX.jpg    → class 7 Bus
    truck_XXXX.jpg  → class 8 Heavy Vehicle
    auto_XXXX.jpg   → class 9 Auto Rickshaw

Usage:
    # From webcam (collect while driving by):
    python collect_real_crops.py --source 0 --target car --n 200

    # From a video file:
    python collect_real_crops.py --source traffic.mp4 --target car --n 500

    # For motorcycles (helmet):
    python collect_real_crops.py --source 0 --target helmet --n 200

    # For motorcycles (no helmet):
    python collect_real_crops.py --source 0 --target bare --n 200
"""
import os, sys, cv2, argparse, time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

CROP_DIR = os.path.join(PROJECT_ROOT, "user_crops")
os.makedirs(CROP_DIR, exist_ok=True)

# YOLO COCO class to prefix mapping
COCO_TO_PREFIX = {
    2: "car",     # car
    5: "bus",     # bus
    7: "truck",   # truck
    1: "helmet",  # bicycle (use for manual compliant tagging)
    3: "helmet",  # motorcycle (use --target bare for violation)
}

TARGET_OPTIONS = {
    "car":    {"coco_classes": [2],    "prefix": "car"},
    "bus":    {"coco_classes": [5],    "prefix": "bus"},
    "truck":  {"coco_classes": [7],    "prefix": "truck"},
    "helmet": {"coco_classes": [1, 3], "prefix": "helmet"},
    "bare":   {"coco_classes": [1, 3], "prefix": "bare"},
}

def count_existing(prefix):
    return sum(1 for f in os.listdir(CROP_DIR) if f.startswith(prefix + "_"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 = webcam, or path to video/image")
    parser.add_argument("--target", required=True, choices=list(TARGET_OPTIONS.keys()),
                        help="What class to collect: car, bus, truck, helmet, bare")
    parser.add_argument("--n", type=int, default=300, help="Number of crops to collect")
    parser.add_argument("--min-size", type=int, default=60, help="Min bounding box size (px)")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO confidence threshold")
    args = parser.parse_args()

    cfg = TARGET_OPTIONS[args.target]
    coco_classes = cfg["coco_classes"]
    prefix = cfg["prefix"]
    start_idx = count_existing(prefix)
    collected = 0
    needed = args.n

    print(f"[COLLECTOR] Target: '{args.target}' → prefix='{prefix}_'")
    print(f"[COLLECTOR] COCO classes to look for: {coco_classes}")
    print(f"[COLLECTOR] Existing: {start_idx} | Need: {needed} more | Min box size: {args.min_size}px")
    print(f"[COLLECTOR] Output directory: {CROP_DIR}")
    print()

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    is_image = args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    if is_image:
        sources = [args.source]
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
        time.sleep(2)
        for _ in range(10): cap.read()
        sources = None
    else:
        cap = cv2.VideoCapture(args.source)
        sources = None

    def process_frame(frame):
        nonlocal collected
        results = model(frame, conf=args.conf, classes=coco_classes, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                if w < args.min_size or h < args.min_size:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
                fname = f"{prefix}_{start_idx + collected:04d}.jpg"
                fpath = os.path.join(CROP_DIR, fname)
                cv2.imwrite(fpath, crop_resized)
                collected += 1
                print(f"  [{collected:4d}/{needed}] Saved: {fname}  (box: {w}x{h}px)")
                if collected >= needed:
                    return True
        return False

    if is_image:
        for src in sources:
            frame = cv2.imread(src)
            if frame is not None:
                process_frame(frame)
    else:
        frame_n = 0
        while collected < needed:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[COLLECTOR] Video ended.")
                break
            frame_n += 1
            if frame_n % 3 != 0:  # Sample every 3rd frame to avoid duplicates
                continue
            # Show preview
            preview = cv2.resize(frame, (640, 360))
            cv2.putText(preview, f"Collecting {prefix}: {collected}/{needed} | Press Q to stop",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imshow(f"AO.SNN Crop Collector — {prefix}", preview)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
            if process_frame(frame):
                break
        cap.release()
        cv2.destroyAllWindows()

    print()
    print(f"[DONE] Collected {collected} '{prefix}' crops.")
    print(f"[DONE] Total '{prefix}' crops in user_crops: {start_idx + collected}")
    print(f"[NEXT] Now retrain the SNN: python training/train_snn.py")

if __name__ == "__main__":
    main()
