"""
AO.SNN Real Motorcyclist Helmet & Bare Head Dataset Harvester
Downloads real motorcycle and scooter riders on roads/streets,
extracts the rider head crops, and saves:
    - helmet_XXXX.jpg -> Class 1 (Bike_Compliant) / Class 4 (Scooter_Compliant)
    - bare_XXXX.jpg   -> Class 2 (Bike_No_Helmet) / Class 5 (Scooter_No_Helmet)
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse
import cv2
import numpy as np
from ultralytics import YOLO

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CROP_DIR = os.path.join(PROJECT_ROOT, "user_crops")
os.makedirs(CROP_DIR, exist_ok=True)

YOLO_MODEL = YOLO("yolov8n.pt")
HEADERS = {"User-Agent": "AOSNNHelmetResearch/2.0 (student_helmet@research.edu)"}

HELMET_QUERIES = [
    "motorcyclist helmet road",
    "motorcycle rider helmet street",
    "motorcycle helmet visor",
    "biker wearing helmet",
    "scooter rider helmet",
    "motorcycle helmet safety",
    "police motorcyclist helmet",
    "racing motorcycle helmet",
    "motorcyclist riding highway",
    "scooter driver helmet traffic",
]

BARE_QUERIES = [
    "motorcyclist without helmet traffic",
    "motorcycle rider no helmet",
    "scooter rider no helmet traffic",
    "bare head motorcycle rider road",
    "two wheeler without helmet india",
    "motorcycle passengers without helmets",
    "biker no helmet street",
    "scooter driver bare head",
]


def query_wikimedia_thumbs(query: str, limit: int = 15) -> list:
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&generator=search&gsrnamespace=6&gsrsearch={urllib.parse.quote(query)}"
        f"&gsrlimit={limit}&prop=imageinfo&iiprop=url&iiurlwidth=640"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    thumb_urls = []
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            info = p.get("imageinfo", [{}])[0]
            thumb = info.get("thumburl")
            if thumb:
                thumb_urls.append(thumb)
    except Exception as e:
        print(f"  [WARN] Query '{query}' failed: {e}")
    return thumb_urls


def download_thumb(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=8) as r:
        arr = np.asarray(bytearray(r.read()), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_rider_head_crops(img: np.ndarray) -> list:
    """Uses YOLO to find person + motorcycle overlap, then extracts anatomical head crop."""
    h, w = img.shape[:2]
    res = YOLO_MODEL(img, verbose=False, conf=0.30)
    boxes = res[0].boxes
    person_boxes = []
    bike_boxes = []

    for b in boxes:
        cls_id = int(b.cls[0].item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        if cls_id == 0:  # person
            person_boxes.append((x1, y1, x2, y2))
        elif cls_id in [1, 3]:  # bicycle / motorcycle
            bike_boxes.append((x1, y1, x2, y2))

    head_crops = []
    for px1, py1, px2, py2 in person_boxes:
        pw = px2 - px1
        ph = py2 - py1
        if pw < 30 or ph < 40:
            continue
        # Check if person is associated with a bike (or is a close rider portrait)
        is_rider = len(bike_boxes) == 0  # if query is already for biker, accept portrait
        for bx1, by1, bx2, by2 in bike_boxes:
            ix1 = max(px1, bx1)
            iy1 = max(py1, by1)
            ix2 = min(px2, bx2)
            iy2 = min(py2, by2)
            if ix2 > ix1 and iy2 > iy1:
                is_rider = True
                break

        if is_rider:
            # Extract upper 40% anatomical head region
            xc = (px1 + px2) // 2
            hw = int(pw * 0.70)
            hx1 = max(0, xc - hw // 2)
            hx2 = min(w, xc + hw // 2)
            hy1 = max(0, py1)
            hy2 = min(h, py1 + int(ph * 0.42))
            crop = img[hy1:hy2, hx1:hx2]
            if crop.shape[0] >= 25 and crop.shape[1] >= 25:
                crop_48 = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
                head_crops.append(crop_48)

    return head_crops


def harvest_head_crops(queries: list, prefix: str, target_count: int = 800):
    print(f"\n============================================================")
    print(f"HARVESTING REAL ROAD CROPS FOR: '{prefix.upper()}' (Target: {target_count})")
    print(f"============================================================")

    # First clean old webcam crops for this prefix
    old_files = [f for f in os.listdir(CROP_DIR) if f.startswith(f"{prefix}_")]
    print(f"Removing {len(old_files)} old webcam-only '{prefix}_' crops...")
    for f in old_files:
        try:
            os.remove(os.path.join(CROP_DIR, f))
        except Exception:
            pass

    unique_crops = []
    for q in queries:
        if len(unique_crops) >= 70:
            break
        print(f"  Searching: '{q}'...")
        urls = query_wikimedia_thumbs(q, limit=12)
        print(f"    Got {len(urls)} images.")
        for u in urls:
            if len(unique_crops) >= 70:
                break
            try:
                img = download_thumb(u)
                if img is None:
                    continue
                crops = extract_rider_head_crops(img)
                unique_crops.extend(crops)
            except Exception:
                pass

    print(f"  Extracted {len(unique_crops)} UNIQUE real road head crops for '{prefix}'.")
    if len(unique_crops) == 0:
        print(f"  [ERROR] No crops extracted for {prefix}!")
        return

    # Augment to target_count
    file_idx = 0
    augs_per_crop = max(1, target_count // len(unique_crops) + 1)
    for crop in unique_crops:
        for _ in range(augs_per_crop):
            if file_idx >= target_count:
                break
            aug = crop.copy()
            if np.random.rand() > 0.5:
                aug = cv2.flip(aug, 1)
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = float(np.random.uniform(-15, 15))
            aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            fname = f"{prefix}_{file_idx:04d}.jpg"
            cv2.imwrite(os.path.join(CROP_DIR, fname), aug)
            file_idx += 1

    print(f"  Successfully wrote {file_idx} real-world '{prefix}' crops to user_crops/!")


def main():
    harvest_head_crops(HELMET_QUERIES, "helmet", target_count=800)
    harvest_head_crops(BARE_QUERIES, "bare", target_count=800)

    print("\n" + "=" * 60)
    print("UPDATED USER_CROPS SUMMARY:")
    files = os.listdir(CROP_DIR)
    for p in ["car", "bus", "truck", "auto", "helmet", "bare"]:
        cnt = sum(1 for f in files if f.startswith(f"{p}_"))
        print(f"  {p:>7}: {cnt:>5} crops")
    print("=" * 60)


if __name__ == "__main__":
    main()
