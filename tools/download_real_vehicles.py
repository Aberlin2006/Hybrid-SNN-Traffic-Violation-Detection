"""
AO.SNN Real Vehicle Dataset Downloader & Crop Pipeline v2.0
Uses Wikimedia Commons thumbnail API (cached CDN) to fetch diverse real vehicles:
    - Car (class 0)
    - Bus (class 7)
    - Heavy Vehicle / Truck / JCB (class 8)
    - Auto Rickshaw (class 9)
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

HEADERS = {"User-Agent": "AOSNNResearchEdu/3.0 (student_research@mit.edu)"}

SEARCH_CATEGORIES = {
    "truck": {
        "queries": [
            "dump truck",
            "semi trailer truck highway",
            "freightliner truck",
            "caterpillar excavator",
            "cement mixer truck",
            "flatbed truck cargo",
            "heavy tipper truck",
            "kenworth truck",
            "volvo truck road",
            "scania truck freight",
            "jcb backhoe loader",
        ],
        "allowed_coco": [7],  # 7 = truck
        "target_count": 800,
        "prefix": "truck"
    },
    "car": {
        "queries": [
            "sedan automobile street",
            "hatchback car street",
            "suv automobile road",
            "toyota corolla car road",
            "white sedan car",
            "black suv vehicle",
            "red automobile street",
            "silver hatchback car",
            "honda civic road",
            "hyundai car traffic",
            "ford car street",
        ],
        "allowed_coco": [2],  # 2 = car
        "target_count": 800,
        "prefix": "car"
    },
    "bus": {
        "queries": [
            "city bus street",
            "double decker bus street",
            "school bus yellow",
            "volvo bus coach highway",
            "transit bus passenger",
            "mercedes bus street",
            "scania bus travel",
            "man lion city bus",
        ],
        "allowed_coco": [5],  # 5 = bus
        "target_count": 800,
        "prefix": "bus"
    },
    "auto": {
        "queries": [
            "auto rickshaw street india",
            "bajaj auto rickshaw",
            "tuk tuk transport road",
            "auto rickshaw delhi traffic",
            "three wheeler auto taxi",
            "cng auto rickshaw",
            "yellow green auto rickshaw",
        ],
        "allowed_coco": [2, 3, 5, 7],
        "target_count": 600,
        "prefix": "auto"
    }
}


def query_wikimedia_thumbnails(query: str, limit: int = 15) -> list:
    """Fetch cached thumbnail URLs from Wikimedia Commons API (fast CDN)."""
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


def download_thumb(thumb_url: str) -> np.ndarray:
    req = urllib.request.Request(thumb_url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=8) as r:
        arr = np.asarray(bytearray(r.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img


def augment_crop(crop_bgr: np.ndarray, num_augs: int = 4) -> list:
    augmented = [crop_bgr]
    h, w = crop_bgr.shape[:2]
    for _ in range(num_augs - 1):
        aug = crop_bgr.copy()
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)
        alpha = float(np.random.uniform(0.85, 1.15))
        beta = float(np.random.uniform(-15, 15))
        aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        pad = int(min(h, w) * 0.05)
        if pad > 1:
            aug = cv2.resize(aug[pad:h-pad, pad:w-pad], (48, 48), interpolation=cv2.INTER_AREA)
        else:
            aug = cv2.resize(aug, (48, 48), interpolation=cv2.INTER_AREA)
        augmented.append(aug)
    return augmented


def collect_category(cat_key: str, cat_cfg: dict):
    prefix = cat_cfg["prefix"]
    target_count = cat_cfg["target_count"]
    allowed_coco = cat_cfg["allowed_coco"]
    queries = cat_cfg["queries"]

    print(f"\n============================================================")
    print(f"COLLECTING REAL VEHICLES FOR: '{cat_key.upper()}' (Target: {target_count})")
    print(f"============================================================")

    old_files = [f for f in os.listdir(CROP_DIR) if f.startswith(f"{prefix}_")]
    print(f"Found {len(old_files)} existing '{prefix}_' files.")

    saved_count = 0
    unique_crops = []

    for q in queries:
        if len(unique_crops) >= 60:
            break
        print(f"  Searching: '{q}'...")
        urls = query_wikimedia_thumbnails(q, limit=12)
        print(f"    Got {len(urls)} thumbnail URLs.")

        for url in urls:
            if len(unique_crops) >= 60:
                break
            try:
                img = download_thumb(url)
                if img is None:
                    continue
                results = YOLO_MODEL(img, verbose=False, conf=0.30)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in allowed_coco:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        bw, bh = x2 - x1, y2 - y1
                        if bw >= 40 and bh >= 40:
                            mx = int(bw * 0.04)
                            my = int(bh * 0.04)
                            x1_c = max(0, x1 - mx)
                            y1_c = max(0, y1 - my)
                            x2_c = min(img.shape[1], x2 + mx)
                            y2_c = min(img.shape[0], y2 + my)
                            crop = img[y1_c:y2_c, x1_c:x2_c]
                            crop_48 = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
                            unique_crops.append(crop_48)
            except Exception:
                pass

    print(f"  Extracted {len(unique_crops)} UNIQUE real vehicle crops for '{prefix}'.")

    if len(unique_crops) == 0:
        print(f"  [WARN] No crops extracted. Keeping existing files.")
        return

    # Delete old bad crops
    for f in old_files:
        try:
            os.remove(os.path.join(CROP_DIR, f))
        except Exception:
            pass

    # Augment to target_count
    file_idx = 0
    augs_per_crop = max(1, target_count // len(unique_crops) + 1)
    for crop in unique_crops:
        augs = augment_crop(crop, num_augs=augs_per_crop)
        for aug in augs:
            if file_idx >= target_count:
                break
            fname = f"{prefix}_{file_idx:04d}.jpg"
            cv2.imwrite(os.path.join(CROP_DIR, fname), aug)
            file_idx += 1

    print(f"  Successfully wrote {file_idx} verified real-world crops for '{prefix}' to user_crops/!")


def main():
    print("=" * 60)
    print("AO.SNN REAL-WORLD DATASET HARVESTER v2.0 (CDN Thumbnails)")
    print("=" * 60)
    for cat_name, cat_cfg in SEARCH_CATEGORIES.items():
        collect_category(cat_name, cat_cfg)
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("SUMMARY OF USER_CROPS DIRECTORY:")
    files = os.listdir(CROP_DIR)
    for p in ["car", "bus", "truck", "auto", "helmet", "bare"]:
        cnt = sum(1 for f in files if f.startswith(f"{p}_"))
        print(f"  {p:>7}: {cnt:>5} crops")
    print("=" * 60)


if __name__ == "__main__":
    main()
