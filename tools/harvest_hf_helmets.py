"""
AO.SNN Benchmark Helmet Harvester
Downloads verified ground-truth motorcycle rider helmet and bare head crops
from the official Roboflow Bike-Helmets benchmark dataset on Hugging Face:
  cute-face/bike-helmet-dataset
  Category 1: 'With Helmet'    -> user_crops/helmet_XXXX.jpg
  Category 2: 'Without Helmet' -> user_crops/bare_XXXX.jpg
"""

import os
import sys
import json
import urllib.request
import concurrent.futures
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CROP_DIR = os.path.join(PROJECT_ROOT, "user_crops")
os.makedirs(CROP_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
JSON_URL = "https://huggingface.co/datasets/cute-face/bike-helmet-dataset/raw/main/train/_annotations.coco.json"
BASE_IMG_URL = "https://huggingface.co/datasets/cute-face/bike-helmet-dataset/resolve/main/train/"

def download_image(file_name: str) -> np.ndarray:
    url = BASE_IMG_URL + file_name
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def main():
    print("=" * 65)
    print("HARVESTING GROUND-TRUTH HELMET CROPS FROM ROBOFLOW BIKE-HELMETS BENCHMARK")
    print("=" * 65)

    print("[1/4] Downloading COCO annotations...")
    req = urllib.request.Request(JSON_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        coco_data = json.loads(resp.read().decode())

    images_map = {img["id"]: img for img in coco_data["images"]}
    annos_by_image = {}
    for a in coco_data["annotations"]:
        img_id = a["image_id"]
        annos_by_image.setdefault(img_id, []).append(a)

    print(f"Loaded {len(images_map)} images and {len(coco_data['annotations'])} annotations.")

    # Remove old helmet and bare crops in user_crops
    print("[2/4] Purging old noisy helmet and bare crops from user_crops...")
    for f in os.listdir(CROP_DIR):
        if f.startswith("helmet_") or f.startswith("bare_"):
            try:
                os.remove(os.path.join(CROP_DIR, f))
            except Exception:
                pass

    helmet_crops = []
    bare_crops = []
    target_count = 800

    image_ids = list(images_map.keys())
    np.random.seed(42)
    np.random.shuffle(image_ids)

    print(f"[3/4] Downloading images and extracting bounding boxes (Target: {target_count} each)...")

    def process_image_id(img_id):
        img_meta = images_map[img_id]
        file_name = img_meta["file_name"]
        annos = annos_by_image.get(img_id, [])
        if not annos:
            return [], []
        
        img = download_image(file_name)
        if img is None:
            return [], []

        h, w = img.shape[:2]
        h_local = []
        b_local = []

        for a in annos:
            cat_id = a["category_id"]
            if cat_id not in [1, 2]:
                continue
            bx, by, bw, bh = a["bbox"]
            if bw < 14 or bh < 14:
                continue

            # Add 10% context margin
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            x1 = max(0, int(bx - pad_x))
            y1 = max(0, int(by - pad_y))
            x2 = min(w, int(bx + bw + pad_x))
            y2 = min(h, int(by + bh + pad_y))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                continue

            crop_48 = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)

            if cat_id == 1:
                h_local.append(crop_48)
            elif cat_id == 2:
                b_local.append(crop_48)

        return h_local, b_local

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_id = {executor.submit(process_image_id, iid): iid for iid in image_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            h_local, b_local = future.result()
            if len(helmet_crops) < target_count:
                helmet_crops.extend(h_local)
            if len(bare_crops) < target_count:
                bare_crops.extend(b_local)

            if len(helmet_crops) >= target_count and len(bare_crops) >= target_count:
                # Cancel remaining
                for f in future_to_id:
                    f.cancel()
                break

    helmet_crops = helmet_crops[:target_count]
    bare_crops = bare_crops[:target_count]

    print(f"[4/4] Writing {len(helmet_crops)} helmet crops and {len(bare_crops)} bare crops...")
    for idx, c in enumerate(helmet_crops):
        cv2.imwrite(os.path.join(CROP_DIR, f"helmet_{idx:04d}.jpg"), c)

    for idx, c in enumerate(bare_crops):
        cv2.imwrite(os.path.join(CROP_DIR, f"bare_{idx:04d}.jpg"), c)

    print("=" * 65)
    print("FINISHED HARVESTING CLEAN GROUND-TRUTH HELMET BENCHMARK DATASET!")
    print(f"  Helmets saved: {len(helmet_crops)} -> user_crops/helmet_0000.jpg..{len(helmet_crops)-1:04d}.jpg")
    print(f"  Bare heads saved: {len(bare_crops)} -> user_crops/bare_0000.jpg..{len(bare_crops)-1:04d}.jpg")
    print("=" * 65)

if __name__ == "__main__":
    main()
