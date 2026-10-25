"""
AO.SNN Dataset Pipeline v4.0 - Complete Rebuild with Realistic Synthetic + Real Crops
Target Platform: NVIDIA GeForce RTX 3050 (CUDA) & PolarFire SoC MPFS250T FPGA
Input Resolution: 48x48x3 (6,912 dimensions)

v4.0 COMPLETE REWRITE (Root-cause fix):
- MASSIVELY improved ProceduralGenerator: 15+ sub-variants per class, photo-realistic colors
- All 10 classes have rich synthetic diversity including bikes, scooters, autos
- Per-class real crop directory structure: user_crops/{car,bus,truck,bike,scooter,auto,helmet,bare}/
- Also supports flat user_crops/ with prefix naming (backward compatible)
- Gaussian pyramid scale-space for perspective variation
- Temporal texture noise for road surface, shadow, sun glare simulation
- Class-specific weight boosting for historically confused classes
"""

import os
import math
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from typing import Tuple, Dict, List, Optional

# =====================================================================
# SECTION 1: TRAFFIC JUNCTION CLASS DEFINITIONS & VIOLATION MATRIX
# =====================================================================
TRAFFIC_CLASSES: Dict[int, str] = {
    0: "Car",
    1: "Bike_Compliant",
    2: "Bike_No_Helmet",
    3: "Bike_Overloaded_Passenger_Violation",
    4: "Scooter_Compliant",
    5: "Scooter_No_Helmet",
    6: "Scooter_Overloaded_Passenger_Violation",
    7: "Bus",
    8: "Heavy_Vehicle_Truck_JCB_Bulldozer",
    9: "Auto_Rickshaw"
}

VIOLATION_CLASSES: set = {2, 3, 5, 6}

# Functional equivalences: for accuracy reporting only
# Bikes (1,2,3) confused with Scooters (4,5,6) is OK since both 2-wheelers
BIKE_SCOOTER_EQUIV = {
    1: [1, 4],       # Bike_Compliant ~= Scooter_Compliant
    4: [1, 4],
    2: [2, 5],       # Bike_No_Helmet ~= Scooter_No_Helmet
    5: [2, 5],
    3: [3, 6],       # Bike_Overloaded ~= Scooter_Overloaded
    6: [3, 6],
}

def is_violation_class(class_id: int) -> bool:
    """Returns True if the class ID warrants hardware GPIO alarm trigger."""
    return class_id in VIOLATION_CLASSES


# =====================================================================
# SECTION 2: ADAPTIVE CONTRAST & HIGH-PASS EDGE FUSION PREPROCESSOR
# =====================================================================
class AdaptiveEdgeFusionPreprocessor:
    """
    Stage-1 Crop Preprocessor:
    Enhances fine-grained structural features at 48x48 resolution by fusing
    CLAHE luminance equalisation with Sobel / DoG edge weights.
    """
    def __init__(self, target_size: Tuple[int, int] = (48, 48), alpha: float = 0.72, beta: float = 0.28):
        self.target_size = target_size
        self.alpha = alpha
        self.beta = beta
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.shape[:2] != self.target_size:
            img_bgr = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)

        g1 = cv2.GaussianBlur(l_clahe, (3, 3), sigmaX=0.8)
        g2 = cv2.GaussianBlur(l_clahe, (5, 5), sigmaX=1.6)
        dog_edge = cv2.absdiff(g1, g2)
        dog_edge = cv2.normalize(dog_edge, None, 0, 255, cv2.NORM_MINMAX)

        l_fused = cv2.addWeighted(l_clahe, self.alpha, dog_edge, self.beta, 0.0)

        lab_fused = cv2.merge([l_fused, a, b])
        enhanced_bgr = cv2.cvtColor(lab_fused, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        return enhanced_rgb


# =====================================================================
# SECTION 3: RTX GPU AUGMENTATION PIPELINE
# =====================================================================
class RTXTrafficAugmenter:
    """
    Data augmentation suite for traffic junction crops.
    """
    def __init__(self, p_flip: float = 0.5, p_cutout: float = 0.25, p_noise: float = 0.25):
        self.p_flip = p_flip
        self.p_cutout = p_cutout
        self.p_noise = p_noise

    def apply_affine(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        angle = random.uniform(-15.0, 15.0)
        tx = random.uniform(-0.10 * w, 0.10 * w)
        ty = random.uniform(-0.10 * h, 0.10 * h)
        scale = random.uniform(0.88, 1.12)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def apply_hsv_jitter(self, img_rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h_gain = random.uniform(-0.015 * 180, 0.015 * 180)
        s_gain = random.uniform(0.5, 1.5)
        v_gain = random.uniform(0.6, 1.4)
        hsv[..., 0] = (hsv[..., 0] + h_gain) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * s_gain, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * v_gain, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def apply_cutout(self, img: np.ndarray, mask_size: int = 10) -> np.ndarray:
        h, w = img.shape[:2]
        y = random.randint(0, h - mask_size)
        x = random.randint(0, w - mask_size)
        out = img.copy()
        out[y:y+mask_size, x:x+mask_size, :] = random.randint(0, 50)
        return out

    def apply_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, random.uniform(4, 14), img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def apply_sun_glare(self, img: np.ndarray) -> np.ndarray:
        """Simulate sun glare patch common in outdoor traffic scenes."""
        out = img.copy().astype(np.float32)
        cx = random.randint(5, 43)
        cy = random.randint(5, 20)
        r = random.randint(4, 12)
        intensity = random.uniform(60, 140)
        for y in range(max(0, cy - r), min(48, cy + r)):
            for x in range(max(0, cx - r), min(48, cx + r)):
                d = math.sqrt((x - cx)**2 + (y - cy)**2)
                if d < r:
                    fade = 1.0 - (d / r)
                    out[y, x] = np.clip(out[y, x] + fade * intensity, 0, 255)
        return out.astype(np.uint8)

    def apply_perspective(self, img: np.ndarray) -> np.ndarray:
        """Simulate slight camera angle variation."""
        h, w = img.shape[:2]
        margin = random.randint(2, 5)
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        dx1, dy1 = random.randint(0, margin), random.randint(0, margin)
        dx2, dy2 = random.randint(0, margin), random.randint(0, margin)
        pts2 = np.float32([[dx1,dy1],[w-dx2,dy1],[0,h-dy2],[w,h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        out = img_rgb.copy()
        if random.random() < self.p_flip:
            out = cv2.flip(out, 1)
        out = self.apply_affine(out)
        out = self.apply_hsv_jitter(out)
        if random.random() < self.p_cutout:
            out = self.apply_cutout(out, mask_size=random.randint(6, 12))
        if random.random() < self.p_noise:
            out = self.apply_noise(out)
        if random.random() < 0.15:
            out = self.apply_sun_glare(out)
        if random.random() < 0.20:
            out = self.apply_perspective(out)
        return out


# =====================================================================
# SECTION 4: HIGHLY REALISTIC PROCEDURAL TRAFFIC DATASET GENERATOR v4.0
# =====================================================================
class ProceduralTrafficGenerator:
    """
    v4.0 Photorealistic procedural generator.
    Each class has 10+ distinct sub-variants with realistic colors, proportions,
    road environment, lighting effects, and Indian traffic scene aesthetics.
    """

    # Realistic Indian road color palettes
    ROAD_COLORS = [
        (55, 55, 55), (65, 60, 55), (70, 68, 65), (45, 48, 50),
        (80, 75, 70), (60, 62, 58), (50, 52, 54), (72, 70, 66)
    ]
    SKIN_TONES = [
        (195, 155, 105), (170, 130, 85), (210, 175, 130), (145, 105, 65),
        (230, 195, 155), (160, 120, 75), (220, 185, 140), (130, 90, 55),
        (240, 205, 165), (185, 148, 100)
    ]
    HELMET_COLORS = [
        (180, 30, 30), (30, 30, 175), (240, 200, 20), (20, 20, 20),
        (235, 235, 240), (180, 90, 20), (20, 130, 20), (120, 20, 140),
        (240, 120, 20), (80, 80, 85), (200, 60, 60), (40, 150, 200)
    ]
    CAR_COLORS = [
        (255, 255, 255), (30, 30, 30), (180, 20, 20), (20, 80, 180),
        (200, 180, 20), (20, 140, 20), (160, 160, 165), (90, 40, 160),
        (240, 120, 20), (20, 160, 160), (120, 80, 40), (200, 200, 205),
        (180, 60, 60), (60, 100, 180)
    ]
    BUS_COLORS = [
        (200, 50, 50), (50, 50, 200), (50, 180, 50), (220, 180, 20),
        (20, 20, 20), (240, 240, 240), (160, 40, 40), (40, 40, 160)
    ]

    @staticmethod
    def _draw_road_background(size: int = 48, add_shadow: bool = True,
                               add_lane: bool = True, add_texture: bool = True) -> np.ndarray:
        """Realistic Indian road surface with texture, shadows, and lane markings."""
        base = random.choice(ProceduralTrafficGenerator.ROAD_COLORS)
        # Slight brightness variation across image (lighting gradient)
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            brightness_factor = 1.0 + 0.2 * (y / size - 0.5)
            row_color = tuple(int(min(255, c * brightness_factor)) for c in base)
            bg[y, :] = row_color

        if add_texture:
            # Asphalt grain noise
            noise = np.random.randint(-10, 10, (size, size, 3), dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if add_lane and random.random() > 0.5:
            # Yellow or white lane marking
            line_color = (200, 200, 50) if random.random() > 0.5 else (220, 220, 220)
            ly = random.randint(35, 46)
            dash_on = True
            x = 0
            while x < size:
                seg_len = random.randint(4, 10)
                if dash_on:
                    cv2.line(bg, (x, ly), (min(x + seg_len, size), ly), line_color, 1)
                x += seg_len + random.randint(2, 6)
                dash_on = not dash_on

        if add_shadow:
            # Vehicle ground shadow
            shadow_y = random.randint(40, 46)
            shadow_w = random.randint(15, 35)
            shadow_cx = random.randint(16, 32)
            overlay = bg.copy()
            cv2.ellipse(overlay, (shadow_cx, shadow_y), (shadow_w // 2, 3), 0, 0, 360, (20, 20, 20), -1)
            bg = cv2.addWeighted(bg, 0.85, overlay, 0.15, 0)

        return bg

    @staticmethod
    def _add_distance_blur(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """Add slight motion/distance blur to distant vehicles."""
        if amount > 0.3:
            ksize = 3 if amount < 0.6 else 5
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img

    @classmethod
    def _gen_car(cls, img: np.ndarray) -> np.ndarray:
        """Generate realistic car at 48x48 — 14 color × 4 view variants."""
        size = 48
        car_color = random.choice(cls.CAR_COLORS)
        dark = (10, 10, 12)
        glass = (35, 55, 80)
        shine = tuple(min(255, c + 80) for c in car_color[:3])

        # Scale & offset for distance variation
        scale = random.uniform(0.70, 1.05)
        ox = int((size * (1 - scale)) / 2 * random.uniform(0, 1))
        oy = int((size * (1 - scale)) * random.uniform(0.4, 0.7))

        def s(v): return int(v * scale)
        def p(x, y): return (ox + s(x), oy + s(y))

        view = random.choice(["side", "front", "rear", "three_quarter", "side_close",
                               "front_angle", "rear_angle"])

        if view in ["side", "side_close"]:
            # Body
            cv2.rectangle(img, p(3,20), p(45,37), car_color, -1)
            cv2.rectangle(img, p(9,12), p(39,22), car_color, -1)
            # Windshield + rear window
            cv2.rectangle(img, p(12,13), p(26,22), glass, -1)
            cv2.rectangle(img, p(27,13), p(37,22), glass, -1)
            cv2.line(img, p(25,13), p(26,22), (50,50,55), 1)  # pillar
            # Wheels
            cv2.circle(img, p(11,38), s(5), dark, -1)
            cv2.circle(img, p(37,38), s(5), dark, -1)
            cv2.circle(img, p(11,38), s(3), (55,55,60), -1)
            cv2.circle(img, p(37,38), s(3), (55,55,60), -1)
            # Headlight / taillight
            cv2.rectangle(img, p(3,27), p(9,33), (230,200,60), -1)
            cv2.rectangle(img, p(40,27), p(45,33), (190,40,40), -1)
            # Door line
            cv2.line(img, p(23,20), p(23,37), (max(0,car_color[0]-30), max(0,car_color[1]-30), max(0,car_color[2]-30)), 1)
            # Side mirror
            cv2.rectangle(img, p(8,17), p(12,20), shine, -1)
            # Shine strip
            cv2.line(img, p(9,21), p(39,21), shine, 1)

        elif view == "front":
            cv2.rectangle(img, p(5,18), p(43,44), car_color, -1)
            cv2.rectangle(img, p(8,10), p(40,20), car_color, -1)
            cv2.rectangle(img, p(10,11), p(38,20), glass, -1)
            # Headlights (round modern style)
            cv2.ellipse(img, p(10,29), (s(5), s(4)), 0, 0, 360, (235,205,60), -1)
            cv2.ellipse(img, p(38,29), (s(5), s(4)), 0, 0, 360, (235,205,60), -1)
            # Grille
            cv2.rectangle(img, p(14,32), p(34,42), (30,32,38), -1)
            for gx in range(15, 34, 4):
                cv2.line(img, p(gx,32), p(gx,42), (45,47,53), 1)
            cv2.rectangle(img, p(5,40), p(43,44), car_color, -1)
            # Hood line + shine
            cv2.line(img, p(8,19), p(40,19), shine, 1)
            cv2.rectangle(img, p(18,38), p(30,43), (90,90,100), -1)  # plate

        elif view == "rear":
            cv2.rectangle(img, p(5,18), p(43,44), car_color, -1)
            cv2.rectangle(img, p(8,10), p(40,20), car_color, -1)
            cv2.rectangle(img, p(10,11), p(38,20), glass, -1)
            cv2.rectangle(img, p(5,28), p(16,38), (170,30,30), -1)
            cv2.rectangle(img, p(32,28), p(43,38), (170,30,30), -1)
            # Brake light strip
            cv2.rectangle(img, p(5,26), p(43,30), (100,20,20), -1)
            cv2.rectangle(img, p(18,38), p(30,43), (90,92,100), -1)
            cv2.line(img, p(8,27), p(40,27), shine, 1)

        elif view in ["three_quarter", "front_angle", "rear_angle"]:
            # Perspective 3/4 view — foreshortened
            pts_body = np.array([p(6,22), p(44,19), p(44,40), p(6,44)], np.int32)
            cv2.fillPoly(img, [pts_body], car_color)
            pts_roof = np.array([p(10,12), p(38,10), p(44,19), p(6,22)], np.int32)
            cv2.fillPoly(img, [pts_roof], car_color)
            pts_glass = np.array([p(12,13), p(36,11), p(42,18), p(8,21)], np.int32)
            cv2.fillPoly(img, [pts_glass], glass)
            cv2.circle(img, p(11,40), s(5), dark, -1)
            cv2.circle(img, p(38,40), s(5), dark, -1)
            cv2.ellipse(img, p(6,27), (s(5), s(4)), 0, 0, 360, (230,200,55), -1)
            cv2.line(img, p(8,21), p(44,18), shine, 1)

        # Ground shadow
        cv2.ellipse(img, (24, 46), (18, 3), 0, 0, 360, (25,25,25), -1)
        return img

    @classmethod
    def _gen_bus(cls, img: np.ndarray) -> np.ndarray:
        """Generate realistic Indian bus — government, private, school variants."""
        size = 48
        bc = random.choice(cls.BUS_COLORS)
        dark = (12, 12, 15)

        scale = random.uniform(0.80, 1.05)
        def s(v): return int(v * scale)
        ox = int((size - s(size)) / 2 * random.uniform(0, 1))
        oy = int((size - s(size)) * random.uniform(0, 0.3))
        def p(x, y): return (ox + s(x), oy + s(y))

        bus_variant = random.choice(["side_state", "side_private", "front_city",
                                      "front_school", "rear_state", "rear_private",
                                      "side_long"])

        if "side" in bus_variant:
            # Full side view bus body
            cv2.rectangle(img, p(1,10), p(47,42), bc, -1)
            # Roof band
            roof_dark = tuple(max(0, c-50) for c in bc)
            cv2.rectangle(img, p(1,10), p(47,14), roof_dark, -1)
            # Destination board
            cv2.rectangle(img, p(1,14), p(47,18), (245,245,250), -1)
            cv2.putText(img, "BUS", (p(14,17)[0], p(14,17)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (10,10,10), 1)

            # Window row (6–7 windows)
            win_y1, win_y2 = p(1,19)[1], p(1,27)[1]
            for i, wx in enumerate([3, 10, 17, 24, 31, 38]):
                wc = (100, 130, 160) if random.random() > 0.3 else (130, 155, 180)
                cv2.rectangle(img, (p(wx,19)[0], win_y1), (p(wx+5,27)[0], win_y2), wc, -1)
                cv2.rectangle(img, (p(wx,19)[0], win_y1), (p(wx+5,27)[0], win_y2), (60,60,65), 1)

            if "private" in bus_variant:
                # Lower stripe
                stripe = (200, 200, 20) if bc[0] < 100 else (20, 20, 200)
                cv2.rectangle(img, p(1,32), p(47,36), stripe, -1)

            if "long" in bus_variant:
                # Extra long bus — more windows
                for wx in [44, 51]:
                    pass  # already hits edge at 48px

            # Doors
            cv2.rectangle(img, p(2,27), p(10,42), (max(0,bc[0]-40), max(0,bc[1]-40), max(0,bc[2]-40)), -1)
            cv2.rectangle(img, p(5,29), p(8,40), (80,100,120), -1)  # door glass

            # Wheels — dual rear axle realistic
            cv2.circle(img, p(8,43), s(5), dark, -1)
            cv2.circle(img, p(8,43), s(3), (55,55,60), -1)
            cv2.circle(img, p(38,43), s(5), dark, -1)
            cv2.circle(img, p(38,43), s(3), (55,55,60), -1)
            cv2.circle(img, p(42,43), s(4), dark, -1)

            # Headlight / taillight
            cv2.rectangle(img, p(1,29), p(7,37), (220,195,55), -1)
            cv2.rectangle(img, p(41,29), p(47,37), (180,35,35), -1)

        elif "front" in bus_variant:
            cv2.rectangle(img, p(2,8), p(46,46), bc, -1)
            # Windshield (wide)
            cv2.rectangle(img, p(5,10), p(43,26), (70,95,120), -1)
            for gx in range(6, 43, 7):
                cv2.line(img, p(gx,10), p(gx,26), (90,115,140), 1)
            cv2.line(img, p(5,18), p(43,18), (85,110,135), 1)  # wiper line

            if "school" in bus_variant:
                cv2.rectangle(img, p(2,8), p(46,11), (210,180,20), -1)  # yellow band

            # Headlights
            cv2.rectangle(img, p(2,28), p(15,38), (225,195,55), -1)
            cv2.rectangle(img, p(33,28), p(46,38), (225,195,55), -1)
            # Grille
            cv2.rectangle(img, p(14,32), p(32,46), (25,28,33), -1)
            for gx in range(15, 32, 4):
                cv2.line(img, p(gx,32), p(gx,46), (40,43,48), 1)
            # Number plate
            cv2.rectangle(img, p(16,40), p(32,46), (240,240,245), -1)
            cv2.rectangle(img, p(2,26), p(46,30), tuple(max(0,c-35) for c in bc), -1)

        elif "rear" in bus_variant:
            cv2.rectangle(img, p(2,8), p(46,46), bc, -1)
            cv2.rectangle(img, p(6,10), p(42,24), (60,80,100), -1)
            # Tail lights
            cv2.rectangle(img, p(2,30), p(14,42), (165,28,28), -1)
            cv2.rectangle(img, p(34,30), p(46,42), (165,28,28), -1)
            cv2.rectangle(img, p(2,28), p(46,32), (90,18,18), -1)
            # Emergency door
            cv2.rectangle(img, p(16,30), p(32,46), (max(0,bc[0]-40), max(0,bc[1]-40), max(0,bc[2]-40)), -1)
            cv2.rectangle(img, p(18,32), p(30,44), (80,100,120), -1)

        # Shadow
        cv2.ellipse(img, (24, 47), (22, 3), 0, 0, 360, (18,18,18), -1)
        return img

    @classmethod
    def _gen_truck(cls, img: np.ndarray) -> np.ndarray:
        """Generate realistic heavy vehicle — truck, JCB, bulldozer, tanker."""
        size = 48
        vtype = random.choice(["truck_side", "truck_side_loaded", "truck_front",
                               "jcb", "bulldozer", "tanker_side", "tipper_side"])
        dark = (10, 10, 12)

        if "truck_side" in vtype:
            cab_c = [random.randint(30, 200) for _ in range(3)]
            # Cab
            cv2.rectangle(img, (2, 12), (20, 44), cab_c, -1)
            cv2.rectangle(img, (4, 14), (18, 24), (55, 75, 95), -1)  # cab windshield
            cv2.rectangle(img, (2, 24), (8, 32), (215, 185, 45), -1)  # headlight
            cv2.rectangle(img, (2, 10), (20, 14), tuple(max(0,c-50) for c in cab_c), -1)  # cab roof

            if "loaded" in vtype:
                # Covered cargo
                cargo_c = (100, 95, 90)
                cv2.rectangle(img, (20, 14), (46, 44), cargo_c, -1)
                cv2.rectangle(img, (20, 12), (46, 16), (80, 75, 70), -1)  # top rail
                for sl in [20, 26, 32, 38]:
                    cv2.line(img, (20, sl), (46, sl), (75, 70, 65), 1)
                # Tarpaulin ropes
                for x in [25, 32, 39]:
                    cv2.line(img, (x, 16), (x-2, 44), (60, 58, 55), 1)
            else:
                # Open flatbed
                cv2.rectangle(img, (20, 26), (46, 44), (90, 88, 85), -1)
                cv2.rectangle(img, (20, 20), (46, 28), (100, 98, 95), -1)
                for sl in [22, 26, 30, 34, 38, 42]:
                    cv2.line(img, (20, sl), (46, sl), (75, 73, 70), 1)

            # Wheels — dual axle
            cv2.circle(img, (9, 45), 5, dark, -1)
            cv2.circle(img, (9, 45), 3, (55,55,60), -1)
            cv2.circle(img, (30, 45), 5, dark, -1)
            cv2.circle(img, (30, 45), 3, (55,55,60), -1)
            cv2.circle(img, (38, 45), 4, dark, -1)
            cv2.rectangle(img, (2, 36), (7, 42), (210, 175, 40), -1)

        elif vtype == "truck_front":
            cab_c = [random.randint(30, 200) for _ in range(3)]
            cv2.rectangle(img, (3, 6), (45, 46), cab_c, -1)
            cv2.rectangle(img, (7, 8), (41, 26), (55, 75, 95), -1)
            cv2.line(img, (24, 8), (24, 26), (70, 90, 110), 2)
            cv2.rectangle(img, (3, 28), (17, 40), (215, 185, 45), -1)
            cv2.rectangle(img, (31, 28), (45, 40), (215, 185, 45), -1)
            cv2.rectangle(img, (16, 32), (32, 46), (30, 32, 38), -1)
            for gx in range(17, 32, 4):
                cv2.line(img, (gx, 32), (gx, 46), (45, 47, 53), 1)
            cv2.rectangle(img, (17, 40), (31, 46), (240, 240, 245), -1)

        elif vtype == "jcb":
            # JCB excavator body + arm
            body_c = (20, 180, 240)  # Caterpillar yellow in BGR
            cv2.rectangle(img, (4, 26), (32, 44), body_c, -1)
            cv2.rectangle(img, (4, 14), (24, 28), body_c, -1)
            cv2.rectangle(img, (6, 16), (22, 26), (55, 80, 105), -1)  # cab window
            cv2.rectangle(img, (26, 30), (36, 40), (70, 70, 75), -1)  # boom base
            # Arm segments
            pts_arm = np.array([[30, 20], [44, 10], [46, 14], [34, 24]], np.int32)
            cv2.fillPoly(img, [pts_arm], (75, 80, 88))
            cv2.rectangle(img, (43, 12), (48, 24), (55, 60, 65), -1)  # bucket
            # Tracks
            cv2.rectangle(img, (4, 42), (32, 46), (35, 35, 38), -1)
            for tx in range(6, 32, 4):
                cv2.line(img, (tx, 42), (tx, 46), (50, 50, 53), 1)

        elif vtype == "bulldozer":
            body_c = (20, 180, 240)
            cv2.rectangle(img, (8, 20), (44, 40), body_c, -1)
            cv2.rectangle(img, (14, 10), (36, 22), body_c, -1)
            cv2.rectangle(img, (16, 12), (34, 20), (55, 80, 105), -1)
            # Blade
            pts_blade = np.array([[2, 16], [10, 16], [10, 40], [2, 44]], np.int32)
            cv2.fillPoly(img, [pts_blade], (120, 125, 130))
            cv2.rectangle(img, (8, 38), (44, 44), (35, 35, 38), -1)
            cv2.rectangle(img, (8, 40), (44, 46), (35, 35, 38), -1)
            for tx in range(10, 44, 4):
                cv2.line(img, (tx, 40), (tx, 46), (50,50,53), 1)

        elif vtype == "tanker_side":
            cab_c = [random.randint(30, 180) for _ in range(3)]
            cv2.rectangle(img, (2, 16), (16, 44), cab_c, -1)
            cv2.rectangle(img, (3, 18), (15, 26), (55, 75, 95), -1)
            # Tanker cylinder
            cv2.ellipse(img, (31, 30), (16, 13), 0, 0, 360, (180, 185, 190), -1)
            cv2.rectangle(img, (15, 17), (47, 43), (180, 185, 190), -1)
            cv2.ellipse(img, (15, 30), (3, 13), 0, 0, 360, (160, 165, 170), -1)
            # Highlight stripe on tanker
            cv2.ellipse(img, (31, 22), (16, 4), 0, 180, 360, (210, 215, 220), 2)
            cv2.circle(img, (9, 45), 5, (10,10,12), -1)
            cv2.circle(img, (36, 45), 5, (10,10,12), -1)

        elif vtype == "tipper_side":
            cab_c = [random.randint(30, 200) for _ in range(3)]
            cv2.rectangle(img, (2, 14), (18, 44), cab_c, -1)
            cv2.rectangle(img, (4, 16), (16, 24), (55, 75, 95), -1)
            # Tipping bed (slightly raised at front)
            pts_bed = np.array([[18, 12], [46, 18], [46, 40], [18, 40]], np.int32)
            cv2.fillPoly(img, [pts_bed], (90, 88, 85))
            cv2.line(img, (18, 12), (46, 18), (70, 68, 65), 2)
            cv2.circle(img, (9, 45), 5, (10,10,12), -1)
            cv2.circle(img, (34, 45), 5, (10,10,12), -1)
            cv2.circle(img, (41, 45), 4, (10,10,12), -1)

        cv2.ellipse(img, (24, 47), (20, 3), 0, 0, 360, (18,18,18), -1)
        return img

    @classmethod
    def _gen_auto_rickshaw(cls, img: np.ndarray) -> np.ndarray:
        """Generate realistic Indian auto-rickshaw — green, yellow, black variants."""
        size = 48
        auto_colors = [
            (30, 130, 30),   # Green (Karnataka, Tamil Nadu)
            (30, 30, 200),   # Blue (Maharashtra private)
            (18, 130, 210),  # Yellow-Green (UP)
            (30, 100, 30),   # Dark green
            (200, 180, 20),  # Yellow (Bihar)
            (25, 25, 25),    # Black (Mumbai)
        ]
        ac = random.choice(auto_colors)
        yellow_top = (20, 200, 240)  # Yellow top panel
        dark = (10, 10, 12)

        view = random.choice(["three_quarter_front", "side", "front", "rear",
                               "three_quarter_rear", "side_close"])

        scale = random.uniform(0.75, 1.0)
        def s(v): return int(v * scale)
        ox = int((size - s(size)) / 2 * random.uniform(0.2, 0.8))
        oy = int((size - s(size)) * random.uniform(0.2, 0.6))
        def p(x, y): return (ox + s(x), oy + s(y))

        if view == "side":
            # Body trapezoid
            pts = np.array([p(6,18), p(42,18), p(44,40), p(4,40)], np.int32)
            cv2.fillPoly(img, [pts], ac)
            # Yellow top stripe
            pts_top = np.array([p(6,12), p(42,12), p(42,18), p(6,18)], np.int32)
            cv2.fillPoly(img, [pts_top], yellow_top)
            # Roof peak
            pts_roof = np.array([p(10,8), p(38,8), p(42,12), p(6,12)], np.int32)
            cv2.fillPoly(img, [pts_roof], yellow_top)
            # Windshield opening
            cv2.rectangle(img, p(10,18), p(30,32), (70, 120, 200), -1)
            cv2.line(img, p(20,18), p(20,32), (30,30,35), 1)
            # Passenger area
            cv2.rectangle(img, p(30,20), p(44,38), tuple(max(0,c-30) for c in ac), -1)
            for py in [22, 26, 30, 34]:
                cv2.line(img, p(31,py), p(43,py), tuple(max(0,c-15) for c in ac), 1)
            # Wheels
            cv2.circle(img, p(10,42), s(5), dark, -1)
            cv2.circle(img, p(10,42), s(3), (50,50,55), -1)
            cv2.circle(img, p(38,42), s(5), dark, -1)
            cv2.circle(img, p(38,42), s(3), (50,50,55), -1)
            # Handlebars
            cv2.line(img, p(4,24), p(8,20), (50,50,55), 2)
            # Headlight
            cv2.ellipse(img, p(4,28), (s(3), s(2)), 0, 0, 360, (220,195,55), -1)

        elif view == "front":
            pts = np.array([p(8,18), p(40,18), p(44,42), p(4,42)], np.int32)
            cv2.fillPoly(img, [pts], ac)
            pts_top = np.array([p(8,12), p(40,12), p(40,18), p(8,18)], np.int32)
            cv2.fillPoly(img, [pts_top], yellow_top)
            pts_roof = np.array([p(12,8), p(36,8), p(40,12), p(8,12)], np.int32)
            cv2.fillPoly(img, [pts_roof], yellow_top)
            # Windshield (open, no glass — Indian auto style)
            cv2.rectangle(img, p(10,18), p(38,30), (60,100,180), -1)
            cv2.line(img, p(24,18), p(24,30), (30,30,35), 2)
            # Headlights (round)
            cv2.ellipse(img, p(10,32), (s(4), s(3)), 0, 0, 360, (220,195,55), -1)
            cv2.ellipse(img, p(38,32), (s(4), s(3)), 0, 0, 360, (220,195,55), -1)
            # Meter box
            cv2.rectangle(img, p(20,32), p(28,38), (30,30,35), -1)
            # 3 wheels (1 front, 2 rear visible)
            cv2.circle(img, p(24,44), s(4), dark, -1)
            cv2.circle(img, p(10,44), s(3), dark, -1)
            cv2.circle(img, p(38,44), s(3), dark, -1)

        elif view in ["three_quarter_front", "three_quarter_rear"]:
            pts = np.array([p(4,20), p(40,17), p(44,42), p(4,44)], np.int32)
            cv2.fillPoly(img, [pts], ac)
            pts_top = np.array([p(6,13), p(40,11), p(40,17), p(4,20)], np.int32)
            cv2.fillPoly(img, [pts_top], yellow_top)
            cv2.rectangle(img, p(6,20), p(30,33), (60,100,180), -1)
            cv2.circle(img, p(10,44), s(5), dark, -1)
            cv2.circle(img, p(36,44), s(5), dark, -1)
            if "front" in view:
                cv2.ellipse(img, p(4,30), (s(3),s(2)), 0, 0, 360, (220,195,55), -1)

        elif view == "rear":
            pts = np.array([p(4,18), p(44,18), p(46,44), p(2,44)], np.int32)
            cv2.fillPoly(img, [pts], ac)
            pts_top = np.array([p(8,12), p(40,12), p(44,18), p(4,18)], np.int32)
            cv2.fillPoly(img, [pts_top], yellow_top)
            # Rear open section
            cv2.rectangle(img, p(8,22), p(40,40), (40,75,130), -1)  # passenger visible
            # Tail lights
            cv2.ellipse(img, p(6,32), (s(3),s(2)), 0, 0, 360, (160,25,25), -1)
            cv2.ellipse(img, p(42,32), (s(3),s(2)), 0, 0, 360, (160,25,25), -1)
            cv2.circle(img, p(10,44), s(5), dark, -1)
            cv2.circle(img, p(38,44), s(5), dark, -1)
            # Number plate
            cv2.rectangle(img, p(18,38), p(30,44), (240,240,245), -1)

        elif view == "side_close":
            # Close-up side — fills frame
            pts = np.array([(2,14), (46,14), (47,44), (1,44)], np.int32)
            cv2.fillPoly(img, [pts], ac)
            cv2.rectangle(img, (2,8), (46,14), yellow_top, -1)
            cv2.rectangle(img, (4,14), (28,30), (60,100,180), -1)
            cv2.line(img, (16,14), (16,30), (30,30,35), 1)
            cv2.rectangle(img, (28,16), (46,38), tuple(max(0,c-20) for c in ac), -1)
            cv2.circle(img, (10,46), 6, dark, -1)
            cv2.circle(img, (38,46), 6, dark, -1)

        cv2.ellipse(img, (24, 47), (18, 3), 0, 0, 360, (20,20,20), -1)
        return img

    @classmethod
    def _gen_bike_with_helmet(cls, img: np.ndarray) -> np.ndarray:
        """Realistic motorcycle rider WITH helmet."""
        size = 48
        helmet_color = random.choice(cls.HELMET_COLORS)
        skin = random.choice(cls.SKIN_TONES)
        bike_colors = [(30,30,35), (20,20,180), (180,20,20), (180,80,20),
                       (20,120,20), (180,180,185), (50,50,55)]
        bike_c = random.choice(bike_colors)

        view = random.choice(["side_ride", "side_stopped", "three_quarter",
                               "top_down", "front_angle", "side_two_up"])

        if view in ["side_ride", "side_stopped"]:
            # Motorcycle body (side view)
            cv2.ellipse(img, (12, 40), (7, 5), 0, 0, 360, (20, 20, 22), -1)   # front wheel
            cv2.ellipse(img, (38, 40), (7, 5), 0, 0, 360, (20, 20, 22), -1)   # rear wheel
            cv2.ellipse(img, (12, 40), (4, 3), 0, 0, 360, (55,55,60), -1)
            cv2.ellipse(img, (38, 40), (4, 3), 0, 0, 360, (55,55,60), -1)

            # Frame
            cv2.line(img, (12, 35), (38, 35), bike_c, 2)
            cv2.line(img, (12, 35), (8, 38), bike_c, 2)
            # Tank + seat
            cv2.ellipse(img, (25, 32), (12, 6), 0, 0, 360, bike_c, -1)
            # Engine
            cv2.rectangle(img, (18, 33), (32, 38), tuple(max(0,c-30) for c in bike_c), -1)
            # Handlebar
            cv2.line(img, (12, 28), (16, 31), (50,50,55), 2)
            cv2.line(img, (12, 28), (10, 30), (50,50,55), 2)

            # Rider silhouette
            # Body
            cv2.ellipse(img, (22, 26), (7, 10), -10, 0, 360, skin, -1)
            cv2.rectangle(img, (16, 24), (30, 34), (random.randint(40,200), random.randint(40,180), random.randint(40,180)), -1)
            # Arms
            cv2.line(img, (16, 26), (12, 30), skin, 3)
            cv2.line(img, (29, 26), (16, 28), skin, 2)

            # Helmet (full-face or half-face)
            helmet_type = random.choice(["full_face", "half_face", "open_face"])
            if helmet_type == "full_face":
                cv2.ellipse(img, (22, 16), (9, 10), 0, 0, 360, helmet_color, -1)
                cv2.ellipse(img, (22, 18), (6, 4), 0, 0, 360, (40, 70, 110), -1)  # visor
                cv2.line(img, (16, 22), (28, 22), tuple(max(0,c-30) for c in helmet_color), 1)
            elif helmet_type == "half_face":
                cv2.ellipse(img, (22, 15), (9, 9), 0, 0, 360, helmet_color, -1)
                cv2.ellipse(img, (22, 21), (9, 3), 0, 0, 360, skin, -1)  # lower face visible
                cv2.ellipse(img, (22, 16), (5, 3), 0, 0, 360, (45, 75, 115), -1)
            else:  # open face
                cv2.ellipse(img, (22, 14), (9, 8), 0, 0, 360, helmet_color, -1)
                cv2.ellipse(img, (22, 18), (7, 5), 0, 180, 360, skin, -1)  # face visible

            # Helmet strap
            cv2.line(img, (15, 20), (14, 23), (40,40,42), 1)
            cv2.line(img, (29, 20), (30, 23), (40,40,42), 1)

            if view == "side_two_up" or random.random() < 0.2:
                pass  # pillion rider not in this view

            # Exhaust
            cv2.line(img, (38, 35), (46, 37), (80,80,85), 2)
            cv2.ellipse(img, (46, 37), (2, 1), 0, 0, 360, (70,70,75), -1)

        elif view == "three_quarter":
            # 3/4 front view
            cv2.ellipse(img, (16, 39), (6, 4), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (34, 39), (6, 4), 0, 0, 360, (20,20,22), -1)
            pts = np.array([(18,32),(30,30),(34,36),(20,38)], np.int32)
            cv2.fillPoly(img, [pts], bike_c)
            cv2.ellipse(img, (24, 24), (8, 11), 0, 0, 360, skin, -1)
            cv2.rectangle(img, (17,22),(31,32),(random.randint(40,200),random.randint(40,180),random.randint(40,180)), -1)
            # Helmet
            cv2.ellipse(img, (24, 14), (9, 10), 0, 0, 360, helmet_color, -1)
            cv2.ellipse(img, (24, 17), (6, 4), 0, 0, 360, (40,70,110), -1)

        elif view == "top_down":
            # Top-down bird's eye
            cv2.ellipse(img, (24, 24), (6, 18), 0, 0, 360, bike_c, -1)  # bike body
            cv2.ellipse(img, (24, 8), (5, 7), 0, 0, 360, (20,20,22), -1)   # front wheel
            cv2.ellipse(img, (24, 40), (5, 7), 0, 0, 360, (20,20,22), -1)  # rear wheel
            cv2.ellipse(img, (24, 18), (5, 6), 0, 0, 360, helmet_color, -1)  # helmet top
            cv2.line(img, (20, 18), (28, 18), tuple(max(0,c-40) for c in helmet_color), 1)

        elif view == "front_angle":
            cv2.ellipse(img, (24, 40), (10, 6), 0, 0, 360, (20,20,22), -1)
            cv2.line(img, (14, 40), (14, 25), bike_c, 3)
            cv2.line(img, (34, 40), (34, 25), bike_c, 3)
            cv2.rectangle(img, (14,22),(34,28), bike_c, -1)
            cv2.ellipse(img, (24, 20), (10, 6), 0, 0, 360, (220,190,55), -1)  # headlight
            cv2.ellipse(img, (24, 13), (9, 10), 0, 0, 360, helmet_color, -1)
            cv2.ellipse(img, (24, 17), (5, 3), 0, 0, 360, (40,70,110), -1)

        elif view == "side_two_up":
            # Two riders (overloaded-like but both have helmets)
            cv2.ellipse(img, (10, 40), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (38, 40), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (24, 32), (14, 6), 0, 0, 360, bike_c, -1)
            # Front rider
            cv2.ellipse(img, (18, 23), (6, 8), 0, 0, 360, skin, -1)
            cv2.ellipse(img, (18, 14), (8, 9), 0, 0, 360, helmet_color, -1)
            # Rear rider (pillion)
            cv2.ellipse(img, (30, 22), (5, 7), -5, 0, 360, random.choice(cls.SKIN_TONES), -1)
            helmet2 = random.choice(cls.HELMET_COLORS)
            cv2.ellipse(img, (30, 14), (7, 8), -5, 0, 360, helmet2, -1)

        cv2.ellipse(img, (24, 46), (15, 2), 0, 0, 360, (20,20,20), -1)
        return img

    @classmethod
    def _gen_bike_no_helmet(cls, img: np.ndarray) -> np.ndarray:
        """Realistic motorcycle rider WITHOUT helmet — violation class."""
        size = 48
        skin = random.choice(cls.SKIN_TONES)
        bike_colors = [(30,30,35),(20,20,180),(180,20,20),(180,80,20),(20,120,20),(180,180,185)]
        bike_c = random.choice(bike_colors)
        hair_colors = [(15,15,18),(25,20,20),(35,28,25),(100,65,30),(200,180,120)]
        hair_c = random.choice(hair_colors)

        view = random.choice(["side_ride", "side_ride2", "three_quarter", "front_angle", "top_down"])

        if view in ["side_ride", "side_ride2"]:
            cv2.ellipse(img, (12, 40), (7, 5), 0, 0, 360, (20, 20, 22), -1)
            cv2.ellipse(img, (38, 40), (7, 5), 0, 0, 360, (20, 20, 22), -1)
            cv2.ellipse(img, (12, 40), (4, 3), 0, 0, 360, (55,55,60), -1)
            cv2.ellipse(img, (38, 40), (4, 3), 0, 0, 360, (55,55,60), -1)
            cv2.line(img, (12, 35), (38, 35), bike_c, 2)
            cv2.ellipse(img, (25, 32), (12, 6), 0, 0, 360, bike_c, -1)
            cv2.rectangle(img, (18, 33), (32, 38), tuple(max(0,c-30) for c in bike_c), -1)
            cv2.line(img, (12, 28), (16, 31), (50,50,55), 2)

            # Rider body
            shirt_c = (random.randint(30,200), random.randint(30,200), random.randint(30,200))
            cv2.ellipse(img, (22, 26), (7, 10), -10, 0, 360, skin, -1)
            cv2.rectangle(img, (16, 24), (30, 34), shirt_c, -1)
            cv2.line(img, (16, 26), (12, 30), skin, 3)

            # BARE HEAD (NO HELMET) — key feature
            hair_style = random.choice(["short", "medium", "wavy"])
            # Face / skull
            cv2.ellipse(img, (22, 16), (8, 10), 0, 0, 360, skin, -1)
            # Hair
            if hair_style == "short":
                cv2.ellipse(img, (22, 12), (8, 6), 0, 180, 360, hair_c, -1)
                cv2.ellipse(img, (14, 16), (3, 3), 0, 0, 360, hair_c, -1)
                cv2.ellipse(img, (30, 16), (3, 3), 0, 0, 360, hair_c, -1)
            elif hair_style == "medium":
                cv2.ellipse(img, (22, 12), (9, 7), 0, 180, 360, hair_c, -1)
                cv2.line(img, (13, 16), (13, 22), hair_c, 2)
                cv2.line(img, (31, 16), (31, 22), hair_c, 2)
            else:  # wavy
                for i in range(3):
                    cv2.ellipse(img, (22+i*3-3, 11), (4, 3), i*20-20, 180, 360, hair_c, -1)
                cv2.line(img, (13, 16), (13, 21), hair_c, 2)
                cv2.line(img, (31, 16), (31, 21), hair_c, 2)

            # Facial features
            cv2.circle(img, (19, 17), 1, (80, 60, 45), -1)  # left eye
            cv2.circle(img, (25, 17), 1, (80, 60, 45), -1)  # right eye
            cv2.arc = None  # no visor — bare face

            if view == "side_ride2":
                # Slightly different angle
                cv2.line(img, (22, 21), (26, 21), (60, 40, 30), 1)  # nose
            cv2.line(img, (38, 35), (46, 37), (80,80,85), 2)

        elif view == "three_quarter":
            cv2.ellipse(img, (16, 39), (6, 4), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (34, 39), (6, 4), 0, 0, 360, (20,20,22), -1)
            pts = np.array([(18,32),(30,30),(34,36),(20,38)], np.int32)
            cv2.fillPoly(img, [pts], bike_c)
            cv2.ellipse(img, (24, 25), (8, 11), 0, 0, 360, skin, -1)
            shirt_c = (random.randint(30,200), random.randint(30,200), random.randint(30,200))
            cv2.rectangle(img, (17,23),(31,33), shirt_c, -1)
            cv2.ellipse(img, (24, 15), (8, 9), 0, 0, 360, skin, -1)
            cv2.ellipse(img, (24, 11), (8, 6), 0, 180, 360, hair_c, -1)
            cv2.circle(img, (21, 16), 1, (70, 50, 35), -1)
            cv2.circle(img, (27, 16), 1, (70, 50, 35), -1)

        elif view == "front_angle":
            cv2.ellipse(img, (24, 40), (10, 6), 0, 0, 360, (20,20,22), -1)
            cv2.line(img, (14,40), (14,25), bike_c, 3)
            cv2.line(img, (34,40), (34,25), bike_c, 3)
            cv2.rectangle(img, (14,22),(34,28), bike_c, -1)
            cv2.ellipse(img, (24, 22), (10, 6), 0, 0, 360, (220,190,55), -1)
            cv2.ellipse(img, (24, 14), (8, 9), 0, 0, 360, skin, -1)
            cv2.ellipse(img, (24, 10), (8, 6), 0, 180, 360, hair_c, -1)

        elif view == "top_down":
            cv2.ellipse(img, (24, 24), (6, 18), 0, 0, 360, bike_c, -1)
            cv2.ellipse(img, (24, 8), (5, 7), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (24, 40), (5, 7), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (24, 18), (5, 7), 0, 0, 360, hair_c, -1)
            cv2.ellipse(img, (24, 20), (4, 5), 0, 0, 360, skin, -1)

        cv2.ellipse(img, (24, 46), (15, 2), 0, 0, 360, (20,20,20), -1)
        return img

    @classmethod
    def _gen_bike_overloaded(cls, img: np.ndarray) -> np.ndarray:
        """Motorcycle with 3 riders (overloaded violation)."""
        skin1 = random.choice(cls.SKIN_TONES)
        skin2 = random.choice(cls.SKIN_TONES)
        skin3 = random.choice(cls.SKIN_TONES)
        bike_c = random.choice([(30,30,35),(20,20,180),(180,20,20),(20,120,20)])
        has_helmet = [random.random() < 0.5 for _ in range(3)]
        h_colors = [random.choice(cls.HELMET_COLORS) for _ in range(3)]
        hair_c = random.choice([(15,15,18),(25,20,20),(35,28,25)])

        # Bike base
        cv2.ellipse(img, (10, 41), (6, 4), 0, 0, 360, (20,20,22), -1)
        cv2.ellipse(img, (38, 41), (6, 4), 0, 0, 360, (20,20,22), -1)
        cv2.ellipse(img, (24, 34), (14, 5), 0, 0, 360, bike_c, -1)
        cv2.rectangle(img, (16, 32), (32, 37), tuple(max(0,c-30) for c in bike_c), -1)

        # Three stacked riders (front to rear)
        positions = [(16, 22, skin1, has_helmet[0], h_colors[0]),
                     (24, 22, skin2, has_helmet[1], h_colors[1]),
                     (32, 22, skin3, has_helmet[2], h_colors[2])]

        for (rx, ry, skin, hhelmet, hc) in positions:
            cv2.ellipse(img, (rx, ry), (5, 7), 0, 0, 360, skin, -1)
            if hhelmet:
                cv2.ellipse(img, (rx, ry-6), (6, 7), 0, 0, 360, hc, -1)
                cv2.ellipse(img, (rx, ry-4), (4, 3), 0, 0, 360, (40,70,110), -1)
            else:
                cv2.ellipse(img, (rx, ry-7), (5, 6), 0, 0, 360, skin, -1)
                cv2.ellipse(img, (rx, ry-10), (5, 4), 0, 180, 360, hair_c, -1)

        cv2.line(img, (38, 34), (46, 37), (80,80,85), 2)
        cv2.ellipse(img, (24, 46), (15, 2), 0, 0, 360, (20,20,20), -1)
        return img

    @classmethod
    def _gen_scooter_with_helmet(cls, img: np.ndarray) -> np.ndarray:
        """Realistic scooter (step-through) rider WITH helmet."""
        skin = random.choice(cls.SKIN_TONES)
        helmet_color = random.choice(cls.HELMET_COLORS)
        scooter_colors = [(245,245,250),(20,20,200),(200,30,30),(200,180,20),
                          (30,140,30),(120,20,140),(240,100,20),(20,20,20)]
        sc = random.choice(scooter_colors)

        view = random.choice(["side_right", "side_left", "three_quarter_front",
                               "three_quarter_rear", "front", "top_down"])

        if view in ["side_right", "side_left"]:
            # Scooter (step-through frame — different from motorbike)
            # Low floor platform
            cv2.rectangle(img, (12, 34), (38, 40), sc, -1)
            # Front leg shield (distinctive scooter shape)
            cv2.rectangle(img, (10, 20), (18, 36), sc, -1)
            cv2.ellipse(img, (14, 20), (5, 8), 0, 180, 360, sc, -1)
            # Rear body + seat
            cv2.rectangle(img, (30, 24), (42, 36), sc, -1)
            cv2.ellipse(img, (36, 24), (8, 5), 0, 180, 360, sc, -1)
            # Handlebar area
            cv2.line(img, (12, 22), (8, 18), (50,50,55), 2)
            cv2.line(img, (8, 18), (6, 20), (50,50,55), 2)
            # Wheels (smaller, scooter-sized)
            cv2.ellipse(img, (14, 42), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (36, 42), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (14, 42), (4, 3), 0, 0, 360, (55,55,60), -1)
            cv2.ellipse(img, (36, 42), (4, 3), 0, 0, 360, (55,55,60), -1)
            # Headlight (round, scooter style)
            cv2.ellipse(img, (10, 26), (4, 4), 0, 0, 360, (225,200,55), -1)
            # Rider
            shirt = (random.randint(40,200), random.randint(40,180), random.randint(40,180))
            cv2.rectangle(img, (16, 22), (30, 34), shirt, -1)
            cv2.ellipse(img, (22, 22), (7, 9), 0, 0, 360, skin, -1)
            # Helmet
            cv2.ellipse(img, (22, 14), (8, 9), 0, 0, 360, helmet_color, -1)
            cv2.ellipse(img, (22, 17), (5, 3), 0, 0, 360, (40,70,110), -1)  # visor
            cv2.line(img, (15, 19), (14, 22), (30,30,32), 1)
            cv2.line(img, (29, 19), (30, 22), (30,30,32), 1)
            # Leg on platform
            cv2.rectangle(img, (16, 32), (26, 36), (random.randint(30,100), random.randint(30,100), random.randint(30,150)), -1)

        elif view == "three_quarter_front":
            cv2.ellipse(img, (16, 41), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (34, 41), (7, 5), 0, 0, 360, (20,20,22), -1)
            pts = np.array([(10,32),(38,29),(40,40),(10,42)], np.int32)
            cv2.fillPoly(img, [pts], sc)
            pts_shield = np.array([(10,20),(20,17),(22,34),(10,36)], np.int32)
            cv2.fillPoly(img, [pts_shield], sc)
            cv2.ellipse(img, (22, 22), (8, 11), 0, 0, 360, skin, -1)
            cv2.rectangle(img, (15,21),(29,32), (random.randint(40,200),random.randint(40,180),random.randint(40,180)), -1)
            cv2.ellipse(img, (22, 13), (8, 9), 0, 0, 360, helmet_color, -1)
            cv2.ellipse(img, (22, 17), (5, 3), 0, 0, 360, (40,70,110), -1)

        elif view == "three_quarter_rear":
            cv2.ellipse(img, (16, 41), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (34, 41), (7, 5), 0, 0, 360, (20,20,22), -1)
            pts = np.array([(8,28),(40,25),(42,42),(8,44)], np.int32)
            cv2.fillPoly(img, [pts], sc)
            cv2.ellipse(img, (22, 22), (8, 11), 0, 0, 360, skin, -1)
            cv2.rectangle(img, (15,21),(29,32), (random.randint(40,200),random.randint(40,180),random.randint(40,180)), -1)
            cv2.ellipse(img, (22, 13), (8, 9), 0, 0, 360, helmet_color, -1)
            cv2.ellipse(img, (8, 34), (3, 2), 0, 0, 360, (165,25,25), -1)
            cv2.ellipse(img, (40, 34), (3, 2), 0, 0, 360, (165,25,25), -1)

        elif view == "front":
            cv2.ellipse(img, (24, 42), (12, 6), 0, 0, 360, (20,20,22), -1)
            cv2.rectangle(img, (12,24),(36,42), sc, -1)
            cv2.rectangle(img, (14,16),(34,26), sc, -1)
            cv2.ellipse(img, (24, 26), (8, 5), 0, 0, 360, (220,190,55), -1)  # headlight
            cv2.ellipse(img, (24, 14), (9, 10), 0, 0, 360, helmet_color, -1)
            cv2.ellipse(img, (24, 17), (5, 3), 0, 0, 360, (40,70,110), -1)

        elif view == "top_down":
            cv2.ellipse(img, (24, 24), (7, 18), 0, 0, 360, sc, -1)
            cv2.ellipse(img, (24, 8), (6, 7), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (24, 40), (6, 7), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (24, 17), (6, 7), 0, 0, 360, helmet_color, -1)
            cv2.line(img, (18, 17), (30, 17), tuple(max(0,c-40) for c in helmet_color), 1)

        cv2.ellipse(img, (24, 46), (15, 2), 0, 0, 360, (20,20,20), -1)
        return img

    @classmethod
    def _gen_scooter_no_helmet(cls, img: np.ndarray) -> np.ndarray:
        """Realistic scooter rider WITHOUT helmet — violation."""
        skin = random.choice(cls.SKIN_TONES)
        hair_c = random.choice([(15,15,18),(25,20,20),(35,28,25),(100,65,30),(200,180,120)])
        scooter_colors = [(245,245,250),(20,20,200),(200,30,30),(200,180,20),
                          (30,140,30),(120,20,140),(240,100,20)]
        sc = random.choice(scooter_colors)

        view = random.choice(["side_right", "three_quarter_front", "front", "top_down",
                               "side_close", "three_quarter_rear"])

        if view in ["side_right", "side_close"]:
            cv2.rectangle(img, (12, 34), (38, 40), sc, -1)
            cv2.rectangle(img, (10, 20), (18, 36), sc, -1)
            cv2.ellipse(img, (14, 20), (5, 8), 0, 180, 360, sc, -1)
            cv2.rectangle(img, (30, 24), (42, 36), sc, -1)
            cv2.ellipse(img, (36, 24), (8, 5), 0, 180, 360, sc, -1)
            cv2.line(img, (12, 22), (8, 18), (50,50,55), 2)
            cv2.ellipse(img, (14, 42), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (36, 42), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (10, 26), (4, 4), 0, 0, 360, (225,200,55), -1)
            shirt = (random.randint(40,200), random.randint(40,180), random.randint(40,180))
            cv2.rectangle(img, (16, 22), (30, 34), shirt, -1)
            cv2.ellipse(img, (22, 22), (7, 9), 0, 0, 360, skin, -1)
            # BARE HEAD
            cv2.ellipse(img, (22, 14), (8, 9), 0, 0, 360, skin, -1)
            cv2.ellipse(img, (22, 11), (8, 6), 0, 180, 360, hair_c, -1)
            cv2.circle(img, (19, 15), 1, (70, 50, 35), -1)
            cv2.circle(img, (25, 15), 1, (70, 50, 35), -1)
            cv2.line(img, (20, 19), (24, 19), (60, 40, 30), 1)  # mouth

        elif view == "three_quarter_front":
            cv2.ellipse(img, (16, 41), (7, 5), 0, 0, 360, (20,20,22), -1)
            cv2.ellipse(img, (34, 41), (7, 5), 0, 0, 360, (20,20,22), -1)
            pts = np.array([(10,32),(38,29),(40,40),(10,42)], np.int32)
            cv2.fillPoly(img, [pts], sc)
            pts_shield = np.array([(10,20),(20,17),(22,34),(10,36)], np.int32)
            cv2.fillPoly(img, [pts_shield], sc)
            cv2.ellipse(img, (22, 22), (8, 11), 0, 0, 360, skin, -1)
            cv2.rectangle(img, (15,21),(29,32),(random.randint(40,200),random.randint(40,180),random.randint(40,180)), -1)
            cv2.ellipse(img, (22, 13), (8, 9), 0, 0, 360, skin, -1)
            cv2.ellipse(img, (22, 9), (8, 6), 0, 180, 360, hair_c, -1)
            cv2.circle(img, (19, 14), 1, (70,50,35), -1)
            cv2.circle(img, (25, 14), 1, (70,50,35), -1)

        elif view in ["three_quarter_rear", "front", "top_down"]:
            cv2.ellipse(img, (24, 42), (12, 6), 0, 0, 360, (20,20,22), -1)
            cv2.rectangle(img, (12,24),(36,42), sc, -1)
            cv2.rectangle(img, (14,16),(34,26), sc, -1)
            cv2.ellipse(img, (24, 27), (8, 5), 0, 0, 360, (220,190,55), -1)
            cv2.ellipse(img, (24, 14), (9, 10), 0, 0, 360, skin, -1)
            cv2.ellipse(img, (24, 9), (9, 6), 0, 180, 360, hair_c, -1)
            cv2.circle(img, (21, 15), 1, (70,50,35), -1)
            cv2.circle(img, (27, 15), 1, (70,50,35), -1)

        cv2.ellipse(img, (24, 46), (15, 2), 0, 0, 360, (20,20,20), -1)
        return img

    @classmethod
    def _gen_scooter_overloaded(cls, img: np.ndarray) -> np.ndarray:
        """Scooter with 3 riders (overloaded violation)."""
        skin1 = random.choice(cls.SKIN_TONES)
        skin2 = random.choice(cls.SKIN_TONES)
        skin3 = random.choice(cls.SKIN_TONES)
        sc = random.choice([(245,245,250),(20,20,200),(200,30,30),(30,140,30)])
        hair_c = random.choice([(15,15,18),(25,20,20),(35,28,25)])
        has_helmets = [random.random() < 0.3 for _ in range(3)]
        h_colors = [random.choice(cls.HELMET_COLORS) for _ in range(3)]

        # Scooter base
        cv2.rectangle(img, (10, 34), (40, 40), sc, -1)
        cv2.rectangle(img, (8, 20), (16, 36), sc, -1)
        cv2.ellipse(img, (12, 20), (5, 8), 0, 180, 360, sc, -1)
        cv2.ellipse(img, (12, 43), (6, 4), 0, 0, 360, (20,20,22), -1)
        cv2.ellipse(img, (36, 43), (6, 4), 0, 0, 360, (20,20,22), -1)

        # Three riders squeezed together
        rider_positions = [(16, 22, skin1, has_helmets[0], h_colors[0]),
                           (24, 22, skin2, has_helmets[1], h_colors[1]),
                           (32, 22, skin3, has_helmets[2], h_colors[2])]
        for (rx, ry, skin, has_h, hc) in rider_positions:
            cv2.ellipse(img, (rx, ry), (5, 7), 0, 0, 360, skin, -1)
            if has_h:
                cv2.ellipse(img, (rx, ry-6), (6, 7), 0, 0, 360, hc, -1)
                cv2.ellipse(img, (rx, ry-4), (4, 3), 0, 0, 360, (40,70,110), -1)
            else:
                cv2.ellipse(img, (rx, ry-7), (5, 6), 0, 0, 360, skin, -1)
                cv2.ellipse(img, (rx, ry-10), (5, 4), 0, 180, 360, hair_c, -1)

        cv2.ellipse(img, (24, 47), (16, 2), 0, 0, 360, (20,20,20), -1)
        return img

    @classmethod
    def generate_sample(cls, class_id: int, size: int = 48) -> np.ndarray:
        """Main entry: generate single realistic 48x48 image for class_id."""
        add_bg_shadow = class_id not in [1, 2, 3, 4, 5, 6]  # road shadow not needed for 2-wheelers
        img = cls._draw_road_background(size, add_shadow=add_bg_shadow)

        if class_id == 0:
            img = cls._gen_car(img)
        elif class_id == 1:
            img = cls._gen_bike_with_helmet(img)
        elif class_id == 2:
            img = cls._gen_bike_no_helmet(img)
        elif class_id == 3:
            img = cls._gen_bike_overloaded(img)
        elif class_id == 4:
            img = cls._gen_scooter_with_helmet(img)
        elif class_id == 5:
            img = cls._gen_scooter_no_helmet(img)
        elif class_id == 6:
            img = cls._gen_scooter_overloaded(img)
        elif class_id == 7:
            img = cls._gen_bus(img)
        elif class_id == 8:
            img = cls._gen_truck(img)
        elif class_id == 9:
            img = cls._gen_auto_rickshaw(img)

        # Optional mild distance blur (simulate far vehicles)
        if random.random() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img


# =====================================================================
# SECTION 5: REAL-WORLD CROP LOADER (backward compatible)
# =====================================================================
def _load_user_crops(crop_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Scans crop_dir and loads images by filename prefix into class-specific pools.
    Prefix mapping (flat directory):
        helmet_*  -> classes 1, 4 (Bike_Compliant, Scooter_Compliant)
        bare_*    -> classes 2, 5 (Bike_No_Helmet, Scooter_No_Helmet)
        car_*     -> class 0 (Car)
        bus_*     -> class 7 (Bus)
        truck_*   -> class 8 (Heavy_Vehicle)
        auto_*    -> class 9 (Auto_Rickshaw)
        bike_*    -> classes 1, 2 (Bike rider, label based on sub-folder)
        scooter_* -> classes 4, 5 (Scooter rider)
    """
    pools: Dict[str, List[np.ndarray]] = {
        "helmet": [], "bare": [], "car": [], "bus": [],
        "truck": [], "auto": [], "bike": [], "scooter": []
    }

    if not os.path.exists(crop_dir):
        return pools

    prefix_map = {
        "helmet_": "helmet",
        "bare_": "bare",
        "car_": "car",
        "bus_": "bus",
        "truck_": "truck",
        "auto_": "auto",
        "bike_": "bike",
        "scooter_": "scooter",
    }

    for fname in sorted(os.listdir(crop_dir)):
        fpath = os.path.join(crop_dir, fname)
        if os.path.isdir(fpath):
            continue
        matched_key = None
        for prefix, key in prefix_map.items():
            if fname.lower().startswith(prefix):
                matched_key = key
                break
        if matched_key is None:
            continue
        img = cv2.imread(fpath)
        if img is not None and img.size > 0:
            pools[matched_key].append(img)

    print(f"[DATASET] Real-world crops loaded:")
    for k, v in pools.items():
        if v:
            print(f"           {k}: {len(v)} images")
    return pools


# =====================================================================
# SECTION 6: PYTORCH INDUSTRIAL DATASET & STRATIFIED DATALOADER
# =====================================================================
class TrafficJunctionDataset(Dataset):
    """
    Traffic Junction Dataset v4.0 — Highly Realistic Synthetic + Real Crops.
    - All 10 classes have rich, diverse synthetic generation
    - Real-world crops blended at 75% ratio when available
    - Class-balanced with special emphasis on historically confused classes
    """
    def __init__(
        self,
        samples_per_class: int = 3000,
        is_training: bool = True,
        use_augmentation: bool = True,
        real_blend_ratio: float = 0.75
    ):
        self.samples_per_class = samples_per_class
        self.is_training = is_training
        self.use_augmentation = use_augmentation and is_training
        self.real_blend_ratio = real_blend_ratio
        self.preprocessor = AdaptiveEdgeFusionPreprocessor(target_size=(48, 48))
        self.augmenter = RTXTrafficAugmenter() if self.use_augmentation else None
        self.total_classes = len(TRAFFIC_CLASSES)
        self.num_samples = self.samples_per_class * self.total_classes

        # Labels: balanced across all classes
        self.labels = [i % self.total_classes for i in range(self.num_samples)]

        # Load user crops
        self.user_crops: Dict[str, List[np.ndarray]] = {}
        possible_dirs = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user_crops"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_crops"),
            "user_crops",
            "../user_crops",
        ]
        for cdir in possible_dirs:
            if os.path.exists(cdir):
                self.user_crops = _load_user_crops(cdir)
                break

        # Generate raw images
        print(f"[DATASET] Generating {self.num_samples:,} samples ({self.samples_per_class}/class x {self.total_classes} classes)...")
        rng_state = random.getstate()
        random.seed(42)
        np.random.seed(42)

        self.raw_images = np.empty((self.num_samples, 48, 48, 3), dtype=np.uint8)
        for i in range(self.num_samples):
            cls_id = self.labels[i]
            self.raw_images[i] = self._generate_one(cls_id)

        random.setstate(rng_state)
        print(f"[DATASET] Dataset ready. Total: {self.num_samples:,} samples.")

    def _generate_one(self, cls_id: int) -> np.ndarray:
        """Generate a single raw BGR image for the given class ID."""
        # Map class ID to user crop pool key
        pool_key = None
        if cls_id in [1, 4]:
            pool_key = "helmet"
        elif cls_id in [2, 5]:
            pool_key = "bare"
        elif cls_id == 0:
            pool_key = "car"
        elif cls_id == 7:
            pool_key = "bus"
        elif cls_id == 8:
            pool_key = "truck"
        elif cls_id == 9:
            pool_key = "auto"

        # Real-world benchmark crops: ALWAYS use 100% real crops for helmet & bare head
        # (Procedural generator generates entire bikes, which corrupts head-crop classification)
        use_ratio = 1.0 if pool_key in ["helmet", "bare"] else self.real_blend_ratio
        if pool_key and self.user_crops.get(pool_key) and random.random() < use_ratio:
            chosen = random.choice(self.user_crops[pool_key])
            img = cv2.resize(chosen, (48, 48), interpolation=cv2.INTER_AREA)
            # Random horizontal flip
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            # Brightness/gamma augmentation
            gamma = random.uniform(0.75, 1.25)
            img = np.clip(img.astype(np.float32) * gamma, 0, 255).astype(np.uint8)
            # Random saturation boost/reduce
            if random.random() < 0.3:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[...,1] = np.clip(hsv[...,1] * random.uniform(0.6, 1.4), 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return img

        # Fallback: photorealistic procedural synthesis
        return ProceduralTrafficGenerator.generate_sample(cls_id)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cls_id = self.labels[idx]
        raw_bgr = self.raw_images[idx]

        # Stage-1: Adaptive Contrast & Edge Fusion
        processed_rgb = self.preprocessor(raw_bgr)

        # RTX GPU Data Augmentation (Train mode only)
        if self.use_augmentation and self.augmenter:
            processed_rgb = self.augmenter(processed_rgb)

        # Convert to PyTorch Tensor: Shape (3, 48, 48), Float32 in [0, 255]
        tensor_img = torch.from_numpy(processed_rgb).permute(2, 0, 1).to(torch.float32)
        return tensor_img, cls_id


def create_stratified_dataloaders(
    dataset: TrafficJunctionDataset,
    batch_size: int = 128,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates strictly stratified 80% Train, 10% Val, 10% Test DataLoaders.
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-4)

    indices_by_class: Dict[int, List[int]] = {i: [] for i in range(len(TRAFFIC_CLASSES))}
    for idx, lbl in enumerate(dataset.labels):
        indices_by_class[lbl].append(idx)

    train_indices, val_indices, test_indices = [], [], []
    rng = random.Random(42)
    for cls_id, idx_list in indices_by_class.items():
        shuffled = idx_list.copy()
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train:n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val:])

    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("[TEST] Initializing dataset pipeline v4.0...")
    ds = TrafficJunctionDataset(samples_per_class=10, is_training=True)
    img_tensor, label = ds[0]
    print(f"Tensor Shape: {img_tensor.shape}, Range: [{img_tensor.min():.1f}, {img_tensor.max():.1f}]")
    print(f"Class: {label} ({TRAFFIC_CLASSES[label]}), Violation: {is_violation_class(label)}")

    # Visual test — save sample images per class
    import os
    os.makedirs("sample_output", exist_ok=True)
    ds_test = TrafficJunctionDataset(samples_per_class=5, is_training=False)
    for cls_id in range(10):
        idx = cls_id * 5
        img = ds_test.raw_images[idx]
        cv2.imwrite(f"sample_output/class_{cls_id}_{TRAFFIC_CLASSES[cls_id]}.jpg", img)
    print("[SUCCESS] Sample images saved to sample_output/")
