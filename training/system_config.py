"""
AO.SNN Canonical System Configuration (Single Source of Truth)
Defines all architecture constants, class taxonomies, fixed-point precision formats,
and hardware constraints across PyTorch software, exports, RTL, and emulators.
"""

from typing import Dict, Set

# ==============================================================================
# SECTION 1: CANONICAL ARCHITECTURE SPECIFICATION
# ==============================================================================
CONFIG_VERSION: str = "3.1.0-canonical"
INPUT_RESOLUTION: tuple = (48, 48, 3)
INPUT_DIMS: int = 6912            # 48 * 48 * 3
HIDDEN_NEURONS: int = 64          # Selected Architecture matching trained checkpoint
OUTPUT_CLASSES: int = 10
TIME_STEPS: int = 8               # Discrete TTFS steps: t in {0, 1, ..., 7}

# ==============================================================================
# SECTION 2: FIXED-POINT & NEUROMORPHIC ARITHMETIC
# ==============================================================================
WEIGHT_FORMAT: str = "Signed Q4.4 (INT8)"
WEIGHT_SCALE: float = 16.0        # 2^4
WEIGHT_MIN: float = -8.0
WEIGHT_MAX: float = 7.9375
WEIGHT_INT_MIN: int = -128
WEIGHT_INT_MAX: int = 127

ACCUMULATOR_BITS: int = 16        # Signed Q8.8 in hardware registers
V_THRESHOLD_FLOAT: float = 1.0
V_THRESHOLD_FIXED: int = 0x0100   # 1.0 * 256 in Q8.8

LEAK_ALPHA_FLOAT: float = 0.75    # Shift-add: (V >> 1) + (V >> 2)
SURROGATE_BETA: float = 10.0

W1_WEIGHT_COUNT: int = INPUT_DIMS * HIDDEN_NEURONS   # 6912 * 64 = 442,368
W2_WEIGHT_COUNT: int = HIDDEN_NEURONS * OUTPUT_CLASSES # 64 * 10 = 640
TOTAL_WEIGHT_BYTES: int = W1_WEIGHT_COUNT + W2_WEIGHT_COUNT # 443,008 bytes

# ==============================================================================
# SECTION 3: TRAFFIC CLASS TAXONOMY & VIOLATION MATRIX
# ==============================================================================
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

VIOLATION_CLASSES: Set[int] = {2, 3, 5, 6}

def is_violation_class(class_id: int) -> bool:
    """Returns True if the class ID warrants hardware GPIO alarm trigger."""
    return class_id in VIOLATION_CLASSES

# ==============================================================================
# SECTION 4: HARDWARE TARGET CONSTRAINTS (Microchip PolarFire SoC MPFS250T)
# ==============================================================================
FPGA_TARGET: str = "Microchip PolarFire SoC MPFS250T-FCVG484E"
CLOCK_FREQUENCY_MHZ: float = 100.0
CLOCK_PERIOD_NS: float = 10.0

# Exact cycle budget:
# 1,728 feature accumulation cycles (6912 features / 4 parallel spikes per cycle)
# + 8 step leak cycles
# + 64 hidden-to-output accumulation cycles (64 hidden neurons)
# + 8 output LIF update cycles
# = 1,808 cycles @ 100MHz = 18.08 us (< 20.0 us budget)
TOTAL_CLOCK_CYCLES: int = 1728 + 8 + HIDDEN_NEURONS + 8 # 1,808
TOTAL_LATENCY_US: float = TOTAL_CLOCK_CYCLES * (CLOCK_PERIOD_NS / 1000.0) # 18.08 us
MAX_LATENCY_BUDGET_US: float = 20.0
MAX_LSRAM_BLOCKS: int = 252       # PolarFire MPFS250T total LSRAM blocks (18Kib each)
