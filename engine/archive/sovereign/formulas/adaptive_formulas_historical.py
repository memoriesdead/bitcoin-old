"""
MULTI-SCALE ADAPTIVE FORMULAS
==================================================
Generated: 2025-12-16T18:24:23.729212
Data: 6184 days of Bitcoin history
Patterns validated: 6

Each formula only fires when its specific (scale, regime) conditions are met.
This is the RenTech methodology: conditional edge exploitation.
"""

from typing import Dict, Optional

SCALES = {1: "1d", 3: "3d", 7: "7d", 14: "14d", 30: "30d", 90: "90d"}
REGIMES = {0: "ACCUMULATION", 1: "DISTRIBUTION", 2: "NEUTRAL", 3: "CAPITULATION", 4: "EUPHORIA"}


ADAPTIVE_FORMULAS = [
    {
        "name": "tx_spike_long",
        "direction": 1,  # LONG
        "total_occurrences": 490,
        "best_scale": 3,
        "best_regime": 0,
        "best_edge": 0.233333,
        "conditions": {
            (1, 0): {"win_rate": 0.5698, "edge": 0.0698, "n": 86, "ci_low": 0.4651},
            (1, 2): {"win_rate": 0.5952, "edge": 0.0952, "n": 126, "ci_low": 0.5079},
            (1, 4): {"win_rate": 0.6250, "edge": 0.1250, "n": 32, "ci_low": 0.4375},
            (3, 0): {"win_rate": 0.7333, "edge": 0.2333, "n": 30, "ci_low": 0.5667},
            (3, 4): {"win_rate": 0.6585, "edge": 0.1585, "n": 41, "ci_low": 0.5122},
        },
    },
    {
        "name": "tx_low_mean_revert",
        "direction": 1,  # LONG
        "total_occurrences": 325,
        "best_scale": 3,
        "best_regime": 3,
        "best_edge": 0.196970,
        "conditions": {
            (1, 3): {"win_rate": 0.5969, "edge": 0.0969, "n": 129, "ci_low": 0.5116},
            (3, 1): {"win_rate": 0.6071, "edge": 0.1071, "n": 28, "ci_low": 0.4286},
            (3, 3): {"win_rate": 0.6970, "edge": 0.1970, "n": 33, "ci_low": 0.5455},
        },
    },
    {
        "name": "whale_spike_long",
        "direction": 1,  # LONG
        "total_occurrences": 432,
        "best_scale": 1,
        "best_regime": 0,
        "best_edge": 0.153543,
        "conditions": {
            (1, 0): {"win_rate": 0.6535, "edge": 0.1535, "n": 127, "ci_low": 0.5669},
            (1, 4): {"win_rate": 0.5349, "edge": 0.0349, "n": 86, "ci_low": 0.4186},
            (3, 4): {"win_rate": 0.5921, "edge": 0.0921, "n": 76, "ci_low": 0.4737},
        },
    },
    {
        "name": "whale_low_accumulate",
        "direction": 1,  # LONG
        "total_occurrences": 184,
        "best_scale": 7,
        "best_regime": 3,
        "best_edge": 0.384615,
        "conditions": {
            (1, 3): {"win_rate": 0.5455, "edge": 0.0455, "n": 110, "ci_low": 0.4455},
            (3, 3): {"win_rate": 0.6875, "edge": 0.1875, "n": 48, "ci_low": 0.5417},
            (7, 3): {"win_rate": 0.8846, "edge": 0.3846, "n": 26, "ci_low": 0.7692},
        },
    },
    {
        "name": "value_spike_long",
        "direction": 1,  # LONG
        "total_occurrences": 403,
        "best_scale": 1,
        "best_regime": 0,
        "best_edge": 0.128205,
        "conditions": {
            (1, 0): {"win_rate": 0.6282, "edge": 0.1282, "n": 78, "ci_low": 0.5256},
            (1, 4): {"win_rate": 0.5887, "edge": 0.0887, "n": 124, "ci_low": 0.5081},
            (3, 0): {"win_rate": 0.5679, "edge": 0.0679, "n": 81, "ci_low": 0.4568},
            (3, 4): {"win_rate": 0.5821, "edge": 0.0821, "n": 67, "ci_low": 0.4478},
            (7, 4): {"win_rate": 0.6038, "edge": 0.1038, "n": 53, "ci_low": 0.4528},
        },
    },
    {
        "name": "all_low_contrarian",
        "direction": 1,  # LONG
        "total_occurrences": 195,
        "best_scale": 7,
        "best_regime": 3,
        "best_edge": 0.285714,
        "conditions": {
            (1, 3): {"win_rate": 0.5514, "edge": 0.0514, "n": 107, "ci_low": 0.4579},
            (3, 3): {"win_rate": 0.6944, "edge": 0.1944, "n": 36, "ci_low": 0.5278},
            (7, 3): {"win_rate": 0.7857, "edge": 0.2857, "n": 28, "ci_low": 0.6429},
            (14, 3): {"win_rate": 0.5833, "edge": 0.0833, "n": 24, "ci_low": 0.3750},
        },
    },
]


def get_signal(pattern_name: str, scale: int, regime: int) -> Optional[Dict]:
    """
    Get signal if conditions are valid for this pattern.
    Returns None if current (scale, regime) is not in valid conditions.
    """
    for formula in ADAPTIVE_FORMULAS:
        if formula["name"] == pattern_name:
            condition = formula["conditions"].get((scale, regime))
            if condition:
                return {
                    "name": formula["name"],
                    "direction": formula["direction"],
                    "edge": condition["edge"],
                    "win_rate": condition["win_rate"],
                    "confidence": condition["ci_low"],
                }
    return None