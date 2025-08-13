# Example usage with visualization
def create_config():
    return {
    "visibility_threshold": 0.5,
    "key_visibility_threshold": 0.7,
    "weights": {
        "visible_ratio": 30,
        "pose_quality": 25,
        "body_coverage_pct": 25,
        "proportion_score": 20
    },
    "expected_ratios": {
        "torso_to_height": 0.35,
        "chest_to_height": 0.15
    },
    "ratio_tolerances": {
        "torso_weight": 300,
        "chest_weight": 500
    }
    }

def create_improved_config():
    return {
        "visibility_threshold": 0.7,
        "key_visibility_threshold": 0.8,
        "weights": {
            "visible_ratio": 0.25,
            "pose_quality": 0.35,
            "body_coverage_pct": 0.25,
            "body_angle": 0.15
        },
        "expected_ratios": {
            "torso_to_height": 0.35,
            "chest_to_height": 0.15
        },
        "ratio_tolerances": {
            "torso_weight": 500,
            "chest_weight": 800
        }
    }

