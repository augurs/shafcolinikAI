import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from typing import Dict, Tuple, Optional

mp_pose = mp.solutions.pose

class BodyMeasurementExtractor:
    def __init__(self, actual_height_cm: float, config: dict):
        self.actual_height_cm = actual_height_cm
        self.config = config
        self.pose = mp_pose.Pose(static_image_mode=True)

    def __del__(self):
        self.pose.close()

    def _get_point(self, landmarks, part, width, height) -> Tuple[int, int]:
        lm = landmarks[mp_pose.PoseLandmark[part].value]
        return int(lm.x * width), int(lm.y * height)

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _calculate_body_height_and_scale(self, landmarks, width, height) -> Tuple[float, float]:
        top = self._get_point(landmarks, "NOSE", width, height)[1]
        bottom = max(
            self._get_point(landmarks, "LEFT_HEEL", width, height)[1],
            self._get_point(landmarks, "RIGHT_HEEL", width, height)[1]
        )
        pixel_body_height = abs(bottom - top)
        cm_per_pixel = self.actual_height_cm / pixel_body_height
        return pixel_body_height, cm_per_pixel

    def _estimate_chest(self, left_shoulder, right_shoulder, to_cm) -> float:
        chest_width = self._distance(left_shoulder, right_shoulder)
        return to_cm(chest_width * 2.2)

    def _estimate_waist(self, left_shoulder, left_hip, right_hip, to_cm) -> float:
        # Use shoulder-to-shoulder distance but adjust for proper chest level
        waist_y = int((left_shoulder[1] + left_hip[1]) / 2)
        waist_left = (left_hip[0], waist_y)
        waist_right = (right_hip[0], waist_y)
        waist_width = self._distance(waist_left, waist_right)

        return to_cm(waist_width * 2.1)

    def _estimate_hips(self, left_hip, right_hip, to_cm) -> float:
        hip_width = self._distance(left_hip, right_hip)
        return to_cm(hip_width * 2.0)

    def _get_size_recommendations(self, chest_cm, waist_cm, inseam_cm) -> Dict[str, str]:
        if chest_cm <= 0:
            return {}
        if chest_cm < 86:
            return {'Recommended Size': 'XS'}
        elif chest_cm < 91:
            return {'Recommended Size': 'S'}
        elif chest_cm < 97:
            return {'Recommended Size': 'M'}
        elif chest_cm < 102:
            return {'Recommended Size': 'L'}
        elif chest_cm < 107:
            return {'Recommended Size': 'XL'}
        else:
            return {'Recommended Size': 'XXL+'}

    def _calculate_pose_quality(self, landmarks) -> float:
        key_landmarks = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_HEEL', 'RIGHT_HEEL'
        ]
        visible_count = sum(
            1 for name in key_landmarks
            if landmarks[mp_pose.PoseLandmark[name].value].visibility > self.config["key_visibility_threshold"]
        )
        return (visible_count / len(key_landmarks)) * 100

    def _calculate_proportion_score(self, shoulder_center, hip_center, chest_px, pixel_body_height) -> float:
        torso_ratio = self._distance(shoulder_center, hip_center) / pixel_body_height
        chest_ratio = chest_px / pixel_body_height

        torso_accuracy = max(0, 100 - abs(
            torso_ratio - self.config["expected_ratios"]["torso_to_height"]
        ) * self.config["ratio_tolerances"]["torso_weight"])

        chest_accuracy = max(0, 100 - abs(
            chest_ratio - self.config["expected_ratios"]["chest_to_height"]
        ) * self.config["ratio_tolerances"]["chest_weight"])

        return (torso_accuracy + chest_accuracy) / 2

    def extract(self, image: Image.Image) -> Dict:
        width, height = image.size
        image_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        results = self.pose.process(image_np)
        if not results.pose_landmarks:
            return {"success": False, "message": "No person detected."}

        landmarks = results.pose_landmarks.landmark
        visible_ratio = sum(1 for lm in landmarks if lm.visibility > self.config["visibility_threshold"]) / len(landmarks)

        try:
            # Get key points
            left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
            right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
            left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
            right_hip = self._get_point(landmarks, "RIGHT_HIP", width, height)

            # Body height and scale
            pixel_body_height, cm_per_pixel = self._calculate_body_height_and_scale(landmarks, width, height)
            to_cm = lambda px: round(px * cm_per_pixel, 1)

            # Chest, waist, hips
            chest_circumference_cm = self._estimate_chest(left_shoulder, right_shoulder, to_cm)
            waist_circumference_cm = self._estimate_waist(left_shoulder, left_hip, right_hip, to_cm)
            hips_circumference_cm = self._estimate_hips(left_hip, right_hip, to_cm)

            # Other linear measurements
            inseam_cm = to_cm(self._distance(left_hip, self._get_point(landmarks, "LEFT_ANKLE", width, height)))
            thigh_cm = to_cm(self._distance(left_hip, self._get_point(landmarks, "LEFT_KNEE", width, height)))
            arm_length_cm = to_cm(self._distance(left_shoulder, self._get_point(landmarks, "LEFT_WRIST", width, height)))
            neck_cm = to_cm(self._distance(left_shoulder, right_shoulder) * 0.4)

            # Scoring
            pose_quality = self._calculate_pose_quality(landmarks)
            body_coverage_pct = min((pixel_body_height / height) * 100, 100)
            shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)
            hip_center = np.mean([left_hip, right_hip], axis=0)
            proportion_score = self._calculate_proportion_score(shoulder_center, hip_center,
                                                                self._distance(left_shoulder, right_shoulder),
                                                                pixel_body_height)

            # Final accuracy score
            weights = self.config["weights"]
            accuracy = round(
                visible_ratio * weights["visible_ratio"] +
                (pose_quality / 100) * weights["pose_quality"] +
                (body_coverage_pct / 100) * weights["body_coverage_pct"] +
                (proportion_score / 100) * weights["proportion_score"], 2
            )

            # Size recommendation
            recommendations = self._get_size_recommendations(chest_circumference_cm, waist_circumference_cm, inseam_cm)
            print("Size recommendations:",recommendations)
            return {
                "success": True,
                "chest_cm": chest_circumference_cm,
                "waist_cm": waist_circumference_cm,
                "hips_cm": hips_circumference_cm,
                "inseam_cm": inseam_cm,
                "thigh_cm": thigh_cm,
                "neck_cm": neck_cm,
                "arm_length_cm": arm_length_cm,
                "accuracy": accuracy,
                "recommendations": recommendations,
                "debug_info": {
                    "visible_ratio": round(visible_ratio, 3),
                    "pose_quality": round(pose_quality, 2),
                    "body_coverage_pct": round(body_coverage_pct, 2),
                    "proportion_score": round(proportion_score, 2),
                    "pixel_body_height": pixel_body_height,
                    "cm_per_pixel": round(cm_per_pixel, 4)
                }
            }

        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
        
