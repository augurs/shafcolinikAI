import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

mp_pose = mp.solutions.pose

class BodyMeasurementExtractor:
    def __init__(self, actual_height_cm: float, config: dict):
        self.actual_height_cm = actual_height_cm
        self.config = config
        self.pose = mp_pose.Pose(static_image_mode=True)
    
    def __del__(self):
        self.pose.close()

    def _get_point(self, landmarks, part, width, height):
        lm = landmarks[mp_pose.PoseLandmark[part].value]
        return int(lm.x * width), int(lm.y * height)

    def _distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def extract(self, image: Image.Image):
        height, width = image.size
        image_np = np.array(image.convert("RGB"))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = self.pose.process(image_np)

        if not results.pose_landmarks:
            return {"success": False, "message": "No person detected."}

        landmarks = results.pose_landmarks.landmark

        # Config shortcuts
        cfg = self.config
        weights = cfg["weights"]
        visibility_threshold = cfg["visibility_threshold"]
        key_visibility_threshold = cfg["key_visibility_threshold"]
        expected_ratios = cfg["expected_ratios"]
        ratio_tolerances = cfg["ratio_tolerances"]

        visible = [lm for lm in landmarks if lm.visibility > visibility_threshold]
    
        visible_ratio = len(visible) / len(landmarks)
        
        for part in mp_pose.PoseLandmark:
            index = part.value
            name = part.name
            lm = landmarks[index]
            x = lm.x * width
            y = lm.y * height
            # print(f"{name}: ({x:.1f}, {y:.1f}), visibility={lm.visibility:.2f}")

        try:
            # Key measurements (pixels)
            chest_px = self._distance(self._get_point(landmarks, "LEFT_SHOULDER", width, height),
                                      self._get_point(landmarks, "RIGHT_SHOULDER", width, height))
            waist_px = self._distance(self._get_point(landmarks, "LEFT_HIP", width, height),
                                      self._get_point(landmarks, "RIGHT_HIP", width, height))
            hips_px = waist_px * 1.1
            inseam_px = self._distance(self._get_point(landmarks, "LEFT_HIP", width, height),
                                       self._get_point(landmarks, "LEFT_ANKLE", width, height))
            thigh_px = self._distance(self._get_point(landmarks, "LEFT_HIP", width, height),
                                      self._get_point(landmarks, "LEFT_KNEE", width, height))
            neck_px = chest_px * 0.4
            arm_px = self._distance(self._get_point(landmarks, "LEFT_SHOULDER", width, height),
                                    self._get_point(landmarks, "LEFT_WRIST", width, height))

            body_top = self._get_point(landmarks, "NOSE", width, height)[1]
            body_bottom = max(self._get_point(landmarks, "LEFT_HEEL", width, height)[1],
                              self._get_point(landmarks, "RIGHT_HEEL", width, height)[1])
            pixel_body_height = abs(body_bottom - body_top)

            if pixel_body_height == 0:
                return {"success": False, "message": "Cannot compute scale (body height = 0)."}

            cm_per_pixel = self.actual_height_cm / pixel_body_height
            to_cm = lambda px: round(px * cm_per_pixel, 1)

            body_coverage_pct = min((pixel_body_height / height) * 100, 100)

            # Key landmarks visibility
            key_landmarks = [
                'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP',
                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL'
            ]
            key_visible = sum(
                1 for landmark_name in key_landmarks
                if landmarks[mp_pose.PoseLandmark[landmark_name].value].visibility > key_visibility_threshold
            )
            print("Key Visibility:", key_visible, "/", len(key_landmarks))
            pose_quality = (key_visible / len(key_landmarks)) * 100
            print("Pose Quality:", pose_quality)
            # Proportion score
            shoulder_center = np.mean([
                self._get_point(landmarks, "LEFT_SHOULDER", width, height),
                self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
            ], axis=0)
            hip_center = np.mean([
                self._get_point(landmarks, "LEFT_HIP", width, height),
                self._get_point(landmarks, "RIGHT_HIP", width, height)
            ], axis=0)
            shoulder_to_hip = self._distance(shoulder_center, hip_center)
            print("Shoulder Center:", shoulder_center, "Hip Center:", hip_center, "Shoulder to Hip:", shoulder_to_hip)

            torso_ratio = shoulder_to_hip / pixel_body_height
            chest_ratio = chest_px / pixel_body_height

            torso_accuracy = max(0, 100 - abs(torso_ratio - expected_ratios["torso_to_height"]) * ratio_tolerances["torso_weight"])
            chest_accuracy = max(0, 100 - abs(chest_ratio - expected_ratios["chest_to_height"]) * ratio_tolerances["chest_weight"])
            proportion_score = (torso_accuracy + chest_accuracy) / 2
            
            accuracy_score = round(
                visible_ratio * weights["visible_ratio"] +
                (pose_quality / 100) * weights["pose_quality"] +
                (body_coverage_pct / 100) * weights["body_coverage_pct"] +
                (proportion_score / 100) * weights["proportion_score"],
                2
            )

            return {
                "success": True,
                "chest_cm": to_cm(chest_px),
                "waist_cm": to_cm(waist_px),
                "hips_cm": to_cm(hips_px),
                "inseam_cm": to_cm(inseam_px),
                "thigh_cm": to_cm(thigh_px),
                "neck_cm": to_cm(neck_px),
                "arm_length_cm": to_cm(arm_px),
                "accuracy": accuracy_score,
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





# visible_ratio = 0.76 # len(visibility_value) / len(landmarks)
# pose_quality = 81.82 # (key_visibility_value / len(key_landmarks)) * 100
# body_coverage_pct = 80
# proportion_score = 85 # (torso_accuracy + chest_accuracy) / 2

# # Pre defined values
# weights_visible_ratio = 0.3
# weights_pose_quality = 0.25
# weights_body_coverage_pct = 0.25
# weights_proportion_score = 0.2

# accuracy = round(
#     visible_ratio * weights_visible_ratio +
#     (pose_quality / 100) * weights_pose_quality +
#     (body_coverage_pct / 100) * weights_body_coverage_pct +
#     (proportion_score / 100) * weights_proportion_score,
#     2
# )
# print(accuracy)

