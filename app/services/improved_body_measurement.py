import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, Optional
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ImprovedBodyMeasurementExtractor:
    def __init__(self, actual_height_cm: float, config: dict):
        self.actual_height_cm = actual_height_cm
        self.config = config
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7)
        self.measurement_points = {}

    def __del__(self):
        self.pose.close()

    # ===== Helper functions =====
    def _get_xyz(self, landmarks, part, width, height):
        """Return (x, y, z) in image-scaled units."""
        lm = landmarks[mp_pose.PoseLandmark[part].value]
        return (lm.x * width, lm.y * height, lm.z * width)  # scale z by width

    def _distance_3d(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_landmark_confidence(self, landmarks, part):
        return landmarks[mp_pose.PoseLandmark[part].value].visibility

    # ===== Core calculations =====
    def _calculate_body_height_and_scale(self, landmarks, width, height):
        head_top_y = self._get_xyz(landmarks, "NOSE", width, height)[1]
        feet_points_y = [
            self._get_xyz(landmarks, "LEFT_HEEL", width, height)[1],
            self._get_xyz(landmarks, "RIGHT_HEEL", width, height)[1],
            self._get_xyz(landmarks, "LEFT_FOOT_INDEX", width, height)[1],
            self._get_xyz(landmarks, "RIGHT_FOOT_INDEX", width, height)[1]
        ]
        bottom_y = max(feet_points_y)
        pixel_body_height = abs(bottom_y - head_top_y)
        cm_per_pixel = self.actual_height_cm / pixel_body_height

        self.measurement_points['height'] = {
            'start': (int(self._get_xyz(landmarks, "NOSE", width, height)[0]), int(head_top_y)),
            'end': (int(self._get_xyz(landmarks, "LEFT_HEEL", width, height)[0]), int(bottom_y)),
            'measurement': self.actual_height_cm
        }
        return pixel_body_height, cm_per_pixel

    def _calculate_body_angle(self, landmarks, width, height):
        left_shoulder = self._get_xyz(landmarks, "LEFT_SHOULDER", width, height)
        right_shoulder = self._get_xyz(landmarks, "RIGHT_SHOULDER", width, height)
        left_hip = self._get_xyz(landmarks, "LEFT_HIP", width, height)
        right_hip = self._get_xyz(landmarks, "RIGHT_HIP", width, height)
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                           (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2,
                      (left_hip[1] + right_hip[1]) / 2)
        angle = math.atan2(hip_center[0] - shoulder_center[0],
                           hip_center[1] - shoulder_center[1])
        return abs(math.degrees(angle))

    def _estimate_chest(self, landmarks, width, height, to_cm, body_angle):
        left_shoulder = self._get_xyz(landmarks, "LEFT_SHOULDER", width, height)
        right_shoulder = self._get_xyz(landmarks, "RIGHT_SHOULDER", width, height)
        shoulder_width = self._distance_3d(left_shoulder, right_shoulder)
        chest_width = shoulder_width * 0.88  # taper from shoulders to chest
        angle_factor = 1.0 if body_angle < 8 else (1.0 + (body_angle - 8) * 0.01)
        adjusted_width = chest_width * angle_factor
        chest_cm = to_cm(adjusted_width * 2.6)  # multiplier for circumference

        # Store for drawing
        cx = (left_shoulder[0] + right_shoulder[0]) / 2
        cy = (left_shoulder[1] + right_shoulder[1]) / 2 + 35
        self.measurement_points['chest'] = {
            'start': (int(cx - chest_width / 2), int(cy)),
            'end': (int(cx + chest_width / 2), int(cy)),
            'measurement': round(chest_cm, 1)
        }
        return round(chest_cm, 1)

    def _estimate_waist(self, landmarks, width, height, to_cm, body_angle):
        left_shoulder = self._get_xyz(landmarks, "LEFT_SHOULDER", width, height)
        right_shoulder = self._get_xyz(landmarks, "RIGHT_SHOULDER", width, height)
        left_hip = self._get_xyz(landmarks, "LEFT_HIP", width, height)
        right_hip = self._get_xyz(landmarks, "RIGHT_HIP", width, height)
        waist_ratio = 0.45
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        waist_y = shoulder_center_y + (hip_center_y - shoulder_center_y) * waist_ratio
        shoulder_width = self._distance_3d(left_shoulder, right_shoulder)
        hip_width = self._distance_3d(left_hip, right_hip)
        waist_width = (shoulder_width * 0.78 + hip_width * 0.85) / 2
        angle_factor = 1.0 if body_angle < 8 else (1.0 + (body_angle - 8) * 0.01)
        waist_cm = to_cm(waist_width * angle_factor * 2.5)

        cx = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
        self.measurement_points['waist'] = {
            'start': (int(cx - waist_width / 2), int(waist_y)),
            'end': (int(cx + waist_width / 2), int(waist_y)),
            'measurement': round(waist_cm, 1)
        }
        return round(waist_cm, 1)

    def _estimate_hips(self, landmarks, width, height, to_cm, body_angle):
        left_hip = self._get_xyz(landmarks, "LEFT_HIP", width, height)
        right_hip = self._get_xyz(landmarks, "RIGHT_HIP", width, height)
        shoulder_width = self._distance_3d(
            self._get_xyz(landmarks, "LEFT_SHOULDER", width, height),
            self._get_xyz(landmarks, "RIGHT_SHOULDER", width, height)
        )
        hip_width = self._distance_3d(left_hip, right_hip)
        hip_width = max(hip_width, shoulder_width * 0.9)
        angle_factor = 1.0 if body_angle < 8 else (1.0 + (body_angle - 8) * 0.01)
        hips_cm = to_cm(hip_width * angle_factor * 2.6)

        self.measurement_points['hips'] = {
            'start': (int(left_hip[0]), int(left_hip[1])),
            'end': (int(right_hip[0]), int(right_hip[1])),
            'measurement': round(hips_cm, 1)
        }
        return round(hips_cm, 1)
    
    def _validate_measurements(self, chest_cm, waist_cm, hips_cm) -> Dict[str, float]:
        """Validate and adjust measurements based on realistic body proportions"""
        measurements = {"chest_cm": chest_cm, "waist_cm": waist_cm, "hips_cm": hips_cm}
        
        # Realistic body proportion checks and adjustments
        if chest_cm > 0 and waist_cm > 0:
            waist_to_chest_ratio = waist_cm / chest_cm
            
            # Waist should typically be 70-90% of chest measurement
            if waist_to_chest_ratio < 0.70:  # Waist too small
                measurements["waist_cm"] = round(chest_cm * 0.75, 1)
            elif waist_to_chest_ratio > 0.95:  # Waist too large
                measurements["waist_cm"] = round(chest_cm * 0.90, 1)
        
        if hips_cm > 0 and waist_cm > 0:
            hip_to_waist_ratio = hips_cm / measurements["waist_cm"]
            
            # Hips should typically be 105-120% of waist measurement
            if hip_to_waist_ratio < 1.05:  # Hips too small
                measurements["hips_cm"] = round(measurements["waist_cm"] * 1.08, 1)
            elif hip_to_waist_ratio > 1.25:  # Hips too large
                measurements["hips_cm"] = round(measurements["waist_cm"] * 1.20, 1)
        
        # Ensure hips are reasonable compared to chest
        if chest_cm > 0 and measurements["hips_cm"] > 0:
            hip_to_chest_ratio = measurements["hips_cm"] / chest_cm
            
            # Hips should typically be 80-100% of chest measurement
            if hip_to_chest_ratio < 0.80:  # Hips too small relative to chest
                measurements["hips_cm"] = round(chest_cm * 0.85, 1)
            elif hip_to_chest_ratio > 1.05:  # Hips too large relative to chest
                measurements["hips_cm"] = round(chest_cm * 1.00, 1)
        
        return measurements

    def _calculate_additional_measurements(self, landmarks, width, height, to_cm) -> Dict[str, float]:
        """Calculate additional body measurements with improved accuracy using XYZ distances."""
        measurements = {}

        # Inseam (crotch to ankle) with proportion check
        try:
            left_hip = self._get_xyz(landmarks, "LEFT_HIP", width, height)
            right_hip = self._get_xyz(landmarks, "RIGHT_HIP", width, height)
            left_ankle = self._get_xyz(landmarks, "LEFT_ANKLE", width, height)

            # Crotch midpoint between hips
            crotch = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
                (left_hip[2] + right_hip[2]) / 2
            )
            raw_inseam_cm = to_cm(self._distance_3d(crotch, left_ankle))

            # Clamp inseam to realistic ratio of total height
            min_ratio, max_ratio = 0.45, 0.48  # 45–48% of height
            min_inseam = self.actual_height_cm * min_ratio
            max_inseam = self.actual_height_cm * max_ratio
            clamped_inseam = max(min_inseam, min(raw_inseam_cm, max_inseam))

            measurements["inseam_cm"] = round(clamped_inseam, 1)

            self.measurement_points['inseam'] = {
                'start': (int(crotch[0]), int(crotch[1])),
                'end': (int(left_ankle[0]), int(left_ankle[1])),
                'measurement': measurements["inseam_cm"]
            }
        except:
            measurements["inseam_cm"] = 0.0


        # Arm length (shoulder to wrist)
        try:
            measurements["arm_length_cm"] = round(
                to_cm(self._distance_3d(
                    self._get_xyz(landmarks, "LEFT_SHOULDER", width, height),
                    self._get_xyz(landmarks, "LEFT_WRIST", width, height)
                )),
                1
            )

            self.measurement_points['arm'] = {
                'start': (int(self._get_xyz(landmarks, "LEFT_SHOULDER", width, height)[0]),
                        int(self._get_xyz(landmarks, "LEFT_SHOULDER", width, height)[1])),
                'end': (int(self._get_xyz(landmarks, "LEFT_WRIST", width, height)[0]),
                        int(self._get_xyz(landmarks, "LEFT_WRIST", width, height)[1])),
                'measurement': measurements["arm_length_cm"]
            }
        except:
            measurements["arm_length_cm"] = 0.0

        # Thigh circumference (60% of hip circumference)
        try:
            hip_circumference = self.measurement_points.get('hips', {}).get('measurement', 0)
            if hip_circumference <= 0:
                hip_width = self._distance_3d(
                    self._get_xyz(landmarks, "LEFT_HIP", width, height),
                    self._get_xyz(landmarks, "RIGHT_HIP", width, height)
                )
                hip_circumference = to_cm(hip_width * 2.6)  # circumference multiplier
            measurements["thigh_cm"] = round(hip_circumference * 0.60, 1)
        except:
            measurements["thigh_cm"] = 0.0

        # Neck circumference (38% of chest or fallback to shoulder width)
        try:
            chest_circumference = self.measurement_points.get('chest', {}).get('measurement', 0)
            if chest_circumference > 0:
                measurements["neck_cm"] = round(chest_circumference * 0.38, 1)
            else:
                shoulder_width = self._distance_3d(
                    self._get_xyz(landmarks, "LEFT_SHOULDER", width, height),
                    self._get_xyz(landmarks, "RIGHT_SHOULDER", width, height)
                )
                measurements["neck_cm"] = round(to_cm(shoulder_width * 0.45), 1)
        except:
            measurements["neck_cm"] = 0.0

        # Sleeve length (shoulder → elbow → wrist)
        try:
            upper_arm = self._distance_3d(
                self._get_xyz(landmarks, "LEFT_SHOULDER", width, height),
                self._get_xyz(landmarks, "LEFT_ELBOW", width, height)
            )
            forearm = self._distance_3d(
                self._get_xyz(landmarks, "LEFT_ELBOW", width, height),
                self._get_xyz(landmarks, "LEFT_WRIST", width, height)
            )
            measurements["sleeve_length_cm"] = round(to_cm(upper_arm + forearm), 1)
        except:
            measurements["sleeve_length_cm"] = measurements.get("arm_length_cm", 0.0)

        return measurements

    def _calculate_pose_quality(self, landmarks) -> float:
        """Enhanced pose quality calculation"""
        key_landmarks = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_HEEL', 'RIGHT_HEEL'
        ]
        
        total_confidence = 0
        for name in key_landmarks:
            confidence = self._get_landmark_confidence(landmarks, name)
            total_confidence += min(confidence, 1.0)  # Cap at 1.0
        
        return (total_confidence / len(key_landmarks)) * 100

    def _get_size_recommendations(self, chest_cm, waist_cm, hips_cm) -> Dict[str, str]:
        """Enhanced size recommendations considering multiple measurements"""
        if chest_cm <= 0:
            return {}
        
        # Primary sizing based on chest
        chest_size = 'XS'
        if chest_cm >= 107:
            chest_size = 'XXL+'
        elif chest_cm >= 102:
            chest_size = 'XL'
        elif chest_cm >= 97:
            chest_size = 'L'
        elif chest_cm >= 91:
            chest_size = 'M'
        elif chest_cm >= 86:
            chest_size = 'S'
        
        # Consider waist for fit adjustment
        size_adjustment = ""
        if waist_cm > 0:
            waist_chest_ratio = waist_cm / chest_cm
            if waist_chest_ratio > 0.9:
                size_adjustment = " (consider regular fit)"
            elif waist_chest_ratio < 0.7:
                size_adjustment = " (consider slim fit)"
        
        return {
            'Recommended Size': chest_size + size_adjustment,
            'Chest': f"{chest_cm:.1f} cm",
            'Waist': f"{waist_cm:.1f} cm" if waist_cm > 0 else "N/A",
            'Hips': f"{hips_cm:.1f} cm" if hips_cm > 0 else "N/A"
        }

    def _draw_measurement_lines(self, image_pil, landmarks, width, height, measurements):
        """Draw measurement lines and labels on the image using EXACT same calculation points"""
        draw = ImageDraw.Draw(image_pil)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Colors for different measurements
        colors = {
            'chest': (255, 0, 0),      # Red
            'waist': (0, 255, 0),      # Green
            'hips': (0, 0, 255),       # Blue
            'height': (255, 255, 0),   # Yellow
            'arm': (255, 0, 255),      # Magenta
            'inseam': (0, 255, 255)    # Cyan
        }
        
        # Draw chest line using stored measurement points
        if 'chest' in self.measurement_points:
            chest_data = self.measurement_points['chest']
            draw.line([chest_data['start'], chest_data['end']], fill=colors['chest'], width=3)
            chest_mid = ((chest_data['start'][0] + chest_data['end'][0]) // 2, 
                        chest_data['start'][1] - 30)
            draw.text(chest_mid, f"Chest: {chest_data['measurement']:.1f}cm", 
                     fill=colors['chest'], font=small_font, anchor="mm")
        
        # Draw waist line using stored measurement points
        if 'waist' in self.measurement_points:
            waist_data = self.measurement_points['waist']
            draw.line([waist_data['start'], waist_data['end']], fill=colors['waist'], width=3)
            waist_mid = ((waist_data['start'][0] + waist_data['end'][0]) // 2, 
                        waist_data['start'][1] - 25)
            draw.text(waist_mid, f"Waist: {waist_data['measurement']:.1f}cm", 
                     fill=colors['waist'], font=small_font, anchor="mm")
        
        # Draw hip line using stored measurement points
        if 'hips' in self.measurement_points:
            hips_data = self.measurement_points['hips']
            draw.line([hips_data['start'], hips_data['end']], fill=colors['hips'], width=3)
            hip_mid = ((hips_data['start'][0] + hips_data['end'][0]) // 2, 
                      hips_data['start'][1] + 30)
            draw.text(hip_mid, f"Hips: {hips_data['measurement']:.1f}cm", 
                     fill=colors['hips'], font=small_font, anchor="mm")
        
        # Draw height line using stored measurement points
        if 'height' in self.measurement_points:
            height_data = self.measurement_points['height']
            draw.line([height_data['start'], height_data['end']], fill=colors['height'], width=2)
            height_mid = (height_data['start'][0] - 50, 
                         (height_data['start'][1] + height_data['end'][1]) // 2)
            draw.text(height_mid, f"Height: {height_data['measurement']:.0f}cm", 
                     fill=colors['height'], font=small_font, anchor="mm")
        
        # Draw arm length line using stored measurement points
        if 'arm' in self.measurement_points:
            arm_data = self.measurement_points['arm']
            draw.line([arm_data['start'], arm_data['end']], fill=colors['arm'], width=2)
            arm_mid = ((arm_data['start'][0] + arm_data['end'][0]) // 2 - 30, 
                      (arm_data['start'][1] + arm_data['end'][1]) // 2)
            draw.text(arm_mid, f"Arm: {arm_data['measurement']:.1f}cm", 
                     fill=colors['arm'], font=small_font, anchor="mm")
        
        # Draw inseam line using stored measurement points
        if 'inseam' in self.measurement_points:
            inseam_data = self.measurement_points['inseam']
            draw.line([inseam_data['start'], inseam_data['end']], fill=colors['inseam'], width=2)
            inseam_mid = ((inseam_data['start'][0] + inseam_data['end'][0]) // 2 + 30, 
                         (inseam_data['start'][1] + inseam_data['end'][1]) // 2)
            draw.text(inseam_mid, f"Inseam: {inseam_data['measurement']:.1f}cm", 
                     fill=colors['inseam'], font=small_font, anchor="mm")
        
        # Add debug info overlay showing measurement accuracy
        debug_y = 30
        draw.text((10, debug_y), f"Accuracy: {measurements.get('accuracy', 0):.1f}%", 
                 fill=(0, 255, 0), font=small_font)
        
        debug_y += 25
        debug_info = measurements.get('debug_info', {})
        draw.text((10, debug_y), f"Pose Quality: {debug_info.get('pose_quality', 0):.1f}%", 
                 fill=(0, 255, 0), font=small_font)
        
        debug_y += 25
        draw.text((10, debug_y), f"Body Angle: {debug_info.get('body_angle', 0):.1f}°", 
                 fill=(0, 255, 0), font=small_font)
        
        debug_y += 25
        draw.text((10, debug_y), f"Confidence: {debug_info.get('pose_confidence', 'Unknown')}", 
                 fill=(0, 255, 0), font=small_font)
        
        return image_pil

    def visualize_pose(self, image: Image.Image, show_measurements: bool = True) -> Image.Image:
        """Create pose visualization with optional measurement overlays"""
        width, height = image.size
        image_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        results = self.pose.process(image_np)
        if not results.pose_landmarks:
            return image
        
        # Draw pose landmarks
        annotated_image = image_np.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Convert back to PIL
        image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
        if show_measurements:
            # Get measurements for overlay (this will populate measurement_points)
            measurements = self.extract(image)
            if measurements["success"]:
                image_pil = self._draw_measurement_lines(
                    image_pil, results.pose_landmarks.landmark, width, height, measurements
                )
        
        return image_pil

    def extract(self, image: Image.Image) -> Dict:
        width, height = image.size
        image_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        results = self.pose.process(image_np)
        if not results.pose_landmarks:
            return {"success": False, "message": "No person detected."}

        landmarks = results.pose_landmarks.landmark
        
        # Clear previous measurement points
        self.measurement_points = {}
        
        # Check overall pose quality first
        pose_quality = self._calculate_pose_quality(landmarks)
        if pose_quality < 70:
            return {"success": False, "message": f"Poor pose quality ({pose_quality:.1f}%). Please ensure the full body is visible and well-lit."}

        try:
            # Calculate body angle for pose assessment
            body_angle = self._calculate_body_angle(landmarks, width, height)
            
            # Body height and scale (this will populate measurement_points['height'])
            pixel_body_height, cm_per_pixel = self._calculate_body_height_and_scale(landmarks, width, height)
            to_cm = lambda px: round(px * cm_per_pixel, 1)

            # Get key body measurements with improved algorithms
            # These will populate measurement_points for each measurement
            chest_circumference_cm = self._estimate_chest(landmarks, width, height, to_cm, body_angle)
            waist_circumference_cm = self._estimate_waist(landmarks, width, height, to_cm, body_angle)
            hips_circumference_cm = self._estimate_hips(landmarks, width, height, to_cm, body_angle)
            
            # Validate and adjust measurements
            validated_measurements = self._validate_measurements(
                chest_circumference_cm, waist_circumference_cm, hips_circumference_cm
            )
            
            # Calculate additional measurements (this will populate more measurement_points)
            additional_measurements = self._calculate_additional_measurements(landmarks, width, height, to_cm)

            # Calculate accuracy metrics
            visible_ratio = sum(1 for lm in landmarks if lm.visibility > 0.7) / len(landmarks)
            body_coverage_pct = min((pixel_body_height / height) * 100, 100)
            
            # Enhanced accuracy calculation
            weights = self.config.get("weights", {
                "visible_ratio": 0.3,
                "pose_quality": 0.3,
                "body_coverage_pct": 0.2,
                "body_angle": 0.2
            })
            
            angle_score = max(0, 100 - body_angle * 2)  # Penalty for angled poses
            
            accuracy = round(
                visible_ratio * weights.get("visible_ratio", 0.3) +
                (pose_quality / 100) * weights.get("pose_quality", 0.3) +
                (body_coverage_pct / 100) * weights.get("body_coverage_pct", 0.2) +
                (angle_score / 100) * weights.get("body_angle", 0.2), 2
            )

            # Size recommendations
            recommendations = self._get_size_recommendations(
                validated_measurements["chest_cm"], 
                validated_measurements["waist_cm"], 
                validated_measurements["hips_cm"]
            )

            # Update measurement_points with final validated measurements
            if 'chest' in self.measurement_points:
                self.measurement_points['chest']['measurement'] = validated_measurements["chest_cm"]
            if 'waist' in self.measurement_points:
                self.measurement_points['waist']['measurement'] = validated_measurements["waist_cm"]
            if 'hips' in self.measurement_points:
                self.measurement_points['hips']['measurement'] = validated_measurements["hips_cm"]

            return {
                "success": True,
                "chest_cm": validated_measurements["chest_cm"],
                "waist_cm": validated_measurements["waist_cm"],
                "hips_cm": validated_measurements["hips_cm"],
                "inseam_cm": additional_measurements["inseam_cm"],
                "thigh_cm": additional_measurements["thigh_cm"],
                "neck_cm": additional_measurements["neck_cm"],
                "arm_length_cm": additional_measurements["arm_length_cm"],
                "sleeve_length_cm": additional_measurements["sleeve_length_cm"],
                "accuracy": round(accuracy*100, 2),
                "recommendations": recommendations,
                "debug_info": {
                    "visible_ratio": round(visible_ratio, 3),
                    "pose_quality": round(pose_quality, 2),
                    "body_coverage_pct": round(body_coverage_pct, 2),
                    "body_angle": round(body_angle, 2),
                    "pixel_body_height": pixel_body_height,
                    "cm_per_pixel": round(cm_per_pixel, 4),
                    "pose_confidence": "Good" if body_angle < 15 else "Fair" if body_angle < 30 else "Poor",
                    "measurement_points_used": list(self.measurement_points.keys())
                }
            }

        except Exception as e:
            return {"success": False, "message": f"Error in measurement calculation: {str(e)}"}


