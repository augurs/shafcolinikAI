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
        
        # Store calculated measurement points for consistent drawing
        self.measurement_points = {}

    def __del__(self):
        self.pose.close()

    def _get_point(self, landmarks, part, width, height) -> Tuple[int, int]:
        lm = landmarks[mp_pose.PoseLandmark[part].value]
        return int(lm.x * width), int(lm.y * height)

    def _get_landmark_confidence(self, landmarks, part) -> float:
        lm = landmarks[mp_pose.PoseLandmark[part].value]
        return lm.visibility

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _calculate_body_height_and_scale(self, landmarks, width, height) -> Tuple[float, float]:
        # Use multiple reference points for better accuracy
        head_top = self._get_point(landmarks, "NOSE", width, height)[1]
        
        # Get feet points with confidence checking
        left_heel = self._get_point(landmarks, "LEFT_HEEL", width, height)
        right_heel = self._get_point(landmarks, "RIGHT_HEEL", width, height)
        left_foot = self._get_point(landmarks, "LEFT_FOOT_INDEX", width, height)
        right_foot = self._get_point(landmarks, "RIGHT_FOOT_INDEX", width, height)
        
        # Use the lowest visible point
        foot_points = [left_heel[1], right_heel[1], left_foot[1], right_foot[1]]
        bottom = max(foot_points)
        
        pixel_body_height = abs(bottom - head_top)
        cm_per_pixel = self.actual_height_cm / pixel_body_height
        
        # Store for drawing
        self.measurement_points['height'] = {
            'start': (self._get_point(landmarks, "NOSE", width, height)),
            'end': (left_heel[0], bottom),
            'measurement': self.actual_height_cm
        }
        
        return pixel_body_height, cm_per_pixel

    def _calculate_body_angle(self, landmarks, width, height) -> float:
        """Calculate if the person is standing straight or at an angle"""
        left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
        right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
        left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
        right_hip = self._get_point(landmarks, "RIGHT_HIP", width, height)
        
        # Calculate the angle of the torso
        shoulder_center = ((left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2)
        hip_center = ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2)
        
        angle = math.atan2(hip_center[0] - shoulder_center[0], hip_center[1] - shoulder_center[1])
        return abs(math.degrees(angle))

    def _estimate_chest(self, landmarks, width, height, to_cm, body_angle) -> float:
        """Improved chest calculation using multiple reference points"""
        left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
        right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
        
        # Get additional chest reference points for better accuracy
        try:
            # Use shoulder-to-shoulder distance but adjust for proper chest level
            shoulder_width = self._distance(left_shoulder, right_shoulder)
            
            # For frontal pose, chest width is typically 85-90% of shoulder width
            # This accounts for the natural tapering from shoulders to chest
            chest_width_factor = 0.88
            chest_width = shoulder_width * chest_width_factor
            
            # Calculate actual chest measurement points for drawing
            center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            # Chest level is typically 15-20% below shoulder line for proper chest measurement
            chest_y_offset = abs(right_shoulder[1] - left_shoulder[1]) * 0.1 + 35
            chest_y = max(left_shoulder[1], right_shoulder[1]) + chest_y_offset
            
            left_chest = (int(center_x - chest_width/2), int(chest_y))
            right_chest = (int(center_x + chest_width/2), int(chest_y))
            
        except:
            # Fallback to shoulder width
            chest_width = self._distance(left_shoulder, right_shoulder) * 0.88
            left_chest = left_shoulder
            right_chest = right_shoulder
        
        # Conservative angle compensation
        angle_factor = 1.0 if body_angle < 12 else (1.0 + (body_angle - 12) * 0.005)
        adjusted_chest_width = chest_width * angle_factor
        
        # More accurate multiplier based on real anthropometric data
        chest_multiplier = 2.6
        chest_measurement = to_cm(adjusted_chest_width * chest_multiplier)
        
        # Store measurement points for drawing
        self.measurement_points['chest'] = {
            'start': left_chest,
            'end': right_chest,
            'measurement': chest_measurement,
            'width_pixels': adjusted_chest_width
        }
        
        return chest_measurement

    def _estimate_waist(self, landmarks, width, height, to_cm, body_angle) -> float:
        """Improved waist calculation using anatomical landmarks"""
        left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
        right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
        left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
        right_hip = self._get_point(landmarks, "RIGHT_HIP", width, height)
        
        # Calculate waist position - natural waist is typically at narrowest point
        # This is usually about 40-50% down from shoulders to hips for better accuracy
        waist_ratio = 0.45
        
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        waist_y = int(shoulder_center_y + (hip_center_y - shoulder_center_y) * waist_ratio)
        
        # More realistic waist width estimation
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        hip_width = abs(right_hip[0] - left_hip[0])
        
        # Natural waist is typically 75-85% of shoulder width for most body types
        # Use a blend approach for better accuracy
        waist_width = (shoulder_width * 0.78 + hip_width * 0.85) / 2
        
        # Calculate waist measurement points for drawing
        center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
        left_waist = (int(center_x - waist_width/2), waist_y)
        right_waist = (int(center_x + waist_width/2), waist_y)
        
        # Conservative angle compensation
        angle_factor = 1.0 if body_angle < 12 else (1.0 + (body_angle - 12) * 0.004)
        adjusted_waist_width = waist_width * angle_factor
        
        # Realistic circumference multiplier for waist
        waist_multiplier = 2.5
        waist_measurement = to_cm(adjusted_waist_width * waist_multiplier)
        
        # Store measurement points for drawing
        self.measurement_points['waist'] = {
            'start': left_waist,
            'end': right_waist,
            'measurement': waist_measurement,
            'width_pixels': adjusted_waist_width
        }
        
        return waist_measurement

    def _estimate_hips(self, landmarks, width, height, to_cm, body_angle) -> float:
        """Improved hip calculation using multiple reference points"""
        left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
        right_hip = self._get_point(landmarks, "RIGHT_HIP", width, height)
        
        # Get shoulder width as reference for proportional calculation
        left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
        right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
        shoulder_width = self._distance(left_shoulder, right_shoulder)
        
        # Try to use additional points for better accuracy
        try:
            left_knee = self._get_point(landmarks, "LEFT_KNEE", width, height)
            right_knee = self._get_point(landmarks, "RIGHT_KNEE", width, height)
            
            # Hip measurement point should be at the widest part of hips/buttocks
            # This is typically 15-20% down from hip joint toward knee for better accuracy
            hip_offset_ratio = 0.18
            
            left_hip_measure = (
                left_hip[0] + (left_knee[0] - left_hip[0]) * hip_offset_ratio,
                left_hip[1] + (left_knee[1] - left_hip[1]) * hip_offset_ratio
            )
            right_hip_measure = (
                right_hip[0] + (right_knee[0] - right_hip[0]) * hip_offset_ratio,
                right_hip[1] + (right_knee[1] - right_hip[1]) * hip_offset_ratio
            )
            
            measured_hip_width = self._distance(left_hip_measure, right_hip_measure)
            
            # Ensure hips are proportional to shoulder width
            proportional_hip_width = shoulder_width * 0.90
            hip_width = max(measured_hip_width, proportional_hip_width)
            
            # Use the calculated measurement points for drawing
            draw_left_hip = left_hip_measure
            draw_right_hip = right_hip_measure
            
        except:
            # Fallback: use proportional calculation based on shoulder width
            hip_width = shoulder_width * 0.90
            draw_left_hip = left_hip
            draw_right_hip = right_hip
        
        # Conservative angle compensation
        angle_factor = 1.0 if body_angle < 12 else (1.0 + (body_angle - 12) * 0.004)
        adjusted_hip_width = hip_width * angle_factor
        
        # More accurate hip circumference multiplier
        hip_multiplier = 2.6
        hip_measurement = to_cm(adjusted_hip_width * hip_multiplier)
        
        # Store measurement points for drawing
        self.measurement_points['hips'] = {
            'start': (int(draw_left_hip[0]), int(draw_left_hip[1])),
            'end': (int(draw_right_hip[0]), int(draw_right_hip[1])),
            'measurement': hip_measurement,
            'width_pixels': adjusted_hip_width
        }
        
        return hip_measurement

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
        """Calculate additional body measurements with improved accuracy"""
        measurements = {}
        
        # Inseam (crotch to ankle) - more accurate calculation
        try:
            left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
            right_hip = self._get_point(landmarks, "RIGHT_HIP", width, height)
            left_ankle = self._get_point(landmarks, "LEFT_ANKLE", width, height)
            
            # Crotch point estimation (midpoint between hips, slightly lower)
            crotch_x = (left_hip[0] + right_hip[0]) / 2
            crotch_y = max(left_hip[1], right_hip[1]) + abs(left_hip[1] - right_hip[1]) * 0.5
            crotch = (int(crotch_x), int(crotch_y))
            
            inseam_measurement = to_cm(self._distance(crotch, left_ankle))
            measurements["inseam_cm"] = inseam_measurement
            
            # Store for drawing
            self.measurement_points['inseam'] = {
                'start': crotch,
                'end': left_ankle,
                'measurement': inseam_measurement
            }
            
        except:
            # Fallback to hip-ankle distance
            left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
            left_ankle = self._get_point(landmarks, "LEFT_ANKLE", width, height)
            inseam_measurement = to_cm(self._distance(left_hip, left_ankle) * 0.85)
            measurements["inseam_cm"] = inseam_measurement
            
            # Store for drawing
            self.measurement_points['inseam'] = {
                'start': left_hip,
                'end': left_ankle,
                'measurement': inseam_measurement
            }
        
        # Arm length (shoulder to wrist)
        try:
            left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
            left_wrist = self._get_point(landmarks, "LEFT_WRIST", width, height)
            arm_measurement = to_cm(self._distance(left_shoulder, left_wrist))
            measurements["arm_length_cm"] = arm_measurement
            
            # Store for drawing
            self.measurement_points['arm'] = {
                'start': left_shoulder,
                'end': left_wrist,
                'measurement': arm_measurement
            }
        except:
            measurements["arm_length_cm"] = 0.0
        
        # Thigh circumference (more accurate)
        try:
            left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
            left_knee = self._get_point(landmarks, "LEFT_KNEE", width, height)
            
            # Thigh measurement is typically at widest part, about 25% down from hip
            thigh_point_ratio = 0.25
            thigh_y = int(left_hip[1] + (left_knee[1] - left_hip[1]) * thigh_point_ratio)
            
            # Estimate thigh width based on hip width
            hip_width = self._distance(self._get_point(landmarks, "LEFT_HIP", width, height),
                                     self._get_point(landmarks, "RIGHT_HIP", width, height))
            thigh_width = hip_width * 0.35  # Thigh is typically 35% of hip width
            
            # Thigh circumference multiplier
            measurements["thigh_cm"] = to_cm(thigh_width * 2.2)
        except:
            measurements["thigh_cm"] = 0.0
        
        # Neck circumference (estimated from shoulder width)
        try:
            left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
            right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
            shoulder_width = self._distance(left_shoulder, right_shoulder)
            # Neck is typically 35-40% of shoulder width
            measurements["neck_cm"] = to_cm(shoulder_width * 0.37 * 2.0)  # circumference
        except:
            measurements["neck_cm"] = 0.0
        
        # Sleeve length (shoulder to elbow + elbow to wrist)
        try:
            left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
            left_elbow = self._get_point(landmarks, "LEFT_ELBOW", width, height)
            left_wrist = self._get_point(landmarks, "LEFT_WRIST", width, height)
            
            upper_arm = self._distance(left_shoulder, left_elbow)
            forearm = self._distance(left_elbow, left_wrist)
            measurements["sleeve_length_cm"] = to_cm(upper_arm + forearm)
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


# Example usage with visualization
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

