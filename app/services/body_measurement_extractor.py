import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from typing import Dict, Tuple, Optional
import logging

# Handle matplotlib import gracefully
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Handle drawing styles import gracefully
try:
    mp_drawing_styles = mp.solutions.drawing_styles
except AttributeError:
    mp_drawing_styles = None

class BodyMeasurementExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Anthropometric ratios for better estimation
        self.body_ratios = {
            'shoulder_width_to_height': 0.235,
            'chest_circumference_to_height': 0.525,
            'waist_circumference_to_height': 0.45,
            'hip_circumference_to_height': 0.53,
            'neck_circumference_to_height': 0.21,
            'arm_span_to_height': 1.0,
            'inseam_to_height': 0.47
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pose.close()

    def _get_point(self, landmarks, part_name: str, width: int, height: int) -> Tuple[int, int]:
        """Get pixel coordinates for a body landmark"""
        lm = landmarks[mp_pose.PoseLandmark[part_name].value]
        return int(lm.x * width), int(lm.y * height)

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_landmark_visibility(self, landmarks, part_name: str) -> float:
        """Get visibility score for a landmark"""
        return landmarks[mp_pose.PoseLandmark[part_name].value].visibility

    def _calculate_circumference_from_width(self, width_px: float, cm_per_pixel: float, 
                                          body_part: str = 'chest') -> float:
        """Calculate circumference from width using body geometry"""
        width_cm = width_px * cm_per_pixel
        
        # Depth-to-width ratios for different body parts
        depth_ratios = {
            'chest': 0.65,
            'waist': 0.7,
            'hip': 0.75,
            'neck': 0.85,
            'thigh': 0.8,
            'upper_arm': 0.9
        }
        
        depth_ratio = depth_ratios.get(body_part, 0.7)
        depth_cm = width_cm * depth_ratio
        
        # Calculate ellipse circumference using Ramanujan's approximation
        a = width_cm / 2
        b = depth_cm / 2
        
        h = ((a - b) ** 2) / ((a + b) ** 2)
        circumference = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        
        return circumference

    def _estimate_body_measurements_from_ratios(self, height_cm: float) -> Dict[str, float]:
        """Estimate measurements using anthropometric ratios as fallback"""
        return {
            'shoulder_width_cm': height_cm * self.body_ratios['shoulder_width_to_height'],
            'chest_circumference_cm': height_cm * self.body_ratios['chest_circumference_to_height'],
            'waist_circumference_cm': height_cm * self.body_ratios['waist_circumference_to_height'],
            'hip_circumference_cm': height_cm * self.body_ratios['hip_circumference_to_height'],
            'neck_circumference_cm': height_cm * self.body_ratios['neck_circumference_to_height'],
            'inseam_cm': height_cm * self.body_ratios['inseam_to_height']
        }

    def _validate_measurement_reasonableness(self, measurements: Dict[str, float], 
                                           height_cm: float) -> Dict[str, float]:
        """Validate measurements against expected ratios and adjust if necessary"""
        ratio_estimates = self._estimate_body_measurements_from_ratios(height_cm)
        validated_measurements = {}
        
        for measurement_name, value in measurements.items():
            if measurement_name in ratio_estimates:
                ratio_estimate = ratio_estimates[measurement_name]
                
                # If measurement is unreasonably different from ratio estimate, blend them
                ratio_diff = abs(value - ratio_estimate) / ratio_estimate
                
                if ratio_diff > 0.3:  # More than 30% difference
                    # Blend the measurements (70% ratio-based, 30% vision-based)
                    validated_measurements[measurement_name] = round(
                        ratio_estimate * 0.7 + value * 0.3, 1
                    )
                else:
                    validated_measurements[measurement_name] = value
            else:
                validated_measurements[measurement_name] = value
        
        return validated_measurements

    def _calculate_dynamic_accuracy_score(self, landmarks, width: int, height: int) -> float:
        """Calculate dynamic accuracy score based on pose quality factors"""
        score = 100.0  # Start with perfect score
        
        # Factor 1: Landmark visibility (25% weight)
        visibility_scores = []
        key_landmarks = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", 
            "LEFT_WRIST", "RIGHT_WRIST"
        ]
        
        for landmark_name in key_landmarks:
            try:
                visibility = self._get_landmark_visibility(landmarks, landmark_name)
                visibility_scores.append(visibility)
            except:
                visibility_scores.append(0.0)
        
        avg_visibility = np.mean(visibility_scores)
        visibility_penalty = (1.0 - avg_visibility) * 25
        score -= visibility_penalty
        
        # Factor 2: Body pose alignment (25% weight)
        try:
            # Check shoulder alignment (should be horizontal)
            left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
            right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
            shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
            
            # Normalize by image height and penalize excessive tilt
            shoulder_tilt_ratio = shoulder_tilt / height
            if shoulder_tilt_ratio > 0.05:  # More than 5% of image height
                alignment_penalty = min(shoulder_tilt_ratio * 200, 15)  # Max 15 point penalty
                score -= alignment_penalty
            
            # Check hip alignment
            left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
            right_hip = self._get_point(landmarks, "RIGHT_HIP", width, height)
            hip_tilt = abs(left_hip[1] - right_hip[1])
            hip_tilt_ratio = hip_tilt / height
            
            if hip_tilt_ratio > 0.05:
                hip_penalty = min(hip_tilt_ratio * 200, 10)  # Max 10 point penalty
                score -= hip_penalty
                
        except Exception as e:
            score -= 10  # Penalty for alignment calculation failure
        
        # Factor 3: Body symmetry (20% weight)
        try:
            symmetry_score = self._calculate_body_symmetry_score(landmarks, width, height)
            symmetry_penalty = (100 - symmetry_score) * 0.2  # Convert to penalty
            score -= symmetry_penalty
        except:
            score -= 20  # Penalty for symmetry calculation failure
        
        # Factor 4: Pose completeness (15% weight)
        try:
            # Check if key body parts are detected
            required_landmarks = [
                "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
                "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
            ]
            
            detected_count = 0
            for landmark_name in required_landmarks:
                try:
                    visibility = self._get_landmark_visibility(landmarks, landmark_name)
                    if visibility > 0.5:
                        detected_count += 1
                except:
                    pass
            
            completeness_ratio = detected_count / len(required_landmarks)
            if completeness_ratio < 1.0:
                completeness_penalty = (1.0 - completeness_ratio) * 15
                score -= completeness_penalty
                
        except:
            score -= 15
        
        # Factor 5: Image quality factors (15% weight)
        try:
            # Calculate body coverage in image
            all_x_coords = []
            all_y_coords = []
            
            for landmark in landmarks:
                if landmark.visibility > 0.5:
                    all_x_coords.append(landmark.x * width)
                    all_y_coords.append(landmark.y * height)
            
            if all_x_coords and all_y_coords:
                body_width = max(all_x_coords) - min(all_x_coords)
                body_height = max(all_y_coords) - min(all_y_coords)
                
                # Penalize if body is too small in frame
                coverage_ratio = (body_width * body_height) / (width * height)
                if coverage_ratio < 0.1:  # Body covers less than 10% of image
                    coverage_penalty = (0.1 - coverage_ratio) * 100
                    score -= min(coverage_penalty, 10)
                elif coverage_ratio > 0.8:  # Body is too large (likely cropped)
                    score -= 5
                    
        except:
            score -= 5
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        return round(score, 1)
    
    def _calculate_body_symmetry_score(self, landmarks, width: int, height: int) -> float:
        """Calculate body symmetry score (0-100)"""
        try:
            # Compare left and right side measurements
            left_arm = self._distance(
                self._get_point(landmarks, "LEFT_SHOULDER", width, height),
                self._get_point(landmarks, "LEFT_WRIST", width, height)
            )
            right_arm = self._distance(
                self._get_point(landmarks, "RIGHT_SHOULDER", width, height),
                self._get_point(landmarks, "RIGHT_WRIST", width, height)
            )
            
            left_leg = self._distance(
                self._get_point(landmarks, "LEFT_HIP", width, height),
                self._get_point(landmarks, "LEFT_ANKLE", width, height)
            )
            right_leg = self._distance(
                self._get_point(landmarks, "RIGHT_HIP", width, height),
                self._get_point(landmarks, "RIGHT_ANKLE", width, height)
            )
            
            # Calculate symmetry (lower difference = higher symmetry)
            arm_symmetry = 1 - abs(left_arm - right_arm) / max(left_arm, right_arm)
            leg_symmetry = 1 - abs(left_leg - right_leg) / max(left_leg, right_leg)
            
            return ((arm_symmetry + leg_symmetry) / 2) * 100
        except:
            return 50.0  # Default moderate symmetry

    def _get_better_height_measurement(self, landmarks, width: int, height: int) -> float:
        """Get more accurate height measurement using multiple reference points"""
        height_measurements = []
        
        # Method 1: Top of head to heel
        try:
            nose_y = self._get_point(landmarks, "NOSE", width, height)[1]
            left_heel_y = self._get_point(landmarks, "LEFT_HEEL", width, height)[1] 
            right_heel_y = self._get_point(landmarks, "RIGHT_HEEL", width, height)[1]
            heel_y = max(left_heel_y, right_heel_y)
            
            head_top_y = nose_y - (heel_y - nose_y) * 0.087
            height_measurements.append(abs(heel_y - head_top_y))
        except:
            pass
        
        # Method 2: Hip to ankle + torso estimation
        try:
            hip_center_y = np.mean([
                self._get_point(landmarks, "LEFT_HIP", width, height)[1],
                self._get_point(landmarks, "RIGHT_HIP", width, height)[1]
            ])
            ankle_y = max(
                self._get_point(landmarks, "LEFT_ANKLE", width, height)[1],
                self._get_point(landmarks, "RIGHT_ANKLE", width, height)[1]
            )
            leg_length = abs(ankle_y - hip_center_y)
            estimated_total_height = leg_length / 0.47
            height_measurements.append(estimated_total_height)
        except:
            pass
        
        # Method 3: Shoulder to ankle + head estimation  
        try:
            shoulder_center_y = np.mean([
                self._get_point(landmarks, "LEFT_SHOULDER", width, height)[1],
                self._get_point(landmarks, "RIGHT_SHOULDER", width, height)[1]
            ])
            ankle_y = max(
                self._get_point(landmarks, "LEFT_ANKLE", width, height)[1],
                self._get_point(landmarks, "RIGHT_ANKLE", width, height)[1]
            )
            shoulder_to_ankle = abs(ankle_y - shoulder_center_y)
            estimated_total_height = shoulder_to_ankle / 0.82
            height_measurements.append(estimated_total_height)
        except:
            pass
        
        if height_measurements:
            return np.median(height_measurements)
        else:
            # Fallback to original method
            nose_y = self._get_point(landmarks, "NOSE", width, height)[1]
            heel_y = max(
                self._get_point(landmarks, "LEFT_HEEL", width, height)[1],
                self._get_point(landmarks, "RIGHT_HEEL", width, height)[1]
            )
            return abs(heel_y - nose_y) * 1.087

    def visualize_measurements(self, image: Image.Image, landmarks, 
                             measurements: Dict[str, float], 
                             show_plot: bool = True) -> Optional[np.ndarray]:
        """Visualize the pose landmarks and measurement annotations on the image"""
        
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib is required for visualization. Please install it with: pip install matplotlib")
            return None
            
        width, height = image.size
        image_np = np.array(image.convert("RGB"))
        
        # Create annotated image
        annotated_image = image_np.copy()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        pose_landmarks = type('obj', (object,), {'landmark': landmarks})()
        
        # Use basic drawing if drawing_styles not available
        if mp_drawing_styles:
            landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
        else:
            landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style
        )
        
        # Convert back to RGB for matplotlib
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Draw measurement lines
        self._draw_measurement_lines(annotated_image, landmarks, width, height)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Annotated image with measurements
        ax2.imshow(annotated_image)
        ax2.set_title('Pose Detection & Measurements', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add measurement text overlay
        self._add_measurement_text(ax2, measurements)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return annotated_image
    
    def _draw_measurement_lines(self, image: np.ndarray, landmarks, width: int, height: int):
        """Draw measurement lines on the image"""
        
        colors = {
        'shoulder': (0, 128, 255),   # blue
        'chest': (0, 255, 0),        # Green
        'waist': (255, 0, 0),        # Red
        'hip': (255, 255, 0),        # Cyan/Yellow
        'inseam': (255, 0, 255),     # Magenta
        'arm': (128, 0, 255)         # Purple
    }
        
        line_thickness = 3
        
        try:
            # Shoulder width line
            left_shoulder = self._get_point(landmarks, "LEFT_SHOULDER", width, height)
            right_shoulder = self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
            cv2.line(image, left_shoulder, right_shoulder, colors['shoulder'], line_thickness)
            
            # Hip width line
            left_hip = self._get_point(landmarks, "LEFT_HIP", width, height)
            right_hip = self._get_point(landmarks, "RIGHT_HIP", width, height)
            cv2.line(image, left_hip, right_hip, colors['hip'], line_thickness)
            
            # Waist line (estimated between chest and hip)
            waist_left = ((left_shoulder[0] + left_hip[0]) // 2, 
                         (left_shoulder[1] + left_hip[1]) // 2)
            waist_right = ((right_shoulder[0] + right_hip[0]) // 2, 
                          (right_shoulder[1] + right_hip[1]) // 2)
            cv2.line(image, waist_left, waist_right, colors['waist'], line_thickness)
            
            # Inseam line (hip to ankle)
            left_ankle = self._get_point(landmarks, "LEFT_ANKLE", width, height)
            cv2.line(image, left_hip, left_ankle, colors['inseam'], line_thickness)
            
            # Arm length line
            left_wrist = self._get_point(landmarks, "LEFT_WRIST", width, height)
            cv2.line(image, left_shoulder, left_wrist, colors['arm'], line_thickness)
            
            # Add small circles at key points
            key_points = [left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, left_wrist]
            for point in key_points:
                cv2.circle(image, point, 8, (0, 255, 0), -1)
                cv2.circle(image, point, 6, (0, 0, 0), -1)
                
        except Exception as e:
            logging.warning(f"Error drawing measurement lines: {e}")
    
    def _add_measurement_text(self, ax, measurements: Dict[str, float]):
        """Add measurement text box to the plot"""
        
        text_lines = ["📏 Body Measurements:"]
        text_lines.append("-" * 25)
        
        measurement_labels = {
            'shoulder_width_cm': 'Shoulder Width',
            'chest_circumference_cm': 'Chest Circumference', 
            'waist_circumference_cm': 'Waist Circumference',
            'hip_circumference_cm': 'Hip Circumference',
            'neck_circumference_cm': 'Neck Circumference',
            'inseam_cm': 'Inseam Length',
            'arm_length_cm': 'Arm Length',
            'thigh_circumference_cm': 'Thigh Circumference'
        }
        
        for key, label in measurement_labels.items():
            if key in measurements:
                text_lines.append(f"{label}: {measurements[key]} cm")
        
        text_content = '\n'.join(text_lines)
        
        ax.text(0.98, 0.98, text_content, 
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor='white', 
                         alpha=0.8,
                         edgecolor='black'),
                fontsize=10,
                fontfamily='monospace')

    
    def _get_size_recommendations(self, measurements: Dict[str, float]) -> Dict[str, str]:
        """Provide clothing size recommendations based on measurements"""
        
        recommendations = {}
        
        # Shirt size (based on chest circumference)
        chest = measurements.get('chest_circumference_cm', 0)
        if chest > 0:
            if chest < 86:
                recommendations['Shirt Size'] = 'XS'
            elif chest < 91:
                recommendations['Shirt Size'] = 'S'
            elif chest < 97:
                recommendations['Shirt Size'] = 'M'
            elif chest < 102:
                recommendations['Shirt Size'] = 'L'
            elif chest < 107:
                recommendations['Shirt Size'] = 'XL'
            else:
                recommendations['Shirt Size'] = 'XXL+'
        
        # Pant size (based on waist circumference)
        waist = measurements.get('waist_circumference_cm', 0)
        if waist > 0:
            waist_inches = waist / 2.54
            recommendations['Pant Waist'] = f'{int(waist_inches)}"'
        
        # Inseam
        inseam = measurements.get('inseam_cm', 0)
        if inseam > 0:
            inseam_inches = inseam / 2.54
            recommendations['Pant Inseam'] = f'{int(inseam_inches)}"'
        
        return recommendations

    def extract_body_measurements(self, image: Image.Image, actual_height_cm: float) -> Dict[str, any]:
        """Extract body measurements from image with enhanced accuracy"""
        width, height = image.size
        image_np = np.array(image.convert("RGB"))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        results = self.pose.process(image_np)
        
        if not results.pose_landmarks:
            return {"success": False, "message": "No person detected in image"}
        
        landmarks = results.pose_landmarks.landmark
        
        # Get better height measurement
        pixel_body_height = self._get_better_height_measurement(landmarks, width, height)
        
        if pixel_body_height == 0:
            return {"success": False, "message": "Cannot determine body height from image"}
        
        cm_per_pixel = actual_height_cm / pixel_body_height
        
        try:
            # Calculate widths first
            shoulder_width_px = self._distance(
                self._get_point(landmarks, "LEFT_SHOULDER", width, height),
                self._get_point(landmarks, "RIGHT_SHOULDER", width, height)
            )
            
            hip_width_px = self._distance(
                self._get_point(landmarks, "LEFT_HIP", width, height),
                self._get_point(landmarks, "RIGHT_HIP", width, height)
            )
            
            # Estimate waist width (typically narrower than hips)
            waist_width_px = hip_width_px * 0.85
            
            # Convert widths to circumferences
            chest_circumference = self._calculate_circumference_from_width(
                shoulder_width_px * 1.1, cm_per_pixel, 'chest'
            )
            
            waist_circumference = self._calculate_circumference_from_width(
                waist_width_px, cm_per_pixel, 'waist'
            )
            
            hip_circumference = self._calculate_circumference_from_width(
                hip_width_px, cm_per_pixel, 'hip'
            )
            
            # Calculate other measurements
            inseam_px = self._distance(
                self._get_point(landmarks, "LEFT_HIP", width, height),
                self._get_point(landmarks, "LEFT_ANKLE", width, height)
            )
            inseam_cm = inseam_px * cm_per_pixel
            
            # Neck circumference (estimate from shoulder width)
            neck_circumference = self._calculate_circumference_from_width(
                shoulder_width_px * 0.4, cm_per_pixel, 'neck'
            )
            
            # Arm length (shoulder to wrist)
            arm_length_px = self._distance(
                self._get_point(landmarks, "LEFT_SHOULDER", width, height),
                self._get_point(landmarks, "LEFT_WRIST", width, height)
            )
            arm_length_cm = arm_length_px * cm_per_pixel
            
            # Thigh circumference estimate
            thigh_width_px = hip_width_px * 0.6
            thigh_circumference = self._calculate_circumference_from_width(
                thigh_width_px, cm_per_pixel, 'thigh'
            )
            
            measurements = {
                "shoulder_width_cm": round(shoulder_width_px * cm_per_pixel, 1),
                "chest_circumference_cm": round(chest_circumference, 1),
                "waist_circumference_cm": round(waist_circumference, 1),
                "hip_circumference_cm": round(hip_circumference, 1),
                "neck_circumference_cm": round(neck_circumference, 1),
                "inseam_cm": round(inseam_cm, 1),
                "arm_length_cm": round(arm_length_cm, 1),
                "thigh_circumference_cm": round(thigh_circumference, 1)
            }
            
            # Validate measurements against anthropometric ratios
            validated_measurements = self._validate_measurement_reasonableness(
                measurements, actual_height_cm
            )
            
            # Calculate dynamic accuracy score based on pose quality
            accuracy_score = self._calculate_dynamic_accuracy_score(landmarks, width, height)
            
            result = {
                "success": True,
                "measurements": validated_measurements,
                "quality_assessment": {
                    "accuracy_score": accuracy_score,
                    "confidence_level": "Medium" if accuracy_score >= 70 else "Low"
                },
                "debug_info": {
                    "pixel_body_height": round(pixel_body_height, 2),
                    "cm_per_pixel": round(cm_per_pixel, 4),
                    "actual_height_cm": actual_height_cm
                }
            }
            self.visualize_measurements(
                    image, 
                    landmarks,
                    validated_measurements,
                    show_plot=True
                )
            return result
            
        except Exception as e:
            return {"success": False, "message": f"Measurement extraction failed: {str(e)}"}


# Usage example with visualization
def create_default_config():
    return {
        "visibility_threshold": 0.5,
        "key_visibility_threshold": 0.7
    }

def feet_inches_to_cm(feet: int, inches: int) -> float:
    """Convert feet and inches to centimeters"""
    total_inches = feet * 12 + inches
    return total_inches * 2.54


# config = create_default_config()
    # with BodyMeasurementExtractor(config) as extractor:
    #     result = extractor.extract_body_measurements(front_img, actual_height_cm=height_cm)
        
    # return result
