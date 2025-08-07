import numpy as np
from PIL import Image
from fastapi import UploadFile, status
from fastapi.responses import JSONResponse
import mediapipe as mp
import cv2
from typing import Dict, Tuple

MAX_IMAGE_SIZE_MB = 5
ALLOWED_FORMATS = ["JPEG", "PNG", "JPG"]
THRESHOLD_COVERAGE = 0.70
MIN_LANDMARKS_REQUIRED = 20  # Minimum number of visible landmarks
MIN_CONFIDENCE = 0.5

mp_pose = mp.solutions.pose

def validate_key_body_parts(landmarks) -> Tuple[bool, int, int]:
    """
    Validate that key body parts are present for full body detection
    Returns: (has_key_parts, visible_count, total_count)
    """
    # Define key landmarks for full body
    key_landmarks = [
        mp_pose.PoseLandmark.NOSE,                    # Head
        mp_pose.PoseLandmark.LEFT_SHOULDER,           # Upper body
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,              # Arms
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,                # Lower body
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,               # Legs
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,              # Feet
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]

    visible_key_parts = 0
    total_key_parts = len(key_landmarks)
    
    for landmark_idx in key_landmarks:
        if landmarks[landmark_idx.value].visibility > MIN_CONFIDENCE:
            visible_key_parts += 1
    
    # Require at least 80% of key body parts to be visible
    has_key_parts = visible_key_parts >= (total_key_parts * 0.8)
    return has_key_parts, visible_key_parts, total_key_parts

def calculate_body_coverage_improved(landmarks, image_height: int, image_width: int) -> float:
    """
    Calculate body coverage using multiple methods for better accuracy
    """
    visible_landmarks = [lm for lm in landmarks if lm.visibility > MIN_CONFIDENCE]
    
    if len(visible_landmarks) < MIN_LANDMARKS_REQUIRED:
        return 0.0
    
    # Method 1: Vertical coverage (head to toe)
    y_coords = [lm.y * image_height for lm in visible_landmarks]
    vertical_span = max(y_coords) - min(y_coords)
    vertical_coverage = vertical_span / image_height
    
    # Method 2: Check specific body segments
    head_present = any(landmarks[i].visibility > MIN_CONFIDENCE 
                      for i in [mp_pose.PoseLandmark.NOSE.value, 
                               mp_pose.PoseLandmark.LEFT_EYE.value,
                               mp_pose.PoseLandmark.RIGHT_EYE.value])
    
    torso_present = any(landmarks[i].visibility > MIN_CONFIDENCE 
                       for i in [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                mp_pose.PoseLandmark.LEFT_HIP.value,
                                mp_pose.PoseLandmark.RIGHT_HIP.value])
    
    legs_present = any(landmarks[i].visibility > MIN_CONFIDENCE 
                      for i in [mp_pose.PoseLandmark.LEFT_KNEE.value,
                               mp_pose.PoseLandmark.RIGHT_KNEE.value,
                               mp_pose.PoseLandmark.LEFT_ANKLE.value,
                               mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    # Bonus for having all body segments
    segment_bonus = 0.1 if (head_present and torso_present and legs_present) else 0.0
    
    # Combine vertical coverage with segment presence
    final_coverage = min(1.0, vertical_coverage + segment_bonus)
    
    return final_coverage

def detect_body_coverage(pil_image: Image.Image) -> Dict:
    """
    Enhanced body coverage detection with better error handling and validation
    """
    try:
        # Convert PIL image to numpy array
        image_rgb = np.array(pil_image.convert("RGB"))
        height, width, _ = image_rgb.shape
        
        # Check minimum image dimensions
        if height < 400 or width < 300:
            return {
                "full_body_detected": False, 
                "coverage": 0.0,
                "error": "Image too small for reliable detection"
            }
        
        # Initialize MediaPipe Pose with better parameters
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Use more complex model for better accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            # Process the image
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return {
                    "full_body_detected": False, 
                    "coverage": 0.0,
                    "error": "No human pose detected in image"
                }
            
            landmarks = results.pose_landmarks.landmark
            
            # Check for minimum number of visible landmarks
            visible_landmarks = [lm for lm in landmarks if lm.visibility > MIN_CONFIDENCE]
        
            if len(visible_landmarks) < MIN_LANDMARKS_REQUIRED:
                return {
                    "full_body_detected": False, 
                    "coverage": 0.0,
                    "error": f"Insufficient landmarks detected ({len(visible_landmarks)}/{MIN_LANDMARKS_REQUIRED})"
                }
            
            # Validate key body parts are present
            has_key_parts, visible_parts, total_parts = validate_key_body_parts(landmarks)
            
            if not has_key_parts:
                return {
                    "full_body_detected": False, 
                    "coverage": 0.0,
                    "error": f"Missing key body parts ({visible_parts}/{total_parts} visible)"
                }
            
            # Calculate improved coverage
            coverage = calculate_body_coverage_improved(landmarks, height, width)
            full_body_detected = coverage >= THRESHOLD_COVERAGE
            
            return {
                "full_body_detected": full_body_detected,
                "coverage": round(coverage * 100, 2),
                "visible_landmarks": len(visible_landmarks),
                "key_parts_visible": f"{visible_parts}/{total_parts}"
            }
            
    except Exception as e:
        return {
            "full_body_detected": False, 
            "coverage": 0.0,
            "error": f"Pose detection failed: {str(e)}"
        }

def validate_image_quality(image: Image.Image) -> Tuple[bool, str]:
    """
    Additional image quality checks
    """
    width, height = image.size
    
    # Check aspect ratio for portrait orientation
    # if height <= width:
    #     return False, "Image must be in portrait orientation (height > width)"
    
    # Check if image is too small
    if width < 300 or height < 400:
        return False, "Image resolution too low for reliable detection"
    
    # Convert to OpenCV format for quality checks
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # # Check if image is too blurry
    # laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # print(f"Laplacian variance: {laplacian_var}")
    # if laplacian_var < 50:
    #     return False, "Image is too blurry for reliable detection"
    
    # Check brightness
    mean_brightness = np.mean(gray)
    print(f"Mean brightness: {mean_brightness}")
    if mean_brightness < 30:
        return False, "Image is too dark"
    elif mean_brightness > 250:
        return False, "Image is too bright/overexposed"
    
    return True, "Image quality acceptable"

def validate_image_upload(image_file: UploadFile, image_label: str = "Image") -> JSONResponse:
    """
    Enhanced image upload validation with comprehensive checks
    """
    try:
        # Check file size
        image_file.file.seek(0, 2)
        size_mb = image_file.file.tell() / (1024 * 1024)
        image_file.file.seek(0)

        if size_mb > MAX_IMAGE_SIZE_MB:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": 400,
                    "success": False,
                    "message": f"{image_label} size exceeds {MAX_IMAGE_SIZE_MB}MB limit.",
                    "accuracy": 0.0
                }
            )

        # Validate and load image
        try:
            image = Image.open(image_file.file)
            image_format = image.format.upper() if image.format else "UNKNOWN"
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": 400,
                    "success": False,
                    "message": f"{image_label} is not a valid image file: {str(e)}",
                    "accuracy": 0.0
                }
            )
        
        # Reset file pointer
        image_file.file.seek(0)

        if image_format not in ALLOWED_FORMATS:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": 400,
                    "success": False,
                    "message": f"{image_label} must be in JPG or PNG format. Got: {image_format}",
                    "accuracy": 0.0
                }
            )

        # Validate image quality
        quality_valid, quality_message = validate_image_quality(image)
        if not quality_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": 400,
                    "success": False,
                    "message": f"{image_label} quality issue: {quality_message}",
                    "accuracy": 0.0
                }
            )

        # Run enhanced pose detection
        result = detect_body_coverage(image)
    
        if "error" in result:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": 400,
                    "success": False,
                    "message": f"{image_label} validation failed: {result['error']}",
                    "accuracy": result['coverage']
                }
            )
        
        if not result["full_body_detected"]:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": 400,
                    "success": False,
                    "message": f"{image_label} must show the full human body. Only {result['coverage']}% coverage detected (minimum {THRESHOLD_COVERAGE*100}% required). Visible landmarks: {result['visible_landmarks']}, Key parts: {result['key_parts_visible']}",
                    "accuracy": result['coverage']
                }
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": 200,
                "success": True,
                "message": f"{image_label} is valid. {result['coverage']}% body coverage detected with {result['visible_landmarks']} visible landmarks.",
                "accuracy": result['coverage'],
                "details": {
                    "visible_landmarks": result['visible_landmarks'],
                    "key_parts_visible": result['key_parts_visible']
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": 400,
                "success": False,
                "message": f"Unexpected error validating {image_label}: {str(e)}",
                "accuracy": 0.0
            }
        )