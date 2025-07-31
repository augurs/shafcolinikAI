from PIL import Image
import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def visualize_pose_landmarks(pil_image: Image.Image) -> Image.Image:
    """
    Detects and draws pose landmarks on the image using MediaPipe's default styles.
    Displays original and pose-detected images side by side in one OpenCV window.
    Returns a PIL image with landmarks overlaid.
    """
    image_rgb = np.array(pil_image.convert("RGB"))
    output_image = image_rgb.copy()

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the image")

        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), circle_radius=2)
        )

    # Concatenate original and pose image side by side
    side_by_side = np.concatenate((image_rgb, output_image), axis=1)

    # Show the combined image in one window
    # cv2.imshow("Original and Pose Landmarks", side_by_side)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return Image.fromarray(output_image)

