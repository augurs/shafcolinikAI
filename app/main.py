from fastapi import FastAPI, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from PIL import Image
import json
import os

from app.utils.height_converter_utils import convert_to_cm
from app.services.image_validator import validate_image_upload
from app.services.visualize_pose import visualize_pose_landmarks
from app.services.body_measurement import BodyMeasurementExtractor

app = FastAPI(title="ShafcoLink - AI", swagger_ui_parameters={"defaultModelsExpandDepth": -1},)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "services", "config.json")

def load_config_from_json(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

@app.post("/extract-measurements/")
async def extract_measurements(front_image: UploadFile = File(...),height_feet: int = Form(0),inches: int = Form(0),height_cm: int = Form(0),):
    
    # Step 1: Validate height
    try:
        height_cm = convert_to_cm(feet=height_feet, inches=inches, centimeters=height_cm)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": 400, "success": False, "message": str(e)})

    # Step 2: Validate front image
    front_result = validate_image_upload(front_image, "Front image")

    if front_result.status_code != 200:
        return front_result

    # Step 3: Load front image
    front_img = Image.open(front_image.file)
    
    # Step 4: Extract body measurements from front image
    try:
        # Load config
        config = load_config_from_json(CONFIG_PATH)
        extractor = BodyMeasurementExtractor(actual_height_cm=height_cm, config=config)
        measurements = extractor.extract(front_img)
        print(measurements.get("debug_info"))

    except Exception as e:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"success": False, "message": f"Measurement error: {str(e)}"})

    if not measurements.get("success"):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=measurements)
    
    # Step 5: Visualize image
    annotated_image = visualize_pose_landmarks(front_img)
    annotated_image.show()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "success": True,
            "message": "Upload successful.",
            "accuracy": measurements.get("accuracy"),
            "body_measurements_cm": {
                "chest": measurements.get("chest_cm"),
                "waist": measurements.get("waist_cm"),
                "hips": measurements.get("hips_cm"),
                "inseam": measurements.get("inseam_cm"),
                "thigh": measurements.get("thigh_cm"),
                "neck": measurements.get("neck_cm"),
                "arm_length": measurements.get("arm_length_cm")
            },
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
