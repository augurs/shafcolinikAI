from fastapi import FastAPI, UploadFile, File, Form, status, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional
from PIL import Image
import base64

from app.utils.height_converter_utils import convert_to_cm
from app.services.image_validator import validate_image_upload
from app.services.visualize_pose import visualize_pose_landmarks
from app.services.body_measurement import BodyMeasurementExtractor
from app.configs.configs import create_improved_config, create_config
from app.services.improved_body_measurement import ImprovedBodyMeasurementExtractor
from app.services.gemini_services import generate_response as gemini_generate_response
from app.services.ollama_services import generate_response as ollama_generate_response

load_dotenv()

app = FastAPI(title="ShafcoLink - AI", swagger_ui_parameters={"defaultModelsExpandDepth": -1},)

@app.post("/custom-logic/extract-measurements/")
async def extract_measurements(
    front_image: UploadFile = File(...),
    height_feet: int = Form(0),
    inches: int = Form(0),
    height_cm: int = Form(0),
):
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
        config = create_improved_config()
        extractor = ImprovedBodyMeasurementExtractor(actual_height_cm=height_cm, config=config)
        measurements = extractor.extract(front_img)
        print(measurements.get("debug_info"))
        # Create visualization
        visualized_image = extractor.visualize_pose(front_img, show_measurements=True)
        # visualized_image.save("pose_visualization.jpg")
        visualized_image.show()
        
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"success": False, "message": f"Measurement error: {str(e)}"})

    if not measurements.get("success"):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=measurements)
    
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

# Helper to encode image to base64
def encode_image_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")

@app.post("/ollama-logic/extract-measurements/")
async def extract_measurements(front_image: UploadFile = File(...), height_cm: int = Form(...)):
    try:
        # Read and encode image
        file_bytes = await front_image.read()
        encoded_image = encode_image_base64(file_bytes)
        response = ollama_generate_response(height_cm, encoded_image)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.post("/gemini-logic/extract-measurements/")
async def extract_measurements(front_image: UploadFile = File(...), height_cm: int = Form(...)):
    try:
        response = gemini_generate_response(front_image.file, height_cm)
        return {
            "status": 200,
            "success": True,
            "message": "Upload successful.",
            "body_measurements_cm": response,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

