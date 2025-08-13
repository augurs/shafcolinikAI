import os
import re
import json
import requests
from fastapi import HTTPException


def generate_response(height_cm, encoded_image):
    # Build prompt
    prompt = (
        f"The person's actual height is {height_cm} cm. "
        "Using this height as a scaling reference, estimate the person's body measurements from the image with the highest possible accuracy. "
        "First, detect the person's gender (male or female). "
        "Analyze the image carefully, considering body shape, proportions, and posture. "
        "Identify and measure the following body parts: chest, waist, hips, inseam, thigh, neck, and arm length. "
        "Provide all measurements in centimeters, scaled proportionally to the given height. "
        "Estimate an overall accuracy score (as a percentage) based on the image quality and measurement confidence. "
        "Recommend the best clothing size (S, M, L, XL) based on the estimated measurements and gender. "
        "Return only the result as a strictly valid JSON object, formatted exactly as follows:\n\n"
        "{\n"
        f'  "height": {height_cm},\n'
        '  "gender": "<male_or_female>",\n'
        '  "chest": <value>,\n'
        '  "waist": <value>,\n'
        '  "hips": <value>,\n'
        '  "inseam": <value>,\n'
        '  "thigh": <value>,\n'
        '  "neck": <value>,\n'
        '  "arm_length": <value>,\n'
        '  "accuracy": <percentage>,\n'
        '  "recommended_size": "<S_or_M_or_L_or_XL>"\n'
        "}\n\n"
        "Only return the JSON — no explanation, no commentary, no extra text."
    )

    payload = {
        "model": os.getenv("MODEL_NAME"),
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False,
    }
    # Send request to model server
    response = requests.post(os.getenv("MODEL_API_URL"), json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Error from model server")

    result = response.json()
    # Try parsing model response
    try:
        # Remove Markdown code fencing (```json ... ```)
        cleaned_response = re.sub(r"^```(?:json)?|```$", "", result["response"], flags=re.MULTILINE).strip()
        measurements = json.loads(cleaned_response)
        res = {
            "status": 200,
            "success": True,
            "message": "Upload successful.",
            "body_measurements_cm": measurements,
        }
        return res
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model response was not valid JSON")


def validate_measurements(measurements, height_cm):
    SYSTEM_PROMPT = """
        You are an expert in human body measurements and fashion sizing for e-commerce clothing. 
        You will receive:
        - The user's actual height in centimeters.
        - A set of 7 body measurements (in centimeters) extracted from an image using MediaPipe.

        Your tasks:

        1. Validation
        - Check each measurement for realism based on the user's actual height.
        - Use proportional human body ranges to determine realistic values:
            - chest: 48-64% of height (example: height 180 cm → chest 86-115 cm)
            - waist: 36-48% of height
            - hips: 48-62% of height
            - inseam: 45-55% of height
            - thigh: 28-38% of height
            - neck: 20-26% of height
            - arm_length: 35-45% of height
        - If a value falls outside its realistic range, correct it by scaling proportionally to the user's height and mark it as "corrected" in the output.

        2. Clothing Size Recommendation
        - Based on the corrected measurements, recommend clothing sizes for:
            - US (S, M, L, XL, etc.)
            - EU (numeric sizes)
            - Asia (S, M, L, XL, etc., adjusted for regional fit differences)
        - Recommendations should prioritize fit for tops, bottoms, and full outfits.

        3. Output Format
        - Always return JSON with this exact structure:
            {
                "sanity_check": {
                "status": "pass" or "fail",
                "issues": [
                    {
                        "measurement": "string",
                        "original_value_cm": number,
                        "corrected_value_cm": number or null,
                        "reason": "string"
                    }
                ]
                },
                "final_measurements_cm": {
                    "height": number,
                    "chest": number,
                    "waist": number,
                    "hips": number,
                    "inseam": number,
                    "thigh": number,
                    "neck": number,
                    "arm_length": number
                },
                "clothing_recommendations": {
                    "US": "string",
                    "EU": "string",
                    "Asia": "string"
                }
            }
        - All measurements in centimeters.
        - No extra text outside the JSON.
        - If a measurement was corrected, include the corrected value in "final_measurements_cm".

        Rules:
        - Do not invent unrelated measurements.
        - Always correct unrealistic values using proportional scaling from the user's height.
        - If all values are realistic, set status to "pass" and leave corrections as null.
        - Ensure the JSON is valid and strictly follows the format.
        """

    # Prepare user input
    user_input = {"height": height_cm, **measurements}

    # Send request to Ollama's local API
    response = requests.post(
        os.getenv("MODEL_API_URL"),
        json={
            "model": os.getenv("MODEL_NAME"),
            "prompt": f"{SYSTEM_PROMPT}\nUser data:\n{json.dumps(user_input)}",
            "stream": False,
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error: {response.text}")

    # Extract text output from Ollama's response
    result_text = response.json().get("response", "")

    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        raise ValueError("Ollama did not return valid JSON")
