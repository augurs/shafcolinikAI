import re
import os
import json
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_response(image_file, height_cm: int) -> dict:
    try:
        GEMINI_PROMPT = (
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
        image = Image.open(image_file)

        model = genai.GenerativeModel(model_name=os.getenv("GEMINI_MODEL"))

        response = model.generate_content(
            [GEMINI_PROMPT.strip(), image],
            generation_config={"temperature": 0.4}
        )
    
        cleaned_response = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.MULTILINE).strip()

        return json.loads(cleaned_response)

    except Exception as e:
        return {"valid": False, "reason": f"Gemini validation failed: {str(e)}"}