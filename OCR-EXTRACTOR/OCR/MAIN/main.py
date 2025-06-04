import os
import google.generativeai as genai
from paddleocr import PaddleOCR
from langdetect import detect

# ========== Setup Gemini ==========
GEMINI_API_KEY = os.getenv(api_key) 
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro')

# ========== OCR Setup ==========
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    texts = []
    for line in result:
        for word in line:
            texts.append(word[1][0])
    return " ".join(texts)

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"

def translate_to_english(text):
    prompt = f"Translate this text to English:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()

def process_image(image_path):
    print(f"[INFO] Processing: {image_path}")
    
    text = extract_text_from_image(image_path)
    print(f"[OCR Output]: {text}")

    lang = detect_language(text)
    print(f"[Detected Language]: {lang}")

    if lang != "en":
        translated_text = translate_to_english(text)
        print(f"[Translated to English]: {translated_text}")
    else:
        print("[Text is already in English]")

# ========== Example ==========
if __name__ == "__main__":
    image_path = input("enter your image path")
    process_image(image_path)
