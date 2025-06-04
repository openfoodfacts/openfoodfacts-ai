# 📙 2. **EasyOCR**

**Overview**:  
EasyOCR is a deep learning–based OCR library built on PyTorch. It supports multiple languages and performs well even with moderately distorted text, including some handwritten or stylized fonts.

**How It Works in Code**:

*   The EasyOCR `Reader` object is initialized with the English language.
    
*   The `readtext()` method reads the image file and detects text along with the bounding boxes and confidence scores.
    
*   The results include the detected text and a score indicating how confident the model is about each detection.
    

**Key Code Snippet**:

```python
# Step 1: Import and initialize the EasyOCR reader with English language support
reader = easyocr.Reader(['en'])

# Step 2: Read the image and extract text along with bounding boxes and confidence scores
result = reader.readtext('/content/application.jpeg')

# Step 3: Loop through results and print the detected text with confidence level
for (bbox, text, prob) in result:
    print(f"Detected text: {text} (Confidence: {prob:.2f})")
```


**Use Case**: Great for multilingual and slightly noisy inputs; it’s also beginner-friendly with built-in visualization options.
