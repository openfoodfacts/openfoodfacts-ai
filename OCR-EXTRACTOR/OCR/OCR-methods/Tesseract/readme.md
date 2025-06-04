# 📘 1. **Tesseract OCR**

**Overview**:  
Tesseract is an open-source OCR engine developed by Google. It's widely used for extracting text from scanned documents and images. It performs best on clean, high-contrast images with clear printed text.

**How It Works in Code**:

*   The image is first read using OpenCV and converted to grayscale.
    
*   Thresholding (using Otsu’s method) is applied to binarize the image, which improves Tesseract’s accuracy.
    
*   The grayscale image is converted into a PIL image format, which Tesseract can process.
    
*   Finally, `pytesseract.image_to_string()` extracts the text from the processed image.
    

**Key Code Snippet**:

```python
# Step 1: Read the image using OpenCV
image = cv2.imread('/content/application.jpeg')

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply binary thresholding (Otsu's method) to improve contrast
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Step 4: Convert the processed image to PIL format
pil_image = Image.fromarray(gray)

# Step 5: Use Tesseract to extract text from the image
text = pytesseract.image_to_string(pil_image)

# Output the extracted text
print("Extracted Text:")
print(text)
```
` 

**Use Case**: Best for clean printed documents with minimal distortion.
