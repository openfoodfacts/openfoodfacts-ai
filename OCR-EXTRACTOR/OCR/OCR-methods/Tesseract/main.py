import cv2
import pytesseract
from PIL import Image

image = '/content/application.jpeg'

image = cv2.imread(image)

# Convert to grayscale (improves accuracy)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#more optimal
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Convert back to PIL image for pytesseract
pil_image = Image.fromarray(gray)

# Run OCR
text = pytesseract.image_to_string(pil_image)

print("Extracted Text:")
print(text)
