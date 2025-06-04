# !pip install easyocr

import easyocr

reader = easyocr.Reader(['en'])

result = reader.readtext('/content/application.jpeg')

for (bbox, text, prob) in result:
  print(f"Detected text: {text} (Confidence: {prob:.2f})")
