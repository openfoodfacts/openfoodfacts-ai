# !pip install paddleocr
# !pip install paddlepaddle

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

results = ocr.ocr('/content/application.jpeg', cls=True)

for line in results[0]:
    print("Detected text:", line[1][0])
