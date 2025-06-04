OCR-EXTRACTOR PROJECT
=====================

INTRODUCTION
------------

OCR-EXTRACTOR is an AI-powered tool built for the OpenFoodFacts open-source community. Its main goal is to make product data globally accessible by extracting and translating text from images uploaded by users around the world.

When someone uploads an image — for example, a food label — the system uses advanced OCR (Optical Character Recognition) to read the text. If the text is not in English, our AI will automatically translate it to English using Google’s Gemini LLM (Large Language Model). This translated text can then be used to create consistent, accessible data entries.

WHY PADDLEOCR?
--------------

Although the project includes multiple OCR engines for testing and comparison, the final production solution is built using **PaddleOCR**. After extensive evaluation, PaddleOCR was chosen because:

*   It supports a wide range of languages out of the box
    
*   It offers high accuracy, especially for real-world images and labels
    
*   It is fast and optimized for performance
    
*   It is well-maintained and open-source
    

Other engines like EasyOCR and Tesseract are included in the project structure for experimentation, but PaddleOCR gave the best results overall.

TRANSLATION POWERED BY GEMINI AI
--------------------------------

To handle the complexity of real-world languages and regional product text, the project uses Google’s Gemini Generative AI (LLM). This model understands context better than traditional translators and produces more natural, accurate English translations.

FOLDER STRUCTURE OVERVIEW
-------------------------

OCR-EXTRACTOR/  
│  
├── OCR/  
│ ├── backup/  
│ │ └── python\_ocr.py ← Older backup version of the script  
│ ├── input image/  
│ │ └── application.jpeg ← Sample input image  
│ └── MAIN/  
│ ├── .env ← Environment variables (e.g., Gemini API key)  
│ └── main.py ← The main production script using PaddleOCR + Gemini  
│  
├── notebook/  
│ └── Python\_OCR.ipynb ← Jupyter notebook for demos or testing  
│  
├── OCR-methods/  
│ ├── EasyOCR/  
│ │ ├── main.py  
│ │ └── readme.md ← Documentation for EasyOCR testing  
│ ├── PaddleOCR/  
│ │ ├── main.py  
│ │ └── readme.md ← ✅ Primary OCR method used in production  
│ └── Tesseract/  
│ ├── main.py  
│ └── readme.md ← Another OCR method tested  
│  
└── output/  
└── ← This folder stores results/output (optional)

GETTING STARTED
---------------

1.  Install all required dependencies using:  
    pip install -r requirements.txt
    
2.  Add your Gemini API key to the `.env` file.
    
3.  Place your image inside the "input image" folder.
    
4.  Run the script from `OCR/MAIN/main.py`.
    
5.  The script will:
    
    *   Extract text using PaddleOCR
        
    *   Detect the language
        
    *   Translate it to English using Gemini AI (if needed)
        
    *   Output the result
        

CONTRIBUTING
------------

This project is designed for global collaboration. Anyone interested in open food data, OCR, or AI-driven translation is welcome to contribute, test, or improve the code.

LICENSE
-------

This project is open-source and intended for public good. See LICENSE file for more details.

CONTACT
-------

Made for OpenFoodFacts by the community. For any questions or collaboration ideas, feel free to raise an issue or reach out through GitHub.