from transformers import AutoModelForTokenClassification, AutoProcessor

from datasets import DatasetDict

base_ds = DatasetDict.load_from_disk("datasets/ingredient-detection-layout-dataset-v1")

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=3
)

example = base_ds["train"][0]
image = example["image"]
words = example["words"]
boxes = example["bboxes"]
word_labels = example["ner_tags"]

width, height = image.size
boxes = [
    (
        int(x_min * 1000 / width),
        int(y_min * 1000 / height),
        int(x_max * 1000 / width),
        int(y_max * 1000 / height),
    )
    for (x_min, y_min, x_max, y_max) in boxes
]
encoding = processor(
    image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt"
)

outputs = model(**encoding)
