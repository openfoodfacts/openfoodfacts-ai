from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline, TokenClassificationPipeline
import wandb

ARTIFACT_NAME = (
    "raphaeloff/ingredient-detection-ner/model-xlm-roberta-large-20-epochs-alpha-v6:v0"
)

DATASET_URLS = {
    "train": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_train.jsonl.gz",
    "test": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_test.jsonl.gz",
}


api = wandb.Api()
artifact = api.artifact(ARTIFACT_NAME, type="model")
checkpoint_path = artifact.download()

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
base_ds = load_dataset("json", data_files=DATASET_URLS)

# classifier = pipeline(
#     "ner", model=model, tokenizer=tokenizer, aggregation_strategy=None
# )

text = """E: LIN INGRÉDIENTS Lait entier de brebis (origine : France) 84 %, riz 9%, sucre 6%, amidon de riz. QUE PEPER Riz au Vous serez séduit par l'onctuosité des RIZ AU LAIT DE BREBIS NATURE SUCRÉ PENSEZ AU TRI ! MAR Poids net : L CONSE A consc sur le d Conser POTS PLASTIQUE À JETER CONSIGNE POUVANT VARIER LOCALEMENT> WW 280 g = 2x140 """
# outputs = classifier(text)

classifier_aggregated = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)
entities = classifier_aggregated(text)
for entity in entities:
    print(text[entity["start"] : entity["end"]])
