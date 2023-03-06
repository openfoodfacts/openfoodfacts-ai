import csv
import json
import os

# Load Ciqual data
ciqual_ingredients = {}
filename = os.path.join(os.path.dirname(__file__), "Ciqual.csv.0")
with open(filename, newline="", encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ciqual_ingredients[row["alim_code"]] = row

# print(ciqual_ingredients['42501'])

# Load OFF Ciqual Nitrient mapping
nutrient_map = {}
filename = os.path.join(os.path.dirname(__file__), "nutrient_map.csv")
with open(filename, newline="", encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["ciqual_id"]:
            nutrient_map[row["off_id"]] = row

# Load ingredients
filename = os.path.join(os.path.dirname(__file__), "ingredients.json")
with open(filename, "r", encoding="utf-8") as ingredients_file:
    ingredients_taxonomy = json.load(ingredients_file)

# Dump ingredients
#with open(filename, "w", encoding="utf-8") as ingredients_file:
#    json.dump(
#        ingredients, ingredients_file, sort_keys=True, indent=4, ensure_ascii=False
#    )
