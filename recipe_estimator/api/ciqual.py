import csv
import json
import os

def parse_value(ciqual_nutrient):
    if not ciqual_nutrient or ciqual_nutrient == '-':
        return None
    return float(ciqual_nutrient.replace(',','.').replace('<','').replace('traces','0'))

# Load Ciqual data
ciqual_ingredients = {}
filename = os.path.join(os.path.dirname(__file__), "Ciqual.csv.0")
with open(filename, newline="", encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        values = list(row.values())
        keys = list(row.keys())
        for i in range(9,len(values)):
            value = parse_value(values[i])
            row[keys[i]] = value
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
