import csv
import os

# Load Ciqual data
ciqual_ingredients = {}
filename = os.path.join(os.path.dirname(__file__), 'Ciqual.csv.0')
with open(filename, newline='', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ciqual_ingredients[row['alim_code']] = row

#print(ciqual_ingredients['42501'])

# Load OFF Ciqual Nitrient mapping
nutrient_map = {}
filename = os.path.join(os.path.dirname(__file__), 'nutrient_map.csv')
with open(filename, newline='', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['ciqual_id']):
            nutrient_map[row['off_id']] = row
