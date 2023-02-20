from pymongo import MongoClient
from ciqual import ciqual_ingredients, nutrient_map

# Connect to local Mongo DB
products = MongoClient(host="localhost", port=27017,).off.products

def parse_value(ciqual_nutrient):
    return float(ciqual_nutrient.replace(',','.').replace('<','').replace('traces','0'))


def setup_ingredients(off_ingredients, nutrients, indent):
    ingredients = []
    
    for off_ingredient in off_ingredients:
        ingredient = {}
        ingredients.append(ingredient)
        ingredient['text'] = off_ingredient['text']
        ingredient['indent'] = indent
        
        if ('ingredients' in off_ingredient):
            # Child ingredients
            child_ingredients = setup_ingredients(off_ingredient['ingredients'], nutrients, indent + ' ')
            if (child_ingredients is None):
                return

            ingredient['ingredients'] = child_ingredients
        else:
            ciqual_code = off_ingredient.get('ciqual_food_code', None)
            if (ciqual_code is None):
                print('Error: ' + off_ingredient['text'] + ' has no ciqual_food_code')
                return
            ingredient['ciqual_code'] = ciqual_code
            ciqual_ingredient = ciqual_ingredients.get(ciqual_code, None)
            if (ciqual_ingredient is None):
                print('Error: ' + off_ingredient['text'] + ' has unknown ciqual_food_code: ' + ciqual_code)
                return
            ingredient['water_content'] = parse_value(ciqual_ingredient['Water (g/100g)'])

            ingredient['nutrients'] = {}

            # Eliminate any nutrients where the ingredient has an unknown or missing value
            for nutrient_key in nutrients:
                nutrinet = nutrients[nutrient_key]
                if (not nutrinet['valid']):
                    continue

                ciqual_nutrient = ciqual_ingredient.get(nutrient_key,None)
                if ciqual_nutrient is None:
                    nutrinet['valid'] = False
                    nutrinet['error'] = 'Some ingredients have no value'
                    continue

                if ciqual_nutrient == '-':
                    nutrinet['valid'] = False
                    nutrinet['error'] = 'Some ingredients have unknown (-) value'
                    continue

                nutrient_value = parse_value(ciqual_nutrient)
                nutrinet['parts'] = max(nutrient_value, nutrinet['parts'])
                ingredient['nutrients'][nutrient_key] = nutrient_value

    return ingredients


def print_recipe(ingredients):
    for ingredient in ingredients:
        lost_water = ingredient.get('evaporation', '')
        if type(lost_water) == float:
            lost_water = '(' + str(lost_water) + ')'
        print(ingredient['indent'], '-', ingredient['text'], ingredient['proportion'], lost_water)
        if 'ingredients' in ingredient:
            print_recipe(ingredient['ingredients'])


def get_product(id):
    query = {"_id": id}
    product = products.find_one(query)
    if not product:
        return {}

    off_ingredients = product['ingredients']
    off_nutrients = product['nutriments']
    print(product['product_name'])
    #print(product_ingredients)
    print(product['ingredients_text'])
    #print(off_nutrients)

    nutrients = {}
    for off_nutrient_key in off_nutrients:
        if off_nutrient_key in nutrient_map:
            ciqual_nutrient = nutrient_map[off_nutrient_key]
            ciqual_unit = ciqual_nutrient['ciqual_unit']
            # Normalise units. OFF units are generally g so need to convert to the
            # Ciqual unit for comparison
            factor = 1.0
            if ciqual_unit == 'mg':
                factor = 1000.0
            elif ciqual_unit == 'µg':
                factor = 1000000.0
            nutrients[ciqual_nutrient['ciqual_id']] = {
                'total': float(off_nutrients[off_nutrient_key]) * factor, 
                'parts': 0.0, 
                'weighting' : float(ciqual_nutrient.get('weighting',1) or 1), 
                'valid' : True
            }
    #print(nutrients)

    ingredients = setup_ingredients(off_ingredients, nutrients, '')

    for nutrient_key in nutrients:
        nutrient = nutrients[nutrient_key]
        if nutrient['total'] == 0 and nutrient['parts'] == 0:
            nutrient['valid'] = False
            nutrient['error'] = 'Product and all ingredients have zero value'

    return {'name': product['product_name'], 'ingredients_text': product['ingredients_text'], 'ingredients': ingredients, 'nutrients':nutrients}

