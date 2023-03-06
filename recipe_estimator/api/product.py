from ciqual import ciqual_ingredients, nutrient_map, ingredients_taxonomy
import requests

def get_ciqual_code(ingredient_id):
    ingredient = ingredients_taxonomy.get(ingredient_id, None)
    if ingredient is None:
        print(ingredient_id + ' not found')
        return None

    ciqual_code = ingredient.get('ciqual_food_code', None)
    if ciqual_code:
        return ciqual_code['en']

    parents = ingredient.get('parents', None)
    if parents:
        for parent_id in parents:
            ciqual_code = get_ciqual_code(parent_id)
            if ciqual_code:
                print('Obtained ciqual_code from ' + parent_id)
                return ciqual_code

    return None

def setup_ingredients(off_ingredients):
    ingredients = []
    
    for off_ingredient in off_ingredients:
        ingredient = {}
        ingredients.append(ingredient)
        ingredient['text'] = off_ingredient['text']
    
        if ('ingredients' in off_ingredient):
            # Child ingredients
            child_ingredients = setup_ingredients(off_ingredient['ingredients'])
            if (child_ingredients is None):
                return

            ingredient['ingredients'] = child_ingredients
            #ingredients = ingredients + child_ingredients
        else:
            ciqual_code = get_ciqual_code(off_ingredient['id'])
            if (ciqual_code is None):
                print(off_ingredient['id'] + ' has no ciqual_food_code')
                continue

            ciqual_ingredient = ciqual_ingredients.get(ciqual_code, None)
            if (ciqual_ingredient is None):
                print(off_ingredient['id'] + ' has unknown ciqual_food_code: ' + ciqual_code)
                continue

            ingredient['ciqual_ingredient'] = ciqual_ingredient

    return ingredients


def prepare_ingredients(ingredients, nutrients):
    count = 0
    for ingredient in ingredients:
        if ('ingredients' in ingredient):
            # Child ingredients
            count = count + prepare_ingredients(ingredient['ingredients'], nutrients)
        else:
            count = count + 1
            ciqual_ingredient = ingredient.get('ciqual_ingredient', None)
            if (ciqual_ingredient is None):
                print('Error: ' + ingredient['text'] + ' has no ciqual ingredient')
                return

            ingredient['water_content'] = ciqual_ingredient['Water (g/100g)']

            ingredient['nutrients'] = {}

            # Eliminate any nutrients where the ingredient has an unknown or missing value
            for nutrient_key in nutrients:
                nutrinet = nutrients[nutrient_key]
                if ('error' in nutrinet):
                    continue

                nutrient_value = ciqual_ingredient.get(nutrient_key,None)
                if nutrient_value is None:
                    nutrinet['error'] = 'Some ingredients have unknown value'
                    continue

                ingredient['nutrients'][nutrient_key] = nutrient_value

                unweighted_total = nutrinet.get('unweighted_total',0)
                nutrinet['unweighted_total'] = unweighted_total + nutrient_value

    return count


def print_recipe(ingredients, indent = ''):
    for ingredient in ingredients:
        lost_water = ingredient.get('evaporation', '')
        if type(lost_water) == float:
            lost_water = '(' + str(lost_water) + ')'
        print(indent, '-', ingredient['text'], ingredient['proportion'], lost_water)
        if 'ingredients' in ingredient:
            print_recipe(ingredient['ingredients'], indent + ' ')


def get_product(id):
    response = requests.get("https://world.openfoodfacts.org/api/v3/product/" + id).json()
    if not 'product' in response:
        return {}

    product = response['product']
    off_ingredients = product['ingredients']
    off_nutrients = product['nutriments']
    #print(product['product_name'])
    #print(product_ingredients)
    #print(product['ingredients_text'])
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
                'weighting' : float(ciqual_nutrient.get('weighting',1) or 1)
            }
    #print(nutrients)
    ingredients = setup_ingredients(off_ingredients)

    return {'name': product['product_name'], 'ingredients_text': product['ingredients_text'], 'ingredients': ingredients, 'nutrients':nutrients}


def prepare_product(product):
    ingredients = product['ingredients']
    nutrients = product['nutrients']
    count = prepare_ingredients(ingredients, nutrients)

    for nutrient_key in nutrients:
        nutrient = nutrients[nutrient_key]
        if nutrient['total'] == 0 and nutrient['unweighted_total'] == 0:
            nutrient['error'] = 'Product and all ingredients have zero value'
        else:
            # Weighting based on size of ingredient, i.e. percentage based
            # Comment out this code to use weighting specified in nutrient_map.csv
            if nutrient['total'] > 0:
                nutrient['weighting'] = 1 / nutrient['total']
            else:
                nutrient['weighting'] = min(0.01, count / nutrient['unweighted_total']) # Weighting below 0.01 causes bad performance, although it isn't that simple as just multiplying all weights doesn't help

    # Favor Sodium over salt if both are present
    if not 'error' in nutrients.get('Sodium (mg/100g)',{}) and not 'error' in nutrients.get('Salt (g/100g)', {}):
        nutrients['Salt (g/100g)']['error'] = 'Prefer sodium where both present'


