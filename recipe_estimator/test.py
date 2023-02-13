from pymongo import MongoClient
from ortools.linear_solver import pywraplp
import csv
import os

# Connect to local Mongo DB
products = MongoClient(host="localhost", port=27017,).off.products

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

precision = 0.01
#print(nutrient_map['pantothenic-acid'])
#print(len(nutrient_map))

def parse_value(ciqual_nutrient):
    return float(ciqual_nutrient.replace(',','.').replace('<','').replace('traces','0'))

def setup_ingredients(off_ingredients,nutrients,solver,total_ingredients):
    ingredients = []
    
    for i,off_ingredient in enumerate(off_ingredients):
        ingredient = {}
        ingredients.append(ingredient)
        ingredient['text'] = off_ingredient['text']
        ingredient['numvar'] = solver.NumVar(0.0, solver.infinity(), off_ingredient['id'])
        if (i > 0):
            # Ingredient should be smaller than the one preceding it
            # i.e. (ingredient n-1) - (ingredient n) >= 0
            relative_constraint = solver.Constraint(0, solver.infinity(), off_ingredient['id'] + ' relative percent')
            relative_constraint.SetCoefficient(ingredients[i - 1]['numvar'], 1.0)
            relative_constraint.SetCoefficient(ingredient['numvar'], -1.0)
            ingredient['relative_constraint'] = relative_constraint
        
        if ('ingredients' in off_ingredient):
            # Child ingredients
            child_ingredients = setup_ingredients(off_ingredient['ingredients'], nutrients, solver, total_ingredients)
            if (child_ingredients is None):
                return

            ingredient['ingredients'] = child_ingredients
            # Parent percent - (sum child percent) = 0 (+/- precision)
            parent_constraint = solver.Constraint(-precision, precision, off_ingredient['id'] + ' parent percent')
            ingredient['parent_constraint'] = parent_constraint
            parent_constraint.SetCoefficient(ingredient['numvar'], 1.0)

            for child_ingredient in child_ingredients:
                parent_constraint.SetCoefficient(child_ingredient['numvar'], -1.0)
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
            ingredient['ciqual_ingredient'] = ciqual_ingredient

            # Constrain water loss. If ingredient is 20% water then
            # raw ingredient - lost water must be greater than 80
            # ingredient - water_loss >= ingredient * (100 - water_ratio) / 100
            # ingredient - water_loss >= ingredient - ingredient * water ratio / 100
            # ingredient * water ratio / 100 - water_loss >= 0
            ingredient['lost_water'] = solver.NumVar(0.0, solver.infinity(), off_ingredient['id'] + ' lost water')
            water_ratio = parse_value(ciqual_ingredient['Water (g/100g)'])

            water_loss_ratio_constraint = solver.Constraint(0, solver.infinity(), off_ingredient['id'] + ' water looss ratio')
            water_loss_ratio_constraint.SetCoefficient(ingredient['numvar'], 0.01 * water_ratio)
            water_loss_ratio_constraint.SetCoefficient(ingredient['lost_water'], -1.0)

            total_ingredients.SetCoefficient(ingredient['numvar'], 1)
            total_ingredients.SetCoefficient(ingredient['lost_water'], -1.0)

            ingredient['nutrients'] = {}

            # Eliminate any nutrients where the ingredient has an unknown or missing value
            for nutrient_key in nutrients:
                if (not nutrients[nutrient_key]['valid']):
                    continue

                ciqual_nutrient = ciqual_ingredient.get(nutrient_key,None)
                if ciqual_nutrient is None:
                    print('Skippping ' + nutrient_key + ' as no value for ' + ingredient['text'] + ' (' + ciqual_code + ')')
                    nutrients[nutrient_key]['valid'] = False
                    continue

                if ciqual_nutrient == '-':
                    print('Skippping ' + nutrient_key + ' as unknown (-) value for ' + ingredient['text'] + ' (' + ciqual_code + ')')
                    nutrients[nutrient_key]['valid'] = False
                    continue

                nutrient_value = parse_value(ciqual_nutrient)
                nutrients[nutrient_key]['parts'] = max(nutrient_value, nutrients[nutrient_key]['parts'])
                ingredient['nutrients'][nutrient_key] = nutrient_value

    return ingredients

def add_nutrient_distance(ingredients, nutrient_key, positive_constraint, negative_constraint, weighting):
    for ingredient in ingredients:
        if 'ingredients' in ingredient:
            add_nutrient_distance(ingredient['ingredients'], nutrient_key, positive_constraint, negative_constraint, weighting)
        else:
            # TODO: Figure out whether to do anything special with < ...
            ingredient_nutrient =  ingredient['nutrients'][nutrient_key]
            print(' - ' + ingredient['text'] + ' (' + ingredient['ciqual_code'] + ') : ' + str(ingredient_nutrient))
            negative_constraint.SetCoefficient(ingredient['numvar'], -weighting * ingredient_nutrient / 100)
            positive_constraint.SetCoefficient(ingredient['numvar'], weighting * ingredient_nutrient / 100)

def print_recipe(ingredients, indent = ''):
    for ingredient in ingredients:
        lost_water = ingredient.get('lost_water', '')
        if lost_water:
            lost_water = '(' + str(lost_water.solution_value()) + ')'
        print(indent, '-', ingredient['text'], ingredient['numvar'].solution_value(), lost_water)
        if 'ingredients' in ingredient:
            print_recipe(ingredient['ingredients'], indent + ' ')


def EstimateRecipe(query):
    product = products.find_one(query)

    off_ingredients = product['ingredients']
    off_nutrients = product['nutriments']
    print(product['product_name'])
    #print(product_ingredients)
    print(product['ingredients_text'])
    #print(off_nutrients)

    """Linear programming sample."""
    # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

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

    # Total of top level ingredients must add up to at least 100 (allow more than 100 to account for loss of water in processing)
    total_ingredients = solver.Constraint(100 - precision, 100 + precision, 'sum')
    ingredients = setup_ingredients(off_ingredients, nutrients, solver, total_ingredients)
    if ingredients is None:
        return

    objective = solver.Objective()
    for nutrient_key in nutrients:
        nutrient = nutrients[nutrient_key]
        if not nutrient['valid']:
            continue
        if nutrient['total'] == 0 and nutrient['parts'] == 0:
            print('Skippping ' + nutrient_key + ' as product and all ingredients have zero value')
            continue

        # We want to minimise the absolute difference between the sum of the ingredient nutients and the total nutrients
        # i.e. minimize(abs(sum(Ni) - Ntot))
        # However we can't do absolute as it isn't linear
        # We get around this by also adding constraints
        # sum(Ni) - Ntot - Ndist >= 0
        # sum(Ni) - Ntot - Ndist <= 0
        # or
        # sum(Ni) - Ndist >= Ntot 
        # sum(Ni) - Ndist <= Ntot

        nutrient_distance = solver.NumVar(0, solver.infinity(), nutrient_key)
        nutrient_total = nutrient['total']
        weighting = nutrient['weighting']
        if nutrient_total > 0:
            weighting = 1 / nutrient_total
        else:
            weighting = 1 / nutrient['parts']

        nutrient_distance = solver.NumVar(0, solver.infinity(), nutrient_key)

        negative_constraint = solver.Constraint(-weighting * nutrient_total,solver.infinity())
        negative_constraint.SetCoefficient(nutrient_distance, 1)
        positive_constraint = solver.Constraint(weighting * nutrient_total, solver.infinity())
        positive_constraint.SetCoefficient(nutrient_distance, 1)
        print(nutrient_key, nutrient_total, weighting)
        add_nutrient_distance(ingredients, nutrient_key, positive_constraint, negative_constraint, weighting)

        objective.SetCoefficient(nutrient_distance, 1)

    objective.SetMinimization()

    status = solver.Solve()

    # Check that the problem has an optimal solution.
    if status == solver.OPTIMAL:
        print('An optimal solution was found in', solver.iterations(), 'iterations')
    else:
        print('The problem does not have an optimal solution!')
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found in', solver.iterations(), 'iterations')
        else:
            print('The solver could not solve the problem.')
            exit(1)

    print_recipe(ingredients)

    # TODO: Print calculated nutrients

#EstimateRecipe({"ingredients_without_ciqual_codes_n": 0,"ingredients_n":{"$gt": 4}})
# Sample with nested ingredients: 
EstimateRecipe({"_id": "0019962035357"})

