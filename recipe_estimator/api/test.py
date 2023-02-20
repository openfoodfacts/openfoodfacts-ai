from pymongo import MongoClient
from ortools.linear_solver import pywraplp
import csv
import os
import sys

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
            ingredient['ciqual_ingredient'] = ciqual_ingredient
            ingredient['water_content'] = parse_value(ciqual_ingredient['Water (g/100g)'])

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


def print_recipe(ingredients):
    for ingredient in ingredients:
        lost_water = ingredient.get('evaporation', '')
        if type(lost_water) == float:
            lost_water = '(' + str(lost_water) + ')'
        print(ingredient['indent'], '-', ingredient['text'], ingredient['proportion'], lost_water)
        if 'ingredients' in ingredient:
            print_recipe(ingredient['ingredients'])


def add_ingredients_to_solver(ingredients, solver, total_ingredients):
    ingredient_numvars = []

    for i,ingredient in enumerate(ingredients):

        ingredient_numvar = {'ingredient': ingredient, 'numvar': solver.NumVar(0.0, solver.infinity(), '')}
        ingredient_numvars.append(ingredient_numvar)
        # TODO: Known percentage or stated range
        if (i > 0):
            # Ingredient should be smaller than the one preceding it
            # i.e. (ingredient n-1) - (ingredient n) >= 0
            relative_constraint = solver.Constraint(0, solver.infinity(), '')
            relative_constraint.SetCoefficient(ingredient_numvars[i - 1]['numvar'], 1.0)
            relative_constraint.SetCoefficient(ingredient_numvar['numvar'], -1.0)

        if ('ingredients' in ingredient):
            # Child ingredients
            child_numvars = add_ingredients_to_solver(ingredient['ingredients'], solver, total_ingredients)

            ingredient_numvar['child_numvars'] = child_numvars
            # Parent percent - (sum child percent) = 0 (+/- precision)
            parent_constraint = solver.Constraint(-precision, precision, '')
            parent_constraint.SetCoefficient(ingredient_numvar['numvar'], 1.0)
            for child_numvar in child_numvars:
                parent_constraint.SetCoefficient(child_numvar['numvar'], -1.0)

        else:
            # Constrain water loss. If ingredient is 20% water then
            # raw ingredient - lost water must be greater than 80
            # ingredient - water_loss >= ingredient * (100 - water_ratio) / 100
            # ingredient - water_loss >= ingredient - ingredient * water ratio / 100
            # ingredient * water ratio / 100 - water_loss >= 0
            ingredient_numvar['lost_water'] = solver.NumVar(0.0, solver.infinity(), '')
            water_ratio = ingredient['water_content']

            water_loss_ratio_constraint = solver.Constraint(0, solver.infinity(),  '')
            water_loss_ratio_constraint.SetCoefficient(ingredient_numvar['numvar'], 0.01 * water_ratio)
            water_loss_ratio_constraint.SetCoefficient(ingredient_numvar['lost_water'], -1.0)

            total_ingredients.SetCoefficient(ingredient_numvar['numvar'], 1)
            total_ingredients.SetCoefficient(ingredient_numvar['lost_water'], -1.0)

    return ingredient_numvars


def set_solution_results(ingredient_numvars):
    for ingredient_numvar in ingredient_numvars:
        ingredient_numvar['ingredient']['proportion'] = ingredient_numvar['numvar'].solution_value()
        if ('child_numvars' in ingredient_numvar):
            set_solution_results(ingredient_numvar['child_numvars'])
        else:
            ingredient_numvar['ingredient']['evaporation'] = ingredient_numvar['lost_water'].solution_value()

    return


def add_nutrient_distance(ingredient_numvars, nutrient_key, positive_constraint, negative_constraint, weighting):
    for ingredient_numvar in ingredient_numvars:
        ingredient = ingredient_numvar['ingredient']
        if 'child_numvars' in ingredient_numvar:
            print(ingredient['indent'] + ' - ' + ingredient['text'] + ':')
            add_nutrient_distance(ingredient_numvar['child_numvars'], nutrient_key, positive_constraint, negative_constraint, weighting)
        else:
            # TODO: Figure out whether to do anything special with < ...
            ingredient_nutrient =  ingredient['nutrients'][nutrient_key]
            print(ingredient['indent'] + ' - ' + ingredient['text'] + ' (' + ingredient['ciqual_code'] + ') : ' + str(ingredient_nutrient))
            negative_constraint.SetCoefficient(ingredient_numvar['numvar'], -ingredient_nutrient / 100)
            positive_constraint.SetCoefficient(ingredient_numvar['numvar'], ingredient_nutrient / 100)


def estimate_recipe(product):
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
    if ingredients is None:
        return

    """Linear programming sample."""
    # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

    # Total of top level ingredients must add up to 100
    total_ingredients = solver.Constraint(100 - precision, 100 + precision, '')
    ingredient_numvars = add_ingredients_to_solver(ingredients, solver, total_ingredients)

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
        #if nutrient_total > 0:
        #    weighting = 1 / nutrient_total
        #else:
        #    weighting = 1 / nutrient['parts']
        weighting = 1

        nutrient_distance = solver.NumVar(0, 100, nutrient_key)

        # not sure this is right as if one ingredient is way over and another is way under
        # then will give a good result
        negative_constraint = solver.Constraint(-solver.infinity(), -nutrient_total)
        negative_constraint.SetCoefficient(nutrient_distance, 1)
        positive_constraint = solver.Constraint(nutrient_total, solver.infinity())
        positive_constraint.SetCoefficient(nutrient_distance, 1)
        print(nutrient_key, nutrient_total, weighting)
        add_nutrient_distance(ingredient_numvars, nutrient_key, positive_constraint, negative_constraint, weighting)

        objective.SetCoefficient(nutrient_distance, weighting)

    objective.SetMinimization()

    status = solver.Solve()

    # Check that the problem has an optimal solution.
    if status == solver.OPTIMAL:
        print('An optimal solution was found in', solver.iterations(), 'iterations')
    else:
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found in', solver.iterations(), 'iterations')
        else:
            print('The solver could not solve the problem.')

    set_solution_results(ingredient_numvars)
    print_recipe(ingredients)

    # TODO: Print calculated nutrients


#query = {"ingredients_without_ciqual_codes_n": 0,"ingredients_n":{"$gt": 4}}
#query = {"_id": "0019962035357"}
# TODO: 0080948630439 looks odd
# TODO: Can't solve 0041268190638, 0677294998018, 0677294998025
# TODO: 0776455940115 doesn't show water loss
# TODO: 24005968 has lots of ingredient percentages but can't solve

query = {"_id": sys.argv[1]}
product = products.find_one(query)
estimate_recipe(product)

