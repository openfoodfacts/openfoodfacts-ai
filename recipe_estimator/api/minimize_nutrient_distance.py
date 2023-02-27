from ortools.linear_solver import pywraplp

precision = 0.01
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
            #print(ingredient['indent'] + ' - ' + ingredient['text'] + ':')
            add_nutrient_distance(ingredient_numvar['child_numvars'], nutrient_key, positive_constraint, negative_constraint, weighting)
        else:
            # TODO: Figure out whether to do anything special with < ...
            ingredient_nutrient =  ingredient['nutrients'][nutrient_key]
            #print(ingredient['indent'] + ' - ' + ingredient['text'] + ' (' + ingredient['ciqual_code'] + ') : ' + str(ingredient_nutrient))
            negative_constraint.SetCoefficient(ingredient_numvar['numvar'], -ingredient_nutrient / 100)
            positive_constraint.SetCoefficient(ingredient_numvar['numvar'], ingredient_nutrient / 100)


def estimate_recipe(product):
    ingredients = product['ingredients']
    nutrients = product['nutrients']
    
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

        # We want to minimise the absolute difference between the sum of the ingredient nutients and the total nutrients
        # i.e. minimize(abs(sum(Ni) - Ntot))
        # However we can't do absolute as it isn't linear
        # We get around this by also adding constraints
        # sum(Ni) - Ntot - Ndist >= 0
        # sum(Ni) - Ntot - Ndist <= 0
        # or
        # sum(Ni) - Ndist >= Ntot 
        # sum(Ni) - Ndist <= Ntot

        nutrient_total = nutrient['total']
        weighting = nutrient['weighting']
        #if nutrient_total > 0:
        #    weighting = 1 / nutrient_total
        #else:
        #    weighting = 1 / nutrient['parts']
        #weighting = 1

        nutrient_distance = solver.NumVar(0, 100, nutrient_key)

        # not sure this is right as if one ingredient is way over and another is way under
        # then will give a good result
        negative_constraint = solver.Constraint(-solver.infinity(), -nutrient_total)
        negative_constraint.SetCoefficient(nutrient_distance, 1)
        positive_constraint = solver.Constraint(nutrient_total, solver.infinity())
        positive_constraint.SetCoefficient(nutrient_distance, 1)
        #print(nutrient_key, nutrient_total, weighting)
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
            return status

    set_solution_results(ingredient_numvars)

    return status
