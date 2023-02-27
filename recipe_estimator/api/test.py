from ortools.linear_solver import pywraplp
import sys
from product import get_product, print_recipe
from minimize_nutrient_distance import estimate_recipe


#query = {"ingredients_without_ciqual_codes_n": 0,"ingredients_n":{"$gt": 4}}
#query = {"_id": "0019962035357"}
# TODO: 0080948630439 looks odd
# TODO: Can't solve 0041268190638, 0677294998018, 0677294998025
# TODO: 0776455940115 doesn't show water loss
# TODO: 24005968 has lots of ingredient percentages but can't solve

product = get_product(sys.argv[1])
estimate_recipe(product)
print_recipe(product['ingredients'])

