#----------------------------LUCAIN SCRIPT FOR TAXONOMY & PNNS----------------------------

import json

#if not installed : pip install python-Levenshtein
from Levenshtein import distance as levenshtein_distance
from robotoff.taxonomy import get_taxonomy

# Get category taxonomy
taxonomy = get_taxonomy("category")

# PNNS categories
categories = {
    "appetizers": 0,
    "artificially sweetened beverages": 1,
    "biscuits and cakes": 2,
    "bread": 3,
    "breakfast cereals": 4,
    "cereals": 5,
    "cheese": 6,
    "chocolate products": 7,
    "dairy desserts": 8,
    "dressings and sauces": 9,
    "dried fruits": 10,
    "eggs": 11,
    "fats": 12,
    "fish and seafood": 13,
    "fruit juices": 14,
    "fruit nectars": 15,
    "fruits": 16,
    "ice cream": 17,
    "legumes": 18,
    "meat": 19,
    "milk and yogurt": 20,
    "nuts": 21,
    "offals": 22,
    "one dish meals": 23,
    "pastries": 24,
    "pizza pies and quiche": 25,
    "plant based milk substitutes": 26,
    "potatoes": 27,
    "processed meat": 28,
    "salty and fatty products": 29,
    "sandwiches": 30,
    "soups": 31,
    "sweetened beverages": 32,
    "sweets": 33,
    "teas and herbal teas and coffees": 34,
    "unsweetened beverages": 35,
    "vegetables": 36,
    "waters and flavored waters": 37,
}


def _clean_str(s):
    """Make the string "searchable"."""
    return "".join(s.lower().split())


# For each PNNS category, we prepare a set of taxonomy suggestion
categories_w_nodes = {
    _clean_str(cat): {"category": cat, "nodes": set()} for cat in categories
}


# Iterate over the nodes of the taxonomy
for node in taxonomy.iter_nodes():
    # Fetch all synonyms of the node in any language
    synonyms = set(_clean_str(word) for lang in node.synonyms.values() for word in lang)
    for synonym in synonyms:
        for category in categories_w_nodes:
            # If category and synonym "looks the same", we add the node to our suggestions
            if synonym in category or category in synonym:
                categories_w_nodes[category]["nodes"].add(node.id)


# Loop through categories, compute a similarity for each node and prepare export
output = []
for cat, details in categories_w_nodes.items():
    if len(details["nodes"]) > 0:
        possibilities = {
            node: levenshtein_distance(node.split(":")[1], details["category"])
            for node in details["nodes"]
        }

        output.append(
            {
                "pnns": details["category"],
                "taxonomy_suggestion": sorted(
                    possibilities.items(), key=lambda x: x[1]
                )[0][0],
                "all_taxonomy_possibilities": possibilities,
            }
        )
    else:
        raise ValueError("Category must have a suggestion :D ")

# export to JSON
with open("taxonomy_pnns.json", "w") as f:
    json.dump(output, f, indent=2, sort_keys=True)