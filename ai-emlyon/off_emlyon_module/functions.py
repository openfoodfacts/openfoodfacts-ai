#Imports
import re
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import robotoff
import seaborn as sns
from robotoff.products import ProductDataset
from robotoff.taxonomy import get_taxonomy

#Init instances
sns.set()
ds = ProductDataset.load()
taxonomy = get_taxonomy('category')

def imports_off(tax=False, dataviz=True, datamanip=True, ml=False, base=True, emlyonmodule=True):
    if base:
        import re
        import time
        import os
    if datamanip:
        import numpy as np
        import pandas as pd
        
    if tax:
        import robotoff
        from robotoff.products import ProductDataset
        from robotoff.taxonomy import get_taxonomy
        ds = ProductDataset.load()
        taxonomy = get_taxonomy('category')
    if dataviz:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        
    
    if ml:
        from sklearn import metrics, ensemble, preprocessing  
        from sklearn.model_selection import train_test_split

    if emlyonmodule:
        from IPython.lib.deepreload import reload as dreload

def get_taxonomy_info (category=str, info_type='tax_list'):
    
    """
    Return taxonomy info for a category tag, with cleaned nodes strings.
    Can be a list of strings with all tags in the taxonomy branch or 
    a dict of integers with distances within the taxonomy branch 
    If the taxonomy does not exist for the category filled, return np.nan
    
    Parameters
    ------------
    - category = str, mandatory 
        Category tag to explore.
    - info_type ='tax_list' or 'tax_distance'
        - tax_list : Return a list with all the nodes (childs and parents) within the category taxonomy.
        - tax_distance : Return a dict with nb_parents, nb_childs and total nodes.
    
    """
    
    nonetype = type(None)
    cat_taxonomy = taxonomy[category]
    if not isinstance(cat_taxonomy, nonetype):
        cat_childs_list = [re.sub('<TaxonomyNode ', '', str(child))[:-1] for child in cat_taxonomy.children]
        cat_parents_list = [re.sub('<TaxonomyNode ', '', str(parent))[:-1] for parent in cat_taxonomy.parents]
        cat_tax_list = cat_childs_list + cat_parents_list
        if info_type == 'tax_list':
            return cat_tax_list
        if info_type == 'tax_distance':
            nb_parents = len(cat_parents_list)
            nb_childs = len(cat_childs_list)
            total_nodes = nb_parents + nb_childs + 1
            tax_distance = {'nb_parents': nb_parents, 'nb_childs': nb_childs, 'total_nodes':total_nodes}
            return tax_distance
    else:
        return np.nan

def pred_is_in_tax (model_prediction=str, category_tag=str):
    model_prediction = model_prediction.lower().strip(' ').strip('')
    category_tag = category_tag.lower().strip(' ').strip('')
    cat_tax_list = get_taxonomy_info(category_tag, info_type='tax_list')
    if not isinstance(cat_tax_list, list):
        return np.nan
    elif model_prediction in cat_tax_list:
        return True
    else:
        return False

def taxonomy_distance(category=str, distance_type='total_nodes'):
    
    """
    Compute the distance in the branch taxonomy
    
    Parameters
    ----------
    - category : str - category tag to use
    - distance type : str ('total_nodes', 'nb_childs', 'nb_parents')
        - total_nodes : total number of nodes, including category filled, in the category branch
        - nb_childs : total number of childs
        - nb_parents : total number of parents

    Return
    ------
    The specified distance as integer

    """
    distances = get_taxonomy_info(category=category, info_type='tax_distance')
    return distances[distance_type]

def find_pnns(row, df_taxonomy):
    output = np.nan
    tags_list = row.split(',')
    tax_list = get_taxonomy_info(tags_list[0])
    if not isinstance(tax_list, float):
        tax_list =  tax_list + tags_list
        tax_list = [tax.strip(' ').strip('') for tax in tax_list]
        tax_list = list(set(tax_list))
        for suggestion, pnns in zip(df_taxonomy.taxonomy_suggestion, df_taxonomy.pnns):
            if suggestion in tax_list:
                output = pnns
                break
            else:
                continue
        if isinstance(output, float):
            for possibilities, pnns in zip(df_taxonomy.all_taxonomy_possibilities, df_tax.pnns):
                for possibility in possibilities.keys():
                    if possibility in tax_list:
                        output = pnns
                        break
                    else :
                        continue
    return output

#----------------------------END----------------------------