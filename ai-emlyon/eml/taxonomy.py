#Imports
import re
import numpy as np
import seaborn as sns
from robotoff.products import ProductDataset
from robotoff.taxonomy import get_taxonomy

#Setup
sns.set()
ds = ProductDataset.load()
taxonomy = get_taxonomy('category')

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
    - info_type = 'childs', 'parents', full_tax_list', 'tax_distance'
        - tax_list : Return a list with all the nodes (childs and parents) within the category taxonomy.
        - tax_distance : Return a dict with nb_parents, nb_childs and total nodes.
    
    """
    
    nonetype = type(None)
    cat_taxonomy = taxonomy[category]
    
    #If taxonomy exist, get information
    if not isinstance(cat_taxonomy, nonetype):
        cat_childs_list = [re.sub('<TaxonomyNode ', '', str(child))[:-1] for child in cat_taxonomy.children]
        cat_parents_list = [re.sub('<TaxonomyNode ', '', str(parent))[:-1] for parent in cat_taxonomy.parents]
        cat_tax_list = cat_childs_list + cat_parents_list
        
        #Set up output depending on information passed in arguments
        if info_type == 'childs': return cat_childs_list
        elif info_type == 'parents': return cat_parents_list
        elif info_type == 'full_tax_list': return cat_tax_list
        elif info_type == 'tax_distance':
            nb_parents = len(cat_parents_list)
            nb_childs = len(cat_childs_list)
            total_nodes = nb_parents + nb_childs + 1
            tax_distance = {'nb_parents': nb_parents, 'nb_childs': nb_childs, 'total_nodes':total_nodes}
            return tax_distance
    
    #If taxonomy does not exist, return nan
    else:
        return np.nan

def pred_is_in_tax (model_prediction, category_tag):
    model_prediction = model_prediction.lower().strip(' ').strip('')
    category_tag = category_tag.lower().strip(' ').strip('')
    cat_tax_list = get_taxonomy_info(category_tag, info_type='full_tax_list')
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

def find_pnns(row=None, df_taxonomy=None, pnns_n=1, nb_groups=4, search_duplicates=True):
    """
    Find a pnns group based on categories tags and taxonomy suggestions.

    Parameters:
    -----------
        - row: df row to apply
        - df_taxonomy: pd.DataFrame with pnns suggestions & matching tags
        - pnns_n:
        - group_idx: 

    Return:
    -------

    Example:
    --------

    """
    #---- Setup variables ----

    #setup idx desired
    pnns_index = pnns_n - 1 
    #setup list of pnns candidates
    pnns_candidates = [] 
    #setup output
    output = np.nan 

    #---- Search Categories Tags ----

    #convert row to a list of tags
    tags_list = row['categories_tags'].split(',') 
    #search parents in taxonomy
    tax_list = [get_taxonomy_info(tags_list[i], info_type='parents') for i in range(len(tags_list))]
    #keep only tags
    tax_list = [item for item in tax_list if not isinstance(item, float)] 
    #convert list of lists to list
    tax_list = [item for sublist in tax_list for item in sublist] 
    #add original tags
    tax_list = tax_list + tags_list 
    #clean strings
    tax_list = [tax.strip(' ').strip('') for tax in tax_list]
    #remove duplicates
    tax_list = list(set(tax_list)) 

    #---- Search pnns candidates ----

    #find candidates in main suggestions
    for suggestion, pnns in zip(df_taxonomy.taxonomy_suggestion, df_taxonomy.pnns): 
        if suggestion in tax_list: pnns_candidates.append(pnns)
    #find candidates in others possibilities
    for possibilities, pnns in zip(df_taxonomy.all_taxonomy_possibilities, df_taxonomy.pnns): 
        for possibility in possibilities.keys():
            if possibility in tax_list: pnns_candidates.append(pnns)

    #---- Setup pnns output ----

    #return nan if no pnns founded
    if not len(pnns_candidates): return output
    #return unique pnns if only one founded
    elif len(pnns_candidates) == 1: 
        output = pnns_candidates[0]
    #return pnns 1 or more
    elif len(pnns_candidates) > 1: 
        output = pnns_candidates[pnns_index]
    
    #---- Search duplicates option ----

    #If pnns already exist in another group
    if search_duplicates:
        existing_values = [row[f'pnns_groups_{i}'] for i in range (1, nb_groups)]
        if output in existing_values: 
        #try to find another pnns
            for i in range(len(pnns_candidates)):
                output = pnns_candidates[i]
                if output not in existing_values:
                    break
    #return output
    return output

def find_pnns_groups_1(df):
    """
    Find unknown pnns groups 1 based on known pnns groups 2.
    """
    data = df.copy()
    vals_to_find = list(data.pnns_groups_1.unique())
    vals_to_find.remove('unknown')
    for val in vals_to_find:
        group_2_vals = list(data['pnns_groups_2'].loc[data['pnns_groups_1'] == val].unique())
        data['pnns_groups_1'].loc[(data['pnns_groups_1'] == 'unknown') & 
        (data['pnns_groups_2'].isin(group_2_vals))] = val
    return data

def get_ingredients_columns(df, ingredients_list):
    """Computational expensive
    Create a new column for each item in ing_list
    and fill with percent_estimate or median if the row contains ingredient and 0 if not"""
    data = df.copy(deep=True) #make a copy of df
    for i in ingredients_list: data[i] = 0 #create 0 columns, 1 per ingredient
    for ingre_list, index in zip(data.ingredients, data.index): #loop over df rows
        for ingre_dict in ingre_list: #loop continue in the dicts in each row
            val = ingre_dict['text'] #get ingredient text
            val_clean = val.replace('_','').replace('-','').strip('').lower()
            if val_clean in ingredients_list: #check if val is in columns added
                try:
                    data.loc[index,val_clean] = ingre_dict['percent_estimate'] #if yes, replace by estimate
                except:
                    pnns_val = data.loc[index,'pnns_groups_1']
                    val_median = data[val_clean].loc[data['pnns_groups_1'] == pnns_val].median()
                    data.loc[index,val_clean] = val_median
    return data #return modified dataframe


#Older function
def depreciated_find_pnns(row, df_taxonomy):
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