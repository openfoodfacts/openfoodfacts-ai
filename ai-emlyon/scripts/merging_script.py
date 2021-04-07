#__________________________Merging Predicitions with OFF Database__________________________

#---------------Libraries---------------

import pandas as pd
from pandas.io.json import json_normalize

#---------------Reading files---------------

#Read OFF full database
full_database = pd.read_csv(r'C:\Users\Antoine\Coding Bootcamp\machine learning\
Open Food Facts\data\en.openfoodfacts.org.products.csv', low_memory = False,
sep='\t',error_bad_lines=False)

#Read all robotoff predictions files
pred1 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page1_by_100000.json')
pred2 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page2_by_100000.json')
pred3 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page3_by_100000.json')
pred4 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page4_by_100000.json')
pred5 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page5_by_100000.json')
pred6 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page6_by_100000.json')
pred7 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page7_by_100000.json')
pred8 = pd.read_json(r'C:\Users\Antoine\Coding Bootcamp\machine learning\Open Food Facts\robotoff_predictions\dump_page8_by_100000.json')

#---------------Concat predictions files---------------

#Save files into a list to concat them
pred_list = [pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8]

#Concat files into one df and normalize with json_normalize (for insights column)
raw_predictions = pd.concat(pred_list)
predictions = json_normalize(raw_predictions['insights'])

#Drop unrelevant columns
cols_pred_to_drop = ['id', 'type', 'timestamp', 'latent', 'process_after', 'value','source_image',
'automatic_processing', 'server_domain', 'server_type','unique_scans_n', 'predictor', 'robotoff_reserved_barcode']
predictions.drop(cols_pred_to_drop, axis=1, inplace=True)

#Add a prefix "robotoff" to the columns (so we can later identify predictions columns into the merged dataframe)
predictions.rename(columns={colname: 'robotoff_' + colname for colname in predictions.columns}, inplace=True)
predictions.rename(columns={'robotoff_barcode' : 'code'}, inplace=True)

#---------------Merging files---------------

#Merge predictions with full database
df_with_predictions = pd.merge(full_database, predictions, on=['code'], how='right')

#---------------Clean columns---------------

#Drop columns with more than 50% nans
cols = df_with_predictions.columns.to_list()
threshold = int(len(df_with_predictions) * 0.5)
cols_removed = [] #cols removed are saved into a list

#We keep columns with category info, even if almost empty
for col in cols :
    if (not 'catego' in col) and (df_with_predictions[col].count() < threshold) : 
        df_with_predictions.drop([col], axis=1, inplace=True)
        cols_removed.append(col)

#Drop some other columns considered unrelevant for the benchmark
cols_to_drop = ['last_modified_t', 'last_modified_datetime', 'image_url', 'image_small_url',
       'image_nutrition_url', 'image_nutrition_small_url', 'energy-kcal_100g',
       'energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g',
       'sugars_100g', 'proteins_100g', 'salt_100g', 'sodium_100g', 'created_t', 'states_en']
cols_removed.append(cols_to_drop)
df_with_predictions.drop(columns=cols_to_drop, inplace=True, errors='ignore')

#---------------Split dataframes---------------

#Create to dataframes : one with categories filled, one without
df_with_cats = df_with_predictions.dropna(subset=['categories', 'categories_tags', 'categories_en', 'main_category', 'main_category_en'])
df_without_cats = df_with_predictions.drop(df_with_cats.index)

#---------------Note---------------
#Predictions are for a category tag
#Some of the products have multiple tags in the OFF database
#Therefore it is difficult to compare them with predictions
#----------------------------------

#Create a col with the number of categories tags for every product
df_with_cats['nb_cats_tags'] = df_with_cats.apply(lambda x: len(x['categories_tags'].split(',')), axis=1)
#We split rows with multiple tags and create a new col "single tag"
single_category_tag = df_with_cats['categories_tags'].str.split(',').apply(pd.Series, 1).stack()
single_category_tag.index = single_category_tag.index.droplevel(-1)
single_category_tag.name = 'single_category_tag'
df_with_cats = df_with_cats.join(single_category_tag)

#---------------Export files to csv---------------

#Full merged dataframe
df_with_predictions.to_csv(r'C:\Users\Antoine\Coding Bootcamp\machine learning\
Open Food Facts\data\merged_predictions_off_full.csv', index=False, header=True)
#Only products with filled categories
df_with_cats.to_csv(r'C:\Users\Antoine\Coding Bootcamp\machine learning\
Open Food Facts\data\merged_predictions_off_with_categories.csv', index=False, header=True)
#Only products with categories columns empty
df_without_cats.to_csv(r'C:\Users\Antoine\Coding Bootcamp\machine learning\
Open Food Facts\data\merged_predictions_off_without_categories.csv', index=False, header=True)

