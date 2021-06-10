import json
import re
import time
from ast import literal_eval

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class XGFood():

    def __init__(self):
        self.model_G1 = XGBClassifier()
        self.model_G1.load_model(r'files\xgboost_G1_m2.model')
        self.model_G2 = XGBClassifier()
        self.model_G2.load_model(r'files\xgboost_G2_m1.model')
        self.le_G1 = None
        self.le_G2 = None
        self._unknown_code_G1 = 9
        self._unknown_code_G2 = 38
        self.predictions = None
        self._X_processed = False
        self._feat_cols_filled = False
        self.X = None

    def process_X(
        self, X_raw, ingredients_column='ingredients', text_column='product_name',verbose=True
        ):
        """
        Given a structured ingredients column and the column 'product_name',
        split information into 948 features (450 most frequent ingredients 
        and 488 most frequent words in product_name.)

        Parameters
        ----------
        X_raw: should be an array 2 cols by N samples where cols are ['ingredients', 'product_name']
        and each sample is a product.
        ingredients_column: name of ingredients feature given (default is 'ingredients')
        text_column: name of text column to search words (default is 'product_name')
        verbose: print running time of preprocessing X (default is True)

        Method
        ------
        1. Create an empty dataframe with the name of all features used to train XGBoost (size is 938*N samples)
        2. Split ingredients related features and text related features
        3. For each sample, append features related to ingredient 
        with percent estimate founded in ingredients col
        4. For each sample, append features related to product_name 
        with dummies (1 if the word is in product_name, 0 else) 

        Return
        ------
        Change self.X to Processed X and self._X_processed from False to True.
        """
        
        time_proc_start = time.time()
        with open(r'files\features.json') as json_file:
            features = json.load(json_file)
        X_empty = pd.DataFrame(columns=features['features_cols'])
        self.ingredients_column = ingredients_column
        self.text_column = text_column


        if not self._feat_cols_filled:
            self.features_cols = X_empty.columns.to_list()
            self.raw_cols_for_ings = [col for col in self.features_cols if 'ing_' in col]
            self.cols_for_ings = [re.sub('ing_', '', col) for col in self.raw_cols_for_ings]
            self.cols_for_text = [col for col in self.features_cols if 'ing_' not in col]
            self._feat_cols_filled = True
        
        self.X_raw = X_empty.append(X_raw).fillna(0)
        self.process_ingredients()
        self.process_names()
        self.X = self.X_raw
        self._X_processed = True

        time_proc_end = (time.time() - time_proc_start) / 60
        if verbose: print(f'Processing done - Total running time : {round(time_proc_end,2)}mn')
        return self.X

    def process_ingredients(self):
        #Convert object to original structured list of dictionnaries
        self.X_raw[self.ingredients_column] = self.X_raw[self.ingredients_column].apply(literal_eval)
        #Loop in the list of dicts
        for ingre_list, index in zip(self.X_raw[self.ingredients_column], self.X_raw.index):
            #Loop in the dicts
            for ingre_dict in ingre_list:
                #get text
                val = ingre_dict['text']
                #Clean text
                val_clean = val.replace('_','').replace('-','').strip('').lower()
                #Check if the ingredient is in features
                if val_clean in self.cols_for_ings:
                    #Try to append with percent_estimate
                    try:
                        self.X_raw.loc[index,'ing_'+val_clean] = ingre_dict['percent_estimate']
                    #If not working, try to append with percent_min    
                    except:
                        try:
                            self.X_raw.loc[index,'ing_'+val_clean] = ingre_dict['percent_min']
                        #If not working, append 1
                        except:
                            self.X_raw.loc[index,'ing_'+val_clean] = 1
        #Finally drop original ingredients col
        self.X_raw.drop(columns=[self.ingredients_column], inplace=True)

    def process_names(self):
        #Loop in text column
        for text, index in zip(self.X_raw[self.text_column], self.X_raw.index):
            row_text = text.lower()
            #Loop in words features selected
            for word in self.cols_for_text:
                #If the word is in text, append 1
                if word in row_text:
                    self.X_raw.loc[index,word] = 1
        #Finally drop original text col
        self.X_raw = self.X_raw.drop(columns=self.text_column)
    
    def filter_preds(self, y_probas, tresholds):
        """ 
        Filter pred with confidence treshold. 
        Fill unkeeped pred with 'y_unkown'.
        """
        preds = []
        y_unknown_code = len(tresholds)
        for p in y_probas:
            result = np.argwhere(p > tresholds)
            if result.size > 0 : preds.append(int(result[0]))
            else : preds.append(y_unknown_code)
        return np.array(preds)

    def get_confidences(self, y_pred, y_probas, unk_code=9):
        """ Get confidence -range (0,1)- for the label predicted"""
        probas = []
        for index, label in enumerate(y_pred):
            if label < unk_code : 
                probas.append(y_probas[index,label])
            else : probas.append(0.0)
        confidences = np.array(probas).reshape(-1,1)
        return confidences

    def decode_labels(self, preds, label_dict): 
        preds_decoded = []
        for pred in preds: preds_decoded.append(label_dict[str(pred)])
        return np.array(preds_decoded)

    def predict(self, X, decode_labels=True, pred_format='pd', get_confidence=True, preprocess=True):
        """
        Get predictions of G1 and G2 with XGBoost models, with optional confidences levels.
        
        Parameters
        ----------
        X (pd.DataFrame or np.array) : Features inputs. 
        Should have one structured column "ingredients" and a column "product_name"
        Can be already preprocessed (shape N samples, 938 features, set preprocess=True)
        decode_labels(Bool, default = True) : Return original labels (string) instead of encoded (int).
        pred_format(default = 'pd'): 
         'pd' : Return a dataframe with predictions for G1, G2 and confidences if get_confidence set to True
         'np' : return a numpy array with predictions and confidences if get_confidence set to True.
        get_confidence(Bool, default = True) : Get level of confidence for each prediction.
        preprocess(Bool, default=True) : preprocess X before fit the model.

        Return
        ------
        An DataFrame (or an array if pred_format = 'np') with 1 row per prediction:
            - y_pred_G1 : predictions for group 1 (9 labels + 'y_unknown')
            - y_pred_G2 : predictions for group 2 (38 labels + 'y_unknown')
            - if get confidences set to True:
                - y_conf_G1 : level of confidence for every G1 prediction
                - y_conf_G2 : level of confidence for every G2 prediction
        
        """

        if preprocess: self.process_X(X)
        else: self.X = X

        with open(r'files\tresholds_G2.json') as json_file:
            tresholds_G2 = json.load(json_file)
                 
        tresholds_G1 = np.array([0.6, 0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5])
        y_probas_G1 = self.model_G1.predict_proba(self.X)
        y_preds_G1 = self.filter_preds(y_probas_G1, tresholds=tresholds_G1)
        y_conf_G1 = self.get_confidences(y_preds_G1, y_probas_G1, unk_code=9)

        X_G2 = np.append(self.X, y_preds_G1.reshape(-1,1), axis=1)

        tresholds_G2 =np.array(list(tresholds_G2.values()))
        y_probas_G2 = self.model_G2.predict_proba(X_G2)
        y_preds_G2 = self.filter_preds(y_probas_G2, tresholds=tresholds_G2)
        y_conf_G2 = self.get_confidences(y_preds_G2, y_probas_G2, unk_code=38)
        
        if decode_labels:
            with open(r'files\labels_G1_code_reference.json') as json_file:
                self.le_G1 = json.load(json_file)
            with open(r'files\labels_G2_code_reference.json') as json_file:
                self.le_G2 = json.load(json_file)
            y_preds_G1 = self.decode_labels(y_preds_G1, self.le_G1)
            y_preds_G2 = self.decode_labels(y_preds_G2, self.le_G2)

        if pred_format == 'np':
            if get_confidence :
                predictions_G1 = np.append(y_preds_G1.reshape(-1,1), y_conf_G1, axis=1)
                predictions_G2 = np.append(y_preds_G2.reshape(-1,1), y_conf_G2, axis=1)
                predictions = np.append(predictions_G1, predictions_G2, axis=1)
            else:
                predictions = np.append(y_preds_G1.reshape(-1,1), y_preds_G2.reshape(-1,1), axis=1)
        
        elif pred_format == 'pd':
            if get_confidence:      
                predictions = pd.DataFrame({
                    'y_pred_G1':y_preds_G1,
                    'y_conf_G1':y_conf_G1.round(2).flatten(),
                    'y_pred_G2':y_preds_G2,
                    'y_conf_G2':y_conf_G2.round(2).flatten(),
                    })
            else:
                predictions = pd.DataFrame({
                    'y_pred_G1':y_preds_G1,
                    'y_pred_G2':y_preds_G2,
                    })
        
        self.predictions = predictions
        return predictions
