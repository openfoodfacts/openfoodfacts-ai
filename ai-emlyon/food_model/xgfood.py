import pandas as pd
import numpy as np
import pickle
from ast import literal_eval
import re
from xgboost import XGBClassifier
import time

class XGFood():
    def __init__(self):
        self.model_G1 = XGBClassifier()
        self.model_G1.load_model(r'files\xgboost_group1.model')
        self.model_G2 = XGBClassifier()
        self.model_G2.load_model(r'files\xgboost_group2.model')
        self.le_G1 = None
        self.le_G2 = None
        self._unk_code_G1 = 9
        self._unk_code_G2 = 38
        self.predictions = None
        self._X_processed = False
        self._feat_cols_filled = False
        self.X = None

    def process_X(
        self, X_raw, ingredients_column='ingredients', text_column='product_name',verbose=True
        ):
        """
        """
        
        time_proc_start = time.time()
        X_empty = pd.read_csv(r'files\Empty_Features.csv')
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

    def process_ingredients(self):
        """
        """
        self.X_raw[self.ingredients_column] = self.X_raw[self.ingredients_column].apply(literal_eval)
        for ingre_list, index in zip(self.X_raw[self.ingredients_column], self.X_raw.index):
            for ingre_dict in ingre_list:
                val = ingre_dict['text']
                val_clean = val.replace('_','').replace('-','').strip('').lower()
                if val_clean in self.cols_for_ings:
                    try:
                        self.X_raw.loc[index,'ing_'+val_clean] = ingre_dict['percent_estimate']
                    except:
                        try:
                            self.X_raw.loc[index,'ing_'+val_clean] = ingre_dict['percent_min']
                        except:
                            self.X_raw.loc[index,'ing_'+val_clean] = 1
        
        self.X_raw.drop(columns=[self.ingredients_column], inplace=True)

    def process_names(self):
        """
        """
        for text, index in zip(self.X_raw[self.text_column], self.X_raw.index):
            row_text = text.lower()
            for word in self.cols_for_text:
                if word in row_text:
                    self.X_raw.loc[index,word] = 1
        
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

    def predict(self, X, decode_labels=True, pred_format='pd', get_confidence=True, preprocess=False):
        
        if preprocess: self.process_X(X)
        else: self.X = X
        
        tresholds_G1 = np.array([0.6, 0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5])
        y_probas_G1 = self.model_G1.predict_proba(self.X)
        y_preds_G1 = self.filter_preds(y_probas_G1, tresholds=tresholds_G1)
        y_conf_G1 = self.get_confidences(y_preds_G1, y_probas_G1, unk_code=9)

        X_G2 = np.append(self.X, y_preds_G1.reshape(-1,1), axis=1)
        tresholds_G2 = np.full(shape=38, fill_value=0.1, dtype=float)
        y_probas_G2 = self.model_G2.predict_proba(X_G2)
        y_preds_G2 = self.filter_preds(y_probas_G2, tresholds=tresholds_G2)
        y_conf_G2 = self.get_confidences(y_preds_G2, y_probas_G2, unk_code=38)
        
        if decode_labels:
            pkl_file_G1 = open(r'files\label_encoder_g1_tresh.pkl', 'rb')
            self.le_G1 = pickle.load(pkl_file_G1)
            pkl_file_G1.close()
            pkl_file_G2 = open(r'files\label_encoder_g2_tresh.pkl', 'rb')
            self.le_G2 = pickle.load(pkl_file_G2)
            pkl_file_G2.close()
            y_preds_G1 = self.le_G1.inverse_transform(y_preds_G1)
            y_preds_G2 = self.le_G2.inverse_transform(y_preds_G2)

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
