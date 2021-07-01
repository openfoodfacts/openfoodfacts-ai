import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import metrics
sns.set()

class Evaluator():
    
    def __init__(self):

        self.data = None
        self._data_ready = False
        self.class_report = None

    def __repr__(self):
        return 'Interpreter object'
        
    def build_data(
        self, y_true, y_pred, 
        y_confidence=None,
        pred_type=None, 
        decode_labels=False, label_encoder='from_existing'
        ):
        """
        Create the dataset used for other methods
        
        Arguments
        ---------
        - y_true = true labels
        - y_pred = predicted labels
        - y_confidence = if passed, the confidence level for preds in y_pred.
        - pred_type = Can be 'G1', 'G2' or other ('tags' for example).
            - 'G1' : PNNS_Groups_1, 9 possible labels. Used for label_decoder
            - 'G2' : PNNS_Groups_2, 38 possible labels. Used for label_decoder
        decode_labels = If true, use a label encoder to decode labels.
        label_encoder = The label encoder used for decode labels if decode_labels == True.
            - if label_encoder = 'from_existing' : load the label encoder for the pred type passed in pred_type.
        
        Return
        ------
        A dataframe with columns :
        - y_true, 
        - y_pred, 
        - pred_is_true : 1 if y_true == y_pred else 0
        - pred_confidence : y_confidence if y_confidence is not None
        """
        
        self.pred_type = pred_type
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_confidence = y_confidence
        self._decode_labels = decode_labels
        
        if self._decode_labels:
            if label_encoder == 'from_existing':
                if pred_type == 'G1':
                    pkl_file = open('label_encoder_g1.pkl', 'rb')
                    self.le = pickle.load(pkl_file)
                    pkl_file.close()
                elif pred_type == 'G2':
                    pkl_file = open('label_encoder_g2.pkl', 'rb')
                    self.le = pickle.load(pkl_file) 
                    pkl_file.close()
            elif label_encoder is not None:
                self.le = label_encoder

        df = pd.DataFrame({'y_true':self.y_true, 'y_pred':self.y_pred})
        df['pred_is_true'] = df.apply(
            lambda x: 1 if x['y_true'] == x['y_pred'] else 0, axis=1)

        if self.y_confidence is not None:
            df['pred_confidence'] = self.y_confidence

        if self._decode_labels:
            df['y_true'] = self.le.inverse_transform(df['y_true'])
            df['y_pred'] = self.le.inverse_transform(df['y_pred'].astype(int))
        
        self.data = df
        self._data_ready = True
        return self.data

    def classification_report(
        self, sortby='precision', name='model',
        save_report=False, report_path='classification_report.csv'
        ):
        """Return a classification report as pd.DataFrame"""
        if not self._data_ready:
            self.build_data()
            
        report = metrics.classification_report(
            self.data.y_true, self.data.y_pred, output_dict=True, zero_division=0
            )
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.sort_values(by=[sortby], ascending=False)
        df_report.rename(
            columns={colname: name + '_' + colname for colname in df_report.columns}, inplace=True
            )
        self.class_report = df_report.round(2)
        
        if save_report:
            df_report.to_csv(report_path, index=True, header=True)
        
        return self.class_report
    
    def global_metrics(
        self, average='weighted', name='Model', zero_div=0
        ):
        """Return Accuracy, Recall, Precision and F-1 score. 
        Average can take two arguments : macro or weighted """
        if not self._data_ready:
            self.build_data()
            self._data_ready = True

        acc = metrics.accuracy_score(self.data.y_true, self.data.y_pred)
        rec = metrics.recall_score(self.data.y_true, self.data.y_pred, average=average, zero_division=zero_div)
        prc = metrics.precision_score(
            self.data.y_true, self.data.y_pred, average=average, zero_division=zero_div
            )
        f1  = metrics.f1_score(self.data.y_true, self.data.y_pred, average=average, zero_division=zero_div)
        print (f"{name} Classification Metrics :")
        print ("-"*(len(name)+25))
        print('Accuracy : {:.2f}%'.format(acc*100))
        print('Recall : {:.2f}%'.format(rec*100))
        print('Precision : {:.2f}%'.format(prc*100))
        print('F1-score : {:.2f}%'.format(f1*100))
        print('\n')

    def plot_categories_scores(
        self, metric='precision', name='Model', figsize=(8,10),
        save_fig=False, fig_path="score_by_category.png"
        ):
        """Point plot with metric score by category, sorted by metric"""
        if not self._data_ready:
            self.build_data()

        report = self.classification_report(sortby=metric, name=name)
        plt.figure(figsize=figsize)
        sns.pointplot(y=report.index, x=report[f'{name}_{metric}'], palette='terrain')
        plt.title(f'{name} {self.pred_type} {metric} by category', fontsize=14)
        if save_fig:
            sns.savefig(fig_path) 

    def plot_confidence(
        self, name='Model', metric='precision', 
        col_wrap=5, save_fig=False, 
        fig_path="confidence_by_category.png"
        ):       
        """Print a KDE Plot with model confidence for every category"""
        
        if not self._data_ready:
            self.build_data()
            self._data_ready = True 

        report = self.classification_report(sortby=metric, name=name)
        labels = self.data.y_true.unique()
        hue_palette = {0:'darkred', 1:'darkgreen'}
        g = sns.FacetGrid(
            self.data, col='y_true', hue='pred_is_true', 
            col_wrap=col_wrap, height=5,palette=hue_palette, xlim=(0,1)
            )
        g.map(sns.kdeplot, 'pred_confidence', fill=True, common_norm=True, alpha=.4)
        g.add_legend()
        for ax, label in zip(g.axes.flat, labels):
            ax.set_title(f'{label} | {metric} : {report[f"{name}_{metric}"][label]}')
        
        if save_fig:
            sns.savefig(fig_path)
    
    def plot_confusion_matrix(
        self, name='Model', figsize=(20,15), annot=True, cmap='Greens',
        save_fig=False, fig_path="confidence_by_category.png"
        ):
        """Return a confusion matrix with seaborn heatmap design"""
        
        if not self._data_ready:
            self.build_data()
            self._data_ready = True 

        cm = metrics.confusion_matrix(self.data.y_true, self.data.y_pred)
        labels = sorted(set(self.data.y_true))
        plt.figure(figsize=figsize)
        plot = sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=annot, cmap=cmap, fmt='g')
        plot.set_title(f"{name} {self.pred_type} Confusion Matrix")
        
        if save_fig:
            sns.savefig(fig_path)