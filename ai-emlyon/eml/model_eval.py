from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

def classifier_metrics (y_test=None, y_preds=None, average='weighted', model=str, zero_division=0):
    """Return Accuracy, Recall, Precision and F-1 score. 
    Average can take two arguments : macro or weighted """

    acc = metrics.accuracy_score(y_test, y_preds)
    rec = metrics.recall_score(y_test, y_preds, average = average, zero_division=zero_division)
    prc = metrics.precision_score(y_test, y_preds, average = average, zero_division=zero_division)
    f1  = metrics.f1_score(y_test, y_preds, average = average, zero_division=zero_division)
    print (f"{model} Classification Metrics :")
    print ("-------------------")
    print('Accuracy : {:.2f}%'.format(acc*100))
    print('Recall : {:.2f}%'.format(rec*100))
    print('Precision : {:.2f}%'.format(prc*100))
    print('F1-score : {:.2f}%'.format(f1*100))
    print('\n')

def get_classification_report(y_test, y_pred, sortby='f1-score', model=str):
    """Return a classification report as pd.DataFrame"""
    report = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=[sortby], ascending=False)
    df_classification_report.rename(columns={colname: model + '_' + colname for colname in df_classification_report.columns}, inplace=True)
    return df_classification_report.round(2)

def plot_confusion_matrix(cm,
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          figsize=(10,10)):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()