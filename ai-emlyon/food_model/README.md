# Models & Evaluation Documentation

## <a name="xgfood"> </a> XGFood

You can use the final model with the class XGFood ([code here](food_model/xgfood.py)).

To try the model, the dataset with 167 548 labelled (`y_true`) and unprocessed products (`X_raw`) 
can be found [on this Google drive](https://drive.google.com/drive/folders/1yg8kT_MpYA2rKGDwiEJ2chs_UNnfq1IC).

Models files `.model` can be downloaded [here](https://drive.google.com/drive/folders/1LyuYedAdya_lGW5fowTtKw8hr1FZhmRh?usp=sharing).

Finally, you can download the small folder [with json files](food_model/xgfood.py).

>**Put the `.model` files in the files folder, and the files folder in the same folder that `xgfood.py` before run the model. 
>Otherwise, you will not be able to load every dependent files.

After loaded the dataset, you can load models by instantiating the XGFood class : 

```python
import xgfood.XGFood
model = XGFood()
```

And then use the predict method : 

```python
model.predict(X_raw)
```
By default, the output is a pandas `DataFrame` with 4 columns :

| y_pred_G1	 | y_conf_G1	| y_pred_G2  | y_conf_G2 |
|-----|-----|-----|-----|
| milk and dairy products | 0.90 | milk and yogurt | 0.93 |
| sugary snacks | 0.99 | sweets | 0.98 |
| composite foods | 0.67 | one dish meals | 0.98 |
| ... | ... | ... | ... |
| fruits and vegetables | 1.00 | fruits | 0.97 |

Where :

- `y_pred_G1` : predictions for group 1 (9 labels + 'y_unknown') 
- `y_pred_G2` : predictions for group 2 (38 labels + 'y_unknown') 
- if `get_confidences` set to `True` (default=`True`): 
    - `y_conf_G1` : level of confidence for G1 prediction
    - `y_conf_G2` : level of confidence for G2 prediction

You can change and get a *numpy `array`* format by passing `pred_format = 'np'` in the predict method :

```python
model.predict(X_raw, pred_format='np')
```

If you want, you can ask for only the predictions (without confidence levels) by setting `get_confidence = False` :

```python
model.predict(X_raw, get_confidence = False)
```

You can also set `preprocessed = False` if you want to preprocess X before with the `.process_X` method :

```python
X = model.process_X(X_raw)

predictions = model.predict(X, preprocess=False)
```

### Details on model class architecture

>This is the structure of `xgfood.py`

Given a X_raw, predictions are obtained with the following processus :

<img src="images/class_arch_2.png" width=850 >

 1. Creation of a null `X` matrix *938_features, n_samples*, where *938* corresponds to the variables and N the number of products to be labeled
 2. Navigation in the `ingredients` column provided to fill the *450* variables relating to the ingredients with estimates on the percentage present of the ingredient for each product
 3. Navigation in the `product_name` column provided to fill the *488* variables relating to the text with 1 if the word is present.
 4. Send data to model 1 for Group 1 prediction
 5. Save predictions and probabilities
 6. Filter with thresholds defined by category (updated by 'y_unknown' if the confidence level is below the threshold)
 7. Update `X` with Group 1 prediction
 8. Sending updated data in model 2 for Group 2 prediction (38 unique labels)
 9. Save predictions and probabilities
 10. Filtering with thresholds defined by category (updated by 'y_unknown' if the confidence level is below the thresholds)
 11. Decode predicted labels
 12. Add predictions + confidence levels in a dataframe
 13. Output : array numpy or DataFrame pandas `[pred_G1, conf_G1, pred_G2, conf_G2],n_predictions`


## <a name="eval"> </a> Evaluator

The Evaluator class is dedicated to evaluate quickly model performances with basic but formated seaborn plot. We used mainly seaborn and matplotlib.
The class is composed of a method `build_data` dedicated to build the dataset used and then different simple plot & metrics methods.
Simple plots showed before in this documents are some examples.

You can access the code here
After downloaded `evaluator.py`, you can instantiate the class and build the dataset with your inputs :

```python
evaluation = Evaluator()

evaluation.build_data(y_true=y_true, y_pred=y_pred, y_confidence=y_conf)
```

Note that y_confidence is optional and default `None` (but required for the `.plot_confidence` method) :

```python
evaluation = Evaluator()

evaluation.build_data(y_true=y_true, y_pred=y_pred)
```

> List of available methods :

<img src="images/eval_class.png" width=650 >


### Global Classification Metrics
```python
#Default arguments
Evaluator.global_metrics(average='weighted', name='Model')
```

Return Accuracy, Recall, Precision and F-1 score. Average can take `macro` or `weighted`

By default, the output start with a header `Model Classification Metrics` : 

```python
Evaluator.global_metrics()
```
```
Model Classification Metrics :
------------------------------
Accuracy : 85.10%
Recall : 85.10%
Precision : 95.23%
F1-score : 89.77%
```

You can change this with the desired text if you want to keep tracability in your output of what you are evaluating :

```python
Evaluator.global_metrics(name='XGBoost G1 Validation Set')
```
```
XGBoost G1 Validation Set Classification Metrics :
--------------------------------------------------
Accuracy : 85.10%
Recall : 85.10%
Precision : 95.23%
F1-score : 89.77%
```

### Structured Classification Report
```python
#Default arguments
Evaluator.classification_report(
sortby='precision', name='model', save_report=False, report_path='classification_report.csv'
)
```
Return a standard `sklearn.metrics.classification_report()` but in `pandas.DataFrame` format. 

Report can be sorted by precision, recall, f1-score or support (`default='precision'`) :
```python
Evaluator.classification_report(sortby='support')
```

Report can also be saved in csv format by setting `save_report=True`. 
Default path is current folder and file name `'classification_report.csv'` but you can change it in `report_path=*your_path*` :
```python
Evaluator.classification_report(save_report=True, report_path='files/classification_report.csv')
```

Here too you can change the name, it will effect on columns name :
```python
Evaluator.classification_report(name='XGBoost G1')
```
|                        | XGBoost G1_precision   | XGBoost G1_recall | XGBoost G1_f1-score | XGBoost G1_support |
|------------------------|------------------------|-------------------|---------------------|--------------------|
|fat and sauces          | 0.98 | 0.90 | 0.94 | 63.00   |
|milk and dairy products | 0.98 | 0.89 | 0.93 | 131.00  |
|fish meat eggs          | 0.97 | 0.88 | 0.92 | 135.00  |
|beverages               | 0.97 | 0.77 | 0.86 | 108.00  |
|salty snacks            | 0.96 | 0.83 | 0.89 | 58.00   |
|sugary snacks           | 0.96 | 0.94 | 0.95 | 214.00  |
|weighted avg            | 0.95 | 0.85 | 0.90 | 1000.00 |
|composite foods         | 0.93 | 0.83 | 0.88 | 101.00  |
|cereals and potatoes    | 0.92 | 0.74 | 0.82 | 111.00  |
|fruits and vegetables   | 0.90 | 0.76 | 0.82 | 79.00   |
|macro avg               | 0.86 | 0.75 | 0.80	| 1000.00 |
|accuracy	               | 0.85 | 0.85 | 0.85	| 0.85    |
|y_unknown               | 0.00 | 0.00 | 0.00 | 0.00    |


### Point plot of sorted categories scores 
```python
#Default arguments
Evaluator.plot_categories_scores(
metric='precision', name='Model', figsize=(8, 10), save_fig=False, fig_path='score_by_category.png'
)
```
Return a point plot with desired metric per category, sorted by metric. Default metric is precision.

Here too you can change the name (it will effect on the title), save the figure. You can also change the figsize :

```python
Evaluator.plot_categories_scores(name=r'XGBoost$G_2$, Valid Set', figsize=(8, 10))
```

<img src="images/xgbg2_valid_prec.png" width=650 >

### Confusion Matrix with seaborn heatmap design
```python
#Default arguments
Evaluator.plot_confusion_matrix(
name='Model', figsize=(20, 15), annot=True, cmap='Greens', save_fig=False, fig_path='confidence_by_category.png'
)
```

Return a confusion matrix with annotations, seaborn heatmap design.

```python
Evaluator.plot_confusion_matrix()
```
<img src="images/conf_mat_G1_tresh.png" width=1000 >

You can also remove annotations by passing `annot=False`.

### Probabilities distribution per category and performance
```python
#Default arguments
Evaluator.plot_confidence(
name='Model', metric='precision', col_wrap=5, save_fig=False, fig_path='confidence_by_category.png'
)
```
Return a KDE plot for every category, with confidence (probabilities) distribution (hue=pred_is_true, with green if pred was true, red else).
The desired metric for is also present on the title of each category plot (default precision).

You can save the figure, and you can change the number of plots per line/row with `col_wrap` (default is `col_wrap=5`) :

```python
Evaluator.plot_confidence(metric='precision', col_wrap=5)
```
<img src="images/xgb_G1_valid_conf.png">
