import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

def plot_correlations(df):
    corr_matrix = df.corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)]= True
    plt.figure(figsize=(25,15))
    heatmap = sns.heatmap(corr_matrix,vmin=-1, vmax=1, annot=True, cmap='RdBu', mask=mask)
    heatmap.set_title("Correlation Heatmap")