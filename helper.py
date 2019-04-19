import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def distplot_w_stats(original_df):
    '''
    Put in a dataframe, 
    return a list of distplots with
    mean, median and mode.
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = original_df.copy().select_dtypes(include=numerics).dropna()
    
    number_of_plots = len(df.columns)
    number_of_columns = [2 if number_of_plots > 1 else 1][0]
    number_of_rows = (number_of_plots + 1) // 2
    
    f, axes = plt.subplots(number_of_rows,
                           number_of_columns,
                           figsize=(14,5*int(number_of_rows)))
    row = 0
    column = 0
    
    if number_of_plots > 2:
        for i, c in enumerate(df.columns):
            mean=df[c].mean()
            median=df[c].median()
            mode=df[c].mode().get_values()[0]

            sns.distplot(df[c], ax=axes[row,column])
            sns.despine()

            axes[row,column].axvline(mean, color='r', linestyle='--')
            axes[row,column].axvline(median, color='g', linestyle='-')
            axes[row,column].axvline(mode, color='b', linestyle='-')
            axes[row,column].legend({'Mean':mean,'Median':median,'Mode':mode})

            if column == 0:
                column += 1
            else:
                column = 0
                row +=1
    elif number_of_plots == 2:
         for i, c in enumerate(df.columns):
            mean=df[c].mean()
            median=df[c].median()
            mode=df[c].mode().get_values()[0]

            sns.distplot(df[c], ax=axes[column])
            sns.despine()

            axes[column].axvline(mean, color='r', linestyle='--')
            axes[column].axvline(median, color='g', linestyle='-')
            axes[column].axvline(mode, color='b', linestyle='-')
            axes[column].legend({'Mean':mean,'Median':median,'Mode':mode})

            if column == 0:
                column += 1
            else:
                column = 0
                row +=1
    else:
        for i, c in enumerate(df.columns):
            mean=df[c].mean()
            median=df[c].median()
            mode=df[c].mode().get_values()[0]

            sns.distplot(df[c], ax=axes)
            sns.despine()

            axes.axvline(mean, color='r', linestyle='--')
            axes.axvline(median, color='g', linestyle='-')
            axes.axvline(mode, color='b', linestyle='-')
            axes.legend({'Mean':mean,'Median':median,'Mode':mode})

            if column == 0:
                column += 1
            else:
                column = 0
                row +=1

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers