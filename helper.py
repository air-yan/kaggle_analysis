import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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