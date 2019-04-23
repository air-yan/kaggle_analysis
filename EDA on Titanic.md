
# Titanic Analysis Practice

The primary targets of this practice:
1. Take this excercise as a preparation for kaggle competitions
2. Hands-on experience to data analysis, cleaning and modeling
3. Review visual analytic skills (matplotlib and seaborn)
4. Use interactive ploting module: plotly (taking a Udemy course now)

Let's do it!


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# filter warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')

%matplotlib inline
sns.set_style('darkgrid')
```

## Define the problem

The problem is to predict the survivals in titanic.

## Gather the Data

As the data is already given in the competition, all we need to do is to read them.


```python
gender_submission = pd.read_csv('titanic/gender_submission.csv')
test_df = pd.read_csv('titanic/test.csv')
train_df = pd.read_csv('titanic/train.csv')
```

## Prepare Data


```python
print(train_df.info())
train_df.sample(3)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    None
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>681</th>
      <td>682</td>
      <td>1</td>
      <td>1</td>
      <td>Hassab, Mr. Hammad</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17572</td>
      <td>76.7292</td>
      <td>D49</td>
      <td>C</td>
    </tr>
    <tr>
      <th>321</th>
      <td>322</td>
      <td>0</td>
      <td>3</td>
      <td>Danoff, Mr. Yoto</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>349219</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>68</th>
      <td>69</td>
      <td>1</td>
      <td>3</td>
      <td>Andersson, Miss. Erna Alexandra</td>
      <td>female</td>
      <td>17.0</td>
      <td>4</td>
      <td>2</td>
      <td>3101281</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891</td>
      <td>891</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891</td>
      <td>891.000000</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>891</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>681</td>
      <td>NaN</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Stephenson, Mrs. Walter Bertram (Martha Eustis)</td>
      <td>male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>347082</td>
      <td>NaN</td>
      <td>B96 B98</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>577</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>4</td>
      <td>644</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>NaN</td>
      <td>32.204208</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>NaN</td>
      <td>49.693429</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>7.910400</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>14.454200</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>NaN</td>
      <td>512.329200</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Understanding the data:

Column Name | Data Type | Comment
-----|----|-----
Survived| binary nominal| it's the training label for our estimator
PassengerId| nominal|it's a random number which is not helpful in the analysis
Name| nominal| it shows information like gender and title
Sex | nominal| need to be converted
Embarked| nominal| the place people embarked to the ship.need to be converted
Ticket| nominal | seems not useful
Cabin| nominal| lot's of missing data
Pclass| ordinal|as stated in kaggle, it stands for ticket class
Age| continuous quantitative datatypes|
Fare| continuous quantitative datatypes|
SibSp| discrete quantitative datatypes|number of siblings and spouses
Parch| discrete quantitative datatypes|number of parents and children



## The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting

### Correcting errors
To see if there's any aberrant or non-acceptable data inputs.


```python
# first, we create a copy of the original training dataset
# as we are now preparing for EDA so we call it df_EDA
df_EDA = train_df.copy()

# The ID is just a random number
df_EDA.drop(['PassengerId'],axis=1,inplace=True)
```

### Completing missing information

Check for null values or missing data.


```python
# missing data
df_EDA.isnull().sum()[df_EDA.isnull().sum() != 0]
```




    Age         177
    Cabin       687
    Embarked      2
    dtype: int64



**Age**


```python
# stats info
print(df_EDA.Age.describe())

# shape of its distribution
f, axes = plt.subplots(1,2,figsize=(12,4))

sns.boxplot(df_EDA.Age.dropna(),ax=axes[0])
sns.distplot(df_EDA.Age.dropna(),ax=axes[1])
```

    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x13333811240>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_15_2.png)


From the data and the graphs above, we can know that its a slightly right-skewed normal distribution. Taking the median to fill all the null value should be fine, but we want more accuracy. First, let's see what it's like to fill with median. *Wei: expectation maximization.*




```python
# fill age with median, save df for comparison
test_df = df_EDA.Age.fillna(df_EDA.Age.median())
```

What if we look into the feature correlations and fill the median according to different situations?


```python
# convert sex into categories
df_EDA['Sex_Code'] = df_EDA['Sex'].astype('category').cat.codes

# show correlation heatmaps
df_EDA.corr().style.background_gradient(cmap='coolwarm')
```




<style  type="text/css" >
    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col0 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col1 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col2 {
            background-color:  #80a3fa;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col3 {
            background-color:  #7ea1fa;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col4 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col5 {
            background-color:  #e2dad5;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col0 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col1 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col2 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col3 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col4 {
            background-color:  #80a3fa;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col5 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col6 {
            background-color:  #ccd9ed;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col0 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col1 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col2 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col4 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col5 {
            background-color:  #c5d6f2;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col6 {
            background-color:  #c4d5f3;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col0 {
            background-color:  #a9c6fd;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col1 {
            background-color:  #c3d5f4;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col2 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col3 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col4 {
            background-color:  #e4d9d2;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col5 {
            background-color:  #d2dbe8;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col6 {
            background-color:  #97b8ff;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col0 {
            background-color:  #c1d4f4;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col1 {
            background-color:  #b5cdfa;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col2 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col3 {
            background-color:  #ead5c9;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col4 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col5 {
            background-color:  #dbdcde;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col6 {
            background-color:  #799cf8;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col0 {
            background-color:  #e1dad6;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col1 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col2 {
            background-color:  #abc8fd;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col3 {
            background-color:  #b2ccfb;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col4 {
            background-color:  #b6cefa;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col5 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col6 {
            background-color:  #86a9fc;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col0 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col1 {
            background-color:  #ccd9ed;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col2 {
            background-color:  #abc8fd;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col3 {
            background-color:  #688aef;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col4 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col5 {
            background-color:  #88abfd;
            color:  #000000;
        }    #T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col6 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }</style><table id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Survived</th>        <th class="col_heading level0 col1" >Pclass</th>        <th class="col_heading level0 col2" >Age</th>        <th class="col_heading level0 col3" >SibSp</th>        <th class="col_heading level0 col4" >Parch</th>        <th class="col_heading level0 col5" >Fare</th>        <th class="col_heading level0 col6" >Sex_Code</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5level0_row0" class="row_heading level0 row0" >Survived</th>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col0" class="data row0 col0" >1</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col1" class="data row0 col1" >-0.338481</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col2" class="data row0 col2" >-0.0772211</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col3" class="data row0 col3" >-0.0353225</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col4" class="data row0 col4" >0.0816294</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col5" class="data row0 col5" >0.257307</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row0_col6" class="data row0 col6" >-0.543351</td>
            </tr>
            <tr>
                        <th id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5level0_row1" class="row_heading level0 row1" >Pclass</th>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col0" class="data row1 col0" >-0.338481</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col1" class="data row1 col1" >1</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col2" class="data row1 col2" >-0.369226</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col3" class="data row1 col3" >0.0830814</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col4" class="data row1 col4" >0.0184427</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col5" class="data row1 col5" >-0.5495</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row1_col6" class="data row1 col6" >0.1319</td>
            </tr>
            <tr>
                        <th id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5level0_row2" class="row_heading level0 row2" >Age</th>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col0" class="data row2 col0" >-0.0772211</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col1" class="data row2 col1" >-0.369226</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col2" class="data row2 col2" >1</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col3" class="data row2 col3" >-0.308247</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col4" class="data row2 col4" >-0.189119</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col5" class="data row2 col5" >0.0960667</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row2_col6" class="data row2 col6" >0.0932536</td>
            </tr>
            <tr>
                        <th id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5level0_row3" class="row_heading level0 row3" >SibSp</th>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col0" class="data row3 col0" >-0.0353225</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col1" class="data row3 col1" >0.0830814</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col2" class="data row3 col2" >-0.308247</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col3" class="data row3 col3" >1</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col4" class="data row3 col4" >0.414838</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col5" class="data row3 col5" >0.159651</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row3_col6" class="data row3 col6" >-0.114631</td>
            </tr>
            <tr>
                        <th id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5level0_row4" class="row_heading level0 row4" >Parch</th>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col0" class="data row4 col0" >0.0816294</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col1" class="data row4 col1" >0.0184427</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col2" class="data row4 col2" >-0.189119</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col3" class="data row4 col3" >0.414838</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col4" class="data row4 col4" >1</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col5" class="data row4 col5" >0.216225</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row4_col6" class="data row4 col6" >-0.245489</td>
            </tr>
            <tr>
                        <th id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5level0_row5" class="row_heading level0 row5" >Fare</th>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col0" class="data row5 col0" >0.257307</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col1" class="data row5 col1" >-0.5495</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col2" class="data row5 col2" >0.0960667</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col3" class="data row5 col3" >0.159651</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col4" class="data row5 col4" >0.216225</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col5" class="data row5 col5" >1</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row5_col6" class="data row5 col6" >-0.182333</td>
            </tr>
            <tr>
                        <th id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5level0_row6" class="row_heading level0 row6" >Sex_Code</th>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col0" class="data row6 col0" >-0.543351</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col1" class="data row6 col1" >0.1319</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col2" class="data row6 col2" >0.0932536</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col3" class="data row6 col3" >-0.114631</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col4" class="data row6 col4" >-0.245489</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col5" class="data row6 col5" >-0.182333</td>
                        <td id="T_285e9f00_63da_11e9_b4b3_b06ebfbf33c5row6_col6" class="data row6 col6" >1</td>
            </tr>
    </tbody></table>



The most correlated columns for age is "Pclass", "SibSp" and "Parch". I'm going to plot and see if that's intuitively ture.


```python
sns.catplot(data=df_EDA,x='Pclass', y='Age', hue='Sex',
            row='SibSp', col='Parch',kind='box',
            row_order=[0,1,2],col_order=[0,1,2])
```




    <seaborn.axisgrid.FacetGrid at 0x13333c31160>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_21_1.png)


We are gonna calculate median age for different sex, pclass, sibch and parch and fill NA with them.


```python
# Fill NA with a complex method
for ind in df_EDA[df_EDA.Age.isnull()].index:
    # filters
    filter1 = df_EDA['Sex'] == df_EDA.loc[ind, 'Sex']
    filter2 = df_EDA['Pclass'] == df_EDA.loc[ind, 'Pclass']
    filter3 = df_EDA['SibSp'] == df_EDA.loc[ind, 'SibSp']
    filter4 = df_EDA['Parch'] == df_EDA.loc[ind, 'Parch']
    fill_value = df_EDA[filter1][filter2][filter3][filter4]['Age'].median()
    
    # if filter result is nan, we fill with the global median
    if pd.isna(fill_value):
        fill_value = df_EDA['Age'].median()
    
    # fill in values
    df_EDA.loc[ind, 'Age'] = fill_value
    
# stats info
print(df_EDA.Age.describe(),"\n")

print("The kurtosis of the complex method is: {}".format(df_EDA.Age.kurtosis()))
print("The kurtosis of the global median method is: {}".format(test_df.kurtosis()))

# shape of its distribution
f, axes = plt.subplots(2,2,figsize=(12,8))

sns.boxplot(df_EDA.Age.dropna(),ax=axes[0,0])
sns.distplot(df_EDA.Age.dropna(),ax=axes[0,1])
sns.boxplot(test_df,ax=axes[1,0],color='Orange')
sns.distplot(test_df,ax=axes[1,1],color='Orange')

axes[0,0].set_title('Box Plot')
axes[0,1].set_title('Dist Plot')
axes[0,0].set_ylabel('More complex method')
axes[1,0].set_ylabel('Fill with median')
```

    count    891.000000
    mean      29.387957
    std       13.467945
    min        0.420000
    25%       22.000000
    50%       27.000000
    75%       36.000000
    max       80.000000
    Name: Age, dtype: float64 
    
    The kurtosis of the complex method is: 0.5874999132251677
    The kurtosis of the global median method is: 0.9938710163801736
    




    Text(0, 0.5, 'Fill with median')




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_23_2.png)


It can be clearly seen that the complex method is much better. It create less outliers, reduce the kurtosis and makes more sense intuitively.

**Cabin**


```python
df_with_cabin = df_EDA[df_EDA.Cabin.notnull()]

# Let us see the survival rate
rate_of_survival_cabin = df_with_cabin.Survived.mean()
print("Survival Rate:","{:.0%}".format(rate_of_survival_cabin))

df_with_cabin['Cabin'] = df_with_cabin.Cabin.apply(lambda x: x[0])

sns.barplot(x='Cabin',y='Survived',data=df_with_cabin)
```

    Survival Rate: 67%
    




    <matplotlib.axes._subplots.AxesSubplot at 0x13334c4dac8>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_26_2.png)


A survival rate of 66% is really high comparing to the overall rate! Remember that we have the overall survival rate of 38%. 

We already know that ticket fare and pclass are the most important factor, so I guess that only higher class people get their Cabin data recorded in the system. To prove the hypothesis, we will plot two figures with one containing all the data and another one containing only the data with Cabin info. We will see the differences in the plots below.

A little practice on the seaborn plot here =)


```python
# style and background
# plt.style.use('seaborn-notebook')
sns.set_style("white")

# figure and sizes
f, axes = plt.subplots(1,2,figsize=(14,4))

# all data
a1 = sns.countplot('Pclass',hue='Survived',data=df_EDA,ax=axes[0],palette='muted')
# data with Cabin available
a2 = sns.countplot('Pclass',hue='Survived',data=df_with_cabin,ax=axes[1],palette='muted')

# title
a1.set_title('All')
a2.set_title('Cabin Available')

# annotation and text
a1.annotate('See how many death here?\nMany Class 3 records here.',
            (1.5,350),
            (0.0,220),
            arrowprops=dict(arrowstyle = 'simple'))
a2.text(.5,50,
        'Pclass 2 and 3 only show few records')

# style
sns.despine(left=True, bottom=False)
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_28_0.png)


Hypothesis proved. I can dig deeper into it but let's now focus on the topic. This column seems to have a high correlation with the fare and pclass and it has many null values. To use that we need more digging and efforts, so we drop that.


```python
df_EDA.drop(['Cabin'], axis=1, inplace=True)

# double check
df_EDA.columns
```




    Index(['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
           'Fare', 'Embarked', 'Sex_Code'],
          dtype='object')



**Embarked**

We got 2 missing value in the column Embarked. We fill it with the most common value in this column.


```python
print('# of NA in Embarked: ',df_EDA.Embarked.isnull().sum())

df_EDA.Embarked.value_counts()

df_EDA.Embarked.fillna(df_EDA['Embarked'].mode()[0],inplace=True)

print('Are all the NAs has been filled: ', df_EDA.Embarked.notnull().all())
```

    # of NA in Embarked:  2
    Are all the NAs has been filled:  True
    

### Creating

1. Generate title from names.
2. Generate family size and single status from Parch and SibSp.
3. Create bins for fare and age for visualizations


```python
# split the Name column twice to get the title
df_EDA['Title'] = df_EDA.Name.apply(lambda x: x.split(', ')[1].split('. ')[0])

g = sns.countplot(df_EDA['Title'])
g = plt.setp(g.get_xticklabels(), rotation=45) 
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_34_0.png)



```python
# Convert to categorical values Title 
df_EDA["Title"] = df_EDA["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_EDA["Title"] = df_EDA["Title"].map({"Master":"Master", "Miss":"Female", "Ms" : "Female" , "Mme":"Female", "Mlle":"Female", "Mrs":"Female", "Mr":"Male", "Rare":"Rare"})

# plot
g = sns.countplot(df_EDA["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
sns.despine()

# create a column of family size and based on it, create another
df_EDA['FamilySize'] = df_EDA['Parch'] + df_EDA['SibSp'] + 1
df_EDA['IsAlone'] = df_EDA['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

df_EDA.head()

# create bins
df_EDA['FareBin'] = pd.qcut(df_EDA['Fare'], 4)
df_EDA['AgeBin'] = pd.cut(df_EDA['Age'].astype(int), 5)
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_35_0.png)


### Converting


```python
# copy it
df_for_model = df_EDA.copy()

from sklearn.preprocessing import LabelEncoder

# transform the interval/category into code, for EDA purpose
# using pandas astype category because according to Dark417, this performs better in the model
df_EDA['FareBin_Code'] = df_EDA.FareBin.astype('category').cat.codes
df_EDA['AgeBin_Code'] = df_EDA.AgeBin.astype('category').cat.codes
df_EDA['Embarked_Code'] = df_EDA['Embarked'].astype('category').cat.codes
```

### Train Test Split


```python
# converting categorical data into one-hot columns
import patsy as pts

y, X = pts.dmatrices('Survived ~ Pclass + C(Sex) + Age + SibSp + Parch + Fare + ' +
                    'C(Embarked) + C(Title) + FamilySize + IsAlone', data=df_for_model,
                    return_type='dataframe')
pd.concat([X,y]).info()

from sklearn.model_selection import train_test_split

# train test split
# X_train, X_test, y_train, y_test = \
#     train_test_split(X, y, test_size=0.2, random_state=42)

# I choose to use kfold cv method, so the train test split above is not used anymore
# To keep the rest of the codes unchanged, I'll reassign the variables.

X_train = X
y_train = y
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1782 entries, 0 to 890
    Data columns (total 15 columns):
    Age                   891 non-null float64
    C(Embarked)[T.Q]      891 non-null float64
    C(Embarked)[T.S]      891 non-null float64
    C(Sex)[T.male]        891 non-null float64
    C(Title)[T.Male]      891 non-null float64
    C(Title)[T.Master]    891 non-null float64
    C(Title)[T.Rare]      891 non-null float64
    FamilySize            891 non-null float64
    Fare                  891 non-null float64
    Intercept             891 non-null float64
    IsAlone               891 non-null float64
    Parch                 891 non-null float64
    Pclass                891 non-null float64
    SibSp                 891 non-null float64
    Survived              891 non-null float64
    dtypes: float64(15)
    memory usage: 222.8 KB
    


```python
# Fare
sns.distplot(df_for_model.Fare)

print("Skewness: {}".format(df_for_model.Fare.skew()))
print("kurtosis: {}".format(df_for_model.Fare.kurtosis()))
```

    Skewness: 4.787316519674893
    kurtosis: 33.39814088089868
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_40_1.png)



```python
df_for_model['Fare'] = df_for_model.Fare.apply(lambda x: np.log(x) if x > 0 else 0)

sns.distplot(df_for_model.Fare)

print("Skewness: {}".format(df_for_model.Fare.skew()))
print("kurtosis: {}".format(df_for_model.Fare.kurtosis()))
```

    Skewness: 0.44310881405404506
    kurtosis: 0.641225603709215
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_41_1.png)


## Exploratory Data Analysis



Take a quick look of the pairplot and see what we can get.

As pairplot will not show categorical information, we are looking at numerical data.


```python
g = sns.pairplot(df_EDA, hue='Survived', plot_kws={'alpha':0.5,'s':13},
             vars=['Sex_Code','Pclass','Fare','Age',\
                   'FamilySize', 'IsAlone'])
# sns.catplot(kind='')

print(df_EDA.Survived.value_counts(normalize=True).apply(lambda x: '{:.0%}'.format(x)))
```

    0    62%
    1    38%
    Name: Survived, dtype: object
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_44_1.png)


Diagonal:
- Fare: a clear peak along the blue line can be seen which means the fatalities are mainly distributed in the low-fare area.
- Age: Children has a higher rate to survive. Mid-age people probably sacrificed themselves?
- Sex: Apparently, women are much more likely to survive than man
- Class: Higher class passenger has a lower rate of death
- Is Alone: stick with family to gain higher survival rate
- Family Size: Don't be alone and don't take too many family friends...

Upper/lower plots:
can't see any thing.

Lastly, remember that we have 38% total survival rate.


```python
# I wrote a helper code to display each numeric column as a distplot.
# This is just to understand the distribution of the data.

import helper
tem_df = df_EDA.drop(['Parch', 'SibSp', 'FareBin_Code', 'AgeBin_Code', 'Title'],axis=1)
helper.distplot_w_stats(tem_df)
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_46_0.png)



```python
tem_df = pd.DataFrame()
tem_df['x'] = pd.cut(df_EDA['Fare'],200).apply(lambda x: round(int(x.mid),0))
tem_df['hue'] = df_EDA['Sex']

# plot
pal = dict(male="#6495ED", female="#F08080")

plt.subplots(figsize=(14,6))
ax = sns.countplot(x='x',hue='hue',data=tem_df,palette=pal)
plt.legend(loc=1)
ax.set_xlim([-.5,30])
ax.set_ylim([1,150])

sns.despine()
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_47_0.png)


### Single and Multi Variable Analysis
#### Fare and Pclass


```python
sns.set_style('white')

a = sns.jointplot(df_EDA['Fare'], df_EDA['Pclass'],  kind="kde", xlim=(-50,150))

sns.despine()
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_49_0.png)



```python
pal = dict(male="#6495ED", female="#F08080")

fig, a = plt.subplots(2,3,figsize=(16,9))

ax0 = sns.distplot(df_EDA.Fare, ax=a[0,0])
ax1 = sns.boxplot(df_EDA.Fare, ax=a[0,1],palette="Oranges")
ax2 = sns.barplot(df_EDA['FareBin'].apply(lambda x:int(x.mid)), df_EDA['Survived'], palette='Oranges',ax=a[0,2], alpha=0.5)
sns.lineplot([x for x in range(-1,6)], 0.38, color='red',ax=a[0,2],alpha=.8)

ax0.set_title('Distribution')
ax1.set_title('Distribution')
ax2.set_title('Survival ratio by fare')
ax2.set_xlim(-1,4)
ax2.annotate('38% survival',
            (0,.38),
            (-.9,.8),
            arrowprops=dict(arrowstyle = 'simple'))

a1 = sns.distplot(df_EDA.Pclass,ax=a[1,0])
a2 = sns.countplot('Pclass', hue='Survived', data=df_EDA, palette="BrBG",ax=a[1,1])
a3 = sns.pointplot('Pclass', 'Survived', data=df_EDA, palette=pal,hue='Sex',ax=a[1,2])
sns.lineplot([x for x in range(-1,4)], 0.38, color='red',ax=a[1,2],alpha=.8)

a1.set_title('Distribution')
a2.set_title('Count')
a3.set_title('Survival Ratio')
a3.set_xlim(-1,3)

sns.despine()
print(df_EDA.Fare.describe())
```

    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_50_1.png)


**Fare Price**
1. Distribution Plot: 

Most people paid 7.91 ~ 31.00 for the ticket, while some paid a lot of money like 512.32... We can see there are some outliers in the box plot. From the stats data showed above, we know that the mean value is influenced by the outliers, and the std is quite high possibly for the same reason. 


2. Line and Bar Plot: 

In the graph, we have 4 groups of bins, and each of them has exactly the same number as each other because we cut the fare using Quantile-based discretization function.

As expected, we found that the more money paid for the ticket, the higher survival rate it will be. 

**Pclass**
1. Most people are in class 3
2. Only class 1 has the survival rate > 50%
3. Class 3 has a low survival rate

Expected. Same reason as fare price.

**Sex**
1. Women > men.
2. Even a first class man, his survival rate < 38%.

Gentle man. My greatest respect.

#### Four Dimensional Analysis

Variables: Age, Pclass, Survived, Sex


```python
pal = dict(male="#6495ED", female="#F08080")

# Show the survival proability as a function of age and sex
g = sns.lmplot(x="Age", y="Survived", row = 'Pclass', col="Sex", hue="Sex", data=df_EDA, aspect=2,
               palette=pal, y_jitter=.02, logistic=True)

g.set(xlim=(0, 80), ylim=(-.05, 1.05))
```




    <seaborn.axisgrid.FacetGrid at 0x133384b1400>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_53_1.png)


**Regression Plot**
- Most ladies in class one were saved, the survival rate them is extremely high
- Older ladies in class 2 have a chance to be abandoned
- Boys in Class 2 have a high rate of survival, maybe class 2 people gives their lives for the young ones.
- As the age increases, the rate of survival decreases (for most cases)
- Class 1 > class 2 > class 3, class 3 sucks
- From the left three charts, we can tell that there has a clear linear relationship between age and survived.


```python
g = sns.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex",
                   data=df_EDA, kind="violin", split=True,
                   bw=0.05, size=7, aspect=.9, s=7)
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_55_0.png)


**Violin plots**

This is another way to display a 4 dimensional chart. While the lmplots emphases on the relationships between x and y, violin plots focuses on showing the distribution of the survivors.

- Class 2 boys are all saved! 
- A peak in female class 3: it shows that most of the ladies on in this age.
- Some peaks can be misleading because of lack of data... e.g. female, pclass 1 and 2, not survived. However, if you know the dataset well enough, you will know there are a few females unfortunately die in this event from class 1 and 2 and these peaks demonstrate that.
- From the tail/head of the distribution, we can see that there are more old men than old women

#### Ticket

This is a tricky one and I don't wanna go too deep on that, but let's try it anyway. From the numbers, we can know the sequence of booking and something more, but I really don't think this will contribute much on the analysis. 

Instead, let's see what's the survival rate for different strings included in the ticket.


```python
# replacing all digit with empty strings, and show the result
def replace_digit(x):
    result = ""
    for i in x:
        if not i.isdigit():
            result = "".join([result,i])
    return result.strip()

# save the result in a variable
converted_ticket = train_df['Ticket'].apply(replace_digit)

# create a slicer including top 5 values of the above variable
slicer = train_df['Ticket'].apply(replace_digit).value_counts().index[:5]
```


```python
j = []
k = []

for i in slicer:
    print('Testing for {}'.format(i))
    print('Number of data: {}'.format(sum(converted_ticket == i)))
    j.append(i)
    try:
        survival_rate = train_df[converted_ticket == i].Survived.value_counts(normalize=True).loc[1]
        print('P(Survival) = {:.0%}'.format(survival_rate))
        k.append(survival_rate)
        
    except:
        print('The rate of survival is 0')
        k.append(0)
        
    print('-'*20)

f, axis = plt.subplots(figsize=(10,4))
a1 = sns.barplot(j,k,palette='GnBu_d')
sns.lineplot(range(-1,len(j) + 1),[.38 for j in range(len(j) + 2)],color='red')
a1.set_xlabel('String in Ticket')
a1.set_ylabel('Survival Rate')
a1.set_xlim(-1,5)

sns.despine()
```

    Testing for 
    Number of data: 661
    P(Survival) = 38%
    --------------------
    Testing for PC
    Number of data: 60
    P(Survival) = 65%
    --------------------
    Testing for C.A.
    Number of data: 27
    P(Survival) = 48%
    --------------------
    Testing for A/
    Number of data: 13
    P(Survival) = 8%
    --------------------
    Testing for STON/O .
    Number of data: 12
    P(Survival) = 42%
    --------------------
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_59_1.png)


As we already know that the rate of survival for the entire training set is 38%, the first result proves that again. The second result shows a **65% of survival rate**. Is that merely a coincidence? Does PC stands for premium class? We should see correlations between different columns later.

The rest of the results cannot prove anything due to lack of data.


```python
# we convert the ticket to another column
# will use it later
converted_ticket = converted_ticket.apply(lambda x: 1 if x == 'PC ' else 0)
```

#### Cabin


```python
Cabin_transformed = train_df.Cabin.apply(lambda x: replace_digit(str(x)))
Cabin_counts = train_df.Cabin.apply(lambda x: replace_digit(str(x))).value_counts()
Cabin_filter = Cabin_counts > 10

Cabin_transformed = Cabin_transformed.fillna('nan').apply(lambda x: x if Cabin_filter[x] else 'misc')
```


```python
a = sns.barplot(Cabin_transformed,train_df.Survived, palette='Blues')
sns.lineplot(range(-1,8),[.38 for x in range(9)],color='red')

a.set_xlim(-1,7)
sns.despine()
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_64_0.png)


Remember, we only have 200+ number of data in Cabin, which only takes 25% of the total.

Very interestingly, We can see the survival rates for Cabin with letters in them all exceed the overall fatality rate. The rest of records that have no Cabin data with them have a lower than overall survival rate - 30%.

#### Embarked


```python
print('*'*35)
print('Survival Rate wrt Embarked')
print(df_EDA.pivot_table(values='Survived',columns=['Embarked']))
print('*'*35)

print('*'*35)
print('Embarked Count')
print(df_EDA.Embarked.value_counts())
print('*'*35)
g = sns.FacetGrid(df_EDA, col = 'Embarked', height =4)
g.map(sns.pointplot,'FareBin', 'Survived', 'Sex', palette='muted')

sns.despine()
```

    ***********************************
    Survival Rate wrt Embarked
    Embarked         C        Q         S
    Survived  0.553571  0.38961  0.339009
    ***********************************
    ***********************************
    Embarked Count
    S    646
    C    168
    Q     77
    Name: Embarked, dtype: int64
    ***********************************
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_67_1.png)


- People embarked at C has a higher rate of survival. Maybe they have higher class ticket?
- Men's survival rate is higher than women in C.

### Correlations between data

- Chart 1: Feature Correlation w.r.t. Survival
- Chart 2: Correlations between all features


```python
corr_data = (df_EDA.corr()
 .drop(['Survived'])['Survived']
 .apply(abs)
 .sort_values(ascending=True)
)

(corr_data
 .apply(lambda x: '{:.2%}'.format(x))
 .to_frame('Correlation with Survival')
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation with Survival</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FamilySize</th>
      <td>1.66%</td>
    </tr>
    <tr>
      <th>AgeBin_Code</th>
      <td>3.49%</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>3.53%</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>6.52%</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>8.16%</td>
    </tr>
    <tr>
      <th>Embarked_Code</th>
      <td>16.77%</td>
    </tr>
    <tr>
      <th>IsAlone</th>
      <td>20.34%</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>25.73%</td>
    </tr>
    <tr>
      <th>FareBin_Code</th>
      <td>29.94%</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>33.85%</td>
    </tr>
    <tr>
      <th>Sex_Code</th>
      <td>54.34%</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(style = 'white')
plt.subplots(figsize=(12,4))

ax = sns.barplot(corr_data.values,
                 corr_data.index,
                 orient='h',
                 palette='RdBu')

ax.set_title('Featrue Correlation with Survival')

sns.despine()
```


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_71_0.png)


- Sex has the largest correlation value with survival. We've seen the difference in EDA. Expected.
- Ticket class and Fare follows. We have fare, fare bins and pclass, and all of them are social class indicators. We should the correlations between these three features.
- Is alone: this is a great column created from Parch + SibSp. Nice job.
- Embarked: we've know that people embarked at C have greater chance to survive.


```python
sns.set(style = 'darkgrid')

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(df_EDA.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(df_EDA.corr(), 
            center=0, 
            cmap = cmap,
            square=True,
            ax=ax,
            mask = mask,
            annot=True, 
            linewidths=0.1,
            linecolor='white',
            annot_kws={'fontsize':12},
            cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1333b07a898>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_73_1.png)


- Family size is high correlated with SibSp and Parch. So as IsAlone.
- Age and title is correlated... Interesting
- Fare and isalone is negatively correlated... 

## Model

### Model Selection


```python
X_train.columns
```




    Index(['Intercept', 'C(Sex)[T.male]', 'C(Embarked)[T.Q]', 'C(Embarked)[T.S]',
           'C(Title)[T.Male]', 'C(Title)[T.Master]', 'C(Title)[T.Rare]', 'Pclass',
           'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone'],
          dtype='object')




```python
# clean names
for i in [X_train]:
    i.columns = ['Intercept', 'C(Sex)T.male', 'C(Embarked)T.Q', 'C(Embarked)T.S',
           'C(Title)T.Male', 'C(Title)T.Master', 'C(Title)T.Rare',
           'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
           'FamilySize', 'IsAlone']
```


```python
from xgboost import XGBClassifier
from sklearn import ensemble, linear_model, svm, naive_bayes, discriminant_analysis, neighbors, tree
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import time



# models
classifers = {'Random Rorest': ensemble.RandomForestClassifier(), # Ensembling Method
              'GradientBoosting': ensemble.GradientBoostingClassifier(),
              'XGB': XGBClassifier(),
              # Linear model
              'LogisticRegression': linear_model.LogisticRegressionCV(),
              'SGD': linear_model.SGDClassifier(),
              'Perception': linear_model.Perceptron(), 
              # Naive bayes
              'BernoulliNB': naive_bayes.BernoulliNB(),
              'GaussianNB': naive_bayes.GaussianNB(),
              'SVM': svm.SVC(probability=True),
              # neighboors
              'KNN': neighbors.KNeighborsClassifier(),
              # Trees
              'Decision Tree': tree.DecisionTreeClassifier(),
              'Discriminant Analysis': discriminant_analysis.LinearDiscriminantAnalysis()}

# create a loop for models to fit and generate scores
classifier_name = []
classifier_score = []
classifier_time = []
classifier_std = []

X_train_norm = normalize(X_train)

kfold = StratifiedKFold(n_splits=10)

for C in classifers:
    start = time.time() # time start
    classifier = classifers[C] #initializing
    cv_scores = cross_val_score(classifier, X_train_norm, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4) #scoring
    score = cv_scores.mean()
    standard_deviation = cv_scores.std()
    end = time.time() # time end
    
    timespent = end - start
    
    classifier_name.append(C)
    classifier_score.append(score)
    classifier_std.append(standard_deviation)
    classifier_time.append(timespent)

number_of_C = len(classifers.keys())
# transform the lists to dataframes (this is not necessary)
temp = np.hstack((np.array(classifier_score).reshape(number_of_C,1),
                  np.array(classifier_std).reshape(number_of_C,1),
                  np.array(classifier_time).reshape(number_of_C,1)))



score_plot = pd.DataFrame(index=classifier_name,data=temp,columns=['Score','Std','Time']).sort_values('Score',ascending=True)

tem_df = score_plot.copy()

tem_df['Score'] = (tem_df
      .loc[:,['Score']]
      .apply(lambda x: "{:.2%}".format(x[0]),axis=1))
print(tem_df.sort_values('Score',ascending=False))

# Barplot for scores and times
sns.set(style = 'white')
f, axes = plt.subplots(2,1,figsize=(12,8))

ax1 = sns.barplot(score_plot['Score'], 
                  score_plot.index,palette='RdBu',
                  ax=axes[0])

ax1.set_xlim(.65,.85)

score_plot.sort_values('Time',ascending=False,inplace=True)
ax2 = sns.barplot(score_plot['Time'], 
                  score_plot.index,
                  palette='YlGn',
                  ax=axes[1])

sns.despine()
```

                            Score       Std      Time
    XGB                    82.72%  0.036745  0.856709
    GradientBoosting       81.93%  0.031963  0.421788
    LogisticRegression     80.93%  0.032503  0.736032
    Discriminant Analysis  80.60%  0.032006  0.046864
    Random Rorest          79.91%  0.036953  0.979242
    BernoulliNB            79.01%  0.037752  0.021942
    Decision Tree          76.43%  0.044866  0.032922
    KNN                    74.65%  0.034153  0.034907
    GaussianNB             73.76%  0.054647  0.022939
    SGD                    70.17%  0.076305  0.038896
    SVM                    68.82%  0.058704  0.346635
    Perception             54.10%  0.160845  0.024933
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_79_1.png)


**Accuracy**
All models were using the default method to learn, so the accuracy is not maximized.

Discriminant Analysis > Logistic Regression > GB > Bayes > XGB > Random Forest > KNN > Decision Tree

**How about normalization?**

We used a normalization process in the last cell. Now, we are going to model it without norming the data and see what we have.


```python
# models
classifers = {'Random Rorest': ensemble.RandomForestClassifier(), # Ensembling Method
              'GradientBoosting': ensemble.GradientBoostingClassifier(),
              'XGB': XGBClassifier(),
              # Linear model
              'LogisticRegression': linear_model.LogisticRegressionCV(),
              'SGD': linear_model.SGDClassifier(),
              'Perception': linear_model.Perceptron(), 
              # Naive bayes
              'BernoulliNB': naive_bayes.BernoulliNB(),
              'GaussianNB': naive_bayes.GaussianNB(),
              'SVM': svm.SVC(probability=True),
              # neighboors
              'KNN': neighbors.KNeighborsClassifier(),
              # Trees
              'Decision Tree': tree.DecisionTreeClassifier(),
              'Discriminant Analysis': discriminant_analysis.LinearDiscriminantAnalysis()}

# create a loop for models to fit and generate scores
classifier_name1 = []
classifier_score1 = []
classifier_time1 = []
classifier_std1 = []

for C in classifers:
    start = time.time() # time start
    classifier = classifers[C] #initializing
    cv_scores = cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4) #scoring
    score = cv_scores.mean()
    standard_deviation = cv_scores.std()
    end = time.time() # time end
    
    timespent = end - start
    
    classifier_name1.append(C)
    classifier_score1.append(score)
    classifier_time1.append(timespent)
    classifier_std1.append(standard_deviation)
    

number_of_C = len(classifers.keys())
# transform the lists to dataframes (this is not necessary)
temp1 = np.hstack((np.array(classifier_score1).reshape(number_of_C,1),
                   np.array(classifier_std1).reshape(number_of_C,1),
                   np.array(classifier_time1).reshape(number_of_C,1)))

score_diff = (pd.DataFrame(index=classifier_name1,data= temp - temp1 ,columns=['Score','Std','Time'])
              .sort_values('Score',ascending=True))
score_plot = (pd.DataFrame(index=classifier_name1,data= temp1 ,columns=['Score','Std','Time'])
              .sort_values('Score',ascending=True))

tem_df = score_plot.copy()

tem_df['Score'] = (tem_df
      .loc[:,['Score']]
      .apply(lambda x: "{:.2%}".format(x[0]),axis=1))
print(tem_df.sort_values('Score',ascending=False))

# Barplot for scores and times
sns.set_style('whitegrid')
f, axes = plt.subplots(2,1,figsize=(12,8))

ax1 = sns.barplot(score_diff['Score'], 
                  score_diff.index,palette='RdBu',
                  ax=axes[0]
                  )
ax1.set_title('Normed score - Not normed score')

ax2 = sns.barplot(score_plot['Score'], 
                  score_plot.index,palette='RdBu',
                  ax=axes[1])
ax2.set_xlim(.65,.85)

sns.despine(left=True,bottom=True)
```

                            Score       Std      Time
    Discriminant Analysis  83.28%  0.032208  0.032912
    GradientBoosting       82.94%  0.035958  0.308062
    XGB                    82.72%  0.040501  0.402932
    LogisticRegression     81.93%  0.024556  1.202883
    GaussianNB             80.14%  0.033692  0.029920
    Random Rorest          79.92%  0.050365  0.074357
    BernoulliNB            79.01%  0.037752  0.028922
    Decision Tree          77.91%  0.050469  0.031914
    SVM                    74.65%  0.035100  0.501659
    KNN                    72.97%  0.030658  0.045878
    SGD                    66.20%  0.086785  0.030917
    Perception             60.20%  0.138871  0.034908
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_81_1.png)


I am so confused. Why modeling on normalized dataset returns a lower score? But look at that SGD model, normalization increase its accuracy by over 10%! KNN got a 5% boost in the normed data. Interesting.

**Support from Wei:** *SVM, Random forest and XGB might not be sensitive to feature normalize. That's because they don't have the process of "weights x features". SGD is based on some random selections, so it is not stable especially in this small dataset. As to why the result turns out that way, there might be two answers:*
1. *The model is not a good selection for this problem*
2. *Errors in the training process*

### Tuning

In this section, I will only tune features for the algorithms that perform great in the last section, which are:

- Discriminant Analysis
- Logistic Regression
- Gradient Boosting
- Bernoulli Naive Bayes
- Extreme Gradient Boosting
- Random Forest


```python
# Naive Bayes Parameters tunning 
NBC = naive_bayes.BernoulliNB()


## Search grid for optimal parameters
nb_param_grid = {
                    'alpha': range(1,100,1),
                    'fit_prior': [True,False]}


gsNBC = GridSearchCV(NBC,param_grid = nb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsNBC.fit(X_train,y_train)

# this is the best classifier
NBC_best = gsNBC.best_estimator_

# Best score
print(gsNBC.best_score_)
print(gsNBC.best_params_)
```

    Fitting 10 folds for each of 198 candidates, totalling 1980 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done 1096 tasks      | elapsed:    1.5s
    

    0.792368125701459
    {'alpha': 74, 'fit_prior': True}
    

    [Parallel(n_jobs=4)]: Done 1980 out of 1980 | elapsed:    2.6s finished
    


```python
# RFC Parameters tunning 
RFC = ensemble.RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,y_train)

# this is the best classifier
RFC_best = gsRFC.best_estimator_

# Best score
print(gsRFC.best_score_)
print(gsRFC.best_params_)
```

    Fitting 10 folds for each of 54 candidates, totalling 540 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  76 tasks      | elapsed:    4.8s
    [Parallel(n_jobs=4)]: Done 376 tasks      | elapsed:   23.7s
    [Parallel(n_jobs=4)]: Done 540 out of 540 | elapsed:   38.4s finished
    




    0.8383838383838383




```python
# XGB Parameters tunning, reference
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
xgb = XGBClassifier()


## Search grid for optimal parameters
xgb_param_grid = {
                'max_depth':[4,6],
                'min_child_weight':[4,6],
                'gamma':[i/10.0 for i in range(0,5,2)],
                'subsample':[i/10.0 for i in range(6,10,2)],
                'colsample_bytree':[i/10.0 for i in range(6,10,2)],
                'reg_alpha':[1e-2, 0.1, 1],
}


gsXGBC = GridSearchCV(xgb,param_grid = xgb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsXGBC.fit(X_train,y_train)

# this is the best classifier
XGBC_best = gsXGBC.best_estimator_

# Best score
print(gsXGBC.best_score_)
print(gsXGBC.best_params_)
```

    Fitting 10 folds for each of 144 candidates, totalling 1440 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  76 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=4)]: Done 376 tasks      | elapsed:   12.5s
    [Parallel(n_jobs=4)]: Done 876 tasks      | elapsed:   29.6s
    [Parallel(n_jobs=4)]: Done 1440 out of 1440 | elapsed:   50.3s finished
    




    0.8439955106621774




```python
# Discriminant Analysis


# DA Parameters tunning 
DAC = discriminant_analysis.LinearDiscriminantAnalysis()


## Search grid for optimal parameters
dac_param_grid = {
                'solver':['svd','lsqr'],
                'n_components':[i for i in range(0,12,1)],
}


gsDAC = GridSearchCV(DAC,param_grid = dac_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsDAC.fit(X_train,y_train)

# this is the best classifier
DAC_best = gsDAC.best_estimator_

# Best score
print(gsDAC.best_score_)
print(gsDAC.best_params_)
```

    Fitting 10 folds for each of 24 candidates, totalling 240 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    

    0.8327721661054994
    {'n_components': 0, 'solver': 'svd'}
    

    [Parallel(n_jobs=4)]: Done 240 out of 240 | elapsed:    0.4s finished
    


```python
# Logistic Regression


# LRC tunning 
LRC = linear_model.LogisticRegression()


## Search grid for optimal parameters
lrc_param_grid = {
    'C': [0.001,0.01,0.1,1,10,100,1000],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#     "penalty":["l1","l2"]

}


gsLRC = GridSearchCV(LRC,param_grid = lrc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsLRC.fit(X_train,y_train)

# this is the best classifier
LRC_best = gsLRC.best_estimator_

# Best score
print(gsLRC.best_score_)
print(gsLRC.best_params_)
```

    Fitting 10 folds for each of 35 candidates, totalling 350 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    

    0.8260381593714927
    {'C': 10, 'solver': 'newton-cg'}
    

    [Parallel(n_jobs=4)]: Done 350 out of 350 | elapsed:    4.5s finished
    


```python
# Gradient Boosting, reference:
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

# GB tunning 
GBC = ensemble.gradient_boosting.GradientBoostingClassifier()


## Search grid for optimal parameters
gbc_param_grid = {
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4,5,6],
              'min_samples_leaf': [31],
              'n_estimators':[30,70],
              'max_features':range(4,14,4),
              'subsample':[0.6,0.75,0.9]
              }

gsGBC = GridSearchCV(GBC ,param_grid = gbc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,y_train)

# this is the best classifier
GBC_best = gsGBC.best_estimator_

# Best score
print(gsGBC.best_score_)
print(gsGBC.best_params_)
```

    Fitting 10 folds for each of 162 candidates, totalling 1620 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done 348 tasks      | elapsed:    4.4s
    

    0.8439955106621774
    {'learning_rate': 0.05, 'max_depth': 6, 'max_features': 8, 'min_samples_leaf': 31, 'n_estimators': 70, 'subsample': 0.75}
    

    [Parallel(n_jobs=4)]: Done 1620 out of 1620 | elapsed:   22.0s finished
    

### Feature Importance, Precision and Recall


```python
# I decide to use the simple train test split in feature importance.
# That's easier for me to plot the figures.

X_train1, X_test1, y_train1, y_test1 = \
    train_test_split(X, y, test_size=0.2, random_state=42)

for i in [X_train1, X_test1]:
    i.columns = ['Intercept', 'C(Sex)T.male', 'C(Embarked)T.Q', 'C(Embarked)T.S',
           'C(Title)T.Male', 'C(Title)T.Master', 'C(Title)T.Rare',
           'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
           'FamilySize', 'IsAlone']
```


```python
from sklearn.metrics import confusion_matrix, classification_report

def plot_FI(clf, model, model_name):
    '''
    clf for classifier
    model:
    1. linear
    2. tree
    '''
    # model
    clf = clf
    clf.fit(X_train1, y_train1)
    
    # preparing feature importance score
    empty_df = X_train1.copy().drop(X_train1.index)
    
    if model == 'tree':
        empty_df.loc[0,] = clf.feature_importances_
    elif model == 'linear':
        empty_df.loc[0,] = abs(clf.coef_)
        total = empty_df.loc[0,].sum()
        empty_df = empty_df.apply(lambda x:x/total)

    plot_df = empty_df.transpose().sort_values(0,ascending=False)
    plot_df.columns = [model_name]
    
    # ploting feature importance
    plt.subplots(figsize=(14,4))
    sns.barplot(plot_df[model_name],plot_df.index)

    pred = clf.predict(X_test1)
    print("The overall score is: {:.2%}\n".format(clf.score(X_test1,y_test1)))
    print(classification_report(y_test1,pred))
    
    
    # Precision and recall
    def inverse_of_oneandzero(x):
        if x == 1:
            return 0
        elif x == 0:
            return 1
        else:
            return np.nan
        
    result = pred.reshape(pred.shape[0],1) == y_test1
    tp_1 = result[y_test1 == 1]
    fn_1 = result[y_test1 == 1]['Survived'].apply(inverse_of_oneandzero).to_frame()
    tn_1 = result[y_test1 == 0]
    fp_1 = result[y_test1 == 0]['Survived'].apply(inverse_of_oneandzero).to_frame()

    tp_0 = result[y_test1 == 0]
    fn_0 = result[y_test1 == 0]['Survived'].apply(inverse_of_oneandzero).to_frame()
    tn_0 = result[y_test1 == 1]
    fp_0 = result[y_test1 == 1]['Survived'].apply(inverse_of_oneandzero).to_frame()
    
    precision_1 = tp_1.sum() / (tp_1.sum() + fp_1.sum())
    precision_0 = tp_0.sum() / (tp_0.sum() + fp_0.sum())
    recall_1 = tp_1.sum() / (tp_1.sum() + fn_1.sum())
    recall_0 = tp_0.sum() / (tp_0.sum() + fn_0.sum())
    
    return plot_df, [precision_0[0], precision_1[0], recall_0[0], recall_1[0]]
```


```python
# XGB
XGB_df, XGB_pr = plot_FI(XGBC_best,'tree','XGB')
```

    The overall score is: 82.12%
    
                  precision    recall  f1-score   support
    
             0.0       0.84      0.86      0.85       105
             1.0       0.79      0.77      0.78        74
    
       micro avg       0.82      0.82      0.82       179
       macro avg       0.82      0.81      0.81       179
    weighted avg       0.82      0.82      0.82       179
    
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_93_1.png)


Seeing the plot, I don't think this algorithm performs well because it shows high importance on Sex and very low importance for others.


```python
# gradient boosting
GB_df, GB_pr = plot_FI(GBC_best,'tree','GradientBoosting')
```

    The overall score is: 81.01%
    
                  precision    recall  f1-score   support
    
             0.0       0.82      0.87      0.84       105
             1.0       0.79      0.73      0.76        74
    
       micro avg       0.81      0.81      0.81       179
       macro avg       0.81      0.80      0.80       179
    weighted avg       0.81      0.81      0.81       179
    
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_95_1.png)



```python
# Random Forest
RF_df, RF_pr = plot_FI(RFC_best,'tree','Random Forest')
```

    The overall score is: 82.68%
    
                  precision    recall  f1-score   support
    
             0.0       0.84      0.88      0.86       105
             1.0       0.81      0.76      0.78        74
    
       micro avg       0.83      0.83      0.83       179
       macro avg       0.82      0.82      0.82       179
    weighted avg       0.83      0.83      0.83       179
    
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_96_1.png)


I like this! It finally takes Fare and Age into consideration! However, the result is lower than the above. Terrible prediction on those survived. This algorithm sacrifices recall score on survived to get a high overall result. That's not cool.

Maybe, it is all about sex? WTF!


```python
# logistic regression
logistic_df, logistic_pr = plot_FI(LRC_best,'linear', 'Logistic regression')
```

    The overall score is: 82.68%
    
                  precision    recall  f1-score   support
    
             0.0       0.84      0.87      0.85       105
             1.0       0.80      0.77      0.79        74
    
       micro avg       0.83      0.83      0.83       179
       macro avg       0.82      0.82      0.82       179
    weighted avg       0.83      0.83      0.83       179
    
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_98_1.png)


Finally, we have something good!
- We need to dig deepter into the Misc type of the title! See the high coefficient~ That's the key.
- Sex is still important. We need to drop some columns like "Miss" and "Mr" because we already have male and female.
- Intercept? Do we even need to have that?
- Age... Why age is not considered at all? And the fare... Interesting.


```python
# Naive Bayes
NB_df, NB_pr = plot_FI(NBC_best,'linear', 'Naive Bayes')
```

    The overall score is: 80.45%
    
                  precision    recall  f1-score   support
    
             0.0       0.84      0.82      0.83       105
             1.0       0.75      0.78      0.77        74
    
       micro avg       0.80      0.80      0.80       179
       macro avg       0.80      0.80      0.80       179
    weighted avg       0.81      0.80      0.80       179
    
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_100_1.png)



```python
# Discriminative Analysis
DA_df, DA_pr = plot_FI(DAC_best,'linear', 'Discriminative Analysis')
```

    The overall score is: 82.68%
    
                  precision    recall  f1-score   support
    
             0.0       0.84      0.87      0.85       105
             1.0       0.80      0.77      0.79        74
    
       micro avg       0.83      0.83      0.83       179
       macro avg       0.82      0.82      0.82       179
    weighted avg       0.83      0.83      0.83       179
    
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_101_1.png)


**Feature Importance**

Very interestingly, fare and age, the most important features, was considered irrelevant to NB's model. However, it did come up with a generally great score.


```python
from IPython.display import HTML

feature_importance_df = pd.concat([XGB_df, GB_df, RF_df, logistic_df, NB_df, DA_df],axis=1)

for col in feature_importance_df.columns:
    feature_importance_df[col] = feature_importance_df[col].apply(lambda x: round(x,4))

feature_importance_df['Total'] = feature_importance_df.apply(lambda x: round(sum(x),4),axis=1)
feature_importance_df.sort_values('Total',ascending=False,inplace=True)
feature_importance_df.drop('Total',axis=1).style.background_gradient()
```




<style  type="text/css" >
    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col0 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col1 {
            background-color:  #045382;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col2 {
            background-color:  #034a74;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col3 {
            background-color:  #046097;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col4 {
            background-color:  #6ba5cd;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col5 {
            background-color:  #046097;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col0 {
            background-color:  #2f8bbe;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col1 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col2 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col3 {
            background-color:  #93b5d6;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col4 {
            background-color:  #3991c1;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col5 {
            background-color:  #93b5d6;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col2 {
            background-color:  #faf3f9;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col3 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col4 {
            background-color:  #045280;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col5 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col0 {
            background-color:  #c2cbe2;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col1 {
            background-color:  #76aad0;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col2 {
            background-color:  #9fbad9;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col3 {
            background-color:  #adc1dd;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col4 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col5 {
            background-color:  #adc1dd;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col2 {
            background-color:  #fcf4fa;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col3 {
            background-color:  #afc1dd;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col4 {
            background-color:  #023858;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col5 {
            background-color:  #afc1dd;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col0 {
            background-color:  #f0eaf4;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col1 {
            background-color:  #81aed2;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col2 {
            background-color:  #9ebad9;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col3 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col4 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col5 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col0 {
            background-color:  #e6e2ef;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col1 {
            background-color:  #e8e4f0;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col2 {
            background-color:  #dbdaeb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col3 {
            background-color:  #e9e5f1;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col4 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col5 {
            background-color:  #e9e5f1;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col0 {
            background-color:  #e7e3f0;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col1 {
            background-color:  #f2ecf5;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col2 {
            background-color:  #f2ecf5;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col3 {
            background-color:  #ede8f3;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col4 {
            background-color:  #83afd3;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col5 {
            background-color:  #ede8f3;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col0 {
            background-color:  #f4eef6;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col2 {
            background-color:  #fef6fa;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col3 {
            background-color:  #e8e4f0;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col4 {
            background-color:  #045a8d;
            color:  #f1f1f1;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col5 {
            background-color:  #e8e4f0;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col0 {
            background-color:  #ebe6f2;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col1 {
            background-color:  #fdf5fa;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col2 {
            background-color:  #faf3f9;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col3 {
            background-color:  #d9d8ea;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col4 {
            background-color:  #adc1dd;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col5 {
            background-color:  #d9d8ea;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col0 {
            background-color:  #ece7f2;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col1 {
            background-color:  #f6eff7;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col2 {
            background-color:  #f8f1f8;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col3 {
            background-color:  #e0deed;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col4 {
            background-color:  #d9d8ea;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col5 {
            background-color:  #e0deed;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col0 {
            background-color:  #f0eaf4;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col1 {
            background-color:  #c4cbe3;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col2 {
            background-color:  #dcdaeb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col3 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col4 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col5 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col0 {
            background-color:  #f4edf6;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col1 {
            background-color:  #fcf4fa;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col2 {
            background-color:  #fbf3f9;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col3 {
            background-color:  #fcf4fa;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col4 {
            background-color:  #65a3cb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col5 {
            background-color:  #fcf4fa;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col0 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col1 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col2 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col3 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col4 {
            background-color:  #fff7fb;
            color:  #000000;
        }    #T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col5 {
            background-color:  #fff7fb;
            color:  #000000;
        }</style><table id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >XGB</th>        <th class="col_heading level0 col1" >GradientBoosting</th>        <th class="col_heading level0 col2" >Random Forest</th>        <th class="col_heading level0 col3" >Logistic regression</th>        <th class="col_heading level0 col4" >Naive Bayes</th>        <th class="col_heading level0 col5" >Discriminative Analysis</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row0" class="row_heading level0 row0" >C(Sex)T.male</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col0" class="data row0 col0" >0.3513</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col1" class="data row0 col1" >0.2591</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col2" class="data row0 col2" >0.2751</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col3" class="data row0 col3" >0.2219</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col4" class="data row0 col4" >0.0888</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row0_col5" class="data row0 col5" >0.2219</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row1" class="row_heading level0 row1" >C(Title)T.Male</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col0" class="data row1 col0" >0.2256</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col1" class="data row1 col1" >0.2875</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col2" class="data row1 col2" >0.2954</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col3" class="data row1 col3" >0.1125</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col4" class="data row1 col4" >0.1026</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row1_col5" class="data row1 col5" >0.1125</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row2" class="row_heading level0 row2" >C(Title)T.Master</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col0" class="data row2 col0" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col1" class="data row2 col1" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col2" class="data row2 col2" >0.01</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col3" class="data row2 col3" >0.2644</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col4" class="data row2 col4" >0.1412</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row2_col5" class="data row2 col5" >0.2644</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row3" class="row_heading level0 row3" >Pclass</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col0" class="data row3 col0" >0.1026</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col1" class="data row3 col1" >0.142</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col2" class="data row3 col2" >0.1162</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col3" class="data row3 col3" >0.0933</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col4" class="data row3 col4" >0.0183</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row3_col5" class="data row3 col5" >0.0933</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row4" class="row_heading level0 row4" >C(Title)T.Rare</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col0" class="data row4 col0" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col1" class="data row4 col1" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col2" class="data row4 col2" >0.0069</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col3" class="data row4 col3" >0.092</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col4" class="data row4 col4" >0.1542</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row4_col5" class="data row4 col5" >0.092</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row5" class="row_heading level0 row5" >Fare</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col0" class="data row5 col0" >0.0349</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col1" class="data row5 col1" >0.1343</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col2" class="data row5 col2" >0.1172</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col3" class="data row5 col3" >0.0003</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col4" class="data row5 col4" >0.0186</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row5_col5" class="data row5 col5" >0.0003</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row6" class="row_heading level0 row6" >FamilySize</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col0" class="data row6 col0" >0.0546</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col1" class="data row6 col1" >0.0408</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col2" class="data row6 col2" >0.0599</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col3" class="data row6 col3" >0.0369</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col4" class="data row6 col4" >0.0183</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row6_col5" class="data row6 col5" >0.0369</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row7" class="row_heading level0 row7" >SibSp</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col0" class="data row7 col0" >0.053</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col1" class="data row7 col1" >0.0251</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col2" class="data row7 col2" >0.025</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col3" class="data row7 col3" >0.0311</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col4" class="data row7 col4" >0.081</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row7_col5" class="data row7 col5" >0.0311</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row8" class="row_heading level0 row8" >C(Embarked)T.Q</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col0" class="data row8 col0" >0.0254</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col1" class="data row8 col1" >0.0001</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col2" class="data row8 col2" >0.0031</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col3" class="data row8 col3" >0.0377</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col4" class="data row8 col4" >0.1372</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row8_col5" class="data row8 col5" >0.0377</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row9" class="row_heading level0 row9" >IsAlone</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col0" class="data row9 col0" >0.0464</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col1" class="data row9 col1" >0.0041</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col2" class="data row9 col2" >0.0095</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col3" class="data row9 col3" >0.0555</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col4" class="data row9 col4" >0.0662</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row9_col5" class="data row9 col5" >0.0555</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row10" class="row_heading level0 row10" >C(Embarked)T.S</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col0" class="data row10 col0" >0.0446</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col1" class="data row10 col1" >0.0179</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col2" class="data row10 col2" >0.0139</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col3" class="data row10 col3" >0.0475</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col4" class="data row10 col4" >0.0469</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row10_col5" class="data row10 col5" >0.0475</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row11" class="row_heading level0 row11" >Age</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col0" class="data row11 col0" >0.0344</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col1" class="data row11 col1" >0.0828</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col2" class="data row11 col2" >0.0585</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col3" class="data row11 col3" >0.001</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col4" class="data row11 col4" >0.0183</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row11_col5" class="data row11 col5" >0.001</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row12" class="row_heading level0 row12" >Parch</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col0" class="data row12 col0" >0.0272</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col1" class="data row12 col1" >0.0063</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col2" class="data row12 col2" >0.0092</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col3" class="data row12 col3" >0.0058</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col4" class="data row12 col4" >0.09</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row12_col5" class="data row12 col5" >0.0058</td>
            </tr>
            <tr>
                        <th id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5level0_row13" class="row_heading level0 row13" >Intercept</th>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col0" class="data row13 col0" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col1" class="data row13 col1" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col2" class="data row13 col2" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col3" class="data row13 col3" >0</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col4" class="data row13 col4" >0.0183</td>
                        <td id="T_9779880c_63e3_11e9_a330_b06ebfbf33c5row13_col5" class="data row13 col5" >0</td>
            </tr>
    </tbody></table>




```python
feature_importance_df.drop('Total',axis=1).plot(kind='bar', figsize=(18,6),width=.8)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x133480a6240>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_104_1.png)


**Result Correlations**


```python
pred_corr = pd.DataFrame({'XGB': XGBC_best.predict(X_test1),
'GBC': GBC_best.predict(X_test1),
'RFC': RFC_best.predict(X_test1),
'NBC': NBC_best.predict(X_test1),
'DAC': DAC_best.predict(X_test1),
'LRC': LRC_best.predict(X_test1)}).corr()

sns.heatmap(pred_corr,
            cmap='coolwarm',
            annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1334f27cbe0>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_106_1.png)



```python
pr_dic = {'XGB_pr': XGB_pr,
          'GB_pr': GB_pr,
          'RF_pr': RF_pr,
          'logistic_pr': logistic_pr,
          'NB_pr': NB_pr,
          'DA_pr': DA_pr}

precision_recall_df = pd.DataFrame(columns=['precision_0','precision_1','recall_0','recall_1'])

for key in pr_dic:
    precision_recall_df.loc[key,:] = pr_dic[key]
```

**Precision and Recall**


```python
precision_recall_df.sort_values('recall_1',ascending=False,inplace=True)
print(precision_recall_df)
precision_recall_df.transpose().plot(kind='bar',ylim=[0.65,0.92],figsize=(15,4),width=.6)
```

                precision_0 precision_1  recall_0  recall_1
    NB_pr          0.843137    0.753247  0.819048  0.783784
    XGB_pr         0.841121    0.791667  0.857143   0.77027
    logistic_pr    0.842593    0.802817  0.866667   0.77027
    DA_pr          0.842593    0.802817  0.866667   0.77027
    RF_pr          0.836364    0.811594   0.87619  0.756757
    GB_pr           0.81982    0.794118  0.866667   0.72973
    




    <matplotlib.axes._subplots.AxesSubplot at 0x13348a834e0>




![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_109_2.png)


**SUMMARY**

Different focuses among models
- Before tuning
    - XGB: XGB is sexist! In its algorithm, sex takes almost 70% of the weight when computing the result.
    - GB and RF: They really love continous data like Fare and Age.
    - LR and NB: A wide interests across the data, especially in categorical or discrete data. What happens to Fare and Age??? haha
- After tuning
    - They tend to focus more and more on the same features like sex and class.

Features
- T1: Sex, fare, age, social class
- T2: Embarked, Parch, SibSp,
- T9: IsAlone... No model needs this!

Models

- The three models (XGB, NBC, DAC) show the largest differences among models. Seeing the feature importance we can know that the three models are focusing on different features to predict the result.

Precision and Recall

- The recall score for survived data is the lowest, that's what we should focus on. Logistic regression has achieved the best score on it.

### Ensemble Voting


```python
from sklearn.ensemble import VotingClassifier

classifiers = { 'XGB': XGBC_best,
                    'GBC': GBC_best,
                    'RFC': RFC_best,
                    'NBC': NBC_best,
                    'DAC': DAC_best,
                    'LRC': LRC_best}

voting_clf = VotingClassifier(estimators=[('XGB', XGBC_best),
                    ('GBC', GBC_best),
                    ('RFC', RFC_best),
                    ('NBC', NBC_best),
                    ('DAC', DAC_best),
                    ('LRC', LRC_best)], voting='soft')

classifiers['Voting'] = voting_clf
```


```python
final_score_df = pd.DataFrame(columns=['Score'])
dc_of_scores = {}

for n_loop in range(30):
    X_train2, X_test2, y_train2, y_test2 = \
        train_test_split(X, y, test_size=0.20)
    
    for clf in classifiers:
        classifiers[clf].fit(X_train2,y_train2)
        sc = classifiers[clf].score(X_test2,y_test2)
        
        if clf not in dc_of_scores.keys():
            dc_of_scores[clf] = [sc]
        elif clf in dc_of_scores.keys():
            dc_of_scores[clf].append(sc)
        else:
            print('what?')
```


```python
sc_df = (pd.DataFrame(dc_of_scores)
 .apply(np.mean)
 .to_frame(name='Average Score')
 .sort_values('Average Score',ascending=False))

print(sc_df)
g = sns.barplot(x=sc_df.index,y=sc_df['Average Score'])
g.set_ylim(bottom=.82,top=.84)
sns.despine()
```

            Average Score
    RFC          0.834451
    Voting       0.834451
    DAC          0.832216
    GBC          0.831471
    XGB          0.831099
    LRC          0.827188
    NBC          0.789199
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_114_1.png)



```python
sc_df = (pd.DataFrame(dc_of_scores)
 .apply(np.std)
 .to_frame(name='std')
 .sort_values('std',ascending=False))

print(sc_df)
g = sns.barplot(x=sc_df.index,y=sc_df['std'])
sns.despine()
```

                 std
    NBC     0.031895
    Voting  0.028649
    LRC     0.027819
    DAC     0.027463
    RFC     0.027273
    GBC     0.025482
    XGB     0.025425
    


![png](EDA%20on%20Titanic_files/EDA%20on%20Titanic_115_1.png)

