
Titatic Data Analytics  - what factor made people more likely to survive the shinking of the Titanic? What ever the amount passerger paid is right or wrong? what is the chances to survive for worker. 

LOAD DATA TO DATAFRAME


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

original_df = pd.read_csv('train.csv')

print ('# of passengers in original data: ',str(len(original_df.index)))

original_df.head()

```

    # of passengers in original data:  891
    




<div>

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
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



DATA WRANGLING

Data Wrangling : Befor analysis, it is importnet to ensure there are no incorrect/missing data that could bias our result. It is called sample correction.


```python
original_df.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



Conclusion1: We Need to remove any passengers who did not have an age[177]/embarking station[2].But we'll not remove the passenger who did not have any cabin,beacuse 1 passenger booked the cabin for multiple passenger. 

Remove passengers who did not have an age


```python
age_wrang_df = original_df[pd.notnull(original_df['Age'])]
print('No. of passenger with age: ',str(len(age_wrang_df.index)))
```

    No. of passenger with age:  714
    

Remove passengers who did not have an embarking station


```python
embark_wrang_df = age_wrang_df[pd.notnull(age_wrang_df['Embarked'])]
print('No. of passenger with Embarked: ',str(len(embark_wrang_df.index)))
```

    No. of passenger with Embarked:  712
    

START ANALYTICS:

Find the Total Survival Rate


```python
print('Total Survival Rate: ',str(round(embark_wrang_df['Survived'].mean(),3)))
```

    Total Survival Rate:  0.404
    

Effect of Gender


```python
gen_data = embark_wrang_df.groupby('Sex',as_index=False)
gen_data_mean=gen_data.mean()
gen_data_mean
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
      <th>Sex</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>440.154440</td>
      <td>0.752896</td>
      <td>2.073359</td>
      <td>27.745174</td>
      <td>0.644788</td>
      <td>0.714286</td>
      <td>47.332433</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>453.412804</td>
      <td>0.205298</td>
      <td>2.335541</td>
      <td>30.726645</td>
      <td>0.439294</td>
      <td>0.271523</td>
      <td>27.268836</td>
    </tr>
  </tbody>
</table>
</div>



HYPOTHESIS

Hypothesis 1: Female survive more [75%] if we compare with male[20%]. 

Hypothesis 2: Average age of survival female and male respectively 27 and 30.

Hypothesis 3: A lot of female passegers has been servived [71%] whoes have the children.

Hypothesis 4: Greater number of female survived whoes have sibling/spouse [64% vs 43%]

Hypothesis 5: Survived Female fair much higher as compared to men [47 vs 27]

START HYPOTHESIS TESTING

Start to find Total no. of male and female pessenger


```python
total_df= gen_data['PassengerId'].count()
total_df
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
      <th>Sex</th>
      <th>PassengerId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>259</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>453</td>
    </tr>
  </tbody>
</table>
</div>



Rename pessenger id to Total


```python
total_df.columns = ['Sex','Total']
total_df
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
      <th>Sex</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>259</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>453</td>
    </tr>
  </tbody>
</table>
</div>



Save 'Sex' column in list for future plot


```python
gen_list = total_df['Sex']
del total_df['Sex']
total_df
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
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>259</td>
    </tr>
    <tr>
      <th>1</th>
      <td>453</td>
    </tr>
  </tbody>
</table>
</div>



Find number of male and female that survived 


```python
gen_surv_df = gen_data['Survived'].sum()
gen_surv_df
del gen_surv_df['Sex']
gen_surv_df
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
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>195</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div>



Create a vector by combining the two dataset as survived and total.


```python
comb_df = total_df.add(gen_surv_df,fill_value=0)
comb_df
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
      <th>Survived</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>195.0</td>
      <td>259.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93.0</td>
      <td>453.0</td>
    </tr>
  </tbody>
</table>
</div>



Conclution of Hypothesis Testing 1:It Appear, on average women more than 3 times likely survived than men. 


```python
comb_df.plot.bar()

```




    <matplotlib.axes._subplots.AxesSubplot at 0x212f0b0a518>




![png](output_34_1.png)


What is the effect of age on survival rate?
what is the effect of company on suvival rate?
what is the effect of socio-economic status on survival rate?


```python
suv_data = embark_wrang_df.groupby('Survived', as_index=False)
suv_mean_data = suv_data.mean()
suv_mean_data
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
      <th>Survived</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>442.299528</td>
      <td>2.485849</td>
      <td>30.626179</td>
      <td>0.525943</td>
      <td>0.365566</td>
      <td>22.965456</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>457.850694</td>
      <td>1.878472</td>
      <td>28.193299</td>
      <td>0.496528</td>
      <td>0.531250</td>
      <td>51.647672</td>
    </tr>
  </tbody>
</table>
</div>



Survivors are younger (28 vs 30) are from higher socio-economic class(Fare [51 vs 22]) with less sibling/spouse [.49 vs .52] and travel with parents/children [.53 vs .36] 
