# Pandas

## Built-in functions

```python
df = pd.read_csv("./wages.csv")
df.head(2).T
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>earn</th>
      <td>79571.299011</td>
      <td>96396.988643</td>
    </tr>
    <tr>
      <th>height</th>
      <td>73.89</td>
      <td>66.23</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>male</td>
      <td>female</td>
    </tr>
    <tr>
      <th>race</th>
      <td>white</td>
      <td>white</td>
    </tr>
    <tr>
      <th>ed</th>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>age</th>
      <td>49</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe() # Numeric Type Data의 요약 정보 (통계) 를 보여줌
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
      <th>earn</th>
      <th>height</th>
      <th>ed</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1379.000000</td>
      <td>1379.000000</td>
      <td>1379.000000</td>
      <td>1379.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>32446.292622</td>
      <td>66.592640</td>
      <td>13.354605</td>
      <td>45.328499</td>
    </tr>
    <tr>
      <th>std</th>
      <td>31257.070006</td>
      <td>3.818108</td>
      <td>2.438741</td>
      <td>15.789715</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-98.580489</td>
      <td>57.340000</td>
      <td>3.000000</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10538.790721</td>
      <td>63.720000</td>
      <td>12.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26877.870178</td>
      <td>66.050000</td>
      <td>13.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>44506.215336</td>
      <td>69.315000</td>
      <td>15.000000</td>
      <td>55.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>317949.127955</td>
      <td>77.210000</td>
      <td>18.000000</td>
      <td>95.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
key = df.race.unique() # 해당 column에서 중복 제거한 value들의 list를 return
print(key)
value = range(len(df.race.unique()))
print(value)
df["race"].replace(to_replace=key, value=value) # 추출한 값을 이용하여 문자열이었던 데이터를 숫자 코드로 변경함
```

    ['white' 'other' 'hispanic' 'black']
    range(0, 4)





    0       0
    1       0
    2       0
    3       1
    4       0
           ..
    1374    0
    1375    0
    1376    0
    1377    0
    1378    0
    Name: race, Length: 1379, dtype: int64




```python
dict(enumerate(sorted(df["race"].unique())))
```




    {0: 'black', 1: 'hispanic', 2: 'other', 3: 'white'}




```python
value = list(map(int, np.array(list(enumerate(df["race"].unique())))[:, 0].tolist()))
key = np.array(list(enumerate(df["race"].unique())), dtype=str)[:, 1].tolist()

value, key
```




    ([0, 1, 2, 3], ['white', 'other', 'hispanic', 'black'])




```python
df["race"].replace(to_replace=key, value=value, inplace=True)
```


```python
df["race"]
```




    0       0
    1       0
    2       0
    3       1
    4       0
           ..
    1374    0
    1375    0
    1376    0
    1377    0
    1378    0
    Name: race, Length: 1379, dtype: int64




```python
value = list(map(int, np.array(list(enumerate(df["sex"].unique())))[:, 0].tolist()))
key = np.array(list(enumerate(df["sex"].unique())), dtype=str)[:, 1].tolist()

value, key
```




    ([0, 1], ['male', 'female'])




```python
df["sex"].replace(to_replace=key, value=value, inplace=True)
df.head(5)
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
      <th>earn</th>
      <th>height</th>
      <th>sex</th>
      <th>race</th>
      <th>ed</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79571.299011</td>
      <td>73.89</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96396.988643</td>
      <td>66.23</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48710.666947</td>
      <td>63.77</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82089.345498</td>
      <td>63.08</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>




```python
numueric_cols = ["earn", "height", "ed", "age"]
df[numueric_cols].sum(axis=1)
```




    0       79710.189011
    1       96541.218643
    2       48823.436947
    3       80652.316153
    4       82212.425498
                ...     
    1374    30290.060363
    1375    25018.829514
    1376    13823.311312
    1377    95563.664410
    1378     9686.681857
    Length: 1379, dtype: float64




```python
df.sum(axis=1) # column, row 연산, add, sub, mean, min, max, count, median, mad, var 등...
```




    0       79710.189011
    1       96542.218643
    2       48824.436947
    3       80654.316153
    4       82213.425498
                ...     
    1374    30290.060363
    1375    25019.829514
    1376    13824.311312
    1377    95563.664410
    1378     9686.681857
    Length: 1379, dtype: float64




```python
df.isnull().sum() / len(df) # NaN 값이 있는 곳의 index를 반환함. sum을 이용하여 빈 곳의 개수를 구할 수 있음
```




    earn      0.0
    height    0.0
    sex       0.0
    race      0.0
    ed        0.0
    age       0.0
    dtype: float64




```python
df.sort_values(["age", "earn"], ascending=True) # 주어진 column을 기준으로 sorting
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
      <th>earn</th>
      <th>height</th>
      <th>sex</th>
      <th>race</th>
      <th>ed</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1038</th>
      <td>-56.321979</td>
      <td>67.81</td>
      <td>0</td>
      <td>2</td>
      <td>10</td>
      <td>22</td>
    </tr>
    <tr>
      <th>800</th>
      <td>-27.876819</td>
      <td>72.29</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>22</td>
    </tr>
    <tr>
      <th>963</th>
      <td>-25.655260</td>
      <td>68.90</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>988.565070</td>
      <td>64.71</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>22</td>
    </tr>
    <tr>
      <th>801</th>
      <td>1000.221504</td>
      <td>64.09</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>22</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>993</th>
      <td>32809.632677</td>
      <td>59.61</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>92</td>
    </tr>
    <tr>
      <th>102</th>
      <td>39751.194030</td>
      <td>67.14</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>93</td>
    </tr>
    <tr>
      <th>331</th>
      <td>39169.750135</td>
      <td>64.79</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>95</td>
    </tr>
    <tr>
      <th>809</th>
      <td>42963.362005</td>
      <td>72.94</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
<p>1379 rows × 6 columns</p>
</div>




```python
df.sort_values("age", ascending=False).head(10) # 나이 기준 sorting
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
      <th>earn</th>
      <th>height</th>
      <th>sex</th>
      <th>race</th>
      <th>ed</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>95</td>
    </tr>
    <tr>
      <th>331</th>
      <td>39169.750135</td>
      <td>64.79</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>95</td>
    </tr>
    <tr>
      <th>809</th>
      <td>42963.362005</td>
      <td>72.94</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>95</td>
    </tr>
    <tr>
      <th>102</th>
      <td>39751.194030</td>
      <td>67.14</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>93</td>
    </tr>
    <tr>
      <th>993</th>
      <td>32809.632677</td>
      <td>59.61</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>92</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>8942.806716</td>
      <td>62.97</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>91</td>
    </tr>
    <tr>
      <th>1192</th>
      <td>39757.947210</td>
      <td>64.79</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>90</td>
    </tr>
    <tr>
      <th>952</th>
      <td>8162.682672</td>
      <td>58.09</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>89</td>
    </tr>
    <tr>
      <th>827</th>
      <td>55712.348432</td>
      <td>70.13</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>88</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>10861.092284</td>
      <td>64.03</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.age.corr(df.earn) # 상관계수를 구하는 함수
```




    0.07400349177836056




```python
df.age[(df.age < 45) & (df.age > 15)].corr(df.earn)
```




    0.31411788725189044




```python
df.age.cov(df.earn) # 공분산을 구하는 함수
```




    36523.69921040891




```python
df["sex_code"] = df["sex"].replace({"male": 1, "female": 0})
```


```python
df.corr()
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
      <th>earn</th>
      <th>height</th>
      <th>sex</th>
      <th>race</th>
      <th>ed</th>
      <th>age</th>
      <th>sex_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>earn</th>
      <td>1.000000</td>
      <td>0.291600</td>
      <td>-0.337328</td>
      <td>-0.063977</td>
      <td>0.350374</td>
      <td>0.074003</td>
      <td>-0.337328</td>
    </tr>
    <tr>
      <th>height</th>
      <td>0.291600</td>
      <td>1.000000</td>
      <td>-0.703672</td>
      <td>-0.045974</td>
      <td>0.114047</td>
      <td>-0.133727</td>
      <td>-0.703672</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>-0.337328</td>
      <td>-0.703672</td>
      <td>1.000000</td>
      <td>0.000858</td>
      <td>-0.061747</td>
      <td>0.070036</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>race</th>
      <td>-0.063977</td>
      <td>-0.045974</td>
      <td>0.000858</td>
      <td>1.000000</td>
      <td>-0.049487</td>
      <td>-0.056879</td>
      <td>0.000858</td>
    </tr>
    <tr>
      <th>ed</th>
      <td>0.350374</td>
      <td>0.114047</td>
      <td>-0.061747</td>
      <td>-0.049487</td>
      <td>1.000000</td>
      <td>-0.129802</td>
      <td>-0.061747</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.074003</td>
      <td>-0.133727</td>
      <td>0.070036</td>
      <td>-0.056879</td>
      <td>-0.129802</td>
      <td>1.000000</td>
      <td>0.070036</td>
    </tr>
    <tr>
      <th>sex_code</th>
      <td>-0.337328</td>
      <td>-0.703672</td>
      <td>1.000000</td>
      <td>0.000858</td>
      <td>-0.061747</td>
      <td>0.070036</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.corrwith(df.earn)
```




    earn        1.000000
    height      0.291600
    sex        -0.337328
    race       -0.063977
    ed          0.350374
    age         0.074003
    sex_code   -0.337328
    dtype: float64




```python
df.sex.value_counts(sort=True)
```




    1    859
    0    520
    Name: sex, dtype: int64


