# Pandas

## map, apply

### map & replace

```python
s1 = Series(np.arange(10))
s1
```




    0    0
    1    1
    2    2
    3    3
    4    4
    5    5
    6    6
    7    7
    8    8
    9    9
    dtype: int64



* ```map()``` with function


```python
# f = lambda x: x**2
def f(x):
    return x + 5

s1.map(f) # Series의 모든 element에 f()를 적용함
```




    0     5
    1     6
    2     7
    3     8
    4     9
    5    10
    6    11
    7    12
    8    13
    9    14
    dtype: int64



* ```map()``` with dict


```python
z = {1: "A", 2: "B", 3: "C"}
s1.map(z) # dict type으로 데이터를 교체함, 없는 값은 NaN
```




    0    NaN
    1      A
    2      B
    3      C
    4    NaN
    5    NaN
    6    NaN
    7    NaN
    8    NaN
    9    NaN
    dtype: object



* ```map()``` with Series


```python
s2 = Series(np.arange(10, 30))
s1.map(s2) # 같은 위치의 데이터로 전환
```




    0    10
    1    11
    2    12
    3    13
    4    14
    5    15
    6    16
    7    17
    8    18
    9    19
    dtype: int64




```python
df = pd.read_csv("./wages.csv")
df.head()
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
      <td>male</td>
      <td>white</td>
      <td>16</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96396.988643</td>
      <td>66.23</td>
      <td>female</td>
      <td>white</td>
      <td>16</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48710.666947</td>
      <td>63.77</td>
      <td>female</td>
      <td>white</td>
      <td>16</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>female</td>
      <td>other</td>
      <td>16</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82089.345498</td>
      <td>63.08</td>
      <td>female</td>
      <td>white</td>
      <td>17</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sex.unique() # 데이터 중복 없애줌
```




    array(['male', 'female'], dtype=object)




```python
def change_sex(x):
    return 0 if x == "male" else 1

df.sex.map(change_sex) # 문자열로 되어있던 성별을 숫자 코드로 변환  - lambda 이용
```




    0       0
    1       1
    2       1
    3       1
    4       1
           ..
    1374    0
    1375    1
    1376    1
    1377    0
    1378    0
    Name: sex, Length: 1379, dtype: int64




```python
df["sex_code"] = df.sex.map({"male": 0, "female": 1}) # 성별을 숫자 코드로 표시하는 새로운 column "sex_code" 추가  - dictionary 이용
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
      <th>sex_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79571.299011</td>
      <td>73.89</td>
      <td>0</td>
      <td>white</td>
      <td>16</td>
      <td>49</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96396.988643</td>
      <td>66.23</td>
      <td>1</td>
      <td>white</td>
      <td>16</td>
      <td>62</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48710.666947</td>
      <td>63.77</td>
      <td>1</td>
      <td>white</td>
      <td>16</td>
      <td>33</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>1</td>
      <td>other</td>
      <td>16</td>
      <td>95</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82089.345498</td>
      <td>63.08</td>
      <td>1</td>
      <td>white</td>
      <td>17</td>
      <td>43</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sex.replace({"male": 0, "female": 1}) # 문자열로 되어있던 성별을 숫자 코드로 변환  - replace() 이용 - 원본은 변하지 않음
```




    0       0
    1       1
    2       1
    3       1
    4       1
           ..
    1374    0
    1375    1
    1376    1
    1377    0
    1378    0
    Name: sex, Length: 1379, dtype: int64




```python
df.sex.head(5)
```




    0    0
    1    1
    2    1
    3    1
    4    1
    Name: sex, dtype: int64




```python
df.sex.replace(["male", "female"], [0, 1], inplace=True) # inplace를 True로 하면 원본 수정됨
```


```python
df
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
      <th>0</th>
      <td>79571.299011</td>
      <td>73.89</td>
      <td>0</td>
      <td>white</td>
      <td>16</td>
      <td>49</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96396.988643</td>
      <td>66.23</td>
      <td>1</td>
      <td>white</td>
      <td>16</td>
      <td>62</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48710.666947</td>
      <td>63.77</td>
      <td>1</td>
      <td>white</td>
      <td>16</td>
      <td>33</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>1</td>
      <td>other</td>
      <td>16</td>
      <td>95</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82089.345498</td>
      <td>63.08</td>
      <td>1</td>
      <td>white</td>
      <td>17</td>
      <td>43</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1374</th>
      <td>30173.380363</td>
      <td>71.68</td>
      <td>0</td>
      <td>white</td>
      <td>12</td>
      <td>33</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1375</th>
      <td>24853.519514</td>
      <td>61.31</td>
      <td>1</td>
      <td>white</td>
      <td>18</td>
      <td>86</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1376</th>
      <td>13710.671312</td>
      <td>63.64</td>
      <td>1</td>
      <td>white</td>
      <td>12</td>
      <td>37</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1377</th>
      <td>95426.014410</td>
      <td>71.65</td>
      <td>0</td>
      <td>white</td>
      <td>12</td>
      <td>54</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1378</th>
      <td>9575.461857</td>
      <td>68.22</td>
      <td>0</td>
      <td>white</td>
      <td>12</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1379 rows × 7 columns</p>
</div>




```python
del df["sex_code"] # column 삭제
```


```python
df.head()
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
      <td>white</td>
      <td>16</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96396.988643</td>
      <td>66.23</td>
      <td>1</td>
      <td>white</td>
      <td>16</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48710.666947</td>
      <td>63.77</td>
      <td>1</td>
      <td>white</td>
      <td>16</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>1</td>
      <td>other</td>
      <td>16</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82089.345498</td>
      <td>63.08</td>
      <td>1</td>
      <td>white</td>
      <td>17</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



### apply & applymap

* map()과 달리 모든 column에 적용됨


```python
df = pd.read_csv("wages.csv")
df.head()
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
      <td>male</td>
      <td>white</td>
      <td>16</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96396.988643</td>
      <td>66.23</td>
      <td>female</td>
      <td>white</td>
      <td>16</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48710.666947</td>
      <td>63.77</td>
      <td>female</td>
      <td>white</td>
      <td>16</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>female</td>
      <td>other</td>
      <td>16</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82089.345498</td>
      <td>63.08</td>
      <td>female</td>
      <td>white</td>
      <td>17</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_info = df[["earn", "height", "age"]]
df_info.head()
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
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79571.299011</td>
      <td>73.89</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96396.988643</td>
      <td>66.23</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48710.666947</td>
      <td>63.77</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80478.096153</td>
      <td>63.22</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82089.345498</td>
      <td>63.08</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>




```python
f = lambda x: np.mean(x)
df_info.apply(f) # 각 column별 평균을 구함
```




    earn      32446.292622
    height       66.592640
    age          45.328499
    dtype: float64




```python
df_info.apply(np.mean)
```




    earn      32446.292622
    height       66.592640
    age          45.328499
    dtype: float64




```python
df_info.mean() # 여러가지 방법으로 평균을 구할 수 있음
```




    earn      32446.292622
    height       66.592640
    age          45.328499
    dtype: float64




```python
def f(x):
    return Series( # 전달된 column에 대하여 여러가지 통계값을 Series에 묶어 반환함
        [x.min(), x.max(), x.mean(), sum(x.isnull())],
        index=["min", "max", "mean", "null"],
    )

df_info.apply(f)
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
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>-98.580489</td>
      <td>57.34000</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>317949.127955</td>
      <td>77.21000</td>
      <td>95.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>32446.292622</td>
      <td>66.59264</td>
      <td>45.328499</td>
    </tr>
    <tr>
      <th>null</th>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
f = lambda x: x // 2
df_info.applymap(f).head(5) # applymap() 함수는 Series 단위가 아닌 element 단위로 함수를 적용함 -> apply()와 동일한 효과 제공
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
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39785.0</td>
      <td>36.0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48198.0</td>
      <td>33.0</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24355.0</td>
      <td>31.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40239.0</td>
      <td>31.0</td>
      <td>47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41044.0</td>
      <td>31.0</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
f = lambda x: x ** 2
df_info["earn"].apply(f) # Series에 apply()를 사용할 수도 있음
```




    0       6.331592e+09
    1       9.292379e+09
    2       2.372729e+09
    3       6.476724e+09
    4       6.738661e+09
                ...     
    1374    9.104329e+08
    1375    6.176974e+08
    1376    1.879825e+08
    1377    9.106124e+09
    1378    9.168947e+07
    Name: earn, Length: 1379, dtype: float64


