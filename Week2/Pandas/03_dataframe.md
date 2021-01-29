# Pandas

## DataFrame


### 생성

* 실제로 DataFrame을 직접 생성하는 일은 거의 없다
* 보통 이미 존재하는 데이터를 불러와서 생성한다


```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
```


```python
# Example from - https://chrisalbon.com/python/pandas_map_values_to_values.html
raw_data = {
    "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
    "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
    "age": [42, 52, 36, 24, 73],
    "city": ["San Francisco", "Baltimore", "Miami", "Douglas", "Boston"],
}
df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "city"])
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
    </tr>
  </tbody>
</table>
</div>




```python
DataFrame(raw_data, columns=["age", "city"]) # 특정 column만 선택해서 DataFrame을 생성할 수 있다
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
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52</td>
      <td>Baltimore</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>Douglas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>73</td>
      <td>Boston</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = DataFrame(raw_data, columns=["first_name", "last_name", "age", "city", "debt"])
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### data 접근, Indexing


```python
df.first_name
```




    0    Jason
    1    Molly
    2     Tina
    3     Jake
    4      Amy
    Name: first_name, dtype: object




```python
df["first_name"]
```




    0    Jason
    1    Molly
    2     Tina
    3     Jake
    4      Amy
    Name: first_name, dtype: object




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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### loc과 iloc

* 두가지 indexing 방식이 존재한다
    * loc은 index의 이름으로 indexing을 한다 (Python Dictionary와 동일한 방식)
    * iloc는 element의 위치로 indexing 한다 (Python list와 동일한 방식)


```python
indexing_df = df.copy() # indexing 테스트를 위해 기존의 DataFrame을 복사한다.
                        # DataFrame을 깊은복사 할 때는 copy()를 사용한다
print(indexing_df is df)# 새로운 객체가 생성된 것을 확인할 수 있다
```

    False



```python
indexing_df.index = range(1, 6) # 테스트를 위해 새로운 인덱스를 지정한다
indexing_df
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
    </tr>
  </tbody>
</table>
</div>




```python
indexing_df.loc[2:] # loc는 인덱스의 이름으로 indexing한다 (Python Dictionary와 동일한 방식)
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
    </tr>
  </tbody>
</table>
</div>




```python
indexing_df.iloc[2:] # iloc는 element의 위치로 indexing 한다 (Python list와 동일한 방식)
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
    </tr>
  </tbody>
</table>
</div>



* 위의 두 코드 모두 2를 가지고 indexing을 하였지만 다른 결과가 나온 것을 유심히 살펴보자


```python
series = indexing_df.loc[2:, "last_name"] # index를 배열에 넣지 않고 그냥 indexing 하면 해당 index의 Series를 반환한다
print(type(series))
series
```

    <class 'pandas.core.series.Series'>





    2    Jacobson
    3         Ali
    4      Milner
    5       Cooze
    Name: last_name, dtype: object




```python
result_df = indexing_df.loc[2:, ["last_name"]] # index를 배열에 넣은 뒤 indexing하면 결과로 DataFrame이 나온다
print(type(result_df))
result_df
```

    <class 'pandas.core.frame.DataFrame'>





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
      <th>last_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Jacobson</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ali</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Milner</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cooze</td>
    </tr>
  </tbody>
</table>
</div>




```python
indexing_df["age"].iloc[1:]
```




    2    52
    3    36
    4    24
    5    73
    Name: age, dtype: int64



### 새로운 Data 추가


```python
df.age > 40
```




    0     True
    1     True
    2    False
    3    False
    4     True
    Name: age, dtype: bool




```python
df["debt"] = df.age > 40 # 새로운 column을 추가한다
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
values = Series(data=["M", "F", "F"], index=[0, 1, 3]) # 새로 추가할 데이터를 담은 Series를 생성한다
values
```




    0    M
    1    F
    3    F
    dtype: object




```python
df["sex"] = values # 위에서 생성한 Series를 DataFrame의 새로운 column으로 추가한다
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>debt</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>True</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>True</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>False</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



데이터가 비어있는 곳은 NaN으로 채워진다

### Data 삭제


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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>debt</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>True</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>True</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>False</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop("debt", axis=1) # drop() 함수를 사용하면 지정한 데이터가 삭제된 새로운 DataFrame을 리턴한다 (원본은 바뀌지 않는다)
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(1, axis = 0)
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>debt</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>True</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>False</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>debt</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>True</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>True</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>False</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
del df["debt"] # del을 사용하면 원본 데이터에서 바로 삭제된다
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
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jason</td>
      <td>Miller</td>
      <td>42</td>
      <td>San Francisco</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Molly</td>
      <td>Jacobson</td>
      <td>52</td>
      <td>Baltimore</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tina</td>
      <td>Ali</td>
      <td>36</td>
      <td>Miami</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Milner</td>
      <td>24</td>
      <td>Douglas</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amy</td>
      <td>Cooze</td>
      <td>73</td>
      <td>Boston</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 기타 기능들


```python
df.head(3).T # Transpose 시킬수도 있다
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
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first_name</th>
      <td>Jason</td>
      <td>Molly</td>
      <td>Tina</td>
    </tr>
    <tr>
      <th>last_name</th>
      <td>Miller</td>
      <td>Jacobson</td>
      <td>Ali</td>
    </tr>
    <tr>
      <th>age</th>
      <td>42</td>
      <td>52</td>
      <td>36</td>
    </tr>
    <tr>
      <th>city</th>
      <td>San Francisco</td>
      <td>Baltimore</td>
      <td>Miami</td>
    </tr>
    <tr>
      <th>debt</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>M</td>
      <td>F</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(type(df.values))
df.values # ndarray 형태로 values에 접근할 수 있다
```

    <class 'numpy.ndarray'>





    array([['Jason', 'Miller', 42, 'San Francisco', True, 'M'],
           ['Molly', 'Jacobson', 52, 'Baltimore', True, 'F'],
           ['Tina', 'Ali', 36, 'Miami', False, nan],
           ['Jake', 'Milner', 24, 'Douglas', False, 'F'],
           ['Amy', 'Cooze', 73, 'Boston', True, nan]], dtype=object)




```python
df.index
```




    RangeIndex(start=0, stop=5, step=1)




```python
df.to_csv() # DataFrame 객체를 바로 CSV 형태로 변환할 수도 있다
```




    ',first_name,last_name,age,city,debt,sex\n0,Jason,Miller,42,San Francisco,True,M\n1,Molly,Jacobson,52,Baltimore,True,F\n2,Tina,Ali,36,Miami,False,\n3,Jake,Milner,24,Douglas,False,F\n4,Amy,Cooze,73,Boston,True,\n'


