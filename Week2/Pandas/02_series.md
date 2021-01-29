# Pandas

## Series

* Series는 Pandas의 DataFrame에서 Column에 해당하는 Object이다

### 생성


```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
```


```python
series = Series(data=[1, 2, 3, 4, 5]) # 리스트를 사용하여 Series 생성
series
```




    0    1
    1    2
    2    3
    3    4
    4    5
    dtype: int64



* Series는 내부적으로 numpy의 ndarray를 사용하여 데이터를 저장한다 (Series는 ndarrya의 subclass)
* ndarray와의 차이점은 Series는 문자열을 인덱스로 사용할 수 있다는 점이다


```python
series = Series(data=[1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"]) # Series를 생성하며 index의 이름도 같이 지정
series
```




    a    1
    b    2
    c    3
    d    4
    e    5
    dtype: int64




```python
type(series.values)
```




    numpy.ndarray




```python
dict_data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
series = Series(dict_data, dtype=np.float32, name="example_series_data") # dictionary를 이용하여 Series를 초기화할 수도 있다
series
```




    a    1.0
    b    2.0
    c    3.0
    d    4.0
    e    5.0
    Name: example_series_data, dtype: float32




```python
series["c"]
```




    3.0




```python
series = series.astype(int) # astpye() 함수를 이용하여 Series에 저장된 value의 data type을 변경할 수 있다
series["a"] = 6.2 # 자동 형변환
series
```




    a    6
    b    2
    c    3
    d    4
    e    5
    Name: example_series_data, dtype: int64



### indexing


```python
cond = series > 3
print(cond)
print()
print(type(cond)) # boolean index 역시 Series로 생성된다
print()
series[cond] # ndarray와 과 같이 boolean index를 사용할 수 있다
```

    a     True
    b    False
    c    False
    d     True
    e     True
    Name: example_series_data, dtype: bool
    
    <class 'pandas.core.series.Series'>
    





    a    6
    d    4
    e    5
    Name: example_series_data, dtype: int64



### 연산


```python
series * 2
```




    a    12
    b     4
    c     6
    d     8
    e    10
    Name: example_series_data, dtype: int64




```python
np.exp(series)  # np.abs , np.log 등의 numpy 함수도 사용 가능
```




    alph
    a    403.428793
    b      7.389056
    c     20.085537
    d     54.598150
    e    148.413159
    Name: new name, dtype: float64



### 기타 기능


```python
"b" in series # in 연산자를 이용해 특정 키가 존재하는지 확인할 수 있다
```




    True




```python
dict = series.to_dict() # python dictionary로 변환할 수 있다
print(type(dict))
dict
```

    <class 'dict'>





    {'a': 6, 'b': 2, 'c': 3, 'd': 4, 'e': 5}




```python
series.name = "new name" # Sereis의 이름을 설정할 수 있다
series.index.name = "alph" # index의 이름을 설정할 수 있다
series
```




    alph
    a    6
    b    2
    c    3
    d    4
    e    5
    Name: new name, dtype: int64


