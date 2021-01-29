# Pandas

## DataFrame Operation

### Series addtion operation


```python
s1 = Series(range(1, 6), index=list("abced"))
s2 = Series(range(5, 11), index=list("bcedef"))
print(s1, s2, sep = "\n")
```

    a    1
    b    2
    c    3
    e    4
    d    5
    dtype: int64
    b     5
    c     6
    e     7
    d     8
    e     9
    f    10
    dtype: int64



```python
s1 + s2 # index를 기준으로 연산을 수행한다. 겹치는 index가 없는 경우엔 NaN이 된다
```




    a     NaN
    b     7.0
    c     9.0
    d    13.0
    e    11.0
    e    13.0
    f     NaN
    dtype: float64




```python
s1.add(s2) # add() 함수를 사용할 수도 있다
```




    a     NaN
    b     7.0
    c     9.0
    d    13.0
    e    11.0
    e    13.0
    f     NaN
    dtype: float64



### dataframe opertion


```python
df1 = DataFrame(np.arange(9).reshape(3, 3), columns=list("abc"))
df2 = DataFrame(np.arange(16).reshape(4, 4), columns=list("abcd"))
print(df1, df2, sep = "\n")
```

       a  b  c
    0  0  1  2
    1  3  4  5
    2  6  7  8
        a   b   c   d
    0   0   1   2   3
    1   4   5   6   7
    2   8   9  10  11
    3  12  13  14  15



```python
df1 + df2 # DataFrame끼리 연산은 index와 Column을 모두 고려한다
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.add(df2, fill_value=0) # fill_value값을 지정하면 연산 시 match되는 data가 없는 경우 fill_value값을 이용하여 연산한다
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.mul(df2, fill_value=1)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.0</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>63.0</td>
      <td>80.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>



### operations for dataframe with series


```python
df = DataFrame(np.arange(16).reshape(4, 4), columns=list("abcd"))
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
s = Series(np.arange(10, 14), index=list("abcd"))
s
```




    a    10
    b    11
    c    12
    d    13
    dtype: int64




```python
df + s
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>12</td>
      <td>14</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
      <td>16</td>
      <td>18</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>20</td>
      <td>22</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>24</td>
      <td>26</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
s2 = Series(np.arange(10, 14))
s2
```




    0    10
    1    11
    2    12
    3    13
    dtype: int64




```python
df + s2
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.add(s2, axis=0)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>26</td>
      <td>27</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>


