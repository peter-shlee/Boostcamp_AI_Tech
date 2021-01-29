# Pandas

## Selection & Drop

#### Data loading
- xlr파일 열려면 추가로 모듈을 설치해야함, conda install openpyxl


```python
import numpy as np
import pandas as pd

df = pd.read_excel("./excel-comp-data.xlsx")
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
      <th>account</th>
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>211829</td>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>320563</td>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>Port Khadijah</td>
      <td>NorthCarolina</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>648336</td>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>109996</td>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Hyattburgh</td>
      <td>Maine</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>121213</td>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
  </tbody>
</table>
</div>



#### data selction with index number and column names


```python
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
      <th>account</th>
      <td>211829</td>
      <td>320563</td>
    </tr>
    <tr>
      <th>name</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>Walter-Trantow</td>
    </tr>
    <tr>
      <th>street</th>
      <td>34456 Sean Highway</td>
      <td>1311 Alvis Tunnel</td>
    </tr>
    <tr>
      <th>city</th>
      <td>New Jaycob</td>
      <td>Port Khadijah</td>
    </tr>
    <tr>
      <th>state</th>
      <td>Texas</td>
      <td>NorthCarolina</td>
    </tr>
    <tr>
      <th>postal-code</th>
      <td>28752</td>
      <td>38365</td>
    </tr>
    <tr>
      <th>Jan</th>
      <td>10000</td>
      <td>95000</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>62000</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>35000</td>
      <td>35000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[["account", "street", "state"]].head() # index로 list를 넣으면 DataFrame을 반환한다
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
      <th>account</th>
      <th>street</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>211829</td>
      <td>34456 Sean Highway</td>
      <td>Texas</td>
    </tr>
    <tr>
      <th>1</th>
      <td>320563</td>
      <td>1311 Alvis Tunnel</td>
      <td>NorthCarolina</td>
    </tr>
    <tr>
      <th>2</th>
      <td>648336</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>Iowa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>109996</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Maine</td>
    </tr>
    <tr>
      <th>4</th>
      <td>121213</td>
      <td>7274 Marissa Common</td>
      <td>California</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[:3]
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
      <th>account</th>
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>211829</td>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>320563</td>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>Port Khadijah</td>
      <td>NorthCarolina</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>648336</td>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["name"][:3] # 해당 column에서만 data를 읽는다
```




    0    Kerluke, Koepp and Hilpert
    1                Walter-Trantow
    2    Bashirian, Kunde and Price
    Name: name, dtype: object




```python
account_serires = df["account"] # 해당 column의 Series를 account_series에 저장한다
account_serires[:3]
```




    0    211829
    1    320563
    2    648336
    Name: account, dtype: int64




```python
account_serires[list(range(0, 15, 2))] # list를 이용해 여러 row를 읽을 수 있다
```




    0     211829
    2     648336
    4     121213
    6     145068
    8     209744
    10    214098
    12    242368
    14    273274
    Name: account, dtype: int64




```python
account_serires[account_serires < 250000] # boolean index도 사용 가능하다
```




    0     211829
    3     109996
    4     121213
    5     132971
    6     145068
    7     205217
    8     209744
    9     212303
    10    214098
    11    231907
    12    242368
    Name: account, dtype: int64




```python
df.index = df["account"] # 기존의 column을 새로운 index로 지정한다
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
      <th>account</th>
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
    <tr>
      <th>account</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211829</th>
      <td>211829</td>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>320563</th>
      <td>320563</td>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>Port Khadijah</td>
      <td>NorthCarolina</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>648336</th>
      <td>648336</td>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>109996</th>
      <td>109996</td>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Hyattburgh</td>
      <td>Maine</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>121213</th>
      <td>121213</td>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
  </tbody>
</table>
</div>




```python
del df["account"] # 인덱스로 지정 뒤 해당 column은 삭제할 수 있다
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
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
    <tr>
      <th>account</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211829</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>320563</th>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>Port Khadijah</td>
      <td>NorthCarolina</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>648336</th>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>109996</th>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Hyattburgh</td>
      <td>Maine</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>121213</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[["name", "street"]][:2]
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
      <th>name</th>
      <th>street</th>
    </tr>
    <tr>
      <th>account</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211829</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
    </tr>
    <tr>
      <th>320563</th>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[[211829, 320563], ["city", "street"]]
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
      <th>city</th>
      <th>street</th>
    </tr>
    <tr>
      <th>account</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211829</th>
      <td>New Jaycob</td>
      <td>34456 Sean Highway</td>
    </tr>
    <tr>
      <th>320563</th>
      <td>Port Khadijah</td>
      <td>1311 Alvis Tunnel</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:10, :3]
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
      <th>name</th>
      <th>street</th>
      <th>city</th>
    </tr>
    <tr>
      <th>account</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211829</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
    </tr>
    <tr>
      <th>320563</th>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>Port Khadijah</td>
    </tr>
    <tr>
      <th>648336</th>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
    </tr>
    <tr>
      <th>109996</th>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Hyattburgh</td>
    </tr>
    <tr>
      <th>121213</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
    </tr>
    <tr>
      <th>132971</th>
      <td>Williamson, Schumm and Hettinger</td>
      <td>89403 Casimer Spring</td>
      <td>Jeremieburgh</td>
    </tr>
    <tr>
      <th>145068</th>
      <td>Casper LLC</td>
      <td>340 Consuela Bridge Apt. 400</td>
      <td>Lake Gabriellaton</td>
    </tr>
    <tr>
      <th>205217</th>
      <td>Kovacek-Johnston</td>
      <td>91971 Cronin Vista Suite 601</td>
      <td>Deronville</td>
    </tr>
    <tr>
      <th>209744</th>
      <td>Champlin-Morar</td>
      <td>26739 Grant Lock</td>
      <td>Lake Juliannton</td>
    </tr>
    <tr>
      <th>212303</th>
      <td>Gerhold-Maggio</td>
      <td>366 Maggio Grove Apt. 998</td>
      <td>North Ras</td>
    </tr>
  </tbody>
</table>
</div>



### reindex


```python
df.index = list(range(0, 15))
# df.reset_index(inplace=True) # inplace를 true로 지정하면 df 자체가 수정된다
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
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>Port Khadijah</td>
      <td>NorthCarolina</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Hyattburgh</td>
      <td>Maine</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
  </tbody>
</table>
</div>



### data drop


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
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>Port Khadijah</td>
      <td>NorthCarolina</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Hyattburgh</td>
      <td>Maine</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Williamson, Schumm and Hettinger</td>
      <td>89403 Casimer Spring</td>
      <td>Jeremieburgh</td>
      <td>Arkansas</td>
      <td>62785</td>
      <td>150000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Casper LLC</td>
      <td>340 Consuela Bridge Apt. 400</td>
      <td>Lake Gabriellaton</td>
      <td>Mississipi</td>
      <td>18008</td>
      <td>62000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Kovacek-Johnston</td>
      <td>91971 Cronin Vista Suite 601</td>
      <td>Deronville</td>
      <td>RhodeIsland</td>
      <td>53461</td>
      <td>145000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Champlin-Morar</td>
      <td>26739 Grant Lock</td>
      <td>Lake Juliannton</td>
      <td>Pennsylvania</td>
      <td>64415</td>
      <td>70000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Gerhold-Maggio</td>
      <td>366 Maggio Grove Apt. 998</td>
      <td>North Ras</td>
      <td>Idaho</td>
      <td>46308</td>
      <td>70000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Goodwin, Homenick and Jerde</td>
      <td>649 Cierra Forks Apt. 078</td>
      <td>Rosaberg</td>
      <td>Tenessee</td>
      <td>47743</td>
      <td>45000</td>
      <td>120000</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hahn-Moore</td>
      <td>18115 Olivine Throughway</td>
      <td>Norbertomouth</td>
      <td>NorthDakota</td>
      <td>31415</td>
      <td>150000</td>
      <td>10000</td>
      <td>162000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Frami, Anderson and Donnelly</td>
      <td>182 Bertie Road</td>
      <td>East Davian</td>
      <td>Iowa</td>
      <td>72686</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Walsh-Haley</td>
      <td>2624 Beatty Parkways</td>
      <td>Goodwinmouth</td>
      <td>RhodeIsland</td>
      <td>31919</td>
      <td>55000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>McDermott PLC</td>
      <td>8917 Bergstrom Meadow</td>
      <td>Kathryneborough</td>
      <td>Delaware</td>
      <td>27933</td>
      <td>150000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(1) # index == 1인 row를 삭제한 DataFrame 리턴
# df.drop(1, inplace=True) # inplace를 True로 하면 df 자체가 수정된다
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
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>New Jaycob</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>New Lilianland</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Hyattburgh</td>
      <td>Maine</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Williamson, Schumm and Hettinger</td>
      <td>89403 Casimer Spring</td>
      <td>Jeremieburgh</td>
      <td>Arkansas</td>
      <td>62785</td>
      <td>150000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Casper LLC</td>
      <td>340 Consuela Bridge Apt. 400</td>
      <td>Lake Gabriellaton</td>
      <td>Mississipi</td>
      <td>18008</td>
      <td>62000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Kovacek-Johnston</td>
      <td>91971 Cronin Vista Suite 601</td>
      <td>Deronville</td>
      <td>RhodeIsland</td>
      <td>53461</td>
      <td>145000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Champlin-Morar</td>
      <td>26739 Grant Lock</td>
      <td>Lake Juliannton</td>
      <td>Pennsylvania</td>
      <td>64415</td>
      <td>70000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Gerhold-Maggio</td>
      <td>366 Maggio Grove Apt. 998</td>
      <td>North Ras</td>
      <td>Idaho</td>
      <td>46308</td>
      <td>70000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Goodwin, Homenick and Jerde</td>
      <td>649 Cierra Forks Apt. 078</td>
      <td>Rosaberg</td>
      <td>Tenessee</td>
      <td>47743</td>
      <td>45000</td>
      <td>120000</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hahn-Moore</td>
      <td>18115 Olivine Throughway</td>
      <td>Norbertomouth</td>
      <td>NorthDakota</td>
      <td>31415</td>
      <td>150000</td>
      <td>10000</td>
      <td>162000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Frami, Anderson and Donnelly</td>
      <td>182 Bertie Road</td>
      <td>East Davian</td>
      <td>Iowa</td>
      <td>72686</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Walsh-Haley</td>
      <td>2624 Beatty Parkways</td>
      <td>Goodwinmouth</td>
      <td>RhodeIsland</td>
      <td>31919</td>
      <td>55000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>McDermott PLC</td>
      <td>8917 Bergstrom Meadow</td>
      <td>Kathryneborough</td>
      <td>Delaware</td>
      <td>27933</td>
      <td>150000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop([0, 1, 2, 3]) # 리스트를 사용하면 여러 row를 동시에 drop한다
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
      <th>name</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>Shanahanchester</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Williamson, Schumm and Hettinger</td>
      <td>89403 Casimer Spring</td>
      <td>Jeremieburgh</td>
      <td>Arkansas</td>
      <td>62785</td>
      <td>150000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Casper LLC</td>
      <td>340 Consuela Bridge Apt. 400</td>
      <td>Lake Gabriellaton</td>
      <td>Mississipi</td>
      <td>18008</td>
      <td>62000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Kovacek-Johnston</td>
      <td>91971 Cronin Vista Suite 601</td>
      <td>Deronville</td>
      <td>RhodeIsland</td>
      <td>53461</td>
      <td>145000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Champlin-Morar</td>
      <td>26739 Grant Lock</td>
      <td>Lake Juliannton</td>
      <td>Pennsylvania</td>
      <td>64415</td>
      <td>70000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Gerhold-Maggio</td>
      <td>366 Maggio Grove Apt. 998</td>
      <td>North Ras</td>
      <td>Idaho</td>
      <td>46308</td>
      <td>70000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Goodwin, Homenick and Jerde</td>
      <td>649 Cierra Forks Apt. 078</td>
      <td>Rosaberg</td>
      <td>Tenessee</td>
      <td>47743</td>
      <td>45000</td>
      <td>120000</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hahn-Moore</td>
      <td>18115 Olivine Throughway</td>
      <td>Norbertomouth</td>
      <td>NorthDakota</td>
      <td>31415</td>
      <td>150000</td>
      <td>10000</td>
      <td>162000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Frami, Anderson and Donnelly</td>
      <td>182 Bertie Road</td>
      <td>East Davian</td>
      <td>Iowa</td>
      <td>72686</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Walsh-Haley</td>
      <td>2624 Beatty Parkways</td>
      <td>Goodwinmouth</td>
      <td>RhodeIsland</td>
      <td>31919</td>
      <td>55000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>McDermott PLC</td>
      <td>8917 Bergstrom Meadow</td>
      <td>Kathryneborough</td>
      <td>Delaware</td>
      <td>27933</td>
      <td>150000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
</div>




```python
matrix = df.values
matrix[:3]
```




    array([['Kerluke, Koepp and Hilpert', '34456 Sean Highway', 'New Jaycob',
            'Texas', 28752, 10000, 62000, 35000],
           ['Walter-Trantow', '1311 Alvis Tunnel', 'Port Khadijah',
            'NorthCarolina', 38365, 95000, 45000, 35000],
           ['Bashirian, Kunde and Price',
            '62184 Schamberger Underpass Apt. 231', 'New Lilianland', 'Iowa',
            76517, 91000, 120000, 35000]], dtype=object)




```python
matrix[:, -3:] # ndarray에서와 동일한 방식으로 indexing 가능
```




    array([[10000, 62000, 35000],
           [95000, 45000, 35000],
           [91000, 120000, 35000],
           [45000, 120000, 10000],
           [162000, 120000, 35000],
           [150000, 120000, 35000],
           [62000, 120000, 70000],
           [145000, 95000, 35000],
           [70000, 95000, 35000],
           [70000, 120000, 35000],
           [45000, 120000, 55000],
           [150000, 10000, 162000],
           [162000, 120000, 35000],
           [55000, 120000, 35000],
           [150000, 120000, 70000]], dtype=object)




```python
matrix[:, -3:].sum(axis=1)
```




    array([107000, 175000, 246000, 175000, 317000, 305000, 252000, 275000,
           200000, 225000, 220000, 322000, 317000, 210000, 340000],
          dtype=object)




```python
df.drop("city", axis=1) # column 삭제
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
      <th>name</th>
      <th>street</th>
      <th>state</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>Texas</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>NorthCarolina</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>Iowa</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>Maine</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>California</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Williamson, Schumm and Hettinger</td>
      <td>89403 Casimer Spring</td>
      <td>Arkansas</td>
      <td>62785</td>
      <td>150000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Casper LLC</td>
      <td>340 Consuela Bridge Apt. 400</td>
      <td>Mississipi</td>
      <td>18008</td>
      <td>62000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Kovacek-Johnston</td>
      <td>91971 Cronin Vista Suite 601</td>
      <td>RhodeIsland</td>
      <td>53461</td>
      <td>145000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Champlin-Morar</td>
      <td>26739 Grant Lock</td>
      <td>Pennsylvania</td>
      <td>64415</td>
      <td>70000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Gerhold-Maggio</td>
      <td>366 Maggio Grove Apt. 998</td>
      <td>Idaho</td>
      <td>46308</td>
      <td>70000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Goodwin, Homenick and Jerde</td>
      <td>649 Cierra Forks Apt. 078</td>
      <td>Tenessee</td>
      <td>47743</td>
      <td>45000</td>
      <td>120000</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hahn-Moore</td>
      <td>18115 Olivine Throughway</td>
      <td>NorthDakota</td>
      <td>31415</td>
      <td>150000</td>
      <td>10000</td>
      <td>162000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Frami, Anderson and Donnelly</td>
      <td>182 Bertie Road</td>
      <td>Iowa</td>
      <td>72686</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Walsh-Haley</td>
      <td>2624 Beatty Parkways</td>
      <td>RhodeIsland</td>
      <td>31919</td>
      <td>55000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>McDermott PLC</td>
      <td>8917 Bergstrom Meadow</td>
      <td>Delaware</td>
      <td>27933</td>
      <td>150000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(["city", "state"], axis=1) # 여러 column 삭제
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
      <th>name</th>
      <th>street</th>
      <th>postal-code</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kerluke, Koepp and Hilpert</td>
      <td>34456 Sean Highway</td>
      <td>28752</td>
      <td>10000</td>
      <td>62000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Walter-Trantow</td>
      <td>1311 Alvis Tunnel</td>
      <td>38365</td>
      <td>95000</td>
      <td>45000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bashirian, Kunde and Price</td>
      <td>62184 Schamberger Underpass Apt. 231</td>
      <td>76517</td>
      <td>91000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D'Amore, Gleichner and Bode</td>
      <td>155 Fadel Crescent Apt. 144</td>
      <td>46021</td>
      <td>45000</td>
      <td>120000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bauch-Goldner</td>
      <td>7274 Marissa Common</td>
      <td>49681</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Williamson, Schumm and Hettinger</td>
      <td>89403 Casimer Spring</td>
      <td>62785</td>
      <td>150000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Casper LLC</td>
      <td>340 Consuela Bridge Apt. 400</td>
      <td>18008</td>
      <td>62000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Kovacek-Johnston</td>
      <td>91971 Cronin Vista Suite 601</td>
      <td>53461</td>
      <td>145000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Champlin-Morar</td>
      <td>26739 Grant Lock</td>
      <td>64415</td>
      <td>70000</td>
      <td>95000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Gerhold-Maggio</td>
      <td>366 Maggio Grove Apt. 998</td>
      <td>46308</td>
      <td>70000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Goodwin, Homenick and Jerde</td>
      <td>649 Cierra Forks Apt. 078</td>
      <td>47743</td>
      <td>45000</td>
      <td>120000</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hahn-Moore</td>
      <td>18115 Olivine Throughway</td>
      <td>31415</td>
      <td>150000</td>
      <td>10000</td>
      <td>162000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Frami, Anderson and Donnelly</td>
      <td>182 Bertie Road</td>
      <td>72686</td>
      <td>162000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Walsh-Haley</td>
      <td>2624 Beatty Parkways</td>
      <td>31919</td>
      <td>55000</td>
      <td>120000</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>McDermott PLC</td>
      <td>8917 Bergstrom Meadow</td>
      <td>27933</td>
      <td>150000</td>
      <td>120000</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
</div>


