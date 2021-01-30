# Pandas

### pivot table


```python
df_phone = pd.read_csv("./phone_data.csv")
df_phone["date"] = df_phone["date"].apply(dateutil.parser.parse, dayfirst=True)
df_phone.head()
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
      <th>index</th>
      <th>date</th>
      <th>duration</th>
      <th>item</th>
      <th>month</th>
      <th>network</th>
      <th>network_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2014-10-15 06:58:00</td>
      <td>34.429</td>
      <td>data</td>
      <td>2014-11</td>
      <td>data</td>
      <td>data</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2014-10-15 06:58:00</td>
      <td>13.000</td>
      <td>call</td>
      <td>2014-11</td>
      <td>Vodafone</td>
      <td>mobile</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2014-10-15 14:46:00</td>
      <td>23.000</td>
      <td>call</td>
      <td>2014-11</td>
      <td>Meteor</td>
      <td>mobile</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2014-10-15 14:48:00</td>
      <td>4.000</td>
      <td>call</td>
      <td>2014-11</td>
      <td>Tesco</td>
      <td>mobile</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2014-10-15 17:27:00</td>
      <td>4.000</td>
      <td>call</td>
      <td>2014-11</td>
      <td>Tesco</td>
      <td>mobile</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_phone.pivot_table(
    values=["duration"], # duration을 value로 사용
    index=[df_phone.month, df_phone.item], # index 지정 (multi index)
    columns=df_phone.network, # column 지정
    aggfunc="sum", # 총합 구하는 연산 적용
    fill_value=0, # 값 없는 경우
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

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="9" halign="left">duration</th>
    </tr>
    <tr>
      <th></th>
      <th>network</th>
      <th>Meteor</th>
      <th>Tesco</th>
      <th>Three</th>
      <th>Vodafone</th>
      <th>data</th>
      <th>landline</th>
      <th>special</th>
      <th>voicemail</th>
      <th>world</th>
    </tr>
    <tr>
      <th>month</th>
      <th>item</th>
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
      <th rowspan="3" valign="top">2014-11</th>
      <th>call</th>
      <td>1521</td>
      <td>4045</td>
      <td>12458</td>
      <td>4316</td>
      <td>0.000</td>
      <td>2906</td>
      <td>0</td>
      <td>301</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>998.441</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sms</th>
      <td>10</td>
      <td>3</td>
      <td>25</td>
      <td>55</td>
      <td>0.000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2014-12</th>
      <th>call</th>
      <td>2010</td>
      <td>1819</td>
      <td>6316</td>
      <td>1302</td>
      <td>0.000</td>
      <td>1424</td>
      <td>0</td>
      <td>690</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1032.870</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sms</th>
      <td>12</td>
      <td>1</td>
      <td>13</td>
      <td>18</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2015-01</th>
      <th>call</th>
      <td>2207</td>
      <td>2904</td>
      <td>6445</td>
      <td>3626</td>
      <td>0.000</td>
      <td>1603</td>
      <td>0</td>
      <td>285</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1067.299</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sms</th>
      <td>10</td>
      <td>3</td>
      <td>33</td>
      <td>40</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2015-02</th>
      <th>call</th>
      <td>1188</td>
      <td>4087</td>
      <td>6279</td>
      <td>1864</td>
      <td>0.000</td>
      <td>730</td>
      <td>0</td>
      <td>268</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1067.299</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sms</th>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>23</td>
      <td>0.000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2015-03</th>
      <th>call</th>
      <td>274</td>
      <td>973</td>
      <td>4966</td>
      <td>3513</td>
      <td>0.000</td>
      <td>11770</td>
      <td>0</td>
      <td>231</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>998.441</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sms</th>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### crosstab

* Pivot table의 특수한 형태
* 두 column에 교차 빈도, 비율, 덧셈 등을 구할 때 사용


```python
df_movie = pd.read_csv("./movie_rating.csv")
df_movie.head()
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
      <th>critic</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jack Matthews</td>
      <td>Lady in the Water</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jack Matthews</td>
      <td>Snakes on a Plane</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jack Matthews</td>
      <td>You Me and Dupree</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jack Matthews</td>
      <td>Superman Returns</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jack Matthews</td>
      <td>The Night Listener</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_movie.pivot_table(
    ["rating"], # rating을 value로 사용
    index=df_movie.critic, # cirtic을 index로 사용
    columns=df_movie.title, # title을 column으로 사용
    aggfunc="sum", # 총합을 더하는 연산을 사용 - 영화별 평점 총합
    fill_value=0, # 데이터 없으면 0으로
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

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th>Just My Luck</th>
      <th>Lady in the Water</th>
      <th>Snakes on a Plane</th>
      <th>Superman Returns</th>
      <th>The Night Listener</th>
      <th>You Me and Dupree</th>
    </tr>
    <tr>
      <th>critic</th>
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
      <th>Claudia Puig</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>Gene Seymour</th>
      <td>1.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>Jack Matthews</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>Lisa Rose</th>
      <td>3.0</td>
      <td>2.5</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>Mick LaSalle</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Toby</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab(
    index=df_movie.critic, # ciritic을 index로 사용
    columns=df_movie.title, # title을 column으로 사용
    values=df_movie.rating, # rating을 value로 사용
    aggfunc="first", # 첫번째로 나온 값을 표시하는 연산 사용
).fillna(0) # NaN을 0으로 바꿈
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
      <th>title</th>
      <th>Just My Luck</th>
      <th>Lady in the Water</th>
      <th>Snakes on a Plane</th>
      <th>Superman Returns</th>
      <th>The Night Listener</th>
      <th>You Me and Dupree</th>
    </tr>
    <tr>
      <th>critic</th>
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
      <th>Claudia Puig</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>Gene Seymour</th>
      <td>1.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>Jack Matthews</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>Lisa Rose</th>
      <td>3.0</td>
      <td>2.5</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>Mick LaSalle</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Toby</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_movie.groupby(["critic", "title"]).agg({"rating": "first"}) # critic과 title로 group 생성 후 연산 적용
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
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>critic</th>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Claudia Puig</th>
      <th>Just My Luck</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Snakes on a Plane</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>Superman Returns</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>The Night Listener</th>
      <td>4.5</td>
    </tr>
    <tr>
      <th>You Me and Dupree</th>
      <td>2.5</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Gene Seymour</th>
      <th>Just My Luck</th>
      <td>1.5</td>
    </tr>
    <tr>
      <th>Lady in the Water</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Snakes on a Plane</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>Superman Returns</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>The Night Listener</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>You Me and Dupree</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Jack Matthews</th>
      <th>Lady in the Water</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Snakes on a Plane</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Superman Returns</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>The Night Listener</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>You Me and Dupree</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Lisa Rose</th>
      <th>Just My Luck</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Lady in the Water</th>
      <td>2.5</td>
    </tr>
    <tr>
      <th>Snakes on a Plane</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>Superman Returns</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>The Night Listener</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>You Me and Dupree</th>
      <td>2.5</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Mick LaSalle</th>
      <th>Just My Luck</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Lady in the Water</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Snakes on a Plane</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Superman Returns</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>The Night Listener</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>You Me and Dupree</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Toby</th>
      <th>Snakes on a Plane</th>
      <td>4.5</td>
    </tr>
    <tr>
      <th>Superman Returns</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>You Me and Dupree</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


