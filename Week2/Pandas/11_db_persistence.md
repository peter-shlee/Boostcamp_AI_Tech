# Pandas

## Load database

* DB 사용

### Example from 
- https://www.dataquest.io/blog/python-pandas-databases/


```python
import sqlite3  # pymysql <- 설치

conn = sqlite3.connect("./data/flights.db")
cur = conn.cursor()
cur.execute("select * from airlines limit 5;")
results = cur.fetchall()
results
```




    [(0, '1', 'Private flight', '\\N', '-', None, None, None, 'Y'),
     (1, '2', '135 Airways', '\\N', None, 'GNL', 'GENERAL', 'United States', 'N'),
     (2, '3', '1Time Airline', '\\N', '1T', 'RNX', 'NEXTIME', 'South Africa', 'Y'),
     (3,
      '4',
      '2 Sqn No 1 Elementary Flying Training School',
      '\\N',
      None,
      'WYT',
      None,
      'United Kingdom',
      'N'),
     (4, '5', '213 Flight Unit', '\\N', None, 'TFU', None, 'Russia', 'N')]



#### Data loading using pandas from DB


```python
df_airplines = pd.read_sql_query("select * from airlines;", conn)  # pandas로 바로 db에서 읽어들일 수 있다
df_airplines
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
      <th>id</th>
      <th>name</th>
      <th>alias</th>
      <th>iata</th>
      <th>icao</th>
      <th>callsign</th>
      <th>country</th>
      <th>active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>Private flight</td>
      <td>\N</td>
      <td>-</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>135 Airways</td>
      <td>\N</td>
      <td>None</td>
      <td>GNL</td>
      <td>GENERAL</td>
      <td>United States</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1Time Airline</td>
      <td>\N</td>
      <td>1T</td>
      <td>RNX</td>
      <td>NEXTIME</td>
      <td>South Africa</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>2 Sqn No 1 Elementary Flying Training School</td>
      <td>\N</td>
      <td>None</td>
      <td>WYT</td>
      <td>None</td>
      <td>United Kingdom</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>213 Flight Unit</td>
      <td>\N</td>
      <td>None</td>
      <td>TFU</td>
      <td>None</td>
      <td>Russia</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>6</td>
      <td>223 Flight Unit State Airline</td>
      <td>\N</td>
      <td>None</td>
      <td>CHD</td>
      <td>CHKALOVSK-AVIA</td>
      <td>Russia</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7</td>
      <td>224th Flight Unit</td>
      <td>\N</td>
      <td>None</td>
      <td>TTF</td>
      <td>CARGO UNIT</td>
      <td>Russia</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>8</td>
      <td>247 Jet Ltd</td>
      <td>\N</td>
      <td>None</td>
      <td>TWF</td>
      <td>CLOUD RUNNER</td>
      <td>United Kingdom</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>9</td>
      <td>3D Aviation</td>
      <td>\N</td>
      <td>None</td>
      <td>SEC</td>
      <td>SECUREX</td>
      <td>United States</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>10</td>
      <td>40-Mile Air</td>
      <td>\N</td>
      <td>Q5</td>
      <td>MLA</td>
      <td>MILE-AIR</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>11</td>
      <td>4D Air</td>
      <td>\N</td>
      <td>None</td>
      <td>QRT</td>
      <td>QUARTET</td>
      <td>Thailand</td>
      <td>N</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>12</td>
      <td>611897 Alberta Limited</td>
      <td>\N</td>
      <td>None</td>
      <td>THD</td>
      <td>DONUT</td>
      <td>Canada</td>
      <td>N</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>13</td>
      <td>Ansett Australia</td>
      <td>\N</td>
      <td>AN</td>
      <td>AAA</td>
      <td>ANSETT</td>
      <td>Australia</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>14</td>
      <td>Abacus International</td>
      <td>\N</td>
      <td>1B</td>
      <td>None</td>
      <td>None</td>
      <td>Singapore</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>15</td>
      <td>Abelag Aviation</td>
      <td>\N</td>
      <td>W9</td>
      <td>AAB</td>
      <td>ABG</td>
      <td>Belgium</td>
      <td>N</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>16</td>
      <td>Army Air Corps</td>
      <td>\N</td>
      <td>None</td>
      <td>AAC</td>
      <td>ARMYAIR</td>
      <td>United Kingdom</td>
      <td>N</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>17</td>
      <td>Aero Aviation Centre Ltd.</td>
      <td>\N</td>
      <td>None</td>
      <td>AAD</td>
      <td>SUNRISE</td>
      <td>Canada</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>18</td>
      <td>Aero Servicios Ejecutivos Internacionales</td>
      <td>\N</td>
      <td>None</td>
      <td>SII</td>
      <td>ASEISA</td>
      <td>Mexico</td>
      <td>N</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>19</td>
      <td>Aero Biniza</td>
      <td>\N</td>
      <td>None</td>
      <td>BZS</td>
      <td>BINIZA</td>
      <td>Mexico</td>
      <td>N</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>20</td>
      <td>Aero Albatros</td>
      <td>\N</td>
      <td>None</td>
      <td>ABM</td>
      <td>ALBATROS ESPANA</td>
      <td>Spain</td>
      <td>N</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>21</td>
      <td>Aigle Azur</td>
      <td>\N</td>
      <td>ZI</td>
      <td>AAF</td>
      <td>AIGLE AZUR</td>
      <td>France</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>22</td>
      <td>Aloha Airlines</td>
      <td>\N</td>
      <td>AQ</td>
      <td>AAH</td>
      <td>ALOHA</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>23</td>
      <td>Alaska Island Air</td>
      <td>\N</td>
      <td>None</td>
      <td>AAK</td>
      <td>ALASKA ISLAND</td>
      <td>United States</td>
      <td>N</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>24</td>
      <td>American Airlines</td>
      <td>\N</td>
      <td>AA</td>
      <td>AAL</td>
      <td>AMERICAN</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>25</td>
      <td>Aviation Management Corporation</td>
      <td>\N</td>
      <td>None</td>
      <td>AAM</td>
      <td>AM CORP</td>
      <td>United States</td>
      <td>N</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>26</td>
      <td>Atlantis Airlines (USA)</td>
      <td>\N</td>
      <td>None</td>
      <td>AAO</td>
      <td>ATLANTIS AIR</td>
      <td>United States</td>
      <td>N</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>27</td>
      <td>Aerovista Airlines</td>
      <td>\N</td>
      <td>None</td>
      <td>AAP</td>
      <td>AEROVISTA GROUP</td>
      <td>United Arab Emirates</td>
      <td>N</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>28</td>
      <td>Asiana Airlines</td>
      <td>\N</td>
      <td>OZ</td>
      <td>AAR</td>
      <td>ASIANA</td>
      <td>Republic of Korea</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>29</td>
      <td>Askari Aviation</td>
      <td>\N</td>
      <td>4K</td>
      <td>AAS</td>
      <td>AL-AAS</td>
      <td>Pakistan</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>30</td>
      <td>Australia Asia Airlines</td>
      <td>\N</td>
      <td>None</td>
      <td>AAU</td>
      <td>AUSTASIA</td>
      <td>Australia</td>
      <td>N</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6018</th>
      <td>6018</td>
      <td>19651</td>
      <td>CARICOM AIRWAYS (BARBADOS) INC.</td>
      <td>CARICOM AIRWAYS</td>
      <td>None</td>
      <td>CCB</td>
      <td>None</td>
      <td>Barbados</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6019</th>
      <td>6019</td>
      <td>19674</td>
      <td>Rainbow Air (RAI)</td>
      <td>Rainbow Air (RAI)</td>
      <td>RN</td>
      <td>RAB</td>
      <td>Rainbow</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6020</th>
      <td>6020</td>
      <td>19675</td>
      <td>Rainbow Air Canada</td>
      <td>Rainbow Air CAN</td>
      <td>RY</td>
      <td>RAY</td>
      <td>Rainbow CAN</td>
      <td>Canada</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6021</th>
      <td>6021</td>
      <td>19676</td>
      <td>Rainbow Air Polynesia</td>
      <td>Rainbow Air POL</td>
      <td>RX</td>
      <td>RPO</td>
      <td>Rainbow Air</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6022</th>
      <td>6022</td>
      <td>19677</td>
      <td>Rainbow Air Euro</td>
      <td>Rainbow Air EU</td>
      <td>RU</td>
      <td>RUE</td>
      <td>Rainbow Air</td>
      <td>United Kingdom</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6023</th>
      <td>6023</td>
      <td>19678</td>
      <td>Rainbow Air US</td>
      <td>Rainbow Air US</td>
      <td>RM</td>
      <td>RNY</td>
      <td>Rainbow Air</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6024</th>
      <td>6024</td>
      <td>19745</td>
      <td>Transilvania</td>
      <td>None</td>
      <td>None</td>
      <td>TNS</td>
      <td>None</td>
      <td>Romania</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6025</th>
      <td>6025</td>
      <td>19751</td>
      <td>Dobrolet</td>
      <td>Добролёт</td>
      <td>QD</td>
      <td>DOB</td>
      <td>DOBROLET</td>
      <td>Russia</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6026</th>
      <td>6026</td>
      <td>19774</td>
      <td>Spike Airlines</td>
      <td>Aero Spike</td>
      <td>S0</td>
      <td>SAL</td>
      <td>Spike Air</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6027</th>
      <td>6027</td>
      <td>19776</td>
      <td>Grand Cru Airlines</td>
      <td>None</td>
      <td>None</td>
      <td>GCA</td>
      <td>None</td>
      <td>Lithuania</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6028</th>
      <td>6028</td>
      <td>19785</td>
      <td>Go2Sky</td>
      <td>None</td>
      <td>None</td>
      <td>RLX</td>
      <td>RELAX</td>
      <td>Slovakia</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6029</th>
      <td>6029</td>
      <td>19803</td>
      <td>All Argentina</td>
      <td>All Argentina</td>
      <td>L1</td>
      <td>AL1</td>
      <td>None</td>
      <td>Argentina</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6030</th>
      <td>6030</td>
      <td>19804</td>
      <td>All America</td>
      <td>All America</td>
      <td>A2</td>
      <td>AL2</td>
      <td>None</td>
      <td>United States</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6031</th>
      <td>6031</td>
      <td>19805</td>
      <td>All Asia</td>
      <td>All Asia</td>
      <td>L9</td>
      <td>AL3</td>
      <td>None</td>
      <td>China</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6032</th>
      <td>6032</td>
      <td>19806</td>
      <td>All Africa</td>
      <td>All Africa</td>
      <td>9A</td>
      <td>99F</td>
      <td>None</td>
      <td>South Africa</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6033</th>
      <td>6033</td>
      <td>19807</td>
      <td>Regionalia México</td>
      <td>Regionalia México</td>
      <td>N4</td>
      <td>J88</td>
      <td>None</td>
      <td>Mexico</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6034</th>
      <td>6034</td>
      <td>19808</td>
      <td>All Europe</td>
      <td>All Europe</td>
      <td>N9</td>
      <td>N99</td>
      <td>None</td>
      <td>United Kingdom</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6035</th>
      <td>6035</td>
      <td>19809</td>
      <td>All Spain</td>
      <td>All Spain</td>
      <td>N7</td>
      <td>N77</td>
      <td>None</td>
      <td>Spain</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6036</th>
      <td>6036</td>
      <td>19810</td>
      <td>Regional Air Iceland</td>
      <td>Regional Air Iceland</td>
      <td>9N</td>
      <td>N78</td>
      <td>None</td>
      <td>Iceland</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6037</th>
      <td>6037</td>
      <td>19811</td>
      <td>British Air Ferries</td>
      <td>None</td>
      <td>??</td>
      <td>??!</td>
      <td>None</td>
      <td>United Kingdom</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6038</th>
      <td>6038</td>
      <td>19812</td>
      <td>Voestar</td>
      <td>Voestar Brasil</td>
      <td>8K</td>
      <td>K88</td>
      <td>None</td>
      <td>Brazil</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6039</th>
      <td>6039</td>
      <td>19813</td>
      <td>All Colombia</td>
      <td>All Colombia</td>
      <td>7O</td>
      <td>7KK</td>
      <td>None</td>
      <td>Colombia</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6040</th>
      <td>6040</td>
      <td>19814</td>
      <td>Regionalia Uruguay</td>
      <td>Regionalia Uruguay</td>
      <td>2X</td>
      <td>2K2</td>
      <td>None</td>
      <td>Uruguay</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6041</th>
      <td>6041</td>
      <td>19815</td>
      <td>Regionalia Venezuela</td>
      <td>Regionalia Venezuela</td>
      <td>9X</td>
      <td>9XX</td>
      <td>None</td>
      <td>Venezuela</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6042</th>
      <td>6042</td>
      <td>19827</td>
      <td>Regionalia Chile</td>
      <td>Regionalia Chile</td>
      <td>9J</td>
      <td>CR1</td>
      <td>None</td>
      <td>Chile</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6043</th>
      <td>6043</td>
      <td>19828</td>
      <td>Vuela Cuba</td>
      <td>Vuela Cuba</td>
      <td>6C</td>
      <td>6CC</td>
      <td>None</td>
      <td>Cuba</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6044</th>
      <td>6044</td>
      <td>19830</td>
      <td>All Australia</td>
      <td>All Australia</td>
      <td>88</td>
      <td>8K8</td>
      <td>None</td>
      <td>Australia</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6045</th>
      <td>6045</td>
      <td>19831</td>
      <td>Fly Europa</td>
      <td>None</td>
      <td>ER</td>
      <td>RWW</td>
      <td>None</td>
      <td>Spain</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6046</th>
      <td>6046</td>
      <td>19834</td>
      <td>FlyPortugal</td>
      <td>None</td>
      <td>PO</td>
      <td>FPT</td>
      <td>FlyPortugal</td>
      <td>Portugal</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6047</th>
      <td>6047</td>
      <td>19845</td>
      <td>FTI Fluggesellschaft</td>
      <td>None</td>
      <td>None</td>
      <td>FTI</td>
      <td>None</td>
      <td>Germany</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>6048 rows × 9 columns</p>
</div>



## Pandas persistence

* 엑셀파일 (.xlsx), pickle 사용

### install
- conda install openpyxl
- conda install XlsxWriter
- see more http://xlsxwriter.readthedocs.io/working_with_pandas.html


```python
writer = pd.ExcelWriter("./data/df_routes.xlsx", engine="xlsxwriter")
df_routes.to_excel(writer, sheet_name="Sheet1") # 엑셀파일로 저장
```


```python
df_routes.to_pickle("./data/df_routes.pickle") # pickle로 저장
```


```python
df_routes_pickle = pd.read_pickle("./data/df_routes.pickle")
df_routes_pickle.head()
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
      <th>airline</th>
      <th>airline_id</th>
      <th>source</th>
      <th>source_id</th>
      <th>dest</th>
      <th>dest_id</th>
      <th>codeshare</th>
      <th>stops</th>
      <th>equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2B</td>
      <td>410</td>
      <td>AER</td>
      <td>2965</td>
      <td>KZN</td>
      <td>2990</td>
      <td>None</td>
      <td>0</td>
      <td>CR2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2B</td>
      <td>410</td>
      <td>ASF</td>
      <td>2966</td>
      <td>KZN</td>
      <td>2990</td>
      <td>None</td>
      <td>0</td>
      <td>CR2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2B</td>
      <td>410</td>
      <td>ASF</td>
      <td>2966</td>
      <td>MRV</td>
      <td>2962</td>
      <td>None</td>
      <td>0</td>
      <td>CR2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2B</td>
      <td>410</td>
      <td>CEK</td>
      <td>2968</td>
      <td>KZN</td>
      <td>2990</td>
      <td>None</td>
      <td>0</td>
      <td>CR2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2B</td>
      <td>410</td>
      <td>CEK</td>
      <td>2968</td>
      <td>OVB</td>
      <td>4078</td>
      <td>None</td>
      <td>0</td>
      <td>CR2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_routes_pickle.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>67663.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33831.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19532.769969</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16915.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33831.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>50746.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>67662.000000</td>
    </tr>
  </tbody>
</table>
</div>


