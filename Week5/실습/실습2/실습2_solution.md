# Lab 2 : Graph Property
- Instuctor : Kijung Shin
- Teaching Assistants : Deukryeol Yoon(main), Hyunju Lee, Shinhwan Kang 
- 본 실습에서는 그래프의 다양한 특성 중 그래프 지름, 전역 군집 계수, 차수 분포를 배우고, 수업에서 배운 small world graph의 그래프 특성을 알아본다.




```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
# Lab 2 : Graph Property
# - Instuctor : Kijung Shin
# - Teaching Assistants : Deukryeol Yoon(main), Hyunju Lee, Shinhwan Kang 
# 본 실습에서는 그래프의 다양한 특성 중 Diameter, Average Clustering Coefficient, Degree Distribution을 배우고. 
# 수업에서 배운 small world graph의 그래프 특성을 알아본다.
#

# 실습에 필요한 library를 import하고 그래프를 초기화합니다.
import networkx as nx
import os
import os.path as osp
import numpy as np
import sys
import matplotlib.pyplot as plt
import collections
np.set_printoptions(threshold=sys.maxsize)

cycle_graph = nx.Graph()
regular_graph = nx.Graph()
small_world_graph = nx.Graph()
random_graph = nx.Graph()
```


```python
# 실습에 사용할 데이터를 읽어옵니다.
print("###### Read Graphs ######")
data = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/lab/lab2/cycle.txt'))
f = open(data)
for line in f:
    v1, v2 = map(int, line.split())
    cycle_graph.add_edge(v1, v2)

data = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/lab/lab2/regular.txt'))
f = open(data)
for line in f:
    v1, v2 = map(int, line.split())
    regular_graph.add_edge(v1, v2)

data = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/lab/lab2/small_world.txt'))
f = open(data)
for line in f:
    v1, v2, = map(int, line.split())
    small_world_graph.add_edge(v1, v2)

data = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/lab/lab2/random.txt'))
f = open(data)
for line in f:
    v1, v2 = map(int, line.split())
    random_graph.add_edge(v1, v2)
```

    ###### Read Graphs ######



```python
# 그래프의 전역 군집 계수를 찾는 함수입니다.
#
# 특정 정점 u의 정점 계수(cc)는 아래와 같이 구할 수 있습니다.
# cc(u) = 2T(u)/(deg(u) * (deg(u) - 1))
#   - cc(u) : 정점 u의 군집계수
#   - T(u)  : 정점 u가 들어있는 삼각형 개수
#   - deg(u): 정점 u의 차수 (degree)
#
# 그리고 전역 군집 계수는 모든 node의 cc(u)의 평균을 의미합니다.
# 전역 군집 계수
# avg_cc(G) = sigma(u in G) cc(u) / n
#   - avg_cc(G) : 그래프 G의 전역 군집 계수
#   - n         : 그래프 G의 정점 개수
#
def getGraphAverageClusteringCoefficient(Graph):
    ccs = [] # 정점 별 지역 군집 계수들을 저장할 list
    for v in Graph.nodes: # graph의 모든 정점을 한번씩 확인
        num_connected_pairs = 0 # 이웃 쌍 중 실제로 이웃인 쌍의 수를 저장할 변수
        for neighbor1 in Graph.neighbors(v):
            for neighbor2 in Graph.neighbors(v):
                if neighbor1 <= neighbor2: # 이미 확인한 이웃 쌍이나, 유효하지 않은 이웃 쌍은 pass
                    continue
                if Graph.has_edge(neighbor1, neighbor2): # 이웃 쌍 사이에 간선이 존재하는지 확인
                    num_connected_pairs = num_connected_pairs + 1
        cc = num_connected_pairs / (Graph.degree(v) * (Graph.degree(v) - 1) / 2) # 지역 군집 계수 확인, 분모는 "d_v C 2" -> 조합 이용하여 이웃 쌍의 수 계산
        ccs.append(cc) # 지역 군집 계수 저장
    return sum(ccs) / len(ccs) # 전역 군집 계수 계산하고 리턴
```


```python
# 본 실습에서는 그래프의 다양한 특성 중 그래프 지름과 전역 군집 계수를 분석해봅니다.
# 그래프에서 Diameter, Average Clustering Coefficient를 찾는 알고리즘을 구현하고, networkx에서 제공하는 라이브러리와 결과를 비교해봅시다.

# 그래프의 지름을 찾는 함수입니다.
# Definition. Graph Diameter
#   The graph diameter of a graph is the length max(u,v)d(u,v) of the "longest shortest path between any two graph vertices (u,v), where d(u,v) is a graph distance.
#
def getGraphDiameter(Graph):
    diameter = 0                                                      # 알고리즘을 시작하기 앞서 diameter 값을 0으로 초기화합니다.
    for v in Graph.nodes:                                             # 그래프의 모든 점점들 대해서 아래와 같은 반복문을 수행합니다.
        length = nx.single_source_shortest_path_length(Graph, v)      #   1. 정점 v로 부터 다른 모든 정점으로 shortest path length를 찾습니다. 
        max_length = max(length.values())                             #   2. 그리고 shortest path length 중 최댓값을 구합니다.
        if max_length > diameter:                                     #   3. 2에서 구한 값이 diameter보다 크다면 diameter를 그 값으로 업데이트 합니다.
            diameter = max_length
    return diameter                                                   # 반복문을 돌고 나온 diameter를 return합니다.
```


```python
# 아래는 위의 함수로 구한 그래프 지름 및 전역 군집 계수 값과 networkX에서 지원하는 library로 구한 값을 비교해봅니다.
#
#                   |     그래프 지름        |     전역 군집 계수
# ------------------+------------------------------------------------------------                    
# Regular Graph     |         High           |              High
# Small World Graph |         Low            |              High
# Random Graph      |         Low            |              Low
#
print("1. Graph Diameter")
print("cycle graph : " + str(nx.diameter(cycle_graph)))
print("cycle graph : " + str(getGraphDiameter(cycle_graph)))

print("regular graph : " + str(nx.diameter(regular_graph)))
print("regular graph : " + str(getGraphDiameter(regular_graph)))

print("small world graph : " + str(nx.diameter(small_world_graph)))
print("small world graph : " + str(getGraphDiameter(small_world_graph)))

print("random graph : " + str(nx.diameter(random_graph)))
print("random graph : " + str(getGraphDiameter(random_graph)) + "\n")

print("2. Average Clustering Coefficient")
print("cycle graph : " + str(nx.average_clustering(cycle_graph)))
print("cycle graph : " + str(getGraphAverageClusteringCoefficient(cycle_graph)))
print("regular graph : " + str(nx.average_clustering(regular_graph)))
print("regular graph : " + str(getGraphAverageClusteringCoefficient(regular_graph)))

print("small world graph : " + str(nx.average_clustering(small_world_graph)))
print("small world graph : " + str(getGraphAverageClusteringCoefficient(small_world_graph)))


print("random graph : " + str(nx.average_clustering(random_graph)))
print("random graph : " + str(getGraphAverageClusteringCoefficient(random_graph)) + "\n")
```

    1. Graph Diameter
    cycle graph : 15
    cycle graph : 15
    regular graph : 8
    regular graph : 8
    small world graph : 6
    small world graph : 6
    random graph : 5
    random graph : 5
    
    2. Average Clustering Coefficient
    cycle graph : 0.0
    cycle graph : 0.0
    regular graph : 0.5
    regular graph : 0.5
    small world graph : 0.42555555555555563
    small world graph : 0.42555555555555563
    random graph : 0.027777777777777776
    random graph : 0.027777777777777776
    



```python
# 그래프의 차수 분포을 그리는 부분입니다.
print("3. Degree Distribution")
degree_sequence = sorted([d for n, d in random_graph.degree()], reverse = True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.bar(deg, cnt, color="b")
plt.xlabel('degree')
plt.ylabel('number of vertices')
plt.xticks([2, 3, 4])
plt.show()

```

    3. Degree Distribution



    
![png](output_7_1.png)
    



```python

```
