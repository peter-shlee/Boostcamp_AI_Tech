# Lab 3 : PageRank
- Instuctor : Kijung Shin
- Teaching Assistants : Deukryeol Yoon(main), Hyunju Lee, Shinhwan Kang
- 본 실습에서는 PageRank 알고리즘에 대해서 배운다.


```python
# -*- coding: utf-8 -*-

# 실습에 필요한 library를 import하고 그래프를 초기화합니다.
import networkx as nx
import os
import os.path as osp
import numpy as np
import sys
import matplotlib.pyplot as plt
import collections
from google.colab import drive
drive.mount('/content/drive')
np.set_printoptions(threshold=sys.maxsize)


G = nx.DiGraph()
```

    Mounted at /content/drive



```python
# 실습에 필요한 데이터셋을 읽어서 저장합니다.
path_v2n = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/others/vertex2name.txt')) # (문서 식별자, 제목)
path_edges = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/others/edges.txt')) # (나가는 문서 식별자, 들어오는 문서 식별자) -> 방향 있는 간선

# keyword : deep_learning.txt (딥러닝), lee.txt (이순신), bong.txt(봉준호)
path_keyword = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/lab/lab3/deep_learning.txt')) # 키워드에 관련된 문서들의 식별자들 -> 딥러닝 키워드로 검색

f = open(path_edges)
for line in f:
    v1, v2 = map(int, line.split())
    G.add_edge(v1, v2) # 간선 추가

n2v = {} # name to vertex -> 문서의 이름으로 문서 식별자를 찾는 dict
v2n = {} # vertex to name -> 문서 식별자로 문서의 이름을 찾는 dict
f = open(path_v2n)
for line in f: # 문서 관리 dict 생성
    v, n = line.split()
    v = int(v)
    n = n.rstrip()
    n2v[n] = v
    v2n[v] = n

node_key = [] # 검색할 키워드에 관련된 문서들의 식별자를 여기에 저장
f = open(path_keyword)
for line in f:
    v = line.rstrip()
    v = int(v)
    node_key.append(v)
```


```python
# 키워드를 포함한 문서들로 이루어진 부분 그래프(subgraph) H를 추출합니다.
H = G.subgraph(node_key) # 검색할 키워드에 관련된 문서들의 식별자로 subgraph 생성
```


```python
# subgraph H에 대해서 pagerank 알고리즘을 시행합니다.
print("###### PageRank Algorithm ######")
pr = nx.pagerank(H, alpha = 0.9) # 페이지랭크 알고리즘을 시행합니다. alpha는 페이지랭크의 damping parameter를 의미합니다.
res = [key for (key, value) in sorted(pr.items(), key=lambda x:x[1], reverse=True)] # 페이지랭크 알고리즘으로 검색한 결과를 ranking에 따라 sorting하고 출력해줍니다.
for item in res[:10]:
    print(v2n[item]) # rank 높은 item들 출력

```

    ###### PageRank Algorithm ######
    딥러닝
    OpenCV
    이스트소프트
    인공지능인문학
    미분기하학
    PyTorch
    라온피플
    자동긴급제동장치
    케플러-90i
    T2d



```python

```
