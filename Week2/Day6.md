# Day6 - Numpy / 벡터 / 행렬

- [Day6 - Numpy / 벡터 / 행렬](#day6---numpy--벡터--행렬)
  - [Numpy](#numpy)
    - [ndarray](#ndarray)
    - [Handling Shape](#handling-shape)
    - [Indexing, Slicing](#indexing-slicing)
    - [Creation Functions](#creation-functions)
    - [Operation Functions](#operation-functions)
    - [Array Operations](#array-operations)
    - [Comparisons](#comparisons)
    - [Boolean index](#boolean-index)
    - [Fancy index](#fancy-index)
    - [Numpy Data I/O](#numpy-data-io)
  - [벡터](#벡터)
    - [벡터란?](#벡터란)
    - [벡터의 연산](#벡터의-연산)
    - [벡터의 노름(norm)](#벡터의-노름norm)
    - [두 벡터 사이의 거리 구하기](#두-벡터-사이의-거리-구하기)
    - [두 벡터 사이의 각도 구하기](#두-벡터-사이의-각도-구하기)
    - [내적의 해석](#내적의-해석)
  - [행렬](#행렬)
    - [행렬이란?](#행렬이란)
    - [행렬의 연산](#행렬의-연산)
    - [행렬을 이해하는 두가지 방법](#행렬을-이해하는-두가지-방법)
    - [역행렬](#역행렬)
  - [Quiz Review](#quiz-review)
    - [벡터 quiz](#벡터-quiz)
    - [행렬 quiz](#행렬-quiz)
  - [TODO](#todo)

## Numpy

* 행렬을 코드로 표현하는 방법
  * list 사용 
    * 다양한 matrix 연산을 직접 구현해야함
    * 굉장히 큰 matrix 표현 시 메모리 효율적이지 않음
    * 처리 속도가 느림
  * 다른 패키지 사용
    * 위의 문제점을 해결
* 파이썬의 과학 처리 패키지 Numpy
  * 파이썬의 고성능 과학 계산용 패키지
  * 일반 list에 비해 빠르고, 메모리 효율적임
  * 반복문 없이 데이터 배열에 대한 처리를 지원함
  * 선형대수와 관련된 다양한 기능을 제공함

### ndarray

* array() 함수를 사용해 배열을 생성함
* 여기에는 list와 달리 한가지 data type만 넣을 수 있음
* 또한 dynamic typing을 지원하지 않음
* C언어의 배열을 사용해 배열을 생성함
  * C언어의 배열은 연속된 메모리 공간에 element들을 저장함
  * 파이썬의 list는 element들의 주소값만 갖고있고, 주소값을 이용해 element에 접근함
  * 따라서 C언어의 배열을 사용하면 연산 속도가 증가함
* ndarray를 생성할 때 data의 type을 지정해 줄 수 있음

    ```Python
    import numpy as np

    ndarray = np.array([1, 3, 5, 7, 9])
    print(ndarray) # [1 3 5 7 9]
    ndarray = np.array([1, 3, 5, 7, 9], np.float64)
    print(ndarray) # [1. 3. 5. 7. 9.]
    ndarray = np.array([1, 3.2, "5", "7.24", 9], np.float64)
    print(ndarray) # [1.   3.2  5.   7.24 9.  ]
    print(ndarray.shape) # (5,) - dimension 구성
    print(ndarray.dtype) # float64 - element의 data type
    ```

* array의 Rank에 따른 이름들

    |Rank|Name|Example|
    |-|-|-|
    |0|scalar|7|
    |1|vector|[10, 10]|
    |2|matrix|[[10, 10], [15, 15]]|
    |3|3-tensor|[[[ 1, 5, 9], [ 2, 6, 10]], [[ 3, 7, 11], [ 4, 8, 12]]]|
    |n|n-tensor||

* Array Shape, dtype 등
  
    ```Python
    import numpy as np

    ndarray = np.array([1, 3, 5])
    print(ndarray.shape) # (3,)
    ndarray = np.array([1, 3, 5, 7, 9])
    print(ndarray.shape) # (5,)
    print(ndarray.ndim) # 1 - array의 차원
    print(ndarray.size) # 5 - data의 개수
    print(ndarray.dtype) # int64 - single element의 data type
    print(ndarray.nbytes) # 40 - ndarray object의 메모리 크기 (8byte * 10 = 40byte)


    ndarray = np.array([[1, 3, 5, 7, 9], [1, 3, 5, 7, 9]])
    print(ndarray.shape) # (2, 5)
    print(ndarray.ndim) # 2
    print(ndarray.size) # 10


    ndarray = np.array([
        [[1, 3, 5, 7, 9], [1, 3, 5, 7, 9]],
        [[1, 3, 5, 7, 9], [1, 3, 5, 7, 9]],
        [[1, 3, 5, 7, 9], [1, 3, 5, 7, 9]]
        ])
    print(ndarray.shape) # (3, 2, 5)
    print(ndarray.ndim) # 3
    print(ndarray.size) # 30
    ```

### Handling Shape

* array의 shape을 변경하는 ```reshape()``` 함수. element의 개수는 동일하게 유지됨

    ```Python
    import numpy as np

    matrix = [[2, 3, 4, 5], [6, 7, 8, 9]]
    print(np.array(matrix)) # [[2 3 4 5]
                            # [6 7 8 9]]
    print(np.array(matrix).shape) # (2, 4)

    # reshape()
    print(np.array(matrix).reshape(8,)) # [2 3 4 5 6 7 8 9]
    print(np.array(matrix).reshape(8,).shape) # (8,)

    print(np.array(matrix).reshape(-1,2)) # [[2 3]
                                          # [4 5]
                                          # [6 7]
                                          # [8 9]] -> -1: size를 기반으로 행/열의 개수 선정
    print(np.array(matrix).reshape(-1,2).shape) # (4, 2)

    print(np.array(matrix).reshape(2, -1, 2)) # [[[2 3]
                                              #  [4 5]]
                                              #
                                              # [[6 7]
                                              #  [8 9]]]
    print(np.array(matrix).reshape(2, -1, 2).shape) # (2, 2, 2)

    # flatten()
    print(np.array(matrix).flatten()) # [2 3 4 5 6 7 8 9]
    print(np.array(matrix).flatten().shape) # (8,)
    ```

### Indexing, Slicing

* Indexing
  * 리스트와 달리 콤마(```,```) 를 사용하여 ```[x, y]```의 형태로 indexing 가능
    * 여기서 matrix의 경우 앞은 row, 뒤는 column을 의미함

* Slicing
  * list와 달리 행과 열 부분을 나눠서 slicing이 가능함
  * matrix의 부분 집합을 추출할 때 유용함

  ```Python
  a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], int) 
  print(a[:,2:]) # 전체 Row의 2열 이상
  print()
  print(a[1,1:3]) # 1 Row의 1열 ~ 2열
  print()
  print(a[1:3]) # 1 Row ~ 2Row의 전체
  ```

  위 코드의 실행 결과는 아래와 같다

  ```
  [[ 3  4  5]
  [ 8  9 10]]

  [7 8]

  [[ 6  7  8  9 10]]
  ```

  ```Python
  a = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], int) 
  print(a) # 전체 array
  print()
  print(a[:, 1:3]) # 전체 Row에서 1, 2열
  print()
  print(a[1, :2]) # 1행에서 0, 1 열
  ```

  위 코드의 실행 결과는 아래와 같다

  ```
    [[1 2 3 4]
    [1 2 3 4]
    [1 2 3 4]
    [1 2 3 4]]

    [[2 3]
    [2 3]
    [2 3]
    [2 3]]

    [1 2]
  ```

  **콤마(```,```)를 기준으로 앞은 열(Row), 뒤는 행(Column)의 slicing 을 나타낸다.**

### Creation Functions

* ```arange()```를 사용하여 list를 생성한다
  * list를 생성할 때 사용하는 range()와 유사한 함수이다
  * 주로 ```reshape()```과 함께 사용한다

  ```Python
  a = np.arange(30)
  print(a)
  print()

  a = np.arange(30).reshape(5, 6)
  print(a)
  ```

  위 코드의 실행 결과는 아래와 같다

  ```
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29]

    [[ 0  1  2  3  4  5]
    [ 6  7  8  9 10 11]
    [12 13 14 15 16 17]
    [18 19 20 21 22 23]
    [24 25 26 27 28 29]]
  ```

* ```ones(shape, dtype, order)```

  ```Python
  print(np.ones(shape = (4, ), dtype = float)) # [1. 1. 1. 1.]
  ```

* ```zeros(shape, dtype, order)```

  ```Python
  print(np.zeros(shape = (2, 4), dtype = int)) # [[0 0 0 0]
                                               # [0 0 0 0]]
  ```

* ```empty(shape, dtype, order)```
  * empty로 초기화 한 array에는 어떤 값이 들어있을지 알 수 없다. 쓰레기 값으로 초기화됨.

  ```Python
  print(np.empty(shape = (4, ))) # [1. 1. 1. 1.] -> 쓰레기값임
  ```

* ```something_like()```
  * 기존 ndarrya의 shape 형태의 array를 생성한다

  ```Python
  a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
  print(a)
  print()

  print(np.ones_like(a))
  print()

  print(np.zeros_like(a))
  print()

  print(np.empty_like(a))
  print()
  ```

  위 코드의 실행 결과는 아래와 같다

  ```
  [[1 2 3 4]
  [5 6 7 8]]

  [[1 1 1 1]
  [1 1 1 1]]

  [[0 0 0 0]
  [0 0 0 0]]

  [[0 0 0 0] # 쓰레기 값
  [0 0 0 0]]
  ```

* ```identity()```
  * 단위 행렬을 생성한다

  ```Python
  print(np.identity(n = 3, dtype = float))
  ```

  위 코드의 실행 결과는 아래와 같다

  ```
  [[1. 0. 0.]
  [0. 1. 0.]
  [0. 0. 1.]]
  ```

* ```eye()```
    * 대각선이 1인 행렬, 대각선 시작 index를 변경할 수 있음 (매개변수 k)

* ```diag()```
  * 대각행렬의 값을 추출함, 추출을 시작할 열을 변경할 수 있음 (매개변수 k)

* random sampling
  * 데이터 분포에 따른 sampling으로 array 생성
  * 균등분포는 ```np.random.uniform(시작 값, 끝 값, data 개수)```
  * 정규분포는 ```np.random.normal(시작 값, 끝 값, data 개수)```

### Operation Functions

* axis
  * 모든 operation function을 실행할 때 기준이 되는 dimension 축
  * 가장 늦게 생성된 축이 0이 된다

* sum()
  * ndarray의 element들 간의 합을 구함. dtype을 지정하여 결과 값으 data type을 설정할 수 있음

* mean()
  * ndarray의 element들 간의 평균을 반환, axis를 이용하면 특정 행 or 열의 평균을 구할 수 있다.

* std()
  * ndarray의 element들 간의 표준편차를 반환, axis를 이용하면 특정 행 or 열의 표준편차를 구할 수 있다.

* 그 외에도 다양한 수학 연산자를 제공함

* ```concatenate()```, ```vstack()```, ```hstack()```
  * ndarray를 합치는 함수.
  * ```vstack()```은 수직 뱡향으로 붙이고, ```hstack()```은 수평 방향으로 붙인다
  * ```concatenate()``` 함수에서는 axis를 이용해 붙일 방향을 설정한다

### Array Operations

* Element-wise Operation
  * 같은 위치에 있는 element간에 사칙 연산이 일어남

* Dot product
  * 두 Matrix를 곱하는 연산. ```dot()``` 함수를 사용함

* Transpose
  * 전치행렬
  * ```transpose()``` 함수 또는 ```T``` attribute를 사용

* broadcasting
  * shape이 다른 배열 간 연산을 지원하는 기능
  * 다른 배열의 shape에 맞춰 배열을 자동으로 확장시킨다

### Comparisons

* ```all(조건)```
  * array의 데이터가 전부 조건에 만족하는지 확인

* ```any(조건)```
  * array의 데이터 일부가 조건에 만족하는지 확인

* 배열간 비교 연산
  * numpy는 배열의 크기가 동일할 때 element간 비교의 결과를 Boolean type의 array로 반환함

* logical_and(), logical_not(), logical_or()
  * 배열의 각 element에 연산들을 적용함
  
* where()
  * 두가지 사용 방법이 있음
    * where(조건, True일 때 값, False일 때 값)
      * 해당 원소가 조건에 대해 True이면 True일 때의 값을, False이면 False일 때의 값을 array에 넣어 리턴
    * where(조건)
      * 조건을 만족하는 element들의 index를 array에 넣어 리턴

* argmax(), argmin()
  * array 내 최댓값 또는 최솟값의 index를 반환
  * axis를 기반으로 행 또는 열에서의 최댓값, 최솟값을 구할 수도 있음

### Boolean index

* 특정 조건에 따른 값을 배열 형태로 추출
* **추출된 boolean array를 index로 사용한다**
* Comparison operation 함수들도 모두 사용 가능
* 반드시 boolean index와 array의 shape이 같아야 함
  
  ```Python
  a = np.arange(10)
  print(a > 7) # [False False False False False False False False  True  True]
  print(a[a > 7]) # [8 9]
  ```

### Fancy index

* array를 index value로 사용해서 값 추출
* fancy index의 원소들을 index로 사용
* fancy index의 모든 element들을 array의 index 범위 내의 값이어야 함

  ```Python
  a = np.arange(10, 1, -1)
  print(a) # [10  9  8  7  6  5  4  3  2]
  b = np.array([2, 4, 6, 8])
  print(a[b]) # [8 6 4 2]
  ```

* matrix 형태의 데이터도 가능

  ```Python
  a = np.array([[1, 2], [3, 4]])
  print(a) # [[1 2]
           # [3 4]]
  b = np.array([0, 0, 1, 1, 0], int)
  c = np.array([0, 1, 1, 1, 1], int)
  print(a[b, c]) # [1 2 4 4 2]
  ```

### Numpy Data I/O

* loadtxt(), savetxt()
  * text 파일 형태로 I/O
* save(), load()
  * numpy 객체를 pickle 형식으로 I/O

## 벡터

### 벡터란?
* 벡터는 숫자를 원소로 가지는 list 또는 array
* 가로로 값들이 나열된 벡터를 **행벡터** 라고 함
* 세로로 값들이 나열된 벡터를 **열벡터** 라고 함
* 벡터는 공간에서 한 점을 나타냄
* 벡터는 원점으로부터 상대적 위치를 표현함(원점을 기준으로 함)
### 벡터의 연산
* 벡터에 숫자를 곱하면(스칼라 곱) 길이만 변함. 단, 0보다 작으면 반대 방향이 됨.
* 벡터끼리 같은 모양을 가지면 덧셈, 뺄셈을 할 수 있음(같은 위치의 값끼리 더하고 뻄).
* 같은 모양의 벡터끼리는 성분곱(element-wise product)을 할 수 있음
  
  벡터의 연산은 파이썬 코드로 다음과 같이 나타낼 수 있다

  ```Python
  x = np.array([1, 7, 2])
  y = np.array([5, 2, 1])
  
  print(x + y) # [6 9 3]
  print(x - y) # [-4  5  1]
  print(x * y) # [ 5 14  2]
  ```

* 벡터의 덧셈은 다른 벡터로부터 상대적 위치 이동을 표현함
* 벡터의 뺄셈은 방향을 반대로 뒤집은(-1 곱한) 벡터로 덧셈을 하는 것

### 벡터의 노름(norm)
* 벡터의 노름(norm)은 **원점과의 거리**를 말함
* 벡터의 노름은 여러 종류가 있음. 우리는 두가지 노름을 배웠음.
  * l1 노름
    * l1 노름은 **각 좌표축을 따라 이동한 거리**를 나타냄
    * 각 성분의 변화량의 절대값을 모두 더하면 구할 수 있음
  * l2 노름
    * l2 노름은 **원점에서 벡터로 일직선으로 가는 거리**를 나타냄
    * 피타고라스 정리를 이용해 유클리드 거리를 계산하면 구할 수 있음

  벡터의 노름을 구하는 수식은 파이썬 코드로 다음과 같이 나타낼 수 있음

  ```Python
  def l1_norm(x): # x는 ndarray
      x_norm = np.abs(x)
      x_norm = np.sum(x_norm)
      return x_norm

  def l2_norm(x):
      x_norm = x * x
      x_norm = np.sum(x_norm)
      x_norm = np.sqrt(x_norm)
      return x_norm
  ```

* 노름의 종류에 따라 기하학적 성질이 달라짐
* 노름의 종류에 따라 원의 모양이 달라짐
  * l1 노름의 원은 마름모 모양 (l1 노름이 각 좌표축을 따라 이동한 거리이기 때문)
  * l2 노름의 원은 우리가 아는 모양의 원
* 머신러닝에선 각 성질들이 필요할 때가 있으므로 둘 다 사용함


### 두 벡터 사이의 거리 구하기
* l1, l2 노름을 이용해 두 벡터 사이의 거리를 계산할 수 있음
* 구하는 방법은 노름의 종류에 따라 달리짐
* 두 벡터 사이의 거리를 계산할 때는 벡터의 뺄셈을 이용함
  * 두 벡터를 뺀 결과로 나오는 벡터와 두 벡터 사이의 거리를 나타내는 벡터가 동일함
  * 두 벡터를 뺼셈해서 구한 벡터의 길이를 구하면 됨
  * 뺄셈을 거꾸로 해도 거리는 같음

### 두 벡터 사이의 각도 구하기
* 각도는 l2 노름에서만 계산할 수 있다
* 제 2 코사인 법칙에 의해 두 벡터 사이의 각도를 계산할 수 있다
* 제 2 코사인 법칙의 분자는 내적을 이용해 쉽게 계산할 수 있다 (np.inner())

  파이썬 코드로는 다음과 같이 표현할 수 있음

  ```Python
  def angle(x, y):
    v = np.inner(x, y) / (l2_norm(x) * l2_norm(y)) # v에 cosΘ 값이 들어감
    theta = np.arccos(v)
    return theta
  ```

  코사인 제 2법칙 증명과, 그 의미는 [여기](http://blog.naver.com/dalsapcho/20130975163)를 참고

### 내적의 해석
* 내적은 정사영된 벡터의 길이와 관련이 있다
* 두 벡터 $x$, $y$가 있을 때 이 두 벡터의 내적은 벡터 $x$를 벡터 $y$에 정사영 시킨 뒤($Proj(x)$), 벡터 $y$의 길이($||y||$)만큼 곱한 값이다.
* 이는 정사영된 벡터의 길이를 $y$벡터의 크기에 맞춰 조정한 것을 의미
* **내적은 두 벡터의 유사도를 측정하는데 사용된다 (두 data가 얼마나 유사한지 측정하는데 사용)**

## 행렬

### 행렬이란?
* 행렬(matrix)은 벡터를 원소로 가지는 2차원 배열이다
* numpy에선 행벡터가 행렬의 기본 단위이다
* 행렬은 행(row)과 열(column)이라는 인덱스를 갖는다
* 행렬의 특정 행(열)을 고정하면 행(열)벡터라고 부른다

### 행렬의 연산
* 전치행렬은 행과 열의 인덱스가 바뀐 행렬을 말한다 (대각선을 기준으로 뒤집음)
* 행렬끼리 같은 모양을 가지면 덧셈, 뺄셈을 할 수 있다
* 성분곱은 벡터와 동일하다(각 인덱스 위치끼리 곱한다)
* 스칼라곱도 백터와 동일하다(모든 성분에 똑같이 숫자를 곱해준다)
* **행렬 곱셈은 i번째 행벡터와 j번째 열벡터 사이의 내적을 성분으로 가지는 행렬을 계산한다**
  * $X$행렬과 $Y$행렬을 곱할 때 $X$행렬의 열의 개수와 $Y$행렬의 행의 개수가 같아야 함
  * numpy에선 ```@``` 연산자와 ```np.inner()``` 함수를 사용해 계산할 수 있다
  * numpy의 ```np.inner()```는 i번째 **행벡터**와 j번째 **행벡터** 사이의 내적을 성분으로 가지는 행렬을 계산한다
    * 수학에서 말하는 내적은 **행백터**와 **열백터** 사이의 내적이다
    * numpy의 ```np.inner()```는 **행백터**와 **행백터**의 내적이다
    * **두개가 서로 다름을 주의하여 사용해야 한다**

  ```Python
  X = np.array([[1, -2, 3],
                [7, 5, 0],
                [-2, -1, 2]])

  Y = np.array([[0, 1, -1],
                [1, -1, 0]])

  print(X @ Y.T) # 행벡터와 열벡터의 내적
  print()
  print(np.inner(X, Y)) # 행벡터와 행벡터의 내적
  ```

  위 코드의 실행 결과는 다음과 같다

  ```
  [[-5  3]
   [ 5  2]
   [-3 -1]]

  [[-5  3]
   [ 5  2]
   [-3 -1]]
  ```
  
### 행렬을 이해하는 두가지 방법
  * 첫번쨰는 행렬을 여러 점의 모임으로 보는 방법이다
    * 벡터가 공간에서 한 점을 의미한다면 행렬은 여러 점을 나타낸다
    * 행렬의 행벡터 $x_i$는 $i$번째 데이터를 의미한다
    * 행렬의 $x_{i, j}$는 $i$번째 데이터의 $j$번째 변수 값을 의미한다
  * 두번쨰는 행렬을 벡터공간에서 사용되는 연산자로 이해하는 것이다 (학교 선형대수 시간에 배웠던 개념)
    * 행렬곱을 통해 벡터를 다른 차원의 공간으로 보낼 수 있다
    * 서로 다른 차원의 두 벡터 $x$와 $z$를 행렬 $A$로 연결시킨다
    * 이를 이용해 패턴을 추출하거나 데이터를 압축한다 (CNN)
    * 모든 선형변환은 행렬곱으로 계산할 수 있다

### 역행렬
* 어떤 행렬 $A$의 연산을 거꾸로 되돌리는 행렬을 역행렬이라 부르고 $A^{-1}$로 표기한다
* 역행렬은 다음 조건을 모두 만족할 때에만 존재한다
  * 1. 행과 열의 숫자가 같아야 한다
  * 2. 행렬식 (determinant)가 0이 아니어야 한다
* $AA^{-1} = A^{-1}A = I$
  
  ```Python
  X = np.array([[1, -2, 3],
                [7, 5, 0],
                [-2, -1, 2]])

  print(np.linalg.inv(X)) # 역행렬을 구한다
  print()
  print(np.inner(X, np.linalg.inv(X).T)) # 역행렬과 원래 행렬을 곱한다
  ```

  위 코드의 실행 결과는 다음과 같다

  ```
  [[ 0.21276596  0.0212766  -0.31914894]
   [-0.29787234  0.17021277  0.44680851]
   [ 0.06382979  0.10638298  0.40425532]]
  
  [[ 1.00000000e+00 -1.38777878e-17  0.00000000e+00]
   [ 0.00000000e+00  1.00000000e+00  0.00000000e+00]
   [-2.77555756e-17  0.00000000e+00  1.00000000e+00]] # 부동소수점 오차때문에 0이 정확히 나오진 않고 0에 근사한 값이 나온다
  ```

* 역행렬을 구할 수 없는 경우엔 **유사역행렬**(pseudo-inverse, moore-penrose inverse)을 이용할 수 있다
  * 유사 역행렬은 행렬의 행과 열의 개수에 따라 구하는 식이 달라진다
  * $n\times m$ 행렬에서 $n ≥ m$ 인 경우 (행의 개수가 열의 개수보다 많은 경우)
    * $A^+ = (A^TA)^{-1}A^T$
  * $n\times m$ 행렬에서 $n ≤ m$ 인 경우 (열의 개수가 행의 개수보다 많은 경우)
    * $A^+ = A^T(AA^T)^{-1}$
  ```Python
  Y = np.array([[0, 1, -1],
                [1, -1, 0]])

  print(np.linalg.pinv(Y)) # 유사역행렬을 구한다
  print()
  print(Y @ np.linalg.pinv(Y)) # 유사역행렬과 원래 행렬을 곱한다, 이 때는 역행렬과는 다르게 곱하는 순서를 고려해야 한다
  ```

  위 코드의 실행 결과는 아래와 같다

  ```
  [[ 0.33333333  0.66666667]
   [ 0.33333333 -0.33333333]
   [-0.66666667 -0.33333333]]
  
  [[ 1.00000000e+00 -3.88578059e-16]
   [ 1.11022302e-16  1.00000000e+00]]
  ```

* 유사 역행렬 응용 1 : 연립방정식 풀기
  * 연립방정식을 풀 때, 식의 개수와 변수의 개수가 같아야 해를 구할 수 있음
  * 변수의 개수 > 식의 개수이면 여러개의 해가 있을 수 있다
  * 이 상황에서 유사 역행렬을 사용하면 여러개의 해들 중 하나를 구할 수 있다

* 유사 역행렬 응용 2 : 선형회귀분석
  * 변수의 개수 < 식의 개수인 경우엔 해를 구할 수 없다
    * 데이터가 모두 선형 식에 올라와있지 않기 때문
    * 이 때는 선형회귀분석을 통해 데이터를 최대한 잘 표현할 수 있는 선형회귀식을 찾아야 한다
  * $X\beta = y$
  * 선형회귀분석은 $X$와 $y$가 주어진 상황에서 계수 $\beta$를 찾아야 한다
  * np.linalg.pinv()를 이용해 데이터를 선형모델로 해석하는 선형회귀식을 찾는다
  * $\beta = X^+y = (X^TX)^{-1}X^Ty$
  
## Quiz Review

### 벡터 quiz

1. l1 벡터의 노름을 구하는 문제 
   * 각 좌표축을 따라 이동한 거리를 구한다
   * 각 요소의 값에 절대값을 취한뒤 모두 더한다
2. l2 벡터의 노름을 구하는 문제
   * 원점에서 벡터로 일직선으로 가는 거리를 구한다
   * 피타고라스의 정리를 이용해 거리를 구한다 (제곱해 더한뒤 루트 취함)
3. 두 벡터 사이의 거리를 구하는 문제
   * 두 벡터의 뺄셈을 이용한다
   * 두 벡터를 뺴서 나온 벡터의 노름을 구한다
4. 두 벡터 사이의 각도를 구하는 문제
   * 제2 코사인 법칙을 이용한다
   * 하지만 이 문제에서는 두 벡터의 방향이 일치하므로 그냥 플었다
5. 두 벡터의 내적을 구하는 문제
   * 문제에 나와있는 수식에 노름을 계산한 값을 넣고, 코사인 값을 넣어 풀었다

### 행렬 quiz

1. 전치행렬을 구하는 문제
   * 행렬 뒤집어서 전치행렬을 구했다
2. 두 행렬의 곱셈은 각 행렬의 모양과 상관없이 가능한가?
   * 성분곱(element-wise 곱)의 경우 두 행렬의 모양 일치해야 함
   * 행렬곱의 경우 앞쪽 행렬의 열의 개수와 뒤쪽 행렬의 행의 개수가 같아야 함
3. 행렬의 역행렬은 항상 계산할 수 있는가?
   * 행과 열의 개수가 일치하고, determinant가 0이 아닌 경우에만 계산 가능
4. 역행렬을 구하는 문제
   * [여기](https://terms.naver.com/entry.nhn?docId=3350377&cid=60210&categoryId=60210)에 있는 역행렬 공식을 이용했다
5. 유사역행렬을 고르는 문제
   * 곱했을 때 단위행렬이 나오는 행렬을 골랐다

## TODO

* 선형대수 수업때 정리해둔 자료에서 오늘 수업때 나왔던 부분들 찾아서 확인해보기