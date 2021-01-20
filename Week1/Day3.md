
# Day 3 - 파이썬 기초 문법 2

## 목차

- [Day 3 - 파이썬 기초 문법 2](#day-3---파이썬-기초-문법-2)
  - [목차](#목차)
  - [Python Data Structure](#python-data-structure)
    - [파이썬 기본 자료 구조](#파이썬-기본-자료-구조)
    - [스택 (Stack)](#스택-stack)
    - [큐 (Queue)](#큐-queue)
    - [튜플 (Tuple)](#튜플-tuple)
    - [집합 (Set)](#집합-set)
    - [사전 (Dictionary)](#사전-dictionary)
    - [Collections 모듈](#collections-모듈)
    - [Deque](#deque)
    - [OrderedDict](#ordereddict)
    - [DefaultDict](#defaultdict)
    - [Counter](#counter)
    - [Namedtuple](#namedtuple)
  - [Pythonic Code](#pythonic-code)
    - [Pythonic Code 목차](#pythonic-code-목차)
    - [split 함수](#split-함수)
    - [join 함수](#join-함수)
    - [list comprehension](#list-comprehension)
    - [enumerate](#enumerate)
    - [zip](#zip)
    - [lambda](#lambda)
    - [map](#map)
    - [reduce](#reduce)
    - [iterable object](#iterable-object)
    - [generator](#generator)
    - [function passing arguments](#function-passing-arguments)
    - [asterisk](#asterisk)

## Python Data Structure

### 파이썬 기본 자료 구조

* 스택과 큐 (Stack & Queue)
* 튜플 (Tuple)
* 집합 (Set)
* 사전 (Dictionary)
* Collection 모듈

### 스택 (Stack)

* list를 사용하여 스택 구조 구현
* push는 ```append()```를, pop은 ```pop()```을 이용
  
    ```Python
    a = [1,2,3,4,5]

    a.append(10)
    a.append(20)
    print(a) # [1, 2, 3, 4, 5, 10, 20]

    top = a.pop()
    print(top) # 20
    top = a.pop()
    print(top) # 10
    print(a) # [1, 2, 3, 4, 5]
    ```

    여기에서 ```pop()```함수는 다른 list 함수와 달리 값을 리턴함과 동시에 list의 내용도 수정하는 함수임을 유의하자

### 큐 (Queue)

* list를 사용하여 큐 구조 구현
* push는 ```append()```를, pop은 ```pop(0)```을 이용  
  (```pop(x)```는 list의 x번째 요소를 리턴하고, 삭제하는 함수)

    ```Python
    a = [1,2,3,4,5]

    a.append(10)
    a.append(20)
    print(a) # [1, 2, 3, 4, 5, 10, 20]

    front = a.pop(0)
    print(front) # 1
    front = a.pop(0)
    print(front) # 2
    print(a) # [3, 4, 5, 10, 20]
    ```

### 튜플 (Tuple)

* 튜플은 값의 변경이 불가능한 list
* 선언 시 ```[]```가 아닌 ```()```를 사용
  * 튜플을 초기화 할 때 ```t = (0)``` 이렇게 하면 괄호로 인식함. ```t = (0,)``` 이런 식으로 초기화 한다
* list의 연산, 인덱싱, 슬라이싱 등을 동일하게 사용 (값 변경만 안됨)
  
### 집합 (Set)

* 값을 순서없이 저장
* 값의 중복 불허
* ```set``` 객체 선언을 이용하여 객체 생성

    ```Python
    # 두 가지 초기화 방식
    s1 = set([1, 2, 3])
    s2 = {1, 2, 3}
    print(s1) # {1, 2, 3}
    print(s2) # {1, 2, 3}

    s1.add(4)
    print(s1) # {1, 2, 3, 4}

    s1.remove(2)
    print(s1) # {1, 3, 4}

    s1.update([1, 3, 5, 7, 9])
    print(s1) # {1, 3, 4, 5, 7, 9}

    s1.discard(9)
    print(s1) # {1, 3, 4, 5, 7}

    s1.clear()
    print(s1) # set()
    ```

* 수학에서 활용하는 다양한 집합연산 가능

    ```Python
    s1 = set([1, 2, 3, 4, 5])
    s2 = set([1, 3, 5, 7, 9])

    # 합집합
    print(s1.union(s2)) # {1, 2, 3, 4, 5, 7, 9}
    print(s1 | s2) # {1, 2, 3, 4, 5, 7, 9}

    # 교집합
    print(s1.intersection(s2)) # {1, 3, 5}
    print(s1 & s2) # {1, 3, 5}

    # 차집합
    print(s1.difference(s2)) # {2, 4}
    print(s1 - s2) # {2, 4}

    # 위 연산들을 해도 원본 집합의 값은 변하지 않음
    print(s1) # {1, 2, 3, 4, 5}
    print(s2) # {1, 3, 5, 7, 9}
    ```

### 사전 (Dictionary)

* key와 value로 데이터를 저장
* key를 이용해 value를 검색
* ```{Key1:Value1, Key2:Value2, Key3:Value3 ...}``` 형태

    ```Python
    country_code = {} # dict()로도 초기화 가능
    country_code = {"America": 1, "Korea": 82, "China": 86, "Japan": 81}
    print(country_code)
    # {'America': 1, 'Korea': 82, 'China': 86, 'Japan': 81}

    print(country_code.items())
    # dict_items([('America', 1), ('Korea', 82), ('China', 86), ('Japan', 81)])
    # 딕셔너리 내의 요소들을 튜플로 묶고, 그것을 리스트에 담아 리턴. 
    # 주로 Unpacking과 함께 사용한다

    print(country_code.keys())
    # dict_keys(['America', 'Korea', 'China', 'Japan'])
    # 딕셔너리에 있는 모든 키값을 리스트에 담아 리턴

    print(country_code.values())
    # dict_values([1, 82, 86, 81])
    # 딕셔너리에 있는 모든 value 값을 리스트에 담아 리턴

    country_code["German"] = 100 # 딕셔너리에 새로운 요소 추가
    print(country_code)
    # {'America': 1, 'Korea': 82, 'China': 86, 'Japan': 81, 'German': 100}

    country_code["German"] = 49 # 기존 값 변경
    print(country_code)
    # {'America': 1, 'Korea': 82, 'China': 86, 'Japan': 81, 'German': 49}}
    ```

### Collections 모듈

* List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조 (모듈)
* 다음과 같은 모듈이 존재함
  * ```from collections import deque```
  * ```from collections import Counter```
  * ```from collections import OrderedDict```
  * ```from collections import defaultdict```
  * ```from collections import namedtuple```

### Deque

* **링크드 리스트 구조**
* stack과 Queue를 지원하는 모듈
* List보다 빠른 자료 저장 방식을 지원
* ```rotate()```, ```reverse()```등 Linked List의 특성을 지원함

    Deque를 Stack으로 사용하는 예제

    ```Python
    from collections import deque
    a = deque([1,2,3,4,5])

    a.append(10)
    a.append(20)
    print(a) # deque([1, 2, 3, 4, 5, 10, 20])

    top = a.pop()
    print(top) # 20
    top = a.pop()
    print(top) # 10
    print(a) # deque([1, 2, 3, 4, 5, 10, 20])
    ```

    Deque를 Queue로 사용하는 예제  

    ```Python
    from collections import deque
    a = deque([1,2,3,4,5][::-1])

    a.appendleft(10)
    a.appendleft(20)
    print(a) # [1, 2, 3, 4, 5, 10, 20]

    front = a.pop()
    print(front) # 20
    front = a.pop()
    print(front) # 10
    print(a) # [1, 2, 3, 4, 5]
    ```

    기존의 list를 Queue로 사용할 때 와의 차이점

    * ```append()```대신 ```appendleft()``` 사용
    * ```pop(0)```대신 ```pop()``` 사용
    * 기존의 list와 push, pop이 반대 방향으로 일어남

    ```rotate()```, ```reverse()``` 예제

    ```Python
    deque_list = ([1, 2, 3, 4, 5])

    deque_list.rotate(2) 
    print(deque_list) # deque([4, 5, 1, 2, 3])
    deque_list.rotate(2) 
    print(deque_list) # deque([2, 3, 4, 5, 1])
    print(deque(reversed(deque_list))) # deque([1, 5, 4, 3, 2])

    deque_list.extend([5, 6, 7]) 
    print(deque_list) # deque([2, 3, 4, 5, 1, 5, 6, 7])
    deque_list.extendleft([5, 6, 7]) 
    print(deque_list) # deque([7, 6, 5, 2, 3, 4, 5, 1, 5, 6, 7])
    ```

* deque는 기존 list보다 효율적인 자료구조를 제공함 (속도 향상)  
(jupyter notebook에서 속도 측정은 ```%timeit <함수>``` 구문 사용)

### OrderedDict

* 데이터를 입력한 순서대로 dict를 반환한다
* **그러나 dict도 Python 3.6 부터 입력한 순서대로 반환하므로 OrderedDict는 잘 사용하지 않는다**

### DefaultDict

* Dict type의 값에 기본 값을 지정한다

    ```Python
    from collections import defaultdict
    d = defaultdict(lambda: 0) # Default 값을 0으로 설정 

    # dict에 없는 값을 요구하면 설정된 기본 값을 리턴함
    print(d["first"]) # 0
    ```

### Counter

* Sequence type의 동일한 원소들의 개수를 dict 형태로 반환
  
    ```Python
    from collections import Counter
    c = Counter('hello world')
    print(c)
    # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})
    ```

* dict type, keyword parameter 등도 처리 가능
  
    ```Python
    from collections import Counter
    c = Counter({"A": 4, "F": 2})
    print(c) # Counter({'A': 4, 'F': 2})
    
    # elements() 메소드
    print(list(c.elements())) # ['A', 'A', 'A', 'A', 'F', 'F']
    ```

* set의 연산들을 지원함

### Namedtuple

* 구조체와 비슷한 자료구조
* tuple 형태로 data 구조체를 저장
* 저장되는 data의 variable은 사전에 지정해서 저장함

    ```Python
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(x = 11, y = 22)

    print(p[0], p[1]) # 11 22 - 인덱스로도 접근 가능
    print(p.x, p.y) # 11 22

    print(p) # Point(x=11, y=22)
    print(Point(2, 3)) # Point(x=2, y=3)
    ```

## Pythonic Code

* 파이썬 스타일의 코딩 기법
* 파이썬 특유의 문법을 활용하여 효율적으로 코드를 표현함
* 고급 코드를 작성할수록 더 많이 필요해짐
* 다른 파이썬 개발자들의 코드를 이해하려면 필수적으로 알아야 함

### Pythonic Code 목차

* split & join
* list comprehension
* enumerate & zip
* lambda & map & reduce
* iterable object
* generator
* function passing arguments
* asterisk

### split 함수

* string type의 값을 특정 문자열을 기준으로 나눠서 List 형태로 변환
  
    ```Python
    string = "Hello my name is Peter"
    print(string.split()) # split의 인자가 없으면 공백문자(' ')를 기준으로 나눔
    print("python, java, c++".split(",")) # ['python', ' java', ' c++']
    print("abcdhiefghijklnhimop".split("hi")) # ['abcd', 'efg', 'jkln', 'mop']
    ```

### join 함수

* string으로 구성된 list를 합쳐 하나의 string으로 반환

    ```Python
    v = ["Hello", "my", "name", "is", "peter"]
    print("".join(v)) # Hellomynameispeter
    print(" ".join(v)) # Hello my name is peter
    print(", ".join(v)) # Hello, my, name, is, peter
    ```

### list comprehension

* 기존 list 사용하여 간단히 다른 list를 만드는 기법
* 포괄적인 list, 포함되는 리스트라는 의미로 사용됨
* 일반적으로 ```for```문과 ```append()```를 사용한 방법보다 속도가 빠름

    ```Python
    v = [i for i in range(10)]
    print(v) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # if문과 함께 사용
    v = [i for i in range(10) if i % 2 == 0]
    print(v) # [0, 2, 4, 6, 8]

    # 이중 for문
    word1 = "hello"
    word2 = "world"
    v = [i + j for i in word1 for j in word2]
    print(v) 
    # ['hw', 'ho', 'hr', 'hl', 'hd', 'ew', 'eo', 'er', 'el', 'ed', 'lw', 'lo', 'lr', 'll', 'ld', 'lw', 'lo', 'lr', 'll', 'ld', 'ow', 'oo', 'or', 'ol', 'od']

    # 이중 for문과 if문
    s = set(["A", "B", "C"])
    v = [i + j for i in s for j in s if i != j]
    print(v) # ['AC', 'AB', 'CA', 'CB', 'BA', 'BC']

    # 2차원 리스트 생성
    v = [[i + j for j in s if i != j] for i in s]
    print(v.sort()) # [['AC', 'AB'], ['BA', 'BC'], ['CA', 'CB']]

    # if - else 문과 함께 사용
    v = [i + j if i != j else "X" for i in s for j in s ]
    print(v) # ['X', 'AC', 'AB', 'CA', 'X', 'CB', 'BA', 'BC', 'X']


    # if - else문을 이용한 2차원 리스트 생성
    v = [[i + j if i != j else "X" for j in s]  for i in s]
    print(v) # [['X', 'AC', 'AB'], ['CA', 'X', 'CB'], ['BA', 'BC', 'X']]
    ```

### enumerate

* list의 element를 추출할 때 번호를 붙여서 추출
* 주로 unpacking과 함께 사용함
  
    ```Python
    v = ["A", "B", "C", "D", "E"]
    for i, word in enumerate(v):
        print(i, word)
    ```

    ```
    0 A
    1 B
    2 C
    3 D
    4 E
    ```

### zip

* 두개의 list에서 값을 병렬적으로 추출함
  
    ```Python
    v1 = ["a1", "a2", "a3"]
    v2 = ["b1", "b2", "b3"]
    for a, b in zip(v1, v2):
        print(a, b)
    ```

    ```
    a1 b1
    a2 b2
    a3 b3
    ```

### lambda

* 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수
* **Python 3 부터는 사용을 권장하진 않으나** 여전히 많이 쓰임
  * 어려운 문법
  * 테스트의 어려움
  * 문서화 docstirng 지원 미비
  * 코드 해석의 어려움
  * 그래도 많이 씀...

  ```Python
  f = lambda x: x * 2
  print(f(3)) # 6
  print((lambda y: y / 2)(10)) # 5.0
  ```

### map

* list에 각 요소에 함수를 적용
* 두개 이상의 list에도 적용 가능
* if filter도 사용 가능
* ```map()``` 또한 python 3에서는 사용을 권장하지 않음
  
  ```Python
  v = [1, 2, 3, 4, 5]
  f = lambda x: x * 2
  print(list(map(f, v))) # [2, 4, 6, 8, 10]

  v2 = [5, 4, 3, 2, 1]
  f2 = lambda x, y: x + y
  print(list(map(f2, v, v2))) # [6, 6, 6, 6, 6]

  f3 = lambda x: x ** 2 if x % 2 == 0 else x
  print(list(map(f3, v))) # [1, 4, 3, 16, 5]
  ```

### reduce

* list에 함수를 적용해서 통합한다
* ```reduce()```도 사용을 권장하지 않음

  ```Python
  from functools import reduce
  v = range(1, 11)
  f = lambda x, y: x + y
  print(reduce(f, v)) # 55
  ```

### iterable object

* sequence형 자료형에서 순서대로 원소를 추출하는 object
* C++의 iterator와 비슷
  
  ```Python
  city = ["Seoul", "Busan", "Pohang"]
  itr = iter(city)

  print(next(itr)) # Seoul
  print(next(itr)) # Busan
  print(next(itr)) # Pohang
  ```

### generator

* 특수한 형태의 iterable object 생성해주는 함수
  * generator는 iterator보다 훨씬 작은 메모리 용량을 사용한다
* element가 사용되는 시점에 값을 메모리에 반환
* element가 사용되기 전 까지는 주소 값만 갖고있음
* ```yield``` 키워드를 사용해 한번에 하나의 element를 반환하는 generator를 만들 수 있음

  ```Python
  # 이 함수는 generator를 리턴함
  def generator_test():
      yield 1
      yield 2
      yield 3

  generator = generator_test()

  print(type(generator)) # <class 'generator'>

  print(next(generator)) # 1
  print(next(generator)) # 2
  print(next(generator)) # 3
  ```

  yield가 호출되면 암시적으로 ```return```이 호출되고 그 값이 리턴된다.  
  한번 더 실행되면 이전에 실행되었던 ```yield``` 다음 코드가 실행된다.

  ```Python
  # 이 함수는 generator를 리턴함
  def generator_list(value):
      for i in range(value):
          yield i

  print(type(generator_list(5))) # <class 'generator'>

  print(list(generator_list(5))) # [0, 1, 2, 3, 4]
  print(next(generator_list(3))) # 0
  ```

* list comprehension과 유사한 형태로도 generator를 생성할 수 있다  
  (list comprehension과는 달리 ```[]``` 대신 ```()```를 사용하여 표현한다)

  ```Python
  g = (n*n for n in range(500))
  print(type(g)) # <class 'generator'>
  print(next(g)) # 0
  print(next(g)) # 1
  print(next(g)) # 4
  ```

* **list 타입의 데이터를 반환하는 함수는 generator로 만들자**
* **큰 데이터를 처리할 때는 generator expression을 고려하자**
* **파일 데이터를 처리할 때도 generator를 쓰자**

### function passing arguments

함수에 입력되는 argument의 다양한 형태

* Keyword arguments
  * 함수의 parameter 명을 이용해 인자 전달

  ```Python
  def func(arg_1, arg_2):
      print(arg_1, arg_2)

  func("ABC", "DEF") # ABC DEF
  func(arg_2 = "ABC", arg_1 = "DEF") # DEF ABC
  ```

* Default arguments

  * parameter에 기본값을 지정해 매개변수가 전달되지 않은 경우에는 기본 값을 사용

  ```Python
  def introduce(name = "Peter", age = 20):
      print(f"Hello, my name is {name} and I'm {age} years old.")
  
  introduce("David", 25) # Hello, my name is David and I'm 25 years old.

  # default arguments 사용
  introduce("Alice") # Hello, my name is Alice and I'm 20 years old.
  introduce() # Hello, my name is Peter and I'm 20 years old.
  ```

* Variable-length arguments  
  * **개수가 정해지지 않은 변수**를 함수의 parameter로 사용하는 방법
  * asterisk(*) 기호를 사용하여 함수의 parameter를 표시함
  * 입력된 값은 tuple type으로 사용할 수 있음
  * **가변인자는 오직 한개만, 맨 마지막 parameter 위치에 사용 가능**
  * 가변인자는 일반적으로 *args를 변수명으로 사용함
  * 기존의 parameter 이후에 나오는 값들을 tuple로 저장함

  ```Python
  def asterisk_func(a, b, *args):
      return a + b + sum(args)
  
  print(asterisk_func(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) # 55
  ```

  * keyword 가변인자 (keyword variable-length)
  * parameter 이름을 따로 지정하지 않고 입력하는 방법
  * asterisk(*) 두개를 사용하여 함수의 parameter를 표시함
  * 입력된 값은 dict type으로 사용할 수 있음
  * keyword 가변인자는 오직 한개만, 기존 가변인자 다음에 사용
    * 기본 가변인자, 가변인자, keyword 가변인자 순으로 사용

  ```Python
  def kwargs_test_3(one,two, *args, **kwargs): 
    print(one + two + sum(args))
    print(kwargs)

  kwargs_test_3(3,4,5,6,7,8,9, first=3, second=4, third=5)
  ```

  ```
  42
  {'first': 3, 'second': 4, 'third': 5}
  ```

### asterisk

* asterisk는 흔히 알고있는 ```*```를 의미함
* 곱셈, 제곱연산, 가변인자, **unpacking**등에 사용됨
  * tuple, dict 등 자료형에 들어가 있는 값을 unpacking
  * unpacking은 ㅎ마수의 입력값, zip 등에 유용하게 사용 가능
  
  unpacking 예제

  ```Python
  def asterisk_test(a, *args): 
      print(a, args)
      print(type(args))
  
  asterisk_test(1, *(2,3,4,5,6))
  ```

  ```
  1 (2, 3, 4, 5, 6)
  <class 'tuple'>
  ```

  ```Python
  def asterisk_test(a, args):
      print(a, *args)
      print(type(args))
  
  asterisk_test(1, (2,3,4,5,6))
  ```

  ```
  1 2 3 4 5 6
  <class 'tuple'>
  ```

  ```zip()```과 함께 사용

  ```Python
  for data in zip(*([1, 2], [3, 4], [5, 6])):
    print(data)
  ```

  ```
  (1, 3, 5)
  (2, 4, 6)
  ```
