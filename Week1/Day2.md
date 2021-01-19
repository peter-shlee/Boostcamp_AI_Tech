# Day 2 - 파이썬 기초 문법

## 목차

- [Day 2 - 파이썬 기초 문법](#day-2---파이썬-기초-문법)
  - [목차](#목차)
  - [변수와 리스트](#변수와-리스트)
    - [변수](#변수)
    - [자료형](#자료형)
    - [연산자](#연산자)
    - [형변환](#형변환)
    - [리스트](#리스트)
  - [함수와 Console I/O](#함수와-console-io)
    - [함수](#함수)
    - [Console I/O](#console-io)
  - [조건문과 반복문](#조건문과-반복문)
    - [조건문](#조건문)
    - [반복문](#반복문)
  - [String and Advanced Function Concept](#string-and-advanced-function-concept)
    - [문자열 (String)](#문자열-string)
    - [함수2 (Advanced Function Concept)](#함수2-advanced-function-concept)

## 변수와 리스트

### 변수

* 변수 이름은 알파벳, 숫자, 언더스코어```_```를 사용해 명명한다
* 변수 이름 첫 자리에는 숫자를 사용할 수 없다
* 파이썬 키워드는 변수 이름으로 사용 불가

### 자료형

* 파이썬에는 다음과 같은 자료형이 있다.

  * 정수형 (```int```)
    * ex) ```1```, ```2```, ```3```, ```100```, ```-9```
  * 실수형 (```float```)
    * ex) ```3.14```, ```3.0```
  * 문자형 (```str```)
    * 따옴표(```'```, ```"```)에 둘러싸인 문자. ex) "hello world"
  * 논리 자료형 (```bool```)
    * ```True```, ```False```
  
  파이썬은 동적 타입 바인딩(Dynamic Type Binding) 언어이기 때문에 런타임에 변수의 자료형이 결정되고, 또한 다른 자료형으로 변경될 수 있다.

* 자료형의 ```True```, ```False```
  
  |자료형|값|
  |-|-|
  |문자열|```""```이면 ```False```|
  |리스트|```[]```이면 ```False```|
  |튜플|```()```이면 ```False```|
  |딕셔너리|```{}```이면 ```False```|
  |숫자형|```0```이면 ```False```|
  ||```None```은 ```False```|

  이 외에는 모두 True

### 연산자

파이썬 연산자의 종류는 [여기](https://www.w3schools.com/python/python_operators.asp)를 참고.

C, Java 등의 언어와 다른 점은 다음과 같다.

* 후위, 전위 증감 연산자(```--```, ```++```)가 없다.
* 정수와 정수를 ```/``` 연산자로 나누면 결과로 실수값이 나온다.
* 몫 연산자 ```//```가 따로 있다.
* 제곱승 연산자 ```**```가 있다.

  ```Python
  3 ** 2　　　# 3의 2승
  ```

### 형변환

* int()
* float()
* str()

실수를 정수로 형변환 하면 소수점 이하는 버린다.  
숫자로만 이루어진 문자열은 ```int()```, ```float()```을 이용하여 수치 자료형으로 형변환 할 수 있다.

### 리스트

* 인덱싱 (indexing)
  
  list에 있는 값들은 주소(offset, index)를 갖고, 주소를 사용해 할당된 값을호출한다.

  ```Python
  colors = ['red', 'blue', 'green']
  print(colors[0])　　　# red
  print(colors[2])　　　# green
  ```

* 슬리이싱 (slicing)

  list의 값들을 자른다. 자르는데는 list의 주소값(offset, index)을 이용한다

  ```Python
  colors = ['red', 'blue', 'green', 'yellow', 'white', 'black']
  print(colors[1:4])　　　# ['blue', 'green', 'yellow']
  print(colors[:3])　　　 # ['red', 'blue', 'green']
  print(colors[2:])　　　 # ['green', 'yellow', 'white', 'black']
  print(colors[:])　　　　# ['red', 'blue', 'green', 'yellow','white', 'black']
  print(colors[::2])　　　# ['red', 'green', 'white'] -> 2칸 단위로
  print(colors[::-1])　　# ['black', 'white', 'yellow', 'green','blue', 'red'] -> 역순으로
  ```

* 리스트 연산

  ```Python
  colors = ['red', 'blue']
  colors2 = ['yellow', 'white']
  print(colors + colors2)　　　# ['red', 'blue', 'yellow', 'white']
  print(colors * 2)　　　　　　 # ['red', 'blue', 'red', 'blue']
  print('blue' in colors)　　　# True
  print('blue' in colors2)　　 # False
  ```

* 추가 삭제

  ```Python
  colors = ['red', 'blue']

  colors.append('green')# colors에 'green' 추가
  print(colors)　　　　　 # ['red', 'blue', 'green']

  colors.remove('blue') # colors에서 'blue' 삭제
  print(colors)　　　　　 # ['red', 'green']

  colors.extend(['yellow', 'white']) # colors에 새로운 리스트 추가
  print(colors)　　　　　 # ['red', 'green', 'yellow', 'white']

  del colors[3]　　　　　 # colors[3]에 있는 원소 삭제
  print(colors)　　　　　 # ['red', 'green', 'yellow']

  colors.insert(1, 'black') # colors의 1번 원소에 'black' 삽입
  print(colors)　　　　　 # ['red', 'black', 'green', 'yellow']
  ```

* 메모리 저장 방식

  파이썬의 리스트 변수에는 해당 리스트의 주소 값이 저장된다.

  ```Python
  a = [2, 1, 3]
  b = [4, 5, 6]
  b = a
  print(b) # [2, 1, 3]
  a.sort()
  print(b) # [1, 2, 3] -> a를 정렬했는데 b도 정렬되어있음
  b = [7, 8, 9]
  print(a, b) # [1, 2, 3] [7, 8, 9]
  ```

* 패킹과 언패킹
  * 패킹 : 한 변수에 여러개의 데이터를 넣는 것
  * 언패킹 : 한 변수의 데이터를 각각의 변수로 반환
  
  ```Python
  v = [1, 2, 3] # 1, 2, 3을 변수 v에 패킹
  a, b, c = v # v의 값을 a, b, c에 언패킹
  print(a, b, c) # 1 2 3
  ```

* 이차원 리스트

  리스트에 리스트를 넣어 행렬 생성
  
  ```Python
  a = [1, 2, 3]
  b = [4, 5, 6]
  c = [7, 8, 9]
  v = [a, b, c]
  print(v) # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  print(v[1][2]) # 6
  ```

* 다양한 Data Type이 하나의 List에 들어갈 수 있다
  
  ```Python
  v = [12, 3.14, "hello world", True]
  ```

  **[Python List/Array Methods](https://www.w3schools.com/python/python_ref_list.asp)** - 파이썬 리스트 함수 목록

## 함수와 Console I/O

### 함수

* 함수 선언 문법
  
  ```Python
  def 함수 이름 (parameter1, parameter2, ... ): 
    수행문 #1(statements)
    수행문 #2(statements)
    return <반환값>
  ```

  매개변수와 리턴문은 없어도 된다.  
  함수 내부의 구문은 들여쓰기로 구분한다

### Console I/O

* 입력 - ```input()```
  
  ```Python
  print("Enter your name: ")
  name = input()
  age = int(input("Enter your age: "))
  ```

* 출력 - ```print()```
  
  ```Python
  age = 20
  name = "Peter"
  print("Hello, My name is", name, "\nI am", age, "years old.")
  ```

  ```
  Hello, My name is Peter 
  I am 20 years old.
  ```

* print formatting
  * 형식(format)에 맞춰서 출력한다
  
  ```Python
  a = 12
  b = 3.14
  c = "hello"
  print("%d %f %s" % (a, b, c)) # %-string
  print("{} {} {}".format(a, b, c)) # format 함수
  print(f"{a} {b} {s}") # f-string
  ```

  %-format과 ```format()``` 함수는 old-school formatting  
  f-string을 주로 사용하자

## 조건문과 반복문

### 조건문

* 조건문 문법

  ```Python
  if <조건1>: # if를 쓰고 조건 삽입 후 “:” 입력
    <수행 명령1-1> # 들여쓰기(indentation)후 수행명령 입력 
    <수행 명령1-2> # 같은 조건하에 실행일 경우 들여쓰기 유지
  elif <조건2>: # 위의 조건이 불일치할 경우 다음 조건 검사. elif는 여러개 가능
    <수행명령2-1> # 조건2에 해당할 경우 수행할 명령 입력
    <수행 명령2-2> # 조건2 일치 시 수행할 명령 들여쓰기 유지
  else: # 위의 조건이 모두 불일치할 경우 수행할 명령 block 
    <수행명령3-1> # 조건 불일치 시 수행할 명령 입력
    <수행 명령3-2> # 조건 불일치 시 수행할 명령 들여쓰기 유지
  ```

* 조건 판단 방법
  * ```if``` 다음에 조건을 표기하여 참 또는 거짓을 판단함
  * 참/거짓의 구분을 위해 비교 연산자 활용
  * 논리 키워드 ```and```, ```or```, ```not``` 사용
  
* 삼항 연산자
  
  ```Python
  value = 30
  is_odd = True if value % 2 == 1 else False
  print(is_odd) # False
  ```

### 반복문

파이썬은 반복문으로 for, while 키워드를 사용함.
  
* ```for```문

  ```Python
  for i in [0, 1, 2, 3, 4]:
    print(i)
  ```

  ```Python
  for i in range(5): # for문은 주로 range() 함수와 함께 사용한다
    print(i)
  ```

  위 두 코드는의 결과는 동일하다.

  ```
  0
  1
  2
  3
  4
  ```

* ```while```문
  
  * 조건을 만족하는 동안 반복해서 내부의 구문들을 실행한다

  ```Python
  while(<조건>):
    <수행 명령1>
    <수행 명령2>
    ...
  ```

  * break와 continue로 반복을 제어한다

  ```Python
  i = 0
  while(True):
    i += 1

    if i % 2 == 0: # i가 짝수이면 continue
      continue

    print("hello") # i가 홀수이면 "hello"를 출력
  
    if (i == 10): # i가 10이면 반복 종료
      break
  ```

  * ```else```
  
  반복문의 조건을 만족시키지 못해 반복이 종료된 경우 ```else```문을 수행한다
  
  ```Python
  i = 0
  while(i < 10):
    i += 1

    if i % 2 == 0: # i가 짝수이면 continue
      continue

    print("hello") # i가 홀수이면 "hello"를 출력
  else:
    print("bye")
  ```

    위 코드는 ```i```가 ```10```이 되어 조건문 ```i < 10```을 만족시키지 못해 반복이 종료된다.  
    조건문을 만족시키지 못해 반복이 종료되었기 때문에 ```else```문이 수행된다.
    
  ```
  hello
  hello
  hello
  hello
  hello
  bye
  ```


  ```Python
  i = 0
  while(i < 10):
    i += 1

    if i % 2 == 0: # i가 짝수이면 continue
      continue

    print("hello") # i가 홀수이면 "hello"를 출력
  
    if (i == 5): # i가 10이면 반복 종료
      break
  else:
    print("bye")
  ```

    위 코드는 ```i```가 ```10```이 되기 전에 ```break```에 의해 반복이 종료된다.  
    조건문을 만족시키지 못해 반복이 종료된 것이 아니기 때문에 ```else```문이 수행되지 않는다.

  ```
  hello
  hello
  hello
  ```

## String and Advanced Function Concept

### 문자열 (String)

* sequence 자료형으로, 문자형 data를 메모리에 저장
* 영문자 한글자는 1byte의 공간을 사용
* 문자열의 각 문자는 개별 주소(offset)를 가짐
* 이 주소를 사용해 인덱싱
* ```list```와 같은 형태로 데이터를 처리함
  
  ```Python
  string = "hello world"
  print(string[2], string[-1]) # l d
  print(string[::-1]) # dlrow olleh
  ```

* 덧셈과 곱셈 연산 가능
  
  ```Python
  a = "hello"
  b = "world"
  print(a + " " + b) # hello world
  print(a * 3) # hellohellohello
  ```

* ```in``` 연산으로 포함 여부 체크

  ```Python
  string = "hello"
  print("a" in string) # False
  print("h" in string) # True
  print("hell" in string) # True
  ```

* ```str```은 immutable
  
  ```Python
  string = "hello"
  string[2] = 'X'
  print(string)
  ```

  ```
  TypeError                                 Traceback (most recent call last)
  <ipython-input-32-40e352df4a96> in <module>
        1 string = "hello"
  ----> 2 string[2] = 'X'
        3 print(string)

  TypeError: 'str' object does not support item assignment
  ```

  **[Python String Methods](https://www.w3schools.com/python/python_ref_string.asp)** - 파이썬 문자열 함수 목록

### 함수2 (Advanced Function Concept)

* 함수에 parameter를 전달하는 방식
  * Call by Value
    * 값만 넘김
    * 함수 내에서 인자 값을 바꿔도, 호출자에게 영향을 주지 않음
  
  * Call by Reference
    * 함수에 인자를 넘길 떄 메모리 주소를 넘김
    * 함수 내에서 인자 값을 변경하면, 호출자의 값도 변경됨

  * Call by Object Reference
    * 파이썬의 함수 호출 방식
    * 파이썬에서는 모든것이 객체이다
    * 파이썬의 객체는 mutable, immutable 두가지 종류로 나뉜다
      * mutable 객체 : int, float, str, tuple...
      * immutable 객체 : dic, list, set...
    * mutable 객체가 인자로 전달된 경우에는 함수 내에서 인자 값을 바꾸면, 호출자의 값도 변경된다
    * immutable 객체가 인자로 전달된 경우에는 함수 내에서 인자값을 바꿔도 호출자에게 영향을 주지 않는다
    * 이처럼 작동하는 것을 call by object reference라고 한다
  
* Function Scoping Rule
  * 지역변수 - 함수 내에서만 사용
  * 전역변수 - 프로그램 전체에서 사용
    * 함수 내에서 전역변수를 사용하려면 ```global``` 키워드를 사용해야함
  
  ```Python
    def f(): 
      global s
      s = "I love London!" 
      print(s)

    s = "I love Paris!" 
    f()
    print(s)
  ```

* Function Type Hints
  * dynamic typing 때문에 함수를 사용하는 사용자가 함수의 interface를 알기 어렵다는 단점이 있다
  * 이를 보완하기 위하여 python 3.5 버전 부터 type hints 기능을 제공한다

  function type hints 문법

  ```Python
    def do_function(var_name: var_type) -> return_type: 
      pass
  ```

  사용 예

  ```Python
    def type_hint_example(name: str) -> str: 
      return f"Hello, {name}"
  ```

  * Type Hints의 장점
    * 사용자에게 interface를 명확히 알려줄 수 있다
    * 함수의 문서화 시 함수에 대한 정보를 명확히 알 수 있다
    * 코드의 발생 가능한 오류를 사전에 확인할 수 있다
    * 시스템의 안정성을 확보할 수 있다

* docstring
  * 파이썬 함수에 대한 상세스펙을 작성
  * 함수명 아래에 세개의 따옴표로 docstring 영역을 표시한다
  
  ```Python
  def func(int: arg1, float arg2) -> string:
      """[summary]

      Args:
          int (arg1): [description]
          floatarg2 ([type]): [description]

      Returns:
          string: [description]
      """    
  ```
  
* 파이썬 코딩 컨벤션
  * flake8 모듈과 black 모듈로 체크 (conda install -c anaconda <모듈명>)
  * 최근에는 black 모듈을 활용하여 pep8 like 수준을 준수한다

* 함수 개발 가이드 라인
  * **함수는 가능하면 짧게 작성한다 (줄 수를 줄이고 여러개의 함수로 쪼갠다)**
  * 함수 이름에 함수의 역할, 의도가 명확히 들어나야 한다
  * 함수의 이름은 동사 + 목적어의 형태로 되도록 작성한다
  * 하나의 함수에는 유사한 역할을 하는 코드만 포함한다
  * **인자로 받은 값 자체를 바꾸지 않는다 (인자를 복사하여 사용한다)**
  * 함수는 언제 만드는가?
    * 공통적으로 사용되는 코드는 함수로 변환한다
    * **복잡한 수식 -> 식별 가능한 이름의 함수로 변환한다**
    * **복잡한 조건 -> 식별 가능한 이름의 함수로 변환한다**

  **좋은 프로그래머는 사람이 이해할 수 있는 코드를 짠다**
