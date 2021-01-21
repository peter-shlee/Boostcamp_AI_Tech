# Day 4 - 파이썬 기초 문법 3

- [Day 4 - 파이썬 기초 문법 3](#day-4---파이썬-기초-문법-3)
  - [객체 지향 프로그래밍](#객체-지향-프로그래밍)
    - [class 선언](#class-선언)
    - [```__init__``` 함수](#__init__-함수)
    - [method 구현](#method-구현)
    - [객체 지향 언어의 특징](#객체-지향-언어의-특징)
    - [First-class objects](#first-class-objects)
    - [Inner function](#inner-function)
    - [**decorator function**](#decorator-function)
  - [Module and Project](#module-and-project)
    - [Module 만들기](#module-만들기)
    - [Package](#package)
    - [오픈소스 라이브러리 사용하기](#오픈소스-라이브러리-사용하기)
      - [conda로 가상환경 사용하는 방법](#conda로-가상환경-사용하는-방법)

## 객체 지향 프로그래밍

### class 선언
  
* class 선언 방법은 다음과 같다
    ```Python
    class <classs 이름> (<상속받는 객체 이름>):
    ```

    상속받는 객체 이름은 생략 가능하다.  
    생략할 시 object class를 자동으로 상속받게 된다.

### ```__init__``` 함수

* attribute 추가는 __init__ 함수 내에서 한다
* 메서드의 인자 중 반드시 ```self```가 있어야 한다

    ```Python
    class Obj:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c
    ```

* ```__``` 는 특수한 예약함수나 변수 그리고 함수명 변경(맨글링)으로 사용  
예) ```__main__```, ```__add__```, ```__str__```, ```__eq__```

    ```Python
    class Obj:
        def __init__(self, num):
            self.num = num

        def __str__(self): # 객체를 str로 변환할 때 사용됨
            return str(self.num)

        def __eq__(self, obj2): # 객체에 비교 연산자를 사용할 때 사용됨
            return self.num == obj2.num

    obj1 = Obj(1)
    obj2 = Obj(2)

    print(str(obj1)) # 1
    print(obj1, obj2) # 1 2
    print(obj1 == obj2) # False

    ```

### method 구현
  * method 추가는 기본 함수 추가 방법과 동일하지만 반드시 ```self```가 인자로 와야 함

    ```Python
    class Obj:
        def method(self, a, b):
            return a + b

    obj = Obj()
    print(obj.method(3, 4)) # 7
    ```

### 객체 지향 언어의 특징
  * 상속 (inheritance)

    ```Python
    class Person(object):
        def __init__(self, name, age):
            self.name = name
            self.age = age

    class Employee(Person):  
        def __init__(self, name, age, wage):
            super().__init__(name, age)
            self.wage = wage

    employee = Employee("Lee", 30, 300)
    print(employee.name, employee.age, employee.wage) # Lee 30 300

    ```

  * 다형성 (polymorphism)

    ```Python
    class Animal:
        def __init__(self, name):
            self.name = name

        def talk(self):
            raise NotImplementedError()

    class Cat(Animal):
        def talk(self):
            return "야옹"

    class Dog(Animal):
        def talk(self):
            return "멍멍"

    cat = Cat("나비")
    dog = Dog("바둑이")

    print(cat.talk()) # 야옹
    print(dog.talk()) # 멍멍

    print(cat.name) # 나비
    print(dog.name) # 바둑이
    ```

  * 가시성 (Visibility)
    * 캡슐화 또는 정보 은닉
    * class간 간섭/정보 공유 최소화
    * 사용자가 임의로 객체의 속성을 수정하지 못하도록 함
    * 인터페이스를 이용해 객체의 속성에 접근
    * ```__``` 을 attribute 이름 앞에 붙여서 private 수로 선언

    ```Python
    class Product:
        pass

    class Inventory(object): 
        def __init__(self):
            self.__items = []

        def add_new_item(self, product): # setter method
            if type(product) == Product:
                self.__items.append(product)
                print("new item added") 
            else:
                raise ValueError("Invalid Item")

        def get_number_of_items(self): # getter method
            return len(self.__items)

    my_inventory = Inventory()
    my_inventory.add_new_item(Product()) # new item added
    my_inventory.add_new_item(Product()) # new item added
    print(my_inventory.get_number_of_items()) # 2
    print(my_inventory.__items) # AttributeError -> private attribute에 직접 접근 불가
    my_inventory.add_new_item(object)
    ```

    ```Python
    class Inventory(object): 
        def __init__(self):
             self.__items = []

        # property decorator는 private attribute를 반환할 수 있게 해줌
        @property
        def items(self):
            return self.__items

        def add_new_item(self, product): # setter method
            if type(product) == Product:
                self.__items.append(product)
                print("new item added") 
            else:
                raise ValueError("Invalid Item")

        def get_number_of_items(self): # getter method
            return len(self.__items)

    my_inventory = Inventory()
    my_inventory.add_new_item(Product()) # new item added
    my_inventory.add_new_item(Product()) # new item added
    print(my_inventory.get_number_of_items()) # 2
    items = my_inventory.items # private attribute에 직접 접근 가능해짐
    items.append(Product()) # private attribute에 직접 접근하여 수정함
    print(my_inventory.get_number_of_items()) # 3
    ```

### First-class objects

* 일등 함수 또는 일급 객체라고 부름
* 변수나 데이터 구조에 할당이 가능한 객체
* parameter로 전달 가능, return 값으로 사용 가능
* **파이썬의 함수는 일급 함수임 -> parameter, return 값으로 사용 가능**

    ```Python
    def sum(a, b):
        return a + b

    func = sum # 변수에 함수를 넣음

    # 변수를 함수로 사용
    print(func(3, 4)) # 7
    ```

    ```Python
    def sum(a, b):
        return a + b

    def formula(method, a, b):
        return method(a, b)

    # 함수를 매개변수로 사용
    print(formula(sum, 3, 4)) # 7
    ```

### Inner function

* 함수 내에 존재하는 함수

    ```Python
    def print_msg(msg):
        def printer():
            print(msg)
        printer()
        
    print_msg("Hello, Python") # Hello, Python
    ```

* closures(클로저): inner function을 return값으로 반환

    ```Python
    def print_msg(msg):
        def printer():
            print(msg)
        return printer

    another = print_msg("Hello, Python")
    another() # Hello, Python
    ```

    closures 활용 예제


    ```Python
    def tag_func(tag, text):
        text = text
        tag = tag
        def inner_func():
            return '<{0}>{1}</{0}>'.format(tag, text)
        return inner_func

    h1_func = tag_func('h1', "This is title")
    p_func = tag_func('p', "this is content")
    print(h1_func()) # <h1>This is title</h1>
    print(p_func()) # <p>this is content</p>
    ```

    이렇게 closure를 이용하면 하나의 함수를 여러 목적에 맞춰서 다양한 형태로 사용할 수 있다

### **decorator function**

* decorator
  * 위에서 봤던 ```@property``` 같은 것을 decorator라고 한다. 자세한 내용은 [여기](https://wikidocs.net/23106)를 참조
* 복잡한 closure 함수를 간단하게 해준다

    ```Python
    def star(func):
        def inner(*args, **kwargs):
            print("*" * 30)
            func(*args, **kwargs)
            print("*" * 30)
        return inner
    
    # 자동으로 printer() 함수를 star() 함수에 전달인자로 넣어줌
    @star
    def printer(msg):
        print(msg)
    printer("Hello")
    ```
    ```
    ******************************
    Hello
    ******************************
    ```

## Module and Project

* Module
  * 작은 프로그램 조각
  * module들을 모아서 하나의 큰 프로그램을 개발함
  * 프로그램을 모듀로하 시키면 다른 프로그램이 사용하기 쉬움
  * 파이썬이 기본 제공하는 Built-in Module도 있음 (random, time 등)

* Package
  * 모듈을 모아놓은 단위, 하나의 프로그램

### Module 만들기

* 파이썬의 Module은 *.py 파일을 의미함
* 같은 폴더 내에 Module에 해당하는 .py 파일을 저장한 후 import문을 사용해서 module을 호출
* from과 import문을 함께 사용하여 namespace를 이용할 수 있음
  * 필요한 내용만 골라서 호출할 수 있음
  * alias를 설정해 모듈명을 별칭으로 바꿔 사용할 수 있음
  * 모듈에서 모든 함수 또는 클래스를 호출하고 싶을 떄는 ```from <모듈명> import *```

### Package

* 패키지는 하나의 대형 프로젝트를 만드는 코드의 묶음
* 다양한 모듈들의 집합, 디렉토리로 연결됨
* ```__init__```, ```__main__``` 등 키워드 파일명이 사용됨
* package 만드는 방법은 [여기](https://wikidocs.net/1418)를 참고

### 오픈소스 라이브러리 사용하기

* 프로젝트를 할 때 PC에 여러 패키지를 설치해야함
* 여러개의 서로 다른 프로젝트를 할 때 설치한 패키지들 끼리 충돌할 수 있음
* 이런 상황을 방지하기 위해 가상환경을 설정해 그곳에서 프로젝트를 진행함
* 가상환경은 필요한 패키지만 설치하는 환경임
* 기본 인터프리터 + 프로젝트 종류별 패키지 설치
* 대표적인 도구로는 virtualev와 conda가 있음

#### conda로 가상환경 사용하는 방법

* 가상환경 생성
  
  ```
  conda create -n <새 가상환경 이름> python=<파이썬 버전>
  ```

* 가상환경 호출
  
  ```
  conda activate <사용할 가상환경 이름>
  ```

* 가상환경 해제
  
  ```
  conda deactivate
  ```

* 패키지 설치  
  Windows에서는 conda, Linux나 Mac에서는 conda와 pip 사용

  ```
  conda install <패키지 이름>
  ```
