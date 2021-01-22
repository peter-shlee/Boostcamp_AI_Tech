# Day 5 - 파이썬으로 데이터 다루기

- [Day 5 - 파이썬으로 데이터 다루기](#day-5---파이썬으로-데이터-다루기)
  - [Exception/File/Log Handling](#exceptionfilelog-handling)
    - [Exception Handling](#exception-handling)
    - [File Handling](#file-handling)
    - [Log Handling](#log-handling)
  - [Data Handling](#data-handling)
    - [CSV (Comma separate Values)](#csv-comma-separate-values)
    - [Web](#web)
    - [XML](#xml)
    - [JSON (JavaScript Object Notation)](#json-javascript-object-notation)


## Exception/File/Log Handling

### Exception Handling

* try ~ except

    ```Python
    try:
        # 예외 발생 가능 코드
    except <Exception Type>:
        # 예외 발생 시 실행할 코드
    ```

    0으로 숫자를 나눌 때의 예외처리
    ```Python
    try:
        print (4 / 0)
    except ZeroDivisionError as e:
        print(e) # division by zero
        print("0으로 나눌 수 없습니다") # 0으로 나눌 수 없습니다
    ```

    Built-in Exception의 종류는 [여기](https://docs.python.org/ko/3/library/exceptions.html)를 참고

* try ~ except ~ else

    ```Python
    try:
        # 예외 발생 가능 코드
    except <Exception Type>:
        # 예외 발생 시 실행할 코드
    else:
        # 예외가 발생하지 않을 때 동작하는 코드
    ```

    ```Python
    for i in range(10): 
        try:
            result = 10 / i 
        except ZeroDivisionError:
            print("Not divided by 0") 
        else:
            print(10 / i)
    ```

    위 코드의 결과는 다음과 같다

    ```
    Not divided by 0
    10.0
    5.0
    3.3333333333333335
    2.5
    2.0
    1.6666666666666667
    1.4285714285714286
    1.25
    1.1111111111111112
    ```

* try ~ except ~ finally

    ```Python
    try:
        # 예외 발생 가능 코드
    except <Exception Type>:
        # 예외 발생 시 실행할 코드
    else:
        # 예외가 발생하지 않을 때 동작하는 코드
    finally:
        # 예외 발생 여부와 상관없이 실행됨
    ```

    ```Python
    for i in range(10): 
        try:
            result = 10 // i 
        except ZeroDivisionError:
            print("Not divided by 0") 
        else:
            print(result)
        finally:
            print("finally test")
    ```

    위 코드의 실행 결과는 다음과 같다

    ```
    Not divided by 0
    finally test
    10
    finally test
    5
    finally test
    3
    finally test
    2
    finally test
    2
    finally test
    1
    finally test
    1
    finally test
    1
    finally test
    1
    finally test
    ```

    exception 발생 여부와 상관 없이 매번 finally statement가 실행되는 것을 확인할 수 있다.

* raise  
    필요에 따라 강제로 Exception을 발생시킨다

    ```Python
    raise <Exception Type>(<예외 정보>)
    ```

* assert  

    assert는 특정 조건에 만족하지 않을 경우 예외를 발생시킨다. 사용법은 다음과 같다.

    ```Python
    assert <예외 조건>
    ```

    아래와 같이 예외 조건이 True인 경우에는 exception 발생되지 않고 계속해서 진행된다

    ```Python
    assert True
    print("ok")
    ```

    ```
    ok
    ```

    아래와 같이 예외조건이 False인 경우에는 Exception이 발생한다

    ```Python
    assert False
    print("ok")
    ```

    ```
    ---------------------------------------------------------------------------
    AssertionError                            Traceback (most recent call last)
    <ipython-input-6-a871fdc9ebee> in <module>
    ----> 1 assert False

    AssertionError: 
    ```

    실제 사용 예는 다음과 같다

    ```Python
    def get_binary_nmubmer(decimal_number):
        assert isinstance(decimal_number, int)
        return bin(decimal_number)

    print(get_binary_numbmer(10))
    ```

### File Handling

* 파일의 종류
  * text 파일
  * binary 파일

* 파이썬의 파일 처리 방법
  
    ```Python
    f = open("<파일 이름>", "<접근 모드>", <encoding>) # encoding은 생략 가능
    f.close()
    ```

  * 파일 접근 모드 종류
  
    |모드|설명|
    |-|-|
    |r|읽기 모드|
    |w|쓰기 모드|
    |a|추가 모드|
    |b|binary 파일 접근시 사용. 단독 사용 불가. 위의 모드들과 함께 사용|

  * encoding

    Mac과 Linux는 UTF-8 또는 UTF-16을 사용하고, Windows는 cp949를 사용함.  
    이 점을 유의해야 하며 저장할 떄는 되도록이면 UTF-8 사용

* File Read
  * ```read()``` - 파일 전체를 읽어 문자열로 반환
  * ```readlines()``` - 파일을 한줄씩 나눠 list로 반환
  * ```readline()``` - 호출할 때마다 한줄씩 읽어서 문자열로 반환

* File Write

    "w" 모드로 write
    ```Python
    f = open("count_log.txt", 'w', encoding="utf8") 
    for i in range(1, 11):
        data = "%d번째 줄입니다.\n" % i
        f.write(data)
    f.close()
    ```

    "a" 모드로 write
    ```Python
    with open("count_log.txt", 'a', encoding="utf8") as f: 
        for i in range(1, 11):
            data = "%d번째 줄입니다.\n" % i 
            f.write(data)
    ```

* Directory Handling

    os 모듈을 사용하여 handling

    ```Python
    import os 
    os.mkdir("log")
    ```

    디렉토리가 있는지 확인하기

    ```Python
    if not os.path.isdir("log"): 
        os.mkdir("log")
    ```

    최근에는 **pathlib 모듈**을 사용하여 path를 객체로 다룸  
    자세한 내용은 [여기](https://docs.python.org/3/library/pathlib.html)를 참고

* Pickle

    * 파이썬의 객체를 영속화(persistence) 하는 built-in 객체
    * 데이터, object 등을 저장, 불러와서 사용

    ```pickle.dump()```를 사용하여 저장

    ```Python
    f = open("list.pickle", "wb") 
    test = [1, 2, 3, 4, 5] 
    pickle.dump(test, f) # f 파일에 test 리스트를 저장
    f.close()
    ```

    ```pickle.load()```를 사용하여 읽어옴

    ```Python
    f = open("list.pickle", "rb") 
    test_pickle = pickle.load(f) # f에 저장되어있던 리스트를 불러옴
    print(test_pickle) # [1, 2, 3, 4, 5] 
    f.close()
    ```


### Log Handling

* logging module  
  Python의 기본 Log 관리 모듈

  * logging level
    * 프로그램 진행 상황에 따라 다른 level의 log를 출력함
    * 개발 시점, 운영 시점마다 다른 log가 남을 수 있도록 지웒마
    * **DEBUG -> INFO -> WARNING -> ERROR -> CRITICAL**

    |Level|개요|
    |-|-|
    |debug|개발시 처리 기록을 남겨야 하는 로그 정보를 남김|
    |info|처리가 진행되는 동안의 정보를 알림|
    |warning|사용자가 잘못 입력한 정보나, 처리는 가능하나 의도치 않은 정보가 들어왔을 때 알림|
    |error|잘못된 처리로 인해 에러가 났으나, 프로그램은 동작할 수 있음을 알림|
    |critical|잘못된 처리로 데이터 손실이나 더 이상 프로그램이 동작할 수 없음을 알림|

    * Python logging 기본 level은 WARNING이다.  
    * 설정된 level보다 낮은 log는 출력 되지 않는다.

  ```Python
  import logging  
  logging.debug("debug log")
  logging.info("info log")
  logging.warning("warning log")
  logging.error("error log")
  logging.critical("critical log")
  ```

  ```
  WARNING:root:warning log
  ERROR:root:error log
  CRITICAL:root:critical log
  ```

    * level 설정

  ```Python
  import logging

  logger = logging.getLogger("main")

  logger.setLevel(logging.DEBUG) # logging level을 debug로 set
  logger.debug("debug log")
  logger.info("info log")
  logger.warning("warning log")
  logger.error("error log")
  logger.critical("critical log")
  ```

  ```
  debug log
  info log
  warning log
  error log
  critical log
  ```

    logging level을 debug로 설정하면 모든 level의 로그가 출력된다

    ```Python
    import logging

    logger = logging.getLogger("main")

    logger.setLevel(logging.CRITICAL) # logging level을 critical로 set
    logger.debug("debug log")
    logger.info("info log")
    logger.warning("warning log")
    logger.error("error log")
    logger.critical("critical log")
    ```

    ```
    critical log
    ```

    logging level을 critical 설정하면 critical level의 로그만 출력된다

    * Handler 설정  
    log를 출력할 위치를 설정한다
  
    ```Python
    import logging

    logger = logging.getLogger("main")
    stream_hander = logging.StreamHandler() # 콘솔에 로그를 출력하기 위한 핸들러
    logger.addHandler(stream_hander)
    file_handler = logging.FileHandler("test.log") # 파일에 로그를 출력하기 위한 핸들러
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO) # logging level을 critical로 set
    logger.debug("debug log")
    logger.info("info log")
    logger.warning("warning log")
    logger.error("error log")
    logger.critical("critical log")
    ```

    아래는 test.log 파일의 내용

    ```
    info log
    warning log
    error log
    critical log
    ```

* 프로그램 설정
  * configparser - 파일에 설정
  * argparser - 실행 시점에 설정

## Data Handling

### CSV (Comma separate Values)

* 필드를 쉼표로 구분한 텍스트 파일
* 탭(TSV), 빈칸(SSV) 등으로 구분해서 만들기도 함
* 일반 text file을 처리하듯 파일을 읽어온 후, 한 줄 한 줄씩 데이터를 처리함
* Python에서는 간단히 CSV 파일을 처리할 수 있게 csv객체를 제공함
(자세한 내용은 [여기](https://docs.python.org/ko/3/library/csv.html)를 참고)

### Web

* HTML을 parsing하는데는 **beautifulsoup** 이나 **정규식**을 사용
* 정규식은 따로 공부
* 정규식 연습장 활용
  * 정규식 연습장(http://www.regexr.com/)을 이용해 연습하기
* 간단한 정규식 문법
  * 문자클래스 ```[ ]``` : ```[``` 와 ```]```사이의 문자들과 match 라는 의미  
  ex) [abc] -> 'a', 'b', 'c' 를 찾는다
  * ```-``` 를 이용해 범위를 지정할 수 있다
  ex) [a-zA-Z] -> 알파벳 전체, [0-9] -> 숫자 전체
  * 정규식 표현을 위해 원래 의미가 아닌 다른 용도로 사용되는 문자는 다음과 같다
  ``` . ^ $ * + ? { } [ ] \ | ( ) ```
* **파이썬에서 정규식 사용**
  * ```re``` (regular expression) 모듈을 ```import``` 하여 사용
  
  ```
  import re
  ```

  * ```search()``` -> 한 개만 찾기
  * ```findall()``` -> 전체 찾기
    * 추출된 패턴은 tuple로 반환됨
  
  ```findall()``` 예제 - "http"로 시작하고 "zip"으로 끝나는 문자열을 찾는다

  ```
  url_list = re.findall(r"(http)(.+)(zip)", html_contents)
  ```

### XML

* 데이터의 구조와 의미를 설명하는 TAG(MarkUp)를 사용하여 표시하는 언어
* TAG와 TAG 사이에 값이 표시되고, 구조적인 정보를 표현할 수 있음 (Tree 구조)

    ```xml
    <?xml version="1.0"?> 
        <고양이>
        <이름>나비</이름> 
        <품종>샴</품종>
        <나이>6</나이> 
        <중성화>예</중성화>
        <발톱 제거>아니요</발톱 제거> 
        <등록 번호>Izz138bod</등록 번호> 
        <소유자>이강주</소유자>
    </고양이>
    ```

* XML도 HTML과 같이 구조적 markup 언어
* 따라서 HTML처럼 Beautifulsoup이나 정규식으로 Parsing 가능
* **beautifulsoup 사용**  
  자세한 사용법은 [여기](https://wikidocs.net/85739)를 참고
  * HTML, XML등 Markup 언어 Scraping을 위한 대표적인 도구
  * 내부적으로는 lxml과 html5lib와 같은 parser를 사용함
  * 속도는 느리지만 간편히 사용할 수 있음
  * conda rktkdghksruddmfh lxml과 beautifulsoup 설치 방법
  
  ```
  activate <가상 환경 이름>
  conda install lxml
  conda install -c beautifulsoup
  ```

  * 모듈 호출

  ```Python
  from bs4 import BeautifulSoup
  ```

  * 객체 생성

  ```Python
  soup = BeautifulSoup(books_xml, "lxml") # beautifulsoup을 사용할 파일과 parser를 설정
  ```

  * Tag 찾는 함수 ```find()```, ```find_all()```
  
  ```Python
  soup.find_all("author")
  ```

  * 태그와 태그 사이의 값 반환 함수 ```get_text()```

### JSON (JavaScript Object Notation)

* 웹 프로그래밍 언어인 JavaScript의 데이터 객체 표현 방식
* 데이터 용량이 적고, code로의 전환이 쉬움
* Python의 Dictionary와 구조가 완전히 동일하여 파이썬에서 사용이 간편함
* Built-in 모듈인 json 모듈을 사용해 손 쉽게 파싱 및 저장 가능
* JSON Read

  ```Python
  import json
  
  with open("json_example.json", "r", encoding="utf8") as f: # 파일 open
      contents = f.read() # 파일 내용 read
      json_data = json.loads(contents) # 파일 내용 (JSON) 읽어옴
      print(json_data["employees"]) # JSON을 읽어온 후 Dictionary type처럼 처리
  ```

* JSON Write

  ```Python
  import json
  
  dict_data = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}

  with open("data.json", "w") as f: 
    json.dump(dict_data, f) # Dictionary를 JSON으로 파일에 Write
  ```