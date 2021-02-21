# 1. Spacy를 이용한 영어 전처리


```python
import spacy
spacy_en = spacy.load('en')
```


```python
nlp = spacy.load('en_core_web_sm')
```

### 1.1 Tokenezation


```python
text = nlp('Naver Connect and Upstage Boostcamp')
print ([token.text for token in text])
```

    ['Naver', 'Connect', 'and', 'Upstage', 'Boostcamp']



```python
doc = nlp('This assignment is about Natural Language Processing.' 'In this assignment, we will do preprocessing')
print ([token.text for token in doc])
```

    ['This', 'assignment', 'is', 'about', 'Natural', 'Language', 'Processing', '.', 'In', 'this', 'assignment', ',', 'we', 'will', 'do', 'preprocessing']



```python
text=nlp("The film's development began when Marvel Studios received a loan from Merrill Lynch in April 2005. After the success of the film Iron Man in May 2008, \
Marvel announced that The Avengers would be released in July 2011 and would bring together Tony Stark, Steve Rogers, Bruce Banner, and Thor from Marvel's previous films. \
With the signing of Johansson as Natasha Romanoff in March 2009, the film was pushed back for a 2012 release. Whedon was brought on board in April 2010 and rewrote the original screenplay by Zak Penn. Production began in April 2011 in Albuquerque, \
New Mexico, before moving to Cleveland, Ohio in August and New York City in September. The film has more than 2,200 visual effects shots.")
```

### 1.2 불용어 (Stopword)


```python
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
for stop_word in list(spacy_stopwords)[:30]:
  print(stop_word)
```

    using
    twenty
    call
    mine
    move
    other
    well
    where
    again
    further
    ’s
    between
    my
    on
    ourselves
    whereas
    therein
    third
    anyway
    've
    here
    wherever
    may
    alone
    his
    yourselves
    did
    even
    n't
    which



```python
stopword_text = [token for token in text if not token.is_stop]
print(stopword_text)
```

    [film, development, began, Marvel, Studios, received, loan, Merrill, Lynch, April, 2005, ., success, film, Iron, Man, 2008, ,, Marvel, announced, Avengers, released, July, 2011, bring, Tony, Stark, ,, Steve, Rogers, ,, Bruce, Banner, ,, Thor, Marvel, previous, films, ., signing, Johansson, Natasha, Romanoff, March, 2009, ,, film, pushed, 2012, release, ., Whedon, brought, board, April, 2010, rewrote, original, screenplay, Zak, Penn, ., Production, began, April, 2011, Albuquerque, ,, New, Mexico, ,, moving, Cleveland, ,, Ohio, August, New, York, City, September, ., film, 2,200, visual, effects, shots, .]


### 1.3 Lemmatization 


```python
for token in text[:20]:
  print (token, "-", token.lemma_) # 사전형 단어
```

    The - the
    film - film
    's - 's
    development - development
    began - begin
    when - when
    Marvel - Marvel
    Studios - Studios
    received - receive
    a - a
    loan - loan
    from - from
    Merrill - Merrill
    Lynch - Lynch
    in - in
    April - April
    2005 - 2005
    . - .
    After - after
    the - the


### 1.4 그외 token class의 attributes 

https://spacy.io/api/token#attributes


```python
print("token \t is_punct \t is_space \t shape_ \t is_stop")
print("="*70)
for token in text[21:31]:
  print(token,"\t", token.is_punct, "\t\t",token.is_space,"\t\t", token.shape_, "\t\t",token.is_stop)
```

    token 	 is_punct 	 is_space 	 shape_ 	 is_stop
    ======================================================================
    of 	 False 		 False 		 xx 		 True
    the 	 False 		 False 		 xxx 		 True
    film 	 False 		 False 		 xxxx 		 False
    Iron 	 False 		 False 		 Xxxx 		 False
    Man 	 False 		 False 		 Xxx 		 False
    in 	 False 		 False 		 xx 		 True
    May 	 False 		 False 		 Xxx 		 True
    2008 	 False 		 False 		 dddd 		 False
    , 	 True 		 False 		 , 		 False
    Marvel 	 False 		 False 		 Xxxxx 		 False


## 빈칸완성 과제1


```python
def is_token_allowed(token):
# stopword와 punctutation을 제거해주세요.

  #if문을 작성해주세요.
  
  ##TODO#
  if token.is_stop or token.is_punct:
  ##TODO##
    return False
  return True

def preprocess_token(token):
  #lemmatization을 실행하는 부분입니다. 
  return token.lemma_.strip().lower()

filtered_tokens = [preprocess_token(token) for token in text if is_token_allowed(token)]
answer=['film', 'development','begin', 'marvel','studios', 'receive','loan', 'merrill','lynch', 'april','2005', 'success','film', 'iron','man', '2008','marvel','announce', 'avengers','release', 'july','2011', 'bring','tony', 'stark','steve', 'rogers','bruce', 'banner','thor', 'marvel','previous', 'film','signing', 'johansson','natasha','romanoff','march','2009','film','push','2012','release','whedon','bring','board','april','2010','rewrote','original','screenplay','zak','penn','production','begin','april','2011','albuquerque','new','mexico','move','cleveland','ohio','august','new','york','city','september','film','2,200','visual','effect','shot']
assert filtered_tokens == answer
```

    ['film', 'development', 'begin', 'marvel', 'studios', 'receive', 'loan', 'merrill', 'lynch', 'april', '2005', 'success', 'film', 'iron', 'man', '2008', 'marvel', 'announce', 'avengers', 'release', 'july', '2011', 'bring', 'tony', 'stark', 'steve', 'rogers', 'bruce', 'banner', 'thor', 'marvel', 'previous', 'film', 'signing', 'johansson', 'natasha', 'romanoff', 'march', '2009', 'film', 'push', '2012', 'release', 'whedon', 'bring', 'board', 'april', '2010', 'rewrote', 'original', 'screenplay', 'zak', 'penn', 'production', 'begin', 'april', '2011', 'albuquerque', 'new', 'mexico', 'move', 'cleveland', 'ohio', 'august', 'new', 'york', 'city', 'september', 'film', '2,200', 'visual', 'effect', 'shot']



```python

```

# 2. 한국어 전처리

### 2.1 Mecab를 이용한 형태소 분석 기반 토크나이징


```python
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
```

    Cloning into 'Mecab-ko-for-Google-Colab'...
    remote: Enumerating objects: 91, done.[K
    remote: Counting objects: 100% (91/91), done.[K
    remote: Compressing objects: 100% (85/85), done.[K
    remote: Total 91 (delta 43), reused 22 (delta 6), pack-reused 0[K
    Unpacking objects: 100% (91/91), done.
    /content/Mecab-ko-for-Google-Colab
    Installing konlpy.....
    Collecting konlpy
    [?25l  Downloading https://files.pythonhosted.org/packages/85/0e/f385566fec837c0b83f216b2da65db9997b35dd675e107752005b7d392b1/konlpy-0.5.2-py2.py3-none-any.whl (19.4MB)
    [K     |████████████████████████████████| 19.4MB 1.2MB/s 
    [?25hCollecting tweepy>=3.7.0
      Downloading https://files.pythonhosted.org/packages/67/c3/6bed87f3b1e5ed2f34bd58bf7978e308c86e255193916be76e5a5ce5dfca/tweepy-3.10.0-py2.py3-none-any.whl
    Collecting beautifulsoup4==4.6.0
    [?25l  Downloading https://files.pythonhosted.org/packages/9e/d4/10f46e5cfac773e22707237bfcd51bbffeaf0a576b0a847ec7ab15bd7ace/beautifulsoup4-4.6.0-py3-none-any.whl (86kB)
    [K     |████████████████████████████████| 92kB 11.0MB/s 
    [?25hRequirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (4.2.6)
    Collecting JPype1>=0.7.0
    [?25l  Downloading https://files.pythonhosted.org/packages/de/af/93f92b38ec1ff3091cd38982ed19cea2800fefb609b5801c41fc43c0781e/JPype1-1.2.1-cp36-cp36m-manylinux2010_x86_64.whl (457kB)
    [K     |████████████████████████████████| 460kB 54.9MB/s 
    [?25hRequirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.6/dist-packages (from konlpy) (1.19.5)
    Collecting colorama
      Downloading https://files.pythonhosted.org/packages/44/98/5b86278fbbf250d239ae0ecb724f8572af1c91f4a11edf4d36a206189440/colorama-0.4.4-py2.py3-none-any.whl
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.15.0)
    Requirement already satisfied: requests[socks]>=2.11.1 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (2.23.0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.3.0)
    Requirement already satisfied: typing-extensions; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from JPype1>=0.7.0->konlpy) (3.7.4.3)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2020.12.5)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (3.0.4)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6; extra == "socks" in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.7.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->tweepy>=3.7.0->konlpy) (3.1.0)
    Installing collected packages: tweepy, beautifulsoup4, JPype1, colorama, konlpy
      Found existing installation: tweepy 3.6.0
        Uninstalling tweepy-3.6.0:
          Successfully uninstalled tweepy-3.6.0
      Found existing installation: beautifulsoup4 4.6.3
        Uninstalling beautifulsoup4-4.6.3:
          Successfully uninstalled beautifulsoup4-4.6.3
    Successfully installed JPype1-1.2.1 beautifulsoup4-4.6.0 colorama-0.4.4 konlpy-0.5.2 tweepy-3.10.0
    Done
    Installing mecab-0.996-ko-0.9.2.tar.gz.....
    Downloading mecab-0.996-ko-0.9.2.tar.gz.......
    from https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
    --2021-02-15 14:49:41--  https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
    Resolving bitbucket.org (bitbucket.org)... 104.192.141.1, 2406:da00:ff00::22e9:9f55, 2406:da00:ff00::3403:4be7, ...
    Connecting to bitbucket.org (bitbucket.org)|104.192.141.1|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://bbuseruploads.s3.amazonaws.com/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz?Signature=aFHEfc3q4rkfwwznbyZZsvSmHG4%3D&Expires=1613402306&AWSAccessKeyId=AKIA6KOSE3BNJRRFUUX6&versionId=null&response-content-disposition=attachment%3B%20filename%3D%22mecab-0.996-ko-0.9.2.tar.gz%22&response-content-encoding=None [following]
    --2021-02-15 14:49:42--  https://bbuseruploads.s3.amazonaws.com/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz?Signature=aFHEfc3q4rkfwwznbyZZsvSmHG4%3D&Expires=1613402306&AWSAccessKeyId=AKIA6KOSE3BNJRRFUUX6&versionId=null&response-content-disposition=attachment%3B%20filename%3D%22mecab-0.996-ko-0.9.2.tar.gz%22&response-content-encoding=None
    Resolving bbuseruploads.s3.amazonaws.com (bbuseruploads.s3.amazonaws.com)... 52.217.17.108
    Connecting to bbuseruploads.s3.amazonaws.com (bbuseruploads.s3.amazonaws.com)|52.217.17.108|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1414979 (1.3M) [application/x-tar]
    Saving to: ‘mecab-0.996-ko-0.9.2.tar.gz’
    
    mecab-0.996-ko-0.9. 100%[===================>]   1.35M  2.64MB/s    in 0.5s    
    
    2021-02-15 14:49:43 (2.64 MB/s) - ‘mecab-0.996-ko-0.9.2.tar.gz’ saved [1414979/1414979]
    
    Done
    Unpacking mecab-0.996-ko-0.9.2.tar.gz.......
    Done
    Change Directory to mecab-0.996-ko-0.9.2.......
    installing mecab-0.996-ko-0.9.2.tar.gz........
    configure
    make
    make check
    make install
    ldconfig
    Done
    Change Directory to /content
    Downloading mecab-ko-dic-2.1.1-20180720.tar.gz.......
    from https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
    --2021-02-15 14:51:08--  https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
    Resolving bitbucket.org (bitbucket.org)... 104.192.141.1, 2406:da00:ff00::22c3:9b0a, 2406:da00:ff00::22e9:9f55, ...
    Connecting to bitbucket.org (bitbucket.org)|104.192.141.1|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://bbuseruploads.s3.amazonaws.com/a4fcd83e-34f1-454e-a6ac-c242c7d434d3/downloads/b5a0c703-7b64-45ed-a2d7-180e962710b6/mecab-ko-dic-2.1.1-20180720.tar.gz?Signature=O5vnnNJwQU44mR4Z2oATlFZopX4%3D&Expires=1613402395&AWSAccessKeyId=AKIA6KOSE3BNJRRFUUX6&versionId=tzyxc1TtnZU_zEuaaQDGN4F76hPDpyFq&response-content-disposition=attachment%3B%20filename%3D%22mecab-ko-dic-2.1.1-20180720.tar.gz%22&response-content-encoding=None [following]
    --2021-02-15 14:51:08--  https://bbuseruploads.s3.amazonaws.com/a4fcd83e-34f1-454e-a6ac-c242c7d434d3/downloads/b5a0c703-7b64-45ed-a2d7-180e962710b6/mecab-ko-dic-2.1.1-20180720.tar.gz?Signature=O5vnnNJwQU44mR4Z2oATlFZopX4%3D&Expires=1613402395&AWSAccessKeyId=AKIA6KOSE3BNJRRFUUX6&versionId=tzyxc1TtnZU_zEuaaQDGN4F76hPDpyFq&response-content-disposition=attachment%3B%20filename%3D%22mecab-ko-dic-2.1.1-20180720.tar.gz%22&response-content-encoding=None
    Resolving bbuseruploads.s3.amazonaws.com (bbuseruploads.s3.amazonaws.com)... 52.216.206.115
    Connecting to bbuseruploads.s3.amazonaws.com (bbuseruploads.s3.amazonaws.com)|52.216.206.115|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 49775061 (47M) [application/x-tar]
    Saving to: ‘mecab-ko-dic-2.1.1-20180720.tar.gz’
    
    mecab-ko-dic-2.1.1- 100%[===================>]  47.47M  24.8MB/s    in 1.9s    
    
    2021-02-15 14:51:10 (24.8 MB/s) - ‘mecab-ko-dic-2.1.1-20180720.tar.gz’ saved [49775061/49775061]
    
    Done
    Unpacking  mecab-ko-dic-2.1.1-20180720.tar.gz.......
    Done
    Change Directory to mecab-ko-dic-2.1.1-20180720
    Done
    installing........
    configure
    make
    make install
    apt-get update
    apt-get upgrade
    apt install curl
    apt install git
    bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
    Done
    Successfully Installed
    Now you can use Mecab
    from konlpy.tag import Mecab
    mecab = Mecab()
    사용자 사전 추가 방법 : https://bit.ly/3k0ZH53
    NameError: name 'Tagger' is not defined 오류 발생 시 런타임을 재실행 해주세요
    블로그에 해결 방법을 남겨주신 tana님 감사합니다.



```python
from konlpy.tag import Mecab
import operator
tokenizer = Mecab()
```


```python
text="최강의 슈퍼히어로들이 모였다! 지구의 운명을 건 거대한 전쟁이 시작된다! 지구의 안보가 위협당하는 위기의 상황에서 슈퍼히어로들을 불러모아 세상을 구하는, 일명 어벤져스 작전. 에너지원 테서랙트를 이용한 적의 등장으로 인류가 위험에 처하자 국제평화유지기구인 쉴드의 국장 닉 퓨리는 어벤져스 작전을 위해 전 세계에 흩어져 있던 슈퍼히어로들을 찾아나선다. 아이언맨부터 토르, 헐크, 캡틴 아메리카는 물론, 쉴드의 요원인 블랙 위도우, 호크아이까지, 최고의 슈퍼히어로들이 어벤져스의 멤버로 모이게 되지만, 각기 개성이 강한 이들의 만남은 예상치 못한 방향으로 흘러가는데… 지구의 운명을 건 거대한 전쟁 앞에 어벤져스 작전은 성공할 수 있을까?"
```


```python
print(tokenizer.morphs(text))
```

    ['최강', '의', '슈퍼', '히어로', '들', '이', '모였', '다', '!', '지구', '의', '운명', '을', '건', '거대', '한', '전쟁', '이', '시작', '된다', '!', '지구', '의', '안보', '가', '위협', '당하', '는', '위기', '의', '상황', '에서', '슈퍼', '히어로', '들', '을', '불러', '모아', '세상', '을', '구하', '는', ',', '일', '명', '어벤져스', '작전', '.', '에너지원', '테', '서', '랙', '트', '를', '이용', '한', '적', '의', '등장', '으로', '인류', '가', '위험', '에', '처하', '자', '국제', '평화', '유지', '기구', '인', '쉴드', '의', '국장', '닉', '퓨리', '는', '어벤져스', '작전', '을', '위해', '전', '세계', '에', '흩어져', '있', '던', '슈퍼', '히어로', '들', '을', '찾', '아', '나선다', '.', '아이언맨', '부터', '토르', ',', '헐크', ',', '캡틴', '아메리카', '는', '물론', ',', '쉴드', '의', '요원', '인', '블랙', '위', '도우', ',', '호크아이', '까지', ',', '최고', '의', '슈퍼', '히어로', '들', '이', '어벤져스', '의', '멤버', '로', '모이', '게', '되', '지만', ',', '각기', '개성', '이', '강한', '이', '들', '의', '만남', '은', '예상', '치', '못한', '방향', '으로', '흘러가', '는데', '…', '지구', '의', '운명', '을', '건', '거대', '한', '전쟁', '앞', '에', '어벤져스', '작전', '은', '성공', '할', '수', '있', '을까', '?']



```python
stopwords=['의','가','이','은','다','들','을','는','인','위해','과','던','도','를','로','게','으로','까지','자','에','을까','는데','치','와','한','하다']
```


```python
tokenized_text = [word for word in tokenizer.morphs(text) if not word in stopwords] # 불용어 제거
print(tokenized_text)
```

    ['최강', '슈퍼', '히어로', '모였', '!', '지구', '운명', '건', '거대', '전쟁', '시작', '된다', '!', '지구', '안보', '위협', '당하', '위기', '상황', '에서', '슈퍼', '히어로', '불러', '모아', '세상', '구하', ',', '일', '명', '어벤져스', '작전', '.', '에너지원', '테', '서', '랙', '트', '이용', '적', '등장', '인류', '위험', '처하', '국제', '평화', '유지', '기구', '쉴드', '국장', '닉', '퓨리', '어벤져스', '작전', '전', '세계', '흩어져', '있', '슈퍼', '히어로', '찾', '아', '나선다', '.', '아이언맨', '부터', '토르', ',', '헐크', ',', '캡틴', '아메리카', '물론', ',', '쉴드', '요원', '블랙', '위', '도우', ',', '호크아이', ',', '최고', '슈퍼', '히어로', '어벤져스', '멤버', '모이', '되', '지만', ',', '각기', '개성', '강한', '만남', '예상', '못한', '방향', '흘러가', '…', '지구', '운명', '건', '거대', '전쟁', '앞', '어벤져스', '작전', '성공', '할', '수', '있', '?']


### 2.2 음절 단위 토크나이징 실습


```python
starry_night=['계절이 지나가는 하늘에는',
'가을로 가득 차 있습니다.',
'나는 아무 걱정도 없이',
'가을 속의 별들을 다 헬 듯합니다.',
'가슴 속에 하나 둘 새겨지는 별을',
'이제 다 못 헤는 것은',
'쉬이 아침이 오는 까닭이요,',
'내일 밤이 남은 까닭이요,',
'아직 나의 청춘이 다하지 않은 까닭입니다.',
'별 하나에 추억과',
'별 하나에 사랑과',
'별 하나에 쓸쓸함과',
'별 하나에 동경과',
'별 하나에 시와',
'별 하나에 어머니, 어머니,',
"어머님, 나는 별 하나에 아름다운 말 한마디씩 불러 봅니다. 소학교 때 책상을 같이 했던 아이들의 이름과, 패, 경, 옥, 이런 이국 소녀들의 이름과, 벌써 아기 어머니 된 계집애들의 이름과, 가난한 이웃 사람들의 이름과, 비둘기, 강아지, 토끼, 노새, 노루, '프랑시스 잠', '라이너 마리아 릴케’ 이런 시인의 이름을 불러 봅니다.",
'이네들은 너무나 멀리 있습니다.',
'별이 아스라이 멀듯이.',
'어머님,',
'그리고 당신은 멀리 북간도에 계십니다.',
'나는 무엇인지 그리워',
'이 많은 별빛이 내린 언덕 위에',
'내 이름자를 써 보고',
'흙으로 덮어 버리었습니다.',
'딴은 밤을 새워 우는 벌레는',
'부끄러운 이름을 슬퍼하는 까닭입니다.',
'그러나 겨울이 지나고 나의 별에도 봄이 오면',
'무덤 위에 파란 잔디가 피어나듯이',
'내 이름자 묻힌 언덕 위에도',
'자랑처럼 풀이 무성할 거외다.',]
```


```python
tokens=[]
for sentence in starry_night:
    tokenezied_text = [token for token in sentence] 
    tokens.extend(tokenezied_text)
```

## 빈칸완성 과제 2


```python
vocab_dict={}
for token in tokens:
  ##TODO##
  '''
  vocab_dict에 token을 key로 빈도수를 value로 채워넣으세요.
  예시) vocab_dict={"나":3,"그":5,"어":3,...}
  '''

  if token not in vocab_dict.keys() :
    vocab_dict[token] = 1
  else :
    vocab_dict[token] = vocab_dict[token] + 1


  ##TODO##
```


```python
sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1),reverse=True)
```


```python
vocab=[]
for token,freq in sorted_vocab:
  ##TODO##
  '''
  정렬된 sorted_vocab에서 빈도수가 2 이상인 token을 vocab에 append하여 vocab을 완성시키세요.
  '''

  if freq >= 2:
    vocab.append(token)

  ##TODO##
  
```


```python
answer=[' ','이',',','나','에','다','니','별','는', '하', '.', '아', '름', '을', '의', '과', '가', '은', '어', '지', '들', '리', '무', '머', '도', '까', '닭', '내', '러', '계', '습', '듯', '새', '랑', '시', "'", '멀', '그', '고', '위', '자', '로', '있', '속', '둘', '겨', '오', '요', '밤', '입', '사', '쓸', '경', '님', '운', '한', '마', '디', '불', '봅', '소', '런', '벌', '써', '기', '노', '스', '라', '너', '인', '워', '언', '덕']
```


```python
assert vocab==answer
```


```python

```
