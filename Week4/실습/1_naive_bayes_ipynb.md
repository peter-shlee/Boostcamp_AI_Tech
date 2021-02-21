## **1. NaiveBayes Classifier**
1. 주어진 데이터를 전처리합니다.
2. NaiveBayes 분류기 모델을 구현하고 학습 데이터로 이를 학습시킵니다.
3. 간단한 test case로 결과를 확인합니다.

### **필요 패키지 import**


```python
!pip install konlpy
```

    Requirement already satisfied: konlpy in /usr/local/lib/python3.6/dist-packages (0.5.2)
    Requirement already satisfied: beautifulsoup4==4.6.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (4.6.0)
    Requirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (4.2.6)
    Requirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.6/dist-packages (from konlpy) (1.19.5)
    Requirement already satisfied: colorama in /usr/local/lib/python3.6/dist-packages (from konlpy) (0.4.4)
    Requirement already satisfied: JPype1>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (1.2.1)
    Requirement already satisfied: tweepy>=3.7.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (3.10.0)
    Requirement already satisfied: typing-extensions; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from JPype1>=0.7.0->konlpy) (3.7.4.3)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.15.0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.3.0)
    Requirement already satisfied: requests[socks]>=2.11.1 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (2.23.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->tweepy>=3.7.0->konlpy) (3.1.0)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2020.12.5)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.24.3)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6; extra == "socks" in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.7.1)



```python
from tqdm import tqdm

# 다양한 한국어 형태소 분석기가 클래스로 구현되어 있음
from konlpy import tag 

from collections import defaultdict

import math
```

### **학습 및 테스트 데이터 전처리**

Sample 데이터를 확인합니다.  
긍정($1$), 부정($0$) 2가지 class로 구성되어 있습니다.


```python
train_data = [
  "정말 맛있습니다. 추천합니다.",
  "기대했던 것보단 별로였네요.",
  "다 좋은데 가격이 너무 비싸서 다시 가고 싶다는 생각이 안 드네요.",
  "완전 최고입니다! 재방문 의사 있습니다.",
  "음식도 서비스도 다 만족스러웠습니다.",
  "위생 상태가 좀 별로였습니다. 좀 더 개선되기를 바랍니다.",
  "맛도 좋았고 직원분들 서비스도 너무 친절했습니다.",
  "기념일에 방문했는데 음식도 분위기도 서비스도 다 좋았습니다.",
  "전반적으로 음식이 너무 짰습니다. 저는 별로였네요.",
  "위생에 조금 더 신경 썼으면 좋겠습니다. 조금 불쾌했습니다."
]
train_labels = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]

test_data = [
  "정말 좋았습니다. 또 가고 싶네요.",
  "별로였습니다. 되도록 가지 마세요.",
  "다른 분들께도 추천드릴 수 있을 만큼 만족했습니다.",
  "서비스가 좀 더 개선되었으면 좋겠습니다. 기분이 좀 나빴습니다."
]
```

KoNLPy 패키지에서 제공하는 Twitter(Okt) tokenizer를 사용하여 tokenization합니다.


```python
tokenizer = tag.Okt()
```


```python
def make_tokenized(data):
  tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

  for sent in tqdm(data):
    tokens = tokenizer.morphs(sent)
    tokenized.append(tokens)

  return tokenized
```


```python
train_tokenized = make_tokenized(train_data)
test_tokenized = make_tokenized(test_data)
```

    100%|██████████| 10/10 [00:00<00:00, 103.47it/s]
    100%|██████████| 4/4 [00:00<00:00, 144.23it/s]



```python
train_tokenized
```




    [['정말', '맛있습니다', '.', '추천', '합니다', '.'],
     ['기대했던', '것', '보단', '별로', '였네요', '.'],
     ['다',
      '좋은데',
      '가격',
      '이',
      '너무',
      '비싸서',
      '다시',
      '가고',
      '싶다는',
      '생각',
      '이',
      '안',
      '드네',
      '요',
      '.'],
     ['완전', '최고', '입니다', '!', '재', '방문', '의사', '있습니다', '.'],
     ['음식', '도', '서비스', '도', '다', '만족스러웠습니다', '.'],
     ['위생',
      '상태',
      '가',
      '좀',
      '별로',
      '였습니다',
      '.',
      '좀',
      '더',
      '개선',
      '되',
      '기를',
      '바랍니다',
      '.'],
     ['맛', '도', '좋았고', '직원', '분들', '서비스', '도', '너무', '친절했습니다', '.'],
     ['기념일',
      '에',
      '방문',
      '했는데',
      '음식',
      '도',
      '분위기',
      '도',
      '서비스',
      '도',
      '다',
      '좋았습니다',
      '.'],
     ['전반', '적', '으로', '음식', '이', '너무', '짰습니다', '.', '저', '는', '별로', '였네요', '.'],
     ['위생', '에', '조금', '더', '신경', '썼으면', '좋겠습니다', '.', '조금', '불쾌했습니다', '.']]



학습데이터 기준으로 가장 많이 등장한 단어부터 순서대로 vocab에 추가합니다.


```python
word_count = defaultdict(int)  # Key: 단어, Value: 등장 횟수

for tokens in tqdm(train_tokenized):
  for token in tokens:
    word_count[token] += 1
```

    100%|██████████| 10/10 [00:00<00:00, 22357.70it/s]



```python
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
print(len(word_count))
```

    66



```python
w2i = {}  # Key: 단어, Value: 단어의 index
for pair in tqdm(word_count):
  if pair[0] not in w2i:
    w2i[pair[0]] = len(w2i)
```

    100%|██████████| 66/66 [00:00<00:00, 61887.79it/s]



```python
len(w2i)
```




    66




```python
w2i
```




    {'!': 35,
     '.': 0,
     '가': 41,
     '가격': 23,
     '가고': 26,
     '개선': 43,
     '것': 20,
     '기념일': 52,
     '기대했던': 19,
     '기를': 45,
     '너무': 5,
     '는': 61,
     '다': 3,
     '다시': 25,
     '더': 12,
     '도': 1,
     '되': 44,
     '드네': 30,
     '만족스러웠습니다': 39,
     '맛': 47,
     '맛있습니다': 16,
     '바랍니다': 46,
     '방문': 9,
     '별로': 2,
     '보단': 21,
     '분들': 50,
     '분위기': 54,
     '불쾌했습니다': 65,
     '비싸서': 24,
     '상태': 40,
     '생각': 28,
     '서비스': 7,
     '신경': 62,
     '싶다는': 27,
     '썼으면': 63,
     '안': 29,
     '에': 13,
     '였네요': 8,
     '였습니다': 42,
     '완전': 32,
     '요': 31,
     '위생': 10,
     '으로': 58,
     '음식': 6,
     '의사': 37,
     '이': 4,
     '입니다': 34,
     '있습니다': 38,
     '재': 36,
     '저': 60,
     '적': 57,
     '전반': 56,
     '정말': 15,
     '조금': 14,
     '좀': 11,
     '좋겠습니다': 64,
     '좋았고': 48,
     '좋았습니다': 55,
     '좋은데': 22,
     '직원': 49,
     '짰습니다': 59,
     '최고': 33,
     '추천': 17,
     '친절했습니다': 51,
     '합니다': 18,
     '했는데': 53}



### **모델 Class 구현**

NaiveBayes Classifier 모델 클래스를 구현합니다.

*   `self.k`: Smoothing을 위한 상수.
*   `self.w2i`: 사전에 구한 vocab.
*   `self.priors`: 각 class의 prior 확률.
*   `self.likelihoods`: 각 token의 특정 class 조건 내에서의 likelihood.



```python
class NaiveBayesClassifier():
  def __init__(self, w2i, k=0.1):
    self.k = k
    self.w2i = w2i
    self.priors = {}
    self.likelihoods = {}

  def train(self, train_tokenized, train_labels):
    self.set_priors(train_labels)  # Priors 계산.
    self.set_likelihoods(train_tokenized, train_labels)  # Likelihoods 계산.

  def inference(self, tokens):
    log_prob0 = 0.0
    log_prob1 = 0.0

    for token in tokens:
      if token in self.likelihoods:  # 학습 당시 추가했던 단어에 대해서만 고려.
        log_prob0 += math.log(self.likelihoods[token][0])
        log_prob1 += math.log(self.likelihoods[token][1])

    # 마지막에 prior를 고려.
    log_prob0 += math.log(self.priors[0])
    log_prob1 += math.log(self.priors[1])

    if log_prob0 >= log_prob1:
      return 0
    else:
      return 1

  def set_priors(self, train_labels):
    class_counts = defaultdict(int)
    for label in tqdm(train_labels):
      class_counts[label] += 1
    
    for label, count in class_counts.items():
      self.priors[label] = class_counts[label] / len(train_labels)

  def set_likelihoods(self, train_tokenized, train_labels):
    token_dists = {}  # 각 단어의 특정 class 조건 하에서의 등장 횟수.
    class_counts = defaultdict(int)  # 특정 class에서 등장한 모든 단어의 등장 횟수.

    for i, label in enumerate(tqdm(train_labels)):
      count = 0
      for token in train_tokenized[i]:
        if token in self.w2i:  # 학습 데이터로 구축한 vocab에 있는 token만 고려.
          if token not in token_dists:
            token_dists[token] = {0:0, 1:0}
          token_dists[token][label] += 1
          count += 1
      class_counts[label] += count

    for token, dist in tqdm(token_dists.items()):
      if token not in self.likelihoods:
        self.likelihoods[token] = {
            0:(token_dists[token][0] + self.k) / (class_counts[0] + len(self.w2i)*self.k),
            1:(token_dists[token][1] + self.k) / (class_counts[1] + len(self.w2i)*self.k),
        }
```

### **모델 학습**

모델 객체를 만들고 학습 데이터로 학습시킵니다.


```python
classifier = NaiveBayesClassifier(w2i)
classifier.train(train_tokenized, train_labels)
```

    100%|██████████| 10/10 [00:00<00:00, 59662.93it/s]
    100%|██████████| 10/10 [00:00<00:00, 1389.53it/s]
    100%|██████████| 66/66 [00:00<00:00, 294807.31it/s]


### **테스트**

Test sample에 대한 결과는 다음과 같습니다.


```python
preds = []
for test_tokens in tqdm(test_tokenized):
  pred = classifier.inference(test_tokens)
  preds.append(pred)
```

    100%|██████████| 4/4 [00:00<00:00, 10155.70it/s]



```python
preds
```




    [1, 0, 1, 0]




```python

```
