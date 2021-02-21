##**2. Word2Vec**
1. 주어진 단어들을 word2vec 모델에 들어갈 수 있는 형태로 만듭니다.
2. CBOW, Skip-gram 모델을 각각 구현합니다.
3. 모델을 실제로 학습해보고 결과를 확인합니다.

### **필요 패키지 import**


```python
!pip install konlpy
```

    Collecting konlpy
    [?25l  Downloading https://files.pythonhosted.org/packages/85/0e/f385566fec837c0b83f216b2da65db9997b35dd675e107752005b7d392b1/konlpy-0.5.2-py2.py3-none-any.whl (19.4MB)
    [K     |████████████████████████████████| 19.4MB 14.0MB/s 
    [?25hCollecting beautifulsoup4==4.6.0
    [?25l  Downloading https://files.pythonhosted.org/packages/9e/d4/10f46e5cfac773e22707237bfcd51bbffeaf0a576b0a847ec7ab15bd7ace/beautifulsoup4-4.6.0-py3-none-any.whl (86kB)
    [K     |████████████████████████████████| 92kB 15.8MB/s 
    [?25hCollecting colorama
      Downloading https://files.pythonhosted.org/packages/44/98/5b86278fbbf250d239ae0ecb724f8572af1c91f4a11edf4d36a206189440/colorama-0.4.4-py2.py3-none-any.whl
    Collecting JPype1>=0.7.0
    [?25l  Downloading https://files.pythonhosted.org/packages/de/af/93f92b38ec1ff3091cd38982ed19cea2800fefb609b5801c41fc43c0781e/JPype1-1.2.1-cp36-cp36m-manylinux2010_x86_64.whl (457kB)
    [K     |████████████████████████████████| 460kB 56.7MB/s 
    [?25hRequirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.6/dist-packages (from konlpy) (1.19.5)
    Requirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (4.2.6)
    Collecting tweepy>=3.7.0
      Downloading https://files.pythonhosted.org/packages/67/c3/6bed87f3b1e5ed2f34bd58bf7978e308c86e255193916be76e5a5ce5dfca/tweepy-3.10.0-py2.py3-none-any.whl
    Requirement already satisfied: typing-extensions; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from JPype1>=0.7.0->konlpy) (3.7.4.3)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.3.0)
    Requirement already satisfied: requests[socks]>=2.11.1 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (2.23.0)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.15.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->tweepy>=3.7.0->konlpy) (3.1.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2020.12.5)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (3.0.4)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6; extra == "socks" in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.7.1)
    Installing collected packages: beautifulsoup4, colorama, JPype1, tweepy, konlpy
      Found existing installation: beautifulsoup4 4.6.3
        Uninstalling beautifulsoup4-4.6.3:
          Successfully uninstalled beautifulsoup4-4.6.3
      Found existing installation: tweepy 3.6.0
        Uninstalling tweepy-3.6.0:
          Successfully uninstalled tweepy-3.6.0
    Successfully installed JPype1-1.2.1 beautifulsoup4-4.6.0 colorama-0.4.4 konlpy-0.5.2 tweepy-3.10.0



```python
from tqdm import tqdm
from konlpy.tag import Okt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import torch
import copy
import numpy as np
```

### **데이터 전처리**



데이터를 확인하고 Word2Vec 형식에 맞게 전처리합니다.  
학습 데이터는 1번 실습과 동일하고, 테스트를 위한 단어를 아래와 같이 가정해봅시다.


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

test_words = ["음식", "맛", "서비스", "위생", "가격"]
```

Tokenization과 vocab을 만드는 과정은 이전 실습과 유사합니다.


```python
tokenizer = Okt()
```


```python
def make_tokenized(data):
  tokenized = []
  for sent in tqdm(data):
    tokens = tokenizer.morphs(sent, stem=True)
    tokenized.append(tokens)

  return tokenized
```


```python
train_tokenized = make_tokenized(train_data)
```

    100%|██████████| 10/10 [00:05<00:00,  1.98it/s]



```python
word_count = defaultdict(int)

for tokens in tqdm(train_tokenized):
  for token in tokens:
    word_count[token] += 1
```

    100%|██████████| 10/10 [00:00<00:00, 4121.76it/s]



```python
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
print(list(word_count))
```

    [('.', 14), ('도', 7), ('이다', 4), ('좋다', 4), ('별로', 3), ('다', 3), ('이', 3), ('너무', 3), ('음식', 3), ('서비스', 3), ('하다', 2), ('방문', 2), ('위생', 2), ('좀', 2), ('더', 2), ('에', 2), ('조금', 2), ('정말', 1), ('맛있다', 1), ('추천', 1), ('기대하다', 1), ('것', 1), ('보단', 1), ('가격', 1), ('비싸다', 1), ('다시', 1), ('가다', 1), ('싶다', 1), ('생각', 1), ('안', 1), ('드네', 1), ('요', 1), ('완전', 1), ('최고', 1), ('!', 1), ('재', 1), ('의사', 1), ('있다', 1), ('만족스럽다', 1), ('상태', 1), ('가', 1), ('개선', 1), ('되다', 1), ('기르다', 1), ('바라다', 1), ('맛', 1), ('직원', 1), ('분들', 1), ('친절하다', 1), ('기념일', 1), ('분위기', 1), ('전반', 1), ('적', 1), ('으로', 1), ('짜다', 1), ('저', 1), ('는', 1), ('신경', 1), ('써다', 1), ('불쾌하다', 1)]



```python
w2i = {}
for pair in tqdm(word_count):
  if pair[0] not in w2i:
    w2i[pair[0]] = len(w2i)
```

    100%|██████████| 60/60 [00:00<00:00, 74831.47it/s]



```python
print(train_tokenized)
print(w2i)
```

    [['정말', '맛있다', '.', '추천', '하다', '.'], ['기대하다', '것', '보단', '별로', '이다', '.'], ['다', '좋다', '가격', '이', '너무', '비싸다', '다시', '가다', '싶다', '생각', '이', '안', '드네', '요', '.'], ['완전', '최고', '이다', '!', '재', '방문', '의사', '있다', '.'], ['음식', '도', '서비스', '도', '다', '만족스럽다', '.'], ['위생', '상태', '가', '좀', '별로', '이다', '.', '좀', '더', '개선', '되다', '기르다', '바라다', '.'], ['맛', '도', '좋다', '직원', '분들', '서비스', '도', '너무', '친절하다', '.'], ['기념일', '에', '방문', '하다', '음식', '도', '분위기', '도', '서비스', '도', '다', '좋다', '.'], ['전반', '적', '으로', '음식', '이', '너무', '짜다', '.', '저', '는', '별로', '이다', '.'], ['위생', '에', '조금', '더', '신경', '써다', '좋다', '.', '조금', '불쾌하다', '.']]
    {'.': 0, '도': 1, '이다': 2, '좋다': 3, '별로': 4, '다': 5, '이': 6, '너무': 7, '음식': 8, '서비스': 9, '하다': 10, '방문': 11, '위생': 12, '좀': 13, '더': 14, '에': 15, '조금': 16, '정말': 17, '맛있다': 18, '추천': 19, '기대하다': 20, '것': 21, '보단': 22, '가격': 23, '비싸다': 24, '다시': 25, '가다': 26, '싶다': 27, '생각': 28, '안': 29, '드네': 30, '요': 31, '완전': 32, '최고': 33, '!': 34, '재': 35, '의사': 36, '있다': 37, '만족스럽다': 38, '상태': 39, '가': 40, '개선': 41, '되다': 42, '기르다': 43, '바라다': 44, '맛': 45, '직원': 46, '분들': 47, '친절하다': 48, '기념일': 49, '분위기': 50, '전반': 51, '적': 52, '으로': 53, '짜다': 54, '저': 55, '는': 56, '신경': 57, '써다': 58, '불쾌하다': 59}


실제 모델에 들어가기 위한 input을 만들기 위해 `Dataset` 클래스를 정의합니다.


```python
class CBOWDataset(Dataset):
  def __init__(self, train_tokenized, window_size=2):
    self.x = []
    self.y = []

    for tokens in tqdm(train_tokenized):
      token_ids = [w2i[token] for token in tokens]
      for i, id in enumerate(token_ids):
        if i-window_size >= 0 and i+window_size < len(token_ids):
          self.x.append(token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
          self.y.append(id)

    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수, 2 * window_size)
    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
```


```python
class SkipGramDataset(Dataset):
  def __init__(self, train_tokenized, window_size=2):
    self.x = []
    self.y = []

    for tokens in tqdm(train_tokenized):
      token_ids = [w2i[token] for token in tokens]
      for i, id in enumerate(token_ids):
        if i-window_size >= 0 and i+window_size < len(token_ids):
          self.y += (token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
          self.x += [id] * 2 * window_size

    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수)
    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
```

각 모델에 맞는 `Dataset` 객체를 생성합니다.


```python
cbow_set = CBOWDataset(train_tokenized)
skipgram_set = SkipGramDataset(train_tokenized)
print(list(skipgram_set))
```

    100%|██████████| 10/10 [00:00<00:00, 6311.97it/s]
    100%|██████████| 10/10 [00:00<00:00, 3347.68it/s]


    [(tensor(0), tensor(17)), (tensor(0), tensor(18)), (tensor(0), tensor(19)), (tensor(0), tensor(10)), (tensor(19), tensor(18)), (tensor(19), tensor(0)), (tensor(19), tensor(10)), (tensor(19), tensor(0)), (tensor(22), tensor(20)), (tensor(22), tensor(21)), (tensor(22), tensor(4)), (tensor(22), tensor(2)), (tensor(4), tensor(21)), (tensor(4), tensor(22)), (tensor(4), tensor(2)), (tensor(4), tensor(0)), (tensor(23), tensor(5)), (tensor(23), tensor(3)), (tensor(23), tensor(6)), (tensor(23), tensor(7)), (tensor(6), tensor(3)), (tensor(6), tensor(23)), (tensor(6), tensor(7)), (tensor(6), tensor(24)), (tensor(7), tensor(23)), (tensor(7), tensor(6)), (tensor(7), tensor(24)), (tensor(7), tensor(25)), (tensor(24), tensor(6)), (tensor(24), tensor(7)), (tensor(24), tensor(25)), (tensor(24), tensor(26)), (tensor(25), tensor(7)), (tensor(25), tensor(24)), (tensor(25), tensor(26)), (tensor(25), tensor(27)), (tensor(26), tensor(24)), (tensor(26), tensor(25)), (tensor(26), tensor(27)), (tensor(26), tensor(28)), (tensor(27), tensor(25)), (tensor(27), tensor(26)), (tensor(27), tensor(28)), (tensor(27), tensor(6)), (tensor(28), tensor(26)), (tensor(28), tensor(27)), (tensor(28), tensor(6)), (tensor(28), tensor(29)), (tensor(6), tensor(27)), (tensor(6), tensor(28)), (tensor(6), tensor(29)), (tensor(6), tensor(30)), (tensor(29), tensor(28)), (tensor(29), tensor(6)), (tensor(29), tensor(30)), (tensor(29), tensor(31)), (tensor(30), tensor(6)), (tensor(30), tensor(29)), (tensor(30), tensor(31)), (tensor(30), tensor(0)), (tensor(2), tensor(32)), (tensor(2), tensor(33)), (tensor(2), tensor(34)), (tensor(2), tensor(35)), (tensor(34), tensor(33)), (tensor(34), tensor(2)), (tensor(34), tensor(35)), (tensor(34), tensor(11)), (tensor(35), tensor(2)), (tensor(35), tensor(34)), (tensor(35), tensor(11)), (tensor(35), tensor(36)), (tensor(11), tensor(34)), (tensor(11), tensor(35)), (tensor(11), tensor(36)), (tensor(11), tensor(37)), (tensor(36), tensor(35)), (tensor(36), tensor(11)), (tensor(36), tensor(37)), (tensor(36), tensor(0)), (tensor(9), tensor(8)), (tensor(9), tensor(1)), (tensor(9), tensor(1)), (tensor(9), tensor(5)), (tensor(1), tensor(1)), (tensor(1), tensor(9)), (tensor(1), tensor(5)), (tensor(1), tensor(38)), (tensor(5), tensor(9)), (tensor(5), tensor(1)), (tensor(5), tensor(38)), (tensor(5), tensor(0)), (tensor(40), tensor(12)), (tensor(40), tensor(39)), (tensor(40), tensor(13)), (tensor(40), tensor(4)), (tensor(13), tensor(39)), (tensor(13), tensor(40)), (tensor(13), tensor(4)), (tensor(13), tensor(2)), (tensor(4), tensor(40)), (tensor(4), tensor(13)), (tensor(4), tensor(2)), (tensor(4), tensor(0)), (tensor(2), tensor(13)), (tensor(2), tensor(4)), (tensor(2), tensor(0)), (tensor(2), tensor(13)), (tensor(0), tensor(4)), (tensor(0), tensor(2)), (tensor(0), tensor(13)), (tensor(0), tensor(14)), (tensor(13), tensor(2)), (tensor(13), tensor(0)), (tensor(13), tensor(14)), (tensor(13), tensor(41)), (tensor(14), tensor(0)), (tensor(14), tensor(13)), (tensor(14), tensor(41)), (tensor(14), tensor(42)), (tensor(41), tensor(13)), (tensor(41), tensor(14)), (tensor(41), tensor(42)), (tensor(41), tensor(43)), (tensor(42), tensor(14)), (tensor(42), tensor(41)), (tensor(42), tensor(43)), (tensor(42), tensor(44)), (tensor(43), tensor(41)), (tensor(43), tensor(42)), (tensor(43), tensor(44)), (tensor(43), tensor(0)), (tensor(3), tensor(45)), (tensor(3), tensor(1)), (tensor(3), tensor(46)), (tensor(3), tensor(47)), (tensor(46), tensor(1)), (tensor(46), tensor(3)), (tensor(46), tensor(47)), (tensor(46), tensor(9)), (tensor(47), tensor(3)), (tensor(47), tensor(46)), (tensor(47), tensor(9)), (tensor(47), tensor(1)), (tensor(9), tensor(46)), (tensor(9), tensor(47)), (tensor(9), tensor(1)), (tensor(9), tensor(7)), (tensor(1), tensor(47)), (tensor(1), tensor(9)), (tensor(1), tensor(7)), (tensor(1), tensor(48)), (tensor(7), tensor(9)), (tensor(7), tensor(1)), (tensor(7), tensor(48)), (tensor(7), tensor(0)), (tensor(11), tensor(49)), (tensor(11), tensor(15)), (tensor(11), tensor(10)), (tensor(11), tensor(8)), (tensor(10), tensor(15)), (tensor(10), tensor(11)), (tensor(10), tensor(8)), (tensor(10), tensor(1)), (tensor(8), tensor(11)), (tensor(8), tensor(10)), (tensor(8), tensor(1)), (tensor(8), tensor(50)), (tensor(1), tensor(10)), (tensor(1), tensor(8)), (tensor(1), tensor(50)), (tensor(1), tensor(1)), (tensor(50), tensor(8)), (tensor(50), tensor(1)), (tensor(50), tensor(1)), (tensor(50), tensor(9)), (tensor(1), tensor(1)), (tensor(1), tensor(50)), (tensor(1), tensor(9)), (tensor(1), tensor(1)), (tensor(9), tensor(50)), (tensor(9), tensor(1)), (tensor(9), tensor(1)), (tensor(9), tensor(5)), (tensor(1), tensor(1)), (tensor(1), tensor(9)), (tensor(1), tensor(5)), (tensor(1), tensor(3)), (tensor(5), tensor(9)), (tensor(5), tensor(1)), (tensor(5), tensor(3)), (tensor(5), tensor(0)), (tensor(53), tensor(51)), (tensor(53), tensor(52)), (tensor(53), tensor(8)), (tensor(53), tensor(6)), (tensor(8), tensor(52)), (tensor(8), tensor(53)), (tensor(8), tensor(6)), (tensor(8), tensor(7)), (tensor(6), tensor(53)), (tensor(6), tensor(8)), (tensor(6), tensor(7)), (tensor(6), tensor(54)), (tensor(7), tensor(8)), (tensor(7), tensor(6)), (tensor(7), tensor(54)), (tensor(7), tensor(0)), (tensor(54), tensor(6)), (tensor(54), tensor(7)), (tensor(54), tensor(0)), (tensor(54), tensor(55)), (tensor(0), tensor(7)), (tensor(0), tensor(54)), (tensor(0), tensor(55)), (tensor(0), tensor(56)), (tensor(55), tensor(54)), (tensor(55), tensor(0)), (tensor(55), tensor(56)), (tensor(55), tensor(4)), (tensor(56), tensor(0)), (tensor(56), tensor(55)), (tensor(56), tensor(4)), (tensor(56), tensor(2)), (tensor(4), tensor(55)), (tensor(4), tensor(56)), (tensor(4), tensor(2)), (tensor(4), tensor(0)), (tensor(16), tensor(12)), (tensor(16), tensor(15)), (tensor(16), tensor(14)), (tensor(16), tensor(57)), (tensor(14), tensor(15)), (tensor(14), tensor(16)), (tensor(14), tensor(57)), (tensor(14), tensor(58)), (tensor(57), tensor(16)), (tensor(57), tensor(14)), (tensor(57), tensor(58)), (tensor(57), tensor(3)), (tensor(58), tensor(14)), (tensor(58), tensor(57)), (tensor(58), tensor(3)), (tensor(58), tensor(0)), (tensor(3), tensor(57)), (tensor(3), tensor(58)), (tensor(3), tensor(0)), (tensor(3), tensor(16)), (tensor(0), tensor(58)), (tensor(0), tensor(3)), (tensor(0), tensor(16)), (tensor(0), tensor(59)), (tensor(16), tensor(3)), (tensor(16), tensor(0)), (tensor(16), tensor(59)), (tensor(16), tensor(0))]


### **모델 Class 구현**

차례대로 두 가지 Word2Vec 모델을 구현합니다.  


*   `self.embedding`: `vocab_size` 크기의 one-hot vector를 특정 크기의 `dim` 차원으로 embedding 시키는 layer.
*   `self.linear`: 변환된 embedding vector를 다시 원래 `vocab_size`로 바꾸는 layer.



```python
class CBOW(nn.Module):
  def __init__(self, vocab_size, dim):
    super(CBOW, self).__init__()
    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
    self.linear = nn.Linear(dim, vocab_size)

  # B: batch size, W: window size, d_w: word embedding size, V: vocab size
  def forward(self, x):  # x: (B, 2W)
    embeddings = self.embedding(x)  # (B, 2W, d_w)
    embeddings = torch.sum(embeddings, dim=1)  # (B, d_w)
    output = self.linear(embeddings)  # (B, V)
    return output
```


```python
class SkipGram(nn.Module):
  def __init__(self, vocab_size, dim):
    super(SkipGram, self).__init__()
    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
    self.linear = nn.Linear(dim, vocab_size)

  # B: batch size, W: window size, d_w: word embedding size, V: vocab size
  def forward(self, x): # x: (B)
    embeddings = self.embedding(x)  # (B, d_w)
    output = self.linear(embeddings)  # (B, V)
    return output
```

두 가지 모델을 생성합니다.


```python
cbow = CBOW(vocab_size=len(w2i), dim=256)
skipgram = SkipGram(vocab_size=len(w2i), dim=256)
```

### **모델 학습**

다음과 같이 hyperparamter를 세팅하고 `DataLoader` 객체를 만듭니다.


```python
batch_size=4
learning_rate = 5e-4
num_epochs = 5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cbow_loader = DataLoader(cbow_set, batch_size=batch_size)
skipgram_loader = DataLoader(skipgram_set, batch_size=batch_size)
```

첫번째로 CBOW 모델 학습입니다.


```python
cbow.train()
cbow = cbow.to(device)
optim = torch.optim.SGD(cbow.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

for e in range(1, num_epochs+1):
  print("#" * 50)
  print(f"Epoch: {e}")
  for batch in tqdm(cbow_loader):
    x, y = batch
    x, y = x.to(device), y.to(device) # (B, W), (B)
    output = cbow(x)  # (B, V)
 
    optim.zero_grad()
    loss = loss_function(output, y)
    loss.backward()
    optim.step()

    print(f"Train loss: {loss.item()}")

print("Finished.")
```

    100%|██████████| 16/16 [00:00<00:00, 82.77it/s]
      0%|          | 0/16 [00:00<?, ?it/s]

    ##################################################
    Epoch: 1
    Train loss: 5.309641361236572
    Train loss: 4.743259429931641
    Train loss: 5.340814113616943
    Train loss: 5.136819839477539
    Train loss: 5.279063701629639
    Train loss: 4.443824291229248
    Train loss: 4.57047176361084
    Train loss: 5.023689270019531
    Train loss: 4.349427223205566
    Train loss: 4.771158218383789
    Train loss: 4.419856071472168
    Train loss: 5.07810115814209
    Train loss: 5.634894371032715
    Train loss: 4.297231674194336
    Train loss: 4.326501846313477
    Train loss: 4.075533866882324
    ##################################################
    Epoch: 2


    100%|██████████| 16/16 [00:00<00:00, 582.82it/s]
    100%|██████████| 16/16 [00:00<00:00, 565.67it/s]
    100%|██████████| 16/16 [00:00<00:00, 573.89it/s]
    100%|██████████| 16/16 [00:00<00:00, 511.56it/s]

    Train loss: 5.155217170715332
    Train loss: 4.614955902099609
    Train loss: 5.215330123901367
    Train loss: 5.010554790496826
    Train loss: 5.148907661437988
    Train loss: 4.188363075256348
    Train loss: 4.404176712036133
    Train loss: 4.896225929260254
    Train loss: 4.226801872253418
    Train loss: 4.578815460205078
    Train loss: 4.258672714233398
    Train loss: 4.704931259155273
    Train loss: 5.483585357666016
    Train loss: 4.199884414672852
    Train loss: 4.202408313751221
    Train loss: 3.9598264694213867
    ##################################################
    Epoch: 3
    Train loss: 5.00381326675415
    Train loss: 4.488359451293945
    Train loss: 5.09112548828125
    Train loss: 4.885776519775391
    Train loss: 5.020363807678223
    Train loss: 3.9388630390167236
    Train loss: 4.241365432739258
    Train loss: 4.770888328552246
    Train loss: 4.1081390380859375
    Train loss: 4.38944149017334
    Train loss: 4.101313591003418
    Train loss: 4.340611934661865
    Train loss: 5.334541320800781
    Train loss: 4.104525089263916
    Train loss: 4.081399917602539
    Train loss: 3.8471240997314453
    ##################################################
    Epoch: 4
    Train loss: 4.855476379394531
    Train loss: 4.363490104675293
    Train loss: 4.968209266662598
    Train loss: 4.762503147125244
    Train loss: 4.893405914306641
    Train loss: 3.696077346801758
    Train loss: 4.082159996032715
    Train loss: 4.647589206695557
    Train loss: 3.993488311767578
    Train loss: 4.203427314758301
    Train loss: 3.9480650424957275
    Train loss: 3.986478805541992
    Train loss: 5.187773704528809
    Train loss: 4.011082649230957
    Train loss: 3.963604688644409
    Train loss: 3.7373907566070557
    ##################################################
    Epoch: 5
    Train loss: 4.710251808166504
    Train loss: 4.240370750427246
    Train loss: 4.846593856811523
    Train loss: 4.640750408172607
    Train loss: 4.768010139465332
    Train loss: 3.4609813690185547
    Train loss: 3.926685333251953
    Train loss: 4.52623176574707
    Train loss: 3.882955551147461
    Train loss: 4.021260738372803
    Train loss: 3.799318552017212
    Train loss: 3.644416332244873
    Train loss: 5.043303489685059
    Train loss: 3.919487953186035
    Train loss: 3.849128246307373
    Train loss: 3.630585193634033
    Finished.


    


다음으로 Skip-gram 모델 학습입니다.


```python
skipgram.train()
skipgram = skipgram.to(device)
optim = torch.optim.SGD(skipgram.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

for e in range(1, num_epochs+1):
  print("#" * 50)
  print(f"Epoch: {e}")
  for batch in tqdm(skipgram_loader):
    x, y = batch
    x, y = x.to(device), y.to(device) # (B, W), (B)
    output = skipgram(x)  # (B, V)

    optim.zero_grad()
    loss = loss_function(output, y)
    loss.backward()
    optim.step()

    print(f"Train loss: {loss.item()}")

print("Finished.")
```

      0%|          | 0/64 [00:00<?, ?it/s]

    ##################################################
    Epoch: 1
    Train loss: 4.457622528076172
    Train loss: 5.073709964752197
    Train loss: 3.949445962905884
    Train loss: 3.4684553146362305
    Train loss: 3.8691606521606445
    Train loss: 4.53884220123291
    Train loss: 4.001818656921387
    Train loss: 4.317654132843018
    Train loss: 3.830134153366089
    Train loss: 3.6843857765197754
    Train loss: 4.163984775543213
    Train loss: 4.576426982879639
    Train loss: 3.8710832595825195
    Train loss: 3.7826437950134277
    Train loss: 4.302430152893066
    Train loss: 4.103245735168457
    Train loss: 4.249519348144531
    Train loss: 4.118566513061523
    Train loss: 4.7026686668396
    Train loss: 4.544737815856934
    Train loss: 4.932601451873779
    Train loss: 4.345537185668945
    Train loss: 4.264388561248779
    Train loss: 3.8496224880218506
    Train loss: 4.824462413787842
    Train loss: 3.9269356727600098
    Train loss: 4.125325679779053
    Train loss: 4.634343147277832
    Train loss: 4.016286373138428
    Train loss: 4.294687271118164
    Train loss: 4.190345764160156
    Train loss: 4.645294189453125
    Train loss: 4.069188594818115
    Train loss: 4.371345520019531
    Train loss: 3.9780333042144775


    100%|██████████| 64/64 [00:00<00:00, 674.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 717.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    Train loss: 4.351551532745361
    Train loss: 4.170408248901367
    Train loss: 4.2906036376953125
    Train loss: 4.0317158699035645
    Train loss: 4.158097267150879
    Train loss: 4.120909690856934
    Train loss: 4.160404205322266
    Train loss: 3.4631404876708984
    Train loss: 4.300404071807861
    Train loss: 4.051157474517822
    Train loss: 4.863903999328613
    Train loss: 4.382681846618652
    Train loss: 4.175360202789307
    Train loss: 3.9118547439575195
    Train loss: 3.935551166534424
    Train loss: 4.441970348358154
    Train loss: 4.046594619750977
    Train loss: 4.306316375732422
    Train loss: 4.149672031402588
    Train loss: 4.146487236022949
    Train loss: 4.298356056213379
    Train loss: 3.9377212524414062
    Train loss: 4.346327304840088
    Train loss: 4.37828254699707
    Train loss: 3.815415859222412
    Train loss: 3.9484877586364746
    Train loss: 4.483412265777588
    Train loss: 4.498593330383301
    Train loss: 4.604999542236328
    ##################################################
    Epoch: 2
    Train loss: 4.4222917556762695
    Train loss: 5.022907257080078
    Train loss: 3.919569492340088
    Train loss: 3.4169397354125977
    Train loss: 3.8423166275024414
    Train loss: 4.499055862426758
    Train loss: 3.9736108779907227
    Train loss: 4.288300514221191
    Train loss: 3.799973487854004
    Train loss: 3.6577539443969727
    Train loss: 4.139537811279297
    Train loss: 4.548243999481201
    Train loss: 3.845059394836426
    Train loss: 3.752389430999756
    Train loss: 4.2681427001953125
    Train loss: 4.076652526855469
    Train loss: 4.222167491912842
    Train loss: 4.090350151062012
    Train loss: 4.674025535583496
    Train loss: 4.507462024688721
    Train loss: 4.838905334472656
    Train loss: 4.252385139465332
    Train loss: 4.217565536499023
    Train loss: 3.8271987438201904
    Train loss: 4.776980400085449
    Train loss: 3.8681716918945312
    Train loss: 4.075407981872559
    Train loss: 4.603514194488525
    Train loss: 3.975273609161377
    Train loss: 4.256750106811523
    Train loss: 4.166794776916504
    Train loss: 4.607666015625
    Train loss: 4.035252094268799
    Train loss: 4.343328952789307
    Train loss: 3.9590578079223633
    Train loss: 4.319196701049805
    Train loss: 4.115006446838379
    Train loss: 4.239194869995117
    Train loss: 3.9918789863586426
    Train loss: 4.130194664001465
    Train loss: 4.100735664367676
    Train loss: 4.131728172302246
    Train loss: 3.4081971645355225
    Train loss: 4.247857570648193
    Train loss: 3.9306793212890625
    Train loss: 4.772850036621094
    Train loss: 4.291577339172363
    Train loss: 4.128632545471191
    Train loss: 3.8834877014160156
    Train loss: 3.9066214561462402
    Train loss: 4.402393341064453
    Train loss: 4.004237651824951
    Train loss: 4.282872676849365
    Train loss: 4.119412422180176
    Train loss: 4.1130170822143555
    Train loss: 4.267981052398682
    Train loss: 3.8796000480651855
    Train loss: 4.31463623046875
    Train loss: 4.337627410888672
    Train loss: 3.788594961166382
    Train loss: 3.916391611099243
    Train loss: 4.452829360961914
    Train loss: 4.464711666107178
    Train loss: 4.557013988494873
    ##################################################
    Epoch: 3
    Train loss: 4.38762903213501
    Train loss: 4.9722819328308105
    Train loss: 3.8898801803588867
    Train loss: 3.3668227195739746
    Train loss: 3.8156282901763916
    Train loss: 4.459796905517578
    Train loss: 3.945904016494751
    Train loss: 4.259095668792725
    Train loss: 3.770035743713379
    Train loss: 3.6313252449035645
    Train loss: 4.115222454071045
    Train loss: 4.520171165466309
    Train loss: 3.8195385932922363
    Train loss: 3.722367525100708
    Train loss: 4.234082221984863
    Train loss: 4.0504255294799805
    Train loss: 4.1949782371521
    Train loss: 4.062277793884277
    Train loss: 4.645583629608154
    Train loss: 4.470441818237305
    Train loss: 4.745842456817627
    Train loss: 4.161134719848633
    Train loss: 4.171115875244141
    Train loss: 3.8049161434173584
    Train loss: 4.729860305786133
    Train loss: 3.8108291625976562
    Train loss: 4.026054382324219
    Train loss: 4.573271751403809
    Train loss: 3.934668779373169
    Train loss: 4.219284534454346
    Train loss: 4.1433515548706055
    Train loss: 4.570213794708252
    Train loss: 4.001532077789307
    Train loss: 4.3155364990234375
    Train loss: 3.940126895904541
    Train loss: 4.287006378173828
    Train loss: 4.060249328613281
    Train loss: 4.189169883728027
    Train loss: 3.9527382850646973
    Train loss: 4.102571964263916
    Train loss: 4.080626010894775
    Train loss: 4.103387832641602
    Train loss: 3.3551526069641113


    100%|██████████| 64/64 [00:00<00:00, 646.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 675.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    Train loss: 4.195611953735352
    Train loss: 3.8126630783081055
    Train loss: 4.6824493408203125
    Train loss: 4.202462673187256
    Train loss: 4.082298755645752
    Train loss: 3.8553109169006348
    Train loss: 3.8780484199523926
    Train loss: 4.363343715667725
    Train loss: 3.962555170059204
    Train loss: 4.259486198425293
    Train loss: 4.089845657348633
    Train loss: 4.079795837402344
    Train loss: 4.2377495765686035
    Train loss: 3.8229291439056396
    Train loss: 4.2833452224731445
    Train loss: 4.297436714172363
    Train loss: 3.761930465698242
    Train loss: 3.8845527172088623
    Train loss: 4.422500133514404
    Train loss: 4.431484222412109
    Train loss: 4.509469985961914
    ##################################################
    Epoch: 4
    Train loss: 4.353627681732178
    Train loss: 4.921830654144287
    Train loss: 3.860379934310913
    Train loss: 3.318145751953125
    Train loss: 3.789097547531128
    Train loss: 4.421069145202637
    Train loss: 3.918701648712158
    Train loss: 4.230039119720459
    Train loss: 3.7403225898742676
    Train loss: 3.6051011085510254
    Train loss: 4.091039657592773
    Train loss: 4.492209434509277
    Train loss: 3.794520616531372
    Train loss: 3.6925806999206543
    Train loss: 4.200254440307617
    Train loss: 4.024567604064941
    Train loss: 4.167953014373779
    Train loss: 4.034350395202637
    Train loss: 4.617340564727783
    Train loss: 4.433677673339844
    Train loss: 4.653477191925049
    Train loss: 4.071924686431885
    Train loss: 4.125047206878662
    Train loss: 3.7827765941619873
    Train loss: 4.683108806610107
    Train loss: 3.7549550533294678
    Train loss: 3.9772777557373047
    Train loss: 4.543608665466309
    Train loss: 3.894477128982544
    Train loss: 4.182295322418213
    Train loss: 4.120016098022461
    Train loss: 4.532938480377197
    Train loss: 3.968033790588379
    Train loss: 4.287969589233398
    Train loss: 3.921236515045166
    Train loss: 4.254982948303223
    Train loss: 4.006185531616211
    Train loss: 4.140598297119141
    Train loss: 3.9143104553222656
    Train loss: 4.075231552124023
    Train loss: 4.060579776763916
    Train loss: 4.07538366317749
    Train loss: 3.3040900230407715
    Train loss: 4.143678188323975
    Train loss: 3.6973040103912354
    Train loss: 4.592772006988525
    Train loss: 4.115483283996582
    Train loss: 4.0363688468933105
    Train loss: 3.8273262977600098
    Train loss: 3.8498330116271973
    Train loss: 4.324826240539551
    Train loss: 3.9215619564056396
    Train loss: 4.236155033111572
    Train loss: 4.060961723327637
    Train loss: 4.04682731628418
    Train loss: 4.207663536071777
    Train loss: 3.7677536010742188
    Train loss: 4.252452850341797
    Train loss: 4.257715702056885
    Train loss: 3.7354235649108887
    Train loss: 3.852975368499756
    Train loss: 4.392427921295166
    Train loss: 4.3989033699035645
    Train loss: 4.462374687194824
    ##################################################
    Epoch: 5
    Train loss: 4.320279121398926
    Train loss: 4.871550559997559
    Train loss: 3.8310704231262207
    Train loss: 3.2709481716156006
    Train loss: 3.762725591659546
    Train loss: 4.382875919342041
    Train loss: 3.8920068740844727
    Train loss: 4.2011308670043945
    Train loss: 3.710836410522461
    Train loss: 3.579082489013672
    Train loss: 4.066988945007324
    Train loss: 4.464360237121582
    Train loss: 3.770005226135254
    Train loss: 3.6630313396453857
    Train loss: 4.166664123535156
    Train loss: 3.9990811347961426
    Train loss: 4.141091346740723
    Train loss: 4.0065693855285645
    Train loss: 4.589297771453857
    Train loss: 4.397171497344971
    Train loss: 4.561882972717285
    Train loss: 3.984901189804077
    Train loss: 4.079366683959961
    Train loss: 3.7607812881469727
    Train loss: 4.6367340087890625
    Train loss: 3.700592517852783
    Train loss: 3.9290943145751953
    Train loss: 4.514517784118652
    Train loss: 3.854703903198242
    Train loss: 4.145786285400391
    Train loss: 4.096789360046387
    Train loss: 4.495845317840576
    Train loss: 3.9347622394561768
    Train loss: 4.260629177093506
    Train loss: 3.902381658554077
    Train loss: 4.223127365112305
    Train loss: 3.9528684616088867
    Train loss: 4.093555450439453


    100%|██████████| 64/64 [00:00<00:00, 595.64it/s]

    Train loss: 3.876612663269043
    Train loss: 4.048171520233154
    Train loss: 4.040597438812256
    Train loss: 4.0477142333984375
    Train loss: 3.2551002502441406
    Train loss: 4.092067241668701
    Train loss: 3.5848121643066406
    Train loss: 4.503894805908203
    Train loss: 4.030795097351074
    Train loss: 3.990849018096924
    Train loss: 3.799534797668457
    Train loss: 3.8219757080078125
    Train loss: 4.286844253540039
    Train loss: 3.8812735080718994
    Train loss: 4.212880611419678
    Train loss: 4.032750129699707
    Train loss: 4.014115333557129
    Train loss: 4.177724838256836
    Train loss: 3.7141144275665283
    Train loss: 4.221958160400391
    Train loss: 4.218469619750977
    Train loss: 3.709078788757324
    Train loss: 3.8216633796691895
    Train loss: 4.362615585327148
    Train loss: 4.3669633865356445
    Train loss: 4.41573429107666
    Finished.


    


### **테스트**

학습된 각 모델을 이용하여 test 단어들의 word embedding을 확인합니다.


```python
for word in test_words:
  input_id = torch.LongTensor([w2i[word]]).to(device)
  emb = cbow.embedding(input_id)

  print(f"Word: {word}")
  print(emb.squeeze(0))
```

    Word: 음식
    tensor([-0.4293, -0.0652,  1.1303, -0.1834, -0.9506,  0.2688,  0.5334,  2.0052,
            -1.1672,  0.4643,  0.8156,  1.0592,  0.2037, -0.3098, -0.7090, -0.1711,
             0.2511,  0.4786, -0.2572, -1.2499, -0.4792, -0.8122, -0.7072, -1.1527,
            -0.4406, -0.0068,  0.3375,  0.8846,  0.1567,  0.0922, -1.4243,  0.4051,
            -0.5042, -1.5514,  0.0887,  0.0284,  0.2890,  1.1689,  0.9363, -0.9722,
             0.2330,  0.8422, -0.0169,  0.0836, -0.9623, -0.3025, -2.4602,  0.2750,
             1.1523, -0.7430, -1.6689, -0.1042, -0.9372, -1.7398,  0.4277,  0.1894,
             1.9748, -0.4577, -1.7439, -0.0395, -1.4147,  0.0040, -0.2036,  1.5260,
             1.2839,  1.2874, -0.7682, -0.0584,  0.2749, -2.5223, -0.0968,  0.3198,
            -1.3087,  0.5760,  0.4635,  0.1187,  1.1968,  0.2728,  0.7372, -0.6834,
             0.4395,  2.8486,  0.8478,  0.5302, -1.1423, -0.6843,  0.5660,  0.0458,
            -1.9329, -0.3572,  0.0966,  1.3619,  0.0436,  0.5284, -1.9949,  0.8913,
            -1.7921, -1.6785,  0.0086, -0.0495, -0.3073, -0.0116,  1.2181,  0.2563,
            -0.3132,  1.3982,  0.1274, -0.2246,  0.5539,  0.6105, -2.2571, -0.0368,
            -0.3172, -1.1902, -0.2495, -1.0905,  2.0062, -0.6963, -0.9694,  0.4169,
             0.7542, -0.5644, -0.6837, -1.7581,  0.4743,  1.7088,  0.2860,  0.7797,
             0.1864,  0.2363,  0.0945,  0.8039, -1.7290,  1.2372, -1.0868, -2.2532,
            -1.6678, -0.8033, -0.4449, -0.8897, -0.6837,  1.4143,  1.2510, -0.6561,
            -1.0781, -0.1390, -0.0105,  1.1475,  0.9395, -0.3808, -1.5697, -1.6498,
            -0.0862, -1.1796,  1.0157, -0.7981,  0.8656, -0.9025,  0.7491,  0.5689,
             1.3790, -1.5314, -0.8128,  0.7381, -0.0767, -1.2771, -0.5712, -2.2171,
            -0.5134, -1.5738, -1.6910, -2.1253, -0.2071, -0.9746, -1.6728, -0.7088,
            -0.6768,  0.4711,  0.5807,  1.0912, -0.0449,  0.0726, -0.8683,  0.0069,
             0.5139, -0.5485, -0.4086,  1.4627, -0.3942,  1.4268,  1.0632, -0.8176,
            -1.7746, -0.7625, -1.1763, -0.3506, -2.5060,  0.0791, -1.2057,  0.0052,
             1.5025, -0.5054, -0.6809,  1.6232,  1.1479,  0.0496,  0.1510,  0.3035,
             0.3519,  0.8686,  0.1055, -0.1115, -0.1085,  0.2102,  1.5273, -0.1849,
             0.9468,  0.7216,  0.4538, -0.3129, -0.4916, -0.4469, -1.9747,  0.2039,
            -0.6337,  0.2793,  0.3792, -1.5641,  0.5255,  0.0938, -0.1751,  0.2507,
            -0.0524,  2.7296, -2.9542, -0.8173,  0.4023,  0.9433, -0.8616,  0.2364,
            -1.4984, -0.1120, -1.3821, -1.1357,  0.6079,  0.2085, -0.6372, -2.1529,
             0.5398,  0.6284,  0.0680,  1.5059, -0.8714,  1.0202,  0.3005,  0.3793],
           device='cuda:0', grad_fn=<SqueezeBackward1>)
    Word: 맛
    tensor([ 1.2373e+00,  6.7238e-01,  1.7686e+00, -9.0980e-02, -6.9252e-02,
             1.9812e-01, -2.7067e+00,  1.4420e+00, -1.9675e+00,  8.3786e-01,
            -1.2479e+00,  6.3312e-01, -1.1295e+00,  2.4114e-01,  6.0786e-01,
             3.3203e-01,  7.7830e-01, -1.5361e+00, -4.8093e-01, -5.8339e-01,
             2.8923e-01, -2.3703e-01,  7.3500e-01, -7.5843e-01, -1.6241e+00,
             9.8620e-01, -1.3967e+00,  5.4592e-01, -1.3051e+00, -7.5074e-01,
            -1.2697e+00,  1.7181e+00, -1.1445e+00, -3.7006e-05,  7.2118e-01,
             1.3838e+00, -6.2925e-01,  1.8026e+00, -2.0295e-01, -3.8608e-01,
            -9.8575e-01,  2.0187e-01, -1.4213e+00, -8.0971e-01,  2.8749e-01,
             4.3190e-01, -6.3175e-02, -1.6092e-02, -1.2613e+00, -1.0037e+00,
             6.9658e-01, -6.6121e-01, -2.7285e-01, -5.9299e-01,  1.5639e+00,
             8.0925e-01,  1.2219e+00,  4.8254e-01, -2.9927e-01, -6.6765e-01,
             1.2843e+00, -1.3290e+00,  6.0622e-01,  9.0710e-01, -5.9743e-01,
             1.1762e+00, -6.3400e-01, -1.6636e+00, -5.6744e-01, -9.9414e-01,
             2.6178e-01,  1.3965e+00,  9.6349e-01, -1.4777e+00, -1.0342e+00,
             3.8337e-01, -1.2168e+00, -1.2041e+00,  1.0428e+00, -9.2089e-02,
             1.3236e-01,  1.8316e-01, -1.7243e-01,  1.1591e+00, -5.1766e-01,
            -6.8013e-01,  1.2873e+00,  9.2035e-01, -2.3102e+00,  8.3291e-03,
             8.5855e-01,  4.3421e-01,  1.1721e+00, -1.3415e+00,  1.1120e+00,
            -4.2345e-01,  5.1603e-02,  9.5064e-02, -1.8468e+00, -1.2210e+00,
            -1.5916e+00, -6.9232e-01, -6.6505e-01,  1.5085e-01, -9.2487e-01,
            -7.7922e-01,  1.2111e+00, -3.3526e-01,  1.0349e+00,  2.5555e-01,
            -1.4304e-01, -1.7962e-01,  3.6251e+00, -1.5564e+00, -1.4931e-01,
            -1.7675e+00,  5.4524e-01, -8.2780e-01,  1.1514e+00, -1.2970e-01,
            -1.1017e+00, -4.7669e-01, -4.8598e-01,  3.3816e-01, -4.3951e-01,
            -1.7197e+00,  1.6794e-01,  2.4669e-01,  1.5328e+00, -2.2887e-01,
             1.9945e-02,  1.1223e+00,  5.9835e-01,  1.2839e+00, -1.6074e+00,
            -2.8443e-01, -2.7866e-01,  2.0659e+00,  1.8377e-01, -3.5889e-01,
             8.6550e-01, -1.8423e+00,  4.5853e-01, -4.4479e-01,  8.4187e-01,
            -1.4521e-01, -1.3413e+00, -8.6657e-01, -9.8299e-02,  3.2194e-01,
            -8.8339e-01,  5.3670e-01, -1.4018e+00,  2.7755e-01,  4.4363e-01,
             3.0684e-01,  1.0060e-01, -4.9218e-01,  5.0764e-01, -1.5078e+00,
            -2.8475e-01,  1.5705e+00,  3.4499e-01, -6.1018e-01,  2.3456e-01,
             6.4025e-01, -1.5149e+00, -6.8592e-01, -3.8844e-01,  6.4095e-02,
             1.9060e+00, -7.9307e-01, -1.1359e+00, -1.0466e+00, -4.8482e-01,
            -8.6480e-01,  1.3715e+00, -2.8230e-01, -4.5067e-01,  5.3153e-01,
            -5.2032e-01,  2.4197e-02, -3.5047e-02,  6.3206e-01, -6.5045e-02,
             7.9742e-01, -1.4531e+00, -1.5548e+00, -3.2226e-01,  7.3327e-01,
            -8.1775e-01,  1.9644e-01,  7.5529e-01,  5.8511e-01,  1.0472e+00,
             1.1659e+00, -1.0043e+00,  1.7306e+00,  6.3741e-01,  5.2219e-01,
            -1.2174e+00, -8.1605e-01,  1.1293e+00, -1.1628e-02,  1.8990e+00,
             7.5925e-01,  8.8833e-02,  1.4605e+00, -2.3455e+00, -8.3351e-01,
            -1.3982e+00,  4.5425e-01, -1.4125e+00, -1.4392e+00, -8.1608e-01,
            -2.6067e-01,  5.1932e-01,  3.5265e-01,  6.0916e-01, -2.6353e-01,
             3.5797e-01,  1.5310e+00, -3.2750e-02, -9.8164e-01,  1.0258e+00,
            -1.2261e+00,  1.9384e-01, -6.3503e-01,  4.3284e-01,  3.2434e-01,
             2.7386e+00,  6.2360e-01, -6.5230e-02,  6.3337e-02,  6.8938e-01,
             1.0918e+00,  1.1181e+00, -6.1721e-01, -4.4350e-01, -4.2928e-01,
            -1.3288e+00, -8.8985e-01, -1.2427e+00, -4.4051e-01,  5.2642e-01,
             6.3956e-01,  1.8802e+00, -1.0490e+00,  4.3924e-01, -1.2841e+00,
             6.6764e-01, -5.4555e-02,  4.5893e-01, -6.5951e-01,  3.1594e-02,
            -8.1846e-01], device='cuda:0', grad_fn=<SqueezeBackward1>)
    Word: 서비스
    tensor([ 0.9379,  2.2923,  1.7285,  1.5569, -2.4416,  0.3962, -0.1212, -0.4675,
            -0.2792,  0.4535,  0.2447,  0.1952,  0.3225,  1.0567, -0.3342,  0.0088,
             1.2934,  0.1748,  0.5883,  0.5528,  0.6584,  1.1421,  1.5276, -0.3962,
            -0.9480, -0.8553,  1.4313,  0.6235, -0.1589,  1.5981,  1.5759, -1.4124,
            -0.8847, -0.2674, -0.1363, -0.8426,  0.2807,  1.2832, -0.5163, -0.1619,
             0.0421,  0.4493,  0.9247, -1.7857,  0.5769, -0.6374, -2.6835, -1.4351,
             0.4493,  0.1349, -1.0015, -0.3386,  1.0401,  1.7429,  0.2047, -0.7440,
            -0.9519,  1.6139,  0.7881, -0.2604, -0.2425,  0.0499,  0.4820,  0.9935,
            -0.1583,  1.2704, -0.5097,  0.2429, -0.3710, -0.0552,  1.5871,  0.0438,
            -1.3367,  1.6052, -0.6792, -1.6698, -0.4328,  0.5978,  1.3129, -0.6844,
            -1.1245,  0.1387,  0.5212,  0.6243,  0.1819, -0.7362, -2.0677,  0.1542,
            -0.6473, -0.7440,  0.8880,  0.5631, -0.3977, -1.4503,  1.4075, -0.0736,
             0.3399,  0.0535, -0.9754, -0.6434, -0.9867, -0.8514, -0.7999, -0.1385,
            -1.5816,  0.5082,  0.2444, -0.5951,  0.0566,  0.7145,  0.1643,  1.1628,
             1.4697, -0.5157,  0.9841, -0.3627, -1.2267, -0.1288, -0.4660,  0.8688,
            -0.3530, -1.0376,  0.9862, -0.4583, -1.4294,  0.3372, -0.0538,  0.3470,
            -0.0332, -0.5169, -0.0924,  1.0310, -0.8623, -1.4362, -1.1436, -0.2503,
            -0.7981,  1.9488,  1.4576, -0.8162,  0.1300,  1.7655,  1.8017, -0.2228,
             0.7136, -1.1429, -1.6536, -0.6257, -0.8150, -0.1253,  1.0197, -0.0051,
            -0.1667, -0.4736,  2.5891,  1.1172,  0.9008,  0.7644, -0.7945, -0.5848,
            -0.2244,  0.4719,  1.7637, -1.0852,  1.6822,  1.3565, -1.6695, -0.8265,
            -0.4961,  0.0887,  1.3352, -0.4060,  0.2844, -1.0139, -0.2695, -0.2852,
            -2.2160, -0.7439,  0.8097,  0.1887, -1.2676, -0.0199,  1.5338, -1.2538,
             1.3621,  0.3515,  1.0395, -1.7128, -1.7673,  0.1697, -0.5146,  0.5697,
             0.3824,  1.4783, -0.2305, -0.4217,  0.2540,  0.8329, -1.8519, -0.8207,
             1.9266,  0.9247, -2.1446, -0.5850,  0.5962,  0.6660,  1.5168, -0.8985,
             0.9553,  1.1893, -0.3302, -0.5162,  2.5800, -0.8693, -1.6737,  0.7298,
            -1.6384,  1.3998, -1.4271,  0.4122, -1.3360, -0.2947,  1.5247,  1.3872,
             0.6283,  0.8699, -1.8608,  1.3045,  1.6834,  0.0126, -1.6372,  0.1186,
            -0.4677, -0.1736, -0.3424, -1.2308, -0.9186, -1.6085, -0.8967, -0.2335,
             0.1294,  0.9463, -1.5035,  0.0113, -0.5842, -0.6599,  0.0188,  0.0302,
            -0.0690, -0.5681, -0.7448,  1.0040,  0.5452, -1.7683,  0.2370, -2.1307],
           device='cuda:0', grad_fn=<SqueezeBackward1>)
    Word: 위생
    tensor([ 0.8314, -1.3401, -1.0074, -0.8731, -1.0666, -0.3455,  1.9160,  0.2094,
            -0.4289, -2.4004,  2.8361,  0.0134, -1.1811,  2.2335,  0.4493,  1.0846,
             0.2690,  1.6403, -0.0099, -0.3255,  1.2135,  0.3207,  0.6812, -0.3444,
             0.5169,  0.9566, -1.0011,  0.5933,  1.7448,  0.2701, -1.7056, -0.9107,
            -0.4263,  0.3892, -0.7011,  1.0246, -0.8322,  0.8497, -0.6231, -0.1914,
             1.1066, -1.0094, -0.8716, -0.2644,  1.1228, -0.8536, -0.6681,  0.6362,
             0.7515, -0.1404, -1.3860, -1.3045,  0.0853, -0.4601,  0.1738,  0.1934,
             0.3550,  0.3732,  0.0280,  1.1068, -0.2721,  0.2533, -0.7745, -0.2674,
             0.4181,  1.0969,  1.2837,  0.1595,  1.3116, -1.9846, -0.3650, -0.8644,
             0.2762,  0.8014,  0.6901, -0.3305, -2.1814, -0.2442,  0.4516,  1.0276,
             0.5743, -0.1494, -0.1550,  0.2897, -0.4407, -0.3978, -0.4429,  0.1443,
            -1.0647, -0.9885,  0.4336,  0.3126,  1.7873, -1.2295, -0.0167, -0.0861,
             0.7964, -1.7957,  0.1244,  0.3704, -0.0670, -1.2359,  2.0190,  0.3068,
             0.1034,  2.3015, -0.6755,  1.4818, -0.2243, -0.0978,  2.1437, -0.1560,
            -0.7105,  0.7916, -1.2005, -2.6775,  0.7241, -1.4705, -0.2064, -0.8489,
             1.3360,  1.0800, -0.1803,  0.1867,  1.3199, -0.3470, -0.2456,  0.6236,
             0.8149,  0.7647,  0.5078, -0.6846,  1.0959,  0.4892, -1.7896, -1.2013,
            -0.4128, -0.6785,  0.8398, -0.3775, -0.1298,  1.3537,  0.2246,  1.1370,
            -0.4896, -0.6662, -0.9217, -0.1896,  1.1277,  0.7447,  2.0553,  0.4742,
            -0.0554,  0.7940,  0.1959,  2.2927, -1.1382,  0.2777, -0.2875,  0.1257,
             0.5657, -0.8408,  1.4260, -0.8763,  0.7298, -0.4821,  1.3029,  0.2513,
            -1.0957,  0.1967, -0.3123, -0.7532,  1.3605,  1.0848,  0.9430, -1.9660,
             2.4634, -0.1952,  0.2036,  0.5759, -0.4736,  0.5523,  2.3403, -0.9002,
             0.2994, -0.5734,  1.0282,  0.2174, -1.0261, -1.1004,  1.4826,  0.1550,
            -0.4850, -0.2801,  2.9342, -1.7135,  1.7057, -0.5857,  0.9845,  1.0706,
             0.8491, -0.6536, -1.6596,  1.8674,  0.2519, -1.1705, -0.3978,  1.1519,
            -1.0064,  0.4048,  0.3336,  0.4920,  0.0881,  0.0100, -0.7818,  0.0849,
            -0.6997, -0.1581,  0.1032, -1.1760,  1.2372, -0.0416,  0.4178, -1.2442,
             1.2740,  0.0948,  0.7317, -0.2205, -0.5536,  0.6827,  0.4230,  0.2637,
            -0.2780,  2.8590, -0.9459, -0.4337, -1.1610,  0.5574,  1.7960,  1.5831,
             0.9526, -0.3345, -0.3412, -0.3886,  0.5468,  0.9768,  0.9341, -0.1199,
             1.1606, -1.3568,  0.3475,  0.6597,  0.0257, -0.4587, -0.1053,  1.1492],
           device='cuda:0', grad_fn=<SqueezeBackward1>)
    Word: 가격
    tensor([ 1.8421e+00,  1.2824e+00,  3.6884e-01, -6.6045e-01,  7.3668e-01,
            -2.0728e-01,  4.1799e-01, -1.3773e-01,  4.6497e-01,  2.4157e+00,
            -9.9831e-01, -7.4885e-01,  8.1607e-01, -5.2786e-01, -8.0978e-01,
             3.6691e-01, -1.2676e+00,  6.2797e-01,  7.2549e-01, -9.0046e-01,
            -1.6004e-01,  7.4205e-02, -3.7895e-01, -1.2470e-01, -3.1495e-01,
             4.0359e-01,  1.1304e+00, -1.5708e+00, -1.5001e+00, -9.0479e-01,
            -2.0906e-01,  1.8730e-01, -2.1511e-01, -4.9057e-01,  1.1790e+00,
            -2.5779e-01, -1.3969e+00, -1.1279e+00,  1.0712e+00,  4.5923e-01,
            -6.5717e-01,  1.9671e-01, -7.0305e-01,  2.2218e+00,  7.2411e-01,
            -1.5047e+00, -6.6234e-01,  9.0419e-02, -7.0391e-01, -4.6506e-01,
            -1.6972e+00, -5.3243e-01, -1.0086e+00, -1.2593e+00,  2.2672e-01,
            -2.7241e-01, -7.8648e-01, -8.4663e-02,  1.1077e+00, -5.7359e-01,
             3.2719e-01, -2.9228e-01, -8.5902e-01, -1.6191e-01,  3.8749e-01,
            -9.9001e-01,  7.1000e-01,  1.9856e-01,  2.6544e-01,  5.3796e-01,
            -3.0658e-02,  1.1173e-02, -1.9265e-01, -1.3656e+00,  1.7247e-01,
            -8.9185e-01, -4.3885e-01,  8.3422e-02,  5.7548e-01, -5.4204e-01,
            -9.4079e-01, -6.8145e-01,  1.0223e-01, -1.2078e-01,  1.2085e+00,
             1.0022e+00, -2.9569e-04, -2.5693e-02, -8.7419e-01, -1.7923e+00,
            -3.8313e-01, -1.0291e+00,  8.3825e-01,  7.2215e-01,  7.8326e-01,
            -9.8549e-01,  1.6736e+00,  1.6090e-01, -4.8489e-01,  1.3288e+00,
             2.0634e-01,  3.3853e-01, -1.2571e+00, -2.4162e+00, -9.4853e-01,
            -1.0262e+00,  6.0385e-02, -1.9265e+00, -1.1857e+00,  3.3838e-01,
             1.2647e-01,  7.1377e-01, -2.6937e-01, -1.0568e+00, -1.0148e-01,
            -1.3934e-01,  7.0799e-01, -2.9905e-01, -7.2564e-01, -7.5597e-02,
            -2.2852e+00, -6.0382e-01,  1.2294e+00,  5.0685e-01, -7.3518e-01,
             1.6930e+00,  5.4813e-01,  4.4600e-01, -3.0217e-01,  9.8996e-01,
             1.0236e+00,  5.9278e-01, -1.4220e+00, -3.4430e-01, -8.4553e-01,
            -8.2942e-01,  1.3650e-01,  5.3497e-01,  6.7417e-03,  6.0273e-02,
            -1.9283e-01, -8.3814e-01,  9.3222e-01,  4.6581e-01,  1.3003e-01,
            -1.4746e+00, -5.1584e-01, -7.2456e-01, -5.3743e-01, -3.5362e-01,
            -1.2355e+00,  3.3902e-01,  1.3149e+00, -1.9835e-01, -1.4906e+00,
             1.2136e+00,  3.6983e-01, -3.9485e-01, -2.9097e-01, -1.0469e-01,
             1.6383e+00,  1.5654e+00, -6.5286e-01, -1.2311e+00, -1.4507e+00,
             4.7414e-01,  1.0784e+00,  7.4089e-01, -9.3085e-01,  9.6048e-02,
             2.0606e-01,  3.9478e-01,  1.8172e+00,  1.3362e+00, -1.0931e+00,
            -2.5555e-01,  8.3154e-01, -7.2003e-01, -6.2983e-01,  1.3404e-01,
             1.8682e-01,  1.1956e+00, -1.8302e-01,  8.8520e-01,  3.7233e-02,
            -6.0848e-01,  6.9271e-01, -9.9551e-01,  3.9791e-01,  1.6233e+00,
            -6.2668e-01,  1.6883e+00,  3.8506e-01, -4.0680e-01,  7.1360e-01,
             1.9366e+00, -1.3344e+00, -1.0319e+00, -2.5030e-01, -4.6979e-01,
            -1.5810e+00, -1.0326e+00,  3.7562e-01,  3.6455e-01,  1.3736e+00,
            -6.6000e-01,  1.1862e+00,  2.5092e-01,  4.7011e-01,  8.6986e-01,
             7.0449e-01,  1.2650e+00, -1.1094e+00,  1.2961e+00,  7.1109e-01,
             7.7496e-01, -1.0887e+00, -2.0511e+00, -1.0782e+00, -1.0635e+00,
             1.2162e+00,  6.6391e-01,  1.6964e-01, -1.0508e+00, -5.3322e-01,
             1.4703e-01,  1.3759e+00,  8.0697e-01, -1.5886e-01, -2.0871e-01,
             9.5959e-01,  3.9533e-01, -5.6915e-01,  1.8703e-02,  1.2580e+00,
            -1.3197e+00, -6.7276e-01, -4.7438e-01, -1.8024e+00, -1.0763e+00,
             1.8754e-01, -8.1936e-01,  7.8343e-01,  4.6790e-02,  1.9677e-01,
            -2.2302e-01, -2.1408e+00,  2.3395e+00,  3.8138e-01,  8.5489e-01,
            -9.8706e-01, -1.6425e+00,  3.1865e-01,  3.8724e-01, -9.9045e-02,
             8.0171e-01], device='cuda:0', grad_fn=<SqueezeBackward1>)



```python
for word in test_words:
  input_id = torch.LongTensor([w2i[word]]).to(device)
  emb = skipgram.embedding(input_id)

  print(f"Word: {word}")
  print(max(emb.squeeze(0)))
```

    Word: 음식
    tensor(2.6777, device='cuda:0', grad_fn=<UnbindBackward>)
    Word: 맛
    tensor(2.7393, device='cuda:0', grad_fn=<UnbindBackward>)
    Word: 서비스
    tensor(2.7793, device='cuda:0', grad_fn=<UnbindBackward>)
    Word: 위생
    tensor(2.2207, device='cuda:0', grad_fn=<UnbindBackward>)
    Word: 가격
    tensor(3.0139, device='cuda:0', grad_fn=<UnbindBackward>)



```python
!apt-get install -qq texlive texlive-xetex texlive-latex-extra pandoc
!pip install -qq pypandoc

from google.colab import drive
drive.mount('/content/drive')

!jupyter nbconvert --to PDF '/content/drive/My Drive/Colab Notebooks/1_naive_bayes.ipynb의 사본'
```

    Extracting templates from packages: 100%
    Preconfiguring packages ...
    Selecting previously unselected package fonts-droid-fallback.
    (Reading database ... 146442 files and directories currently installed.)
    Preparing to unpack .../00-fonts-droid-fallback_1%3a6.0.1r16-1.1_all.deb ...
    Unpacking fonts-droid-fallback (1:6.0.1r16-1.1) ...
    Selecting previously unselected package fonts-lato.
    Preparing to unpack .../01-fonts-lato_2.0-2_all.deb ...
    Unpacking fonts-lato (2.0-2) ...
    Selecting previously unselected package poppler-data.
    Preparing to unpack .../02-poppler-data_0.4.8-2_all.deb ...
    Unpacking poppler-data (0.4.8-2) ...
    Selecting previously unselected package tex-common.
    Preparing to unpack .../03-tex-common_6.09_all.deb ...
    Unpacking tex-common (6.09) ...
    Selecting previously unselected package fonts-lmodern.
    Preparing to unpack .../04-fonts-lmodern_2.004.5-3_all.deb ...
    Unpacking fonts-lmodern (2.004.5-3) ...
    Selecting previously unselected package fonts-noto-mono.
    Preparing to unpack .../05-fonts-noto-mono_20171026-2_all.deb ...
    Unpacking fonts-noto-mono (20171026-2) ...
    Selecting previously unselected package fonts-texgyre.
    Preparing to unpack .../06-fonts-texgyre_20160520-1_all.deb ...
    Unpacking fonts-texgyre (20160520-1) ...
    Selecting previously unselected package javascript-common.
    Preparing to unpack .../07-javascript-common_11_all.deb ...
    Unpacking javascript-common (11) ...
    Selecting previously unselected package libcupsfilters1:amd64.
    Preparing to unpack .../08-libcupsfilters1_1.20.2-0ubuntu3.1_amd64.deb ...
    Unpacking libcupsfilters1:amd64 (1.20.2-0ubuntu3.1) ...
    Selecting previously unselected package libcupsimage2:amd64.
    Preparing to unpack .../09-libcupsimage2_2.2.7-1ubuntu2.8_amd64.deb ...
    Unpacking libcupsimage2:amd64 (2.2.7-1ubuntu2.8) ...
    Selecting previously unselected package libijs-0.35:amd64.
    Preparing to unpack .../10-libijs-0.35_0.35-13_amd64.deb ...
    Unpacking libijs-0.35:amd64 (0.35-13) ...
    Selecting previously unselected package libjbig2dec0:amd64.
    Preparing to unpack .../11-libjbig2dec0_0.13-6_amd64.deb ...
    Unpacking libjbig2dec0:amd64 (0.13-6) ...
    Selecting previously unselected package libgs9-common.
    Preparing to unpack .../12-libgs9-common_9.26~dfsg+0-0ubuntu0.18.04.14_all.deb ...
    Unpacking libgs9-common (9.26~dfsg+0-0ubuntu0.18.04.14) ...
    Selecting previously unselected package libgs9:amd64.
    Preparing to unpack .../13-libgs9_9.26~dfsg+0-0ubuntu0.18.04.14_amd64.deb ...
    Unpacking libgs9:amd64 (9.26~dfsg+0-0ubuntu0.18.04.14) ...
    Selecting previously unselected package libjs-jquery.
    Preparing to unpack .../14-libjs-jquery_3.2.1-1_all.deb ...
    Unpacking libjs-jquery (3.2.1-1) ...
    Selecting previously unselected package libkpathsea6:amd64.
    Preparing to unpack .../15-libkpathsea6_2017.20170613.44572-8ubuntu0.1_amd64.deb ...
    Unpacking libkpathsea6:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Selecting previously unselected package libpotrace0.
    Preparing to unpack .../16-libpotrace0_1.14-2_amd64.deb ...
    Unpacking libpotrace0 (1.14-2) ...
    Selecting previously unselected package libptexenc1:amd64.
    Preparing to unpack .../17-libptexenc1_2017.20170613.44572-8ubuntu0.1_amd64.deb ...
    Unpacking libptexenc1:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Selecting previously unselected package rubygems-integration.
    Preparing to unpack .../18-rubygems-integration_1.11_all.deb ...
    Unpacking rubygems-integration (1.11) ...
    Selecting previously unselected package ruby2.5.
    Preparing to unpack .../19-ruby2.5_2.5.1-1ubuntu1.7_amd64.deb ...
    Unpacking ruby2.5 (2.5.1-1ubuntu1.7) ...
    Selecting previously unselected package ruby.
    Preparing to unpack .../20-ruby_1%3a2.5.1_amd64.deb ...
    Unpacking ruby (1:2.5.1) ...
    Selecting previously unselected package rake.
    Preparing to unpack .../21-rake_12.3.1-1ubuntu0.1_all.deb ...
    Unpacking rake (12.3.1-1ubuntu0.1) ...
    Selecting previously unselected package ruby-did-you-mean.
    Preparing to unpack .../22-ruby-did-you-mean_1.2.0-2_all.deb ...
    Unpacking ruby-did-you-mean (1.2.0-2) ...
    Selecting previously unselected package ruby-minitest.
    Preparing to unpack .../23-ruby-minitest_5.10.3-1_all.deb ...
    Unpacking ruby-minitest (5.10.3-1) ...
    Selecting previously unselected package ruby-net-telnet.
    Preparing to unpack .../24-ruby-net-telnet_0.1.1-2_all.deb ...
    Unpacking ruby-net-telnet (0.1.1-2) ...
    Selecting previously unselected package ruby-power-assert.
    Preparing to unpack .../25-ruby-power-assert_0.3.0-1_all.deb ...
    Unpacking ruby-power-assert (0.3.0-1) ...
    Selecting previously unselected package ruby-test-unit.
    Preparing to unpack .../26-ruby-test-unit_3.2.5-1_all.deb ...
    Unpacking ruby-test-unit (3.2.5-1) ...
    Selecting previously unselected package libruby2.5:amd64.
    Preparing to unpack .../27-libruby2.5_2.5.1-1ubuntu1.7_amd64.deb ...
    Unpacking libruby2.5:amd64 (2.5.1-1ubuntu1.7) ...
    Selecting previously unselected package libsynctex1:amd64.
    Preparing to unpack .../28-libsynctex1_2017.20170613.44572-8ubuntu0.1_amd64.deb ...
    Unpacking libsynctex1:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Selecting previously unselected package libtexlua52:amd64.
    Preparing to unpack .../29-libtexlua52_2017.20170613.44572-8ubuntu0.1_amd64.deb ...
    Unpacking libtexlua52:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Selecting previously unselected package libtexluajit2:amd64.
    Preparing to unpack .../30-libtexluajit2_2017.20170613.44572-8ubuntu0.1_amd64.deb ...
    Unpacking libtexluajit2:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Selecting previously unselected package libzzip-0-13:amd64.
    Preparing to unpack .../31-libzzip-0-13_0.13.62-3.1ubuntu0.18.04.1_amd64.deb ...
    Unpacking libzzip-0-13:amd64 (0.13.62-3.1ubuntu0.18.04.1) ...
    Selecting previously unselected package lmodern.
    Preparing to unpack .../32-lmodern_2.004.5-3_all.deb ...
    Unpacking lmodern (2.004.5-3) ...
    Selecting previously unselected package preview-latex-style.
    Preparing to unpack .../33-preview-latex-style_11.91-1ubuntu1_all.deb ...
    Unpacking preview-latex-style (11.91-1ubuntu1) ...
    Selecting previously unselected package t1utils.
    Preparing to unpack .../34-t1utils_1.41-2_amd64.deb ...
    Unpacking t1utils (1.41-2) ...
    Selecting previously unselected package tex-gyre.
    Preparing to unpack .../35-tex-gyre_20160520-1_all.deb ...
    Unpacking tex-gyre (20160520-1) ...
    Selecting previously unselected package texlive-binaries.
    Preparing to unpack .../36-texlive-binaries_2017.20170613.44572-8ubuntu0.1_amd64.deb ...
    Unpacking texlive-binaries (2017.20170613.44572-8ubuntu0.1) ...
    Selecting previously unselected package texlive-base.
    Preparing to unpack .../37-texlive-base_2017.20180305-1_all.deb ...
    Unpacking texlive-base (2017.20180305-1) ...
    Selecting previously unselected package texlive-fonts-recommended.
    Preparing to unpack .../38-texlive-fonts-recommended_2017.20180305-1_all.deb ...
    Unpacking texlive-fonts-recommended (2017.20180305-1) ...
    Selecting previously unselected package texlive-latex-base.
    Preparing to unpack .../39-texlive-latex-base_2017.20180305-1_all.deb ...
    Unpacking texlive-latex-base (2017.20180305-1) ...
    Selecting previously unselected package texlive-latex-recommended.
    Preparing to unpack .../40-texlive-latex-recommended_2017.20180305-1_all.deb ...
    Unpacking texlive-latex-recommended (2017.20180305-1) ...
    Selecting previously unselected package texlive.
    Preparing to unpack .../41-texlive_2017.20180305-1_all.deb ...
    Unpacking texlive (2017.20180305-1) ...
    Selecting previously unselected package texlive-pictures.
    Preparing to unpack .../42-texlive-pictures_2017.20180305-1_all.deb ...
    Unpacking texlive-pictures (2017.20180305-1) ...
    Selecting previously unselected package texlive-latex-extra.
    Preparing to unpack .../43-texlive-latex-extra_2017.20180305-2_all.deb ...
    Unpacking texlive-latex-extra (2017.20180305-2) ...
    Selecting previously unselected package texlive-plain-generic.
    Preparing to unpack .../44-texlive-plain-generic_2017.20180305-2_all.deb ...
    Unpacking texlive-plain-generic (2017.20180305-2) ...
    Selecting previously unselected package tipa.
    Preparing to unpack .../45-tipa_2%3a1.3-20_all.deb ...
    Unpacking tipa (2:1.3-20) ...
    Selecting previously unselected package texlive-xetex.
    Preparing to unpack .../46-texlive-xetex_2017.20180305-1_all.deb ...
    Unpacking texlive-xetex (2017.20180305-1) ...
    Setting up libgs9-common (9.26~dfsg+0-0ubuntu0.18.04.14) ...
    Setting up libkpathsea6:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Setting up libjs-jquery (3.2.1-1) ...
    Setting up libtexlua52:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Setting up fonts-droid-fallback (1:6.0.1r16-1.1) ...
    Setting up libsynctex1:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Setting up libptexenc1:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Setting up tex-common (6.09) ...
    update-language: texlive-base not installed and configured, doing nothing!
    Setting up poppler-data (0.4.8-2) ...
    Setting up tex-gyre (20160520-1) ...
    Setting up preview-latex-style (11.91-1ubuntu1) ...
    Setting up fonts-texgyre (20160520-1) ...
    Setting up fonts-noto-mono (20171026-2) ...
    Setting up fonts-lato (2.0-2) ...
    Setting up libcupsfilters1:amd64 (1.20.2-0ubuntu3.1) ...
    Setting up libcupsimage2:amd64 (2.2.7-1ubuntu2.8) ...
    Setting up libjbig2dec0:amd64 (0.13-6) ...
    Setting up ruby-did-you-mean (1.2.0-2) ...
    Setting up t1utils (1.41-2) ...
    Setting up ruby-net-telnet (0.1.1-2) ...
    Setting up libijs-0.35:amd64 (0.35-13) ...
    Setting up rubygems-integration (1.11) ...
    Setting up libpotrace0 (1.14-2) ...
    Setting up javascript-common (11) ...
    Setting up ruby-minitest (5.10.3-1) ...
    Setting up libzzip-0-13:amd64 (0.13.62-3.1ubuntu0.18.04.1) ...
    Setting up libgs9:amd64 (9.26~dfsg+0-0ubuntu0.18.04.14) ...
    Setting up libtexluajit2:amd64 (2017.20170613.44572-8ubuntu0.1) ...
    Setting up fonts-lmodern (2.004.5-3) ...
    Setting up ruby-power-assert (0.3.0-1) ...
    Setting up texlive-binaries (2017.20170613.44572-8ubuntu0.1) ...
    update-alternatives: using /usr/bin/xdvi-xaw to provide /usr/bin/xdvi.bin (xdvi.bin) in auto mode
    update-alternatives: using /usr/bin/bibtex.original to provide /usr/bin/bibtex (bibtex) in auto mode
    Setting up texlive-base (2017.20180305-1) ...
    mktexlsr: Updating /var/lib/texmf/ls-R-TEXLIVEDIST... 
    mktexlsr: Updating /var/lib/texmf/ls-R-TEXMFMAIN... 
    mktexlsr: Updating /var/lib/texmf/ls-R... 
    mktexlsr: Done.
    tl-paper: setting paper size for dvips to a4: /var/lib/texmf/dvips/config/config-paper.ps
    tl-paper: setting paper size for dvipdfmx to a4: /var/lib/texmf/dvipdfmx/dvipdfmx-paper.cfg
    tl-paper: setting paper size for xdvi to a4: /var/lib/texmf/xdvi/XDvi-paper
    tl-paper: setting paper size for pdftex to a4: /var/lib/texmf/tex/generic/config/pdftexconfig.tex
    Setting up texlive-fonts-recommended (2017.20180305-1) ...
    Setting up texlive-plain-generic (2017.20180305-2) ...
    Setting up texlive-latex-base (2017.20180305-1) ...
    Setting up lmodern (2.004.5-3) ...
    Setting up texlive-latex-recommended (2017.20180305-1) ...
    Setting up texlive-pictures (2017.20180305-1) ...
    Setting up tipa (2:1.3-20) ...
    Regenerating '/var/lib/texmf/fmtutil.cnf-DEBIAN'... done.
    Regenerating '/var/lib/texmf/fmtutil.cnf-TEXLIVEDIST'... done.
    update-fmtutil has updated the following file(s):
    	/var/lib/texmf/fmtutil.cnf-DEBIAN
    	/var/lib/texmf/fmtutil.cnf-TEXLIVEDIST
    If you want to activate the changes in the above file(s),
    you should run fmtutil-sys or fmtutil.
    Setting up texlive (2017.20180305-1) ...
    Setting up texlive-latex-extra (2017.20180305-2) ...
    Setting up texlive-xetex (2017.20180305-1) ...
    Setting up ruby2.5 (2.5.1-1ubuntu1.7) ...
    Setting up ruby (1:2.5.1) ...
    Setting up ruby-test-unit (3.2.5-1) ...
    Setting up rake (12.3.1-1ubuntu0.1) ...
    Setting up libruby2.5:amd64 (2.5.1-1ubuntu1.7) ...
    Processing triggers for mime-support (3.60ubuntu1) ...
    Processing triggers for libc-bin (2.27-3ubuntu1.4) ...
    /sbin/ldconfig.real: /usr/local/lib/python3.6/dist-packages/ideep4py/lib/libmkldnn.so.0 is not a symbolic link
    
    Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
    Processing triggers for fontconfig (2.12.6-0ubuntu2) ...
    Processing triggers for tex-common (6.09) ...
    Running updmap-sys. This may take some time... done.
    Running mktexlsr /var/lib/texmf ... done.
    Building format(s) --all.
    	This may take some time... done.
      Building wheel for pypandoc (setup.py) ... [?25l[?25hdone
    Mounted at /content/drive
    [NbConvertApp] Converting notebook /content/drive/My Drive/Colab Notebooks/1_naive_bayes.ipynb의 사본 to PDF
    [NbConvertApp] Writing 41442 bytes to ./notebook.tex
    [NbConvertApp] Building PDF
    [NbConvertApp] Running xelatex 3 times: [u'xelatex', u'./notebook.tex', '-quiet']
    [NbConvertApp] Running bibtex 1 time: [u'bibtex', u'./notebook']
    [NbConvertApp] WARNING | bibtex had problems, most likely because there were no citations
    [NbConvertApp] PDF successfully created
    [NbConvertApp] Writing 38690 bytes to /content/drive/My Drive/Colab Notebooks/1_naive_bayes.pdf



```python

```
