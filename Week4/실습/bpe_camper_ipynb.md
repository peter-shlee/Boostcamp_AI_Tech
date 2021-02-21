# Natural Language Processing
## Assignment 4: Byte Pair Encoding

### 1. Introduction

- 일반적으로 하나의 단어에 대해 하나의 embedding을 생성할 경우 out-of-vocabulary(OOV)라는 치명적인 문제를 갖게 됩니다. 학습 데이터에서 등장하지 않은 단어가 나오는 경우 Unknown token으로 처리해주어 모델의 입력으로 넣게 되면서 전체적으로 모델의 성능이 저하될 수 있습니다. 반면 모든 단어의 embedding을 만들기에는 필요한 embedding parameter의 수가 지나치게 많습니다.
이러한 문제를 해결하기 위해 컴퓨터가 이해하는 단어를 표현하는 데에 데이터 압축 알고리즘 중 하나인 byte pair encoding 기법을 적용한 sub-word tokenizaiton이라는 개념이 나타났습니다. 
- 본 과제에서는 byte pair encoding을 이용한 간단한 sub-word tokenizer를 구현해봅니다.
과제 노트북의 지시사항과 각 함수의 docstring과 [논문](https://arxiv.org/pdf/1508.07909.pdf)의 3페이지 algorithm 1 참고하여 build_bpe 함수를 완성하고 모든 test case를 통과해주세요.

## 2-1.build_bpe 함수를 완성해주세요.


```python
from typing import List, Dict, Set
from itertools import chain
import re
from collections import defaultdict, Counter


def build_bpe(
        corpus: List[str],
        max_vocab_size: int
) -> List[int]:
    """ BPE Vocabulary Builder
    Implement vocabulary builder for byte pair encoding.
    Please sort your idx2word by subword length in descending manner.

    Hint: Counter in collection library would be helpful

    Note: If you convert sentences list to word frequence dictionary,
          building speed is enhanced significantly because duplicated words are
          preprocessed together

    Arguments:
    corpus -- List of words to build vocab
    max_vocab_size -- The maximum size of vocab

    Return:
    idx2word -- Subword list
    """
    # Special tokens
    PAD = BytePairEncoding.PAD_token  # Index of <PAD> must be 0
    UNK = BytePairEncoding.UNK_token  # Index of <UNK> must be 1
    CLS = BytePairEncoding.CLS_token  # Index of <CLS> must be 2
    SEP = BytePairEncoding.SEP_token  # Index of <SEP> must be 3
    MSK = BytePairEncoding.MSK_token  # Index of <MSK> must be 4
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]

    WORD_END = BytePairEncoding.WORD_END  # Use this token as the end of a word
    # YOUR CODE HERE
    idx2word = None

    def get_stats(dictionary):
      pairs = defaultdict(int)
      for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
          pairs[symbols[i],symbols[i+1]] += freq
      return pairs

    def merge_dictionary(pair, v_in):
      v_out = {}
      bigram = re.escape(' '.join(pair))
      p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
      for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
      return v_out

    vocab = set() # vocabulary에는 중복된 값이 들어가면 안되므로 set을 이용
    for i in range(0, len(corpus)): # vocabulary 초기화
      corpus[i] = " ".join(corpus[i]) + " _"
      vocab.update(corpus[i].split())
    dictionary = dict(Counter(corpus)) # dictionary 초기화

    while len(vocab) < max_vocab_size - len(SPECIAL): # vocabulary의 element 수가 max_vocab_size가 될 때까지 반복
      pairs = get_stats(dictionary)

      try:
        best = max(pairs, key=pairs.get)
      except ValueError:
        break
      
      dictionary = merge_dictionary(best, dictionary)
      vocab.add("".join(best)) # vocabulary 갱신

    vocab = list(vocab) # set이었던 vocabulary를 list로 변환
    vocab.sort(key=len, reverse=True) # 길이 내림차순으로 정렬
    idx2word = SPECIAL + vocab # special token들을 vocabulary의 앞에 넣어줌
    # print(idx2word)
    
    return idx2word
```

## 2-2. build_bpe 함수 평가


```python
#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class BytePairEncoding(object):
    """ Byte Pair Encoding class
    We aren't gonna use this class for encoding. Because it is too slow......
    We will use sentence piece Google have made.
    Thus, this class is just for special token index reference.
    """
    PAD_token = '<pad>'
    PAD_token_idx = 0
    UNK_token = '<unk>'
    UNK_token_idx = 1
    CLS_token = '<cls>'
    CLS_token_idx = 2
    SEP_token = '<sep>'
    SEP_token_idx = 3
    MSK_token = '<msk>'
    MSK_token_idx = 4

    WORD_END = '_'

    def __init__(self, corpus: List[List[str]], max_vocab_size: int) -> None:
        self.idx2word = build_bpe(corpus, max_vocab_size)

    def encode(self, sentence: List[str]) -> List[int]:
        return encode(sentence, self.idx2word)

    def decoder(self, tokens: List[int]) -> List[str]:
        return decode(tokens, self.idx2word)


#############################################
# Testing functions below.                  #
#############################################


def test_build_bpe():
    print("======Building BPE Vocab Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    WORD_END = BytePairEncoding.WORD_END

    # First test
    corpus = ['abcde']
    vocab = build_bpe(corpus, max_vocab_size=15)
    assert vocab[:5] == [PAD, UNK, CLS, SEP, MSK], \
        "Please insert the special tokens properly"
    print("The first test passed!")

    # Second test
    assert sorted(vocab[5:], key=len, reverse=True) == vocab[5:], \
        "Please sort your idx2word by subword length in decsending manner."
    print("The second test passed!")

    # Third test
    corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
    vocab = set(build_bpe(corpus, max_vocab_size=24))
    assert vocab > {PAD, UNK, CLS, SEP, MSK, 'est_', 'low', 'newest_', \
                    'i', 'e', 'n', 't', 'd', 's', 'o', 'l', 'r', 'w',
                    WORD_END} and \
           "low_" not in vocab and "wi" not in vocab and "id" not in vocab, \
        "Your bpe result does not match expected result"
    print("The third test passed!")

    # forth test
    corpus = ['aaaaaaaaaaaa', 'abababab']
    vocab = set(build_bpe(corpus, max_vocab_size=13))
    assert vocab == {PAD, UNK, CLS, SEP, MSK, 'aaaaaaaa', 'aaaa', 'abab', 'aa',
                     'ab', 'a', 'b', WORD_END}, \
        "Your bpe result does not match expected result"
    print("The forth test passed!")

    # fifth test
    corpus = ['abc', 'bcd']
    vocab = build_bpe(corpus, max_vocab_size=10000)
    assert len(vocab) == 15, \
        "Your bpe result does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")
test_build_bpe()
```

    ======Building BPE Vocab Test Case======
    The first test passed!
    The second test passed!
    The third test passed!
    The forth test passed!
    The fifth test passed!
    All 5 tests passed!



```python

```
