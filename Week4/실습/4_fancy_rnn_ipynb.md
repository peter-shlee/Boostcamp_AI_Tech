##**4. LSTM, GRU**
1. 기존 RNN과 다른 부분에 대해서 배웁니다.
2. 이전 실습에 이어 다양한 적용법을 배웁니다.

### **필요 패키지 import**


```python
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
```

### **데이터 전처리**

아래의 sample data를 확인해봅시다.  
이전 실습과 동일합니다.


```python
vocab_size = 100
pad_id = 0

data = [
  [85,14,80,34,99,20,31,65,53,86,3,58,30,4,11,6,50,71,74,13],
  [62,76,79,66,32],
  [93,77,16,67,46,74,24,70],
  [19,83,88,22,57,40,75,82,4,46],
  [70,28,30,24,76,84,92,76,77,51,7,20,82,94,57],
  [58,13,40,61,88,18,92,89,8,14,61,67,49,59,45,12,47,5],
  [22,5,21,84,39,6,9,84,36,59,32,30,69,70,82,56,1],
  [94,21,79,24,3,86],
  [80,80,33,63,34,63],
  [87,32,79,65,2,96,43,80,85,20,41,52,95,50,35,96,24,80]
]
```


```python
max_len = len(max(data, key=len))
print(f"Maximum sequence length: {max_len}")

valid_lens = []
for i, seq in enumerate(tqdm(data)):
  valid_lens.append(len(seq))
  if len(seq) < max_len:
    data[i] = seq + [pad_id] * (max_len - len(seq))
```

    100%|██████████| 10/10 [00:00<00:00, 8776.53it/s]

    Maximum sequence length: 20


    



```python
# B: batch size, L: maximum sequence length
batch = torch.LongTensor(data)  # (B, L)
batch_lens = torch.LongTensor(valid_lens)  # (B)

batch_lens, sorted_idx = batch_lens.sort(descending=True)
batch = batch[sorted_idx]

print(batch)
print(batch_lens)
```

    tensor([[85, 14, 80, 34, 99, 20, 31, 65, 53, 86,  3, 58, 30,  4, 11,  6, 50, 71,
             74, 13],
            [58, 13, 40, 61, 88, 18, 92, 89,  8, 14, 61, 67, 49, 59, 45, 12, 47,  5,
              0,  0],
            [87, 32, 79, 65,  2, 96, 43, 80, 85, 20, 41, 52, 95, 50, 35, 96, 24, 80,
              0,  0],
            [22,  5, 21, 84, 39,  6,  9, 84, 36, 59, 32, 30, 69, 70, 82, 56,  1,  0,
              0,  0],
            [70, 28, 30, 24, 76, 84, 92, 76, 77, 51,  7, 20, 82, 94, 57,  0,  0,  0,
              0,  0],
            [19, 83, 88, 22, 57, 40, 75, 82,  4, 46,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0],
            [93, 77, 16, 67, 46, 74, 24, 70,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0],
            [94, 21, 79, 24,  3, 86,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0],
            [80, 80, 33, 63, 34, 63,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0],
            [62, 76, 79, 66, 32,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0]])
    tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])


### **LSTM 사용**

LSTM에선 cell state가 추가됩니다.  
Cell state의 shape는 hidden state의 그것과 동일합니다.


```python
embedding_size = 256
hidden_size = 512
num_layers = 1
num_dirs = 1

embedding = nn.Embedding(vocab_size, embedding_size)
lstm = nn.LSTM(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=True if num_dirs > 1 else False
)

h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)
c_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)
```


```python
# d_w: word embedding size
batch_emb = embedding(batch)  # (B, L, d_w)

packed_batch = pack_padded_sequence(batch_emb.transpose(0, 1), batch_lens)

packed_outputs, (h_n, c_n) = lstm(packed_batch, (h_0, c_0))
print(packed_outputs)
print(packed_outputs[0].shape)
print(h_n.shape)
print(c_n.shape)
```

    PackedSequence(data=tensor([[-0.0368,  0.0854, -0.0029,  ...,  0.0294, -0.0300, -0.0560],
            [-0.1581,  0.1212, -0.0262,  ..., -0.0973, -0.0504,  0.0385],
            [ 0.1136, -0.1164,  0.1244,  ..., -0.0621,  0.0655,  0.0678],
            ...,
            [-0.0588, -0.1139, -0.0181,  ..., -0.2025,  0.0807,  0.0765],
            [ 0.0571, -0.0030, -0.0957,  ...,  0.0196, -0.0941,  0.0816],
            [ 0.0593,  0.1179, -0.0242,  ..., -0.0141, -0.0780,  0.0111]],
           grad_fn=<CatBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
             1,  1]), sorted_indices=None, unsorted_indices=None)
    torch.Size([123, 512])
    torch.Size([1, 10, 512])
    torch.Size([1, 10, 512])



```python
outputs, output_lens = pad_packed_sequence(packed_outputs)
print(outputs.shape)
print(output_lens)
```

    torch.Size([20, 10, 512])
    tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])


### **GRU 사용**

GRU는 cell state가 없어 RNN과 동일하게 사용 가능합니다.   
GRU를 이용하여 LM task를 수행해봅시다.


```python
gru = nn.GRU(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=True if num_dirs > 1 else False
)
```


```python
output_layer = nn.Linear(hidden_size, vocab_size)
```


```python
input_id = batch.transpose(0, 1)[0, :]  # (B)
hidden = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (1, B, d_h)
```

Teacher forcing 없이 이전에 얻은 결과를 다음 input으로 이용합니다.


```python
for t in range(max_len):
  input_emb = embedding(input_id).unsqueeze(0)  # (1, B, d_w)
  output, hidden = gru(input_emb, hidden)  # output: (1, B, d_h), hidden: (1, B, d_h)

  # V: vocab size
  output = output_layer(output)  # (1, B, V)
  probs, top_id = torch.max(output, dim=-1)  # probs: (1, B), top_id: (1, B)

  print("*" * 50)
  print(f"Time step: {t}")
  print(output.shape)
  print(probs.shape)
  print(top_id.shape)

  input_id = top_id.squeeze(0)  # (B)
```

    **************************************************
    Time step: 0
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 1
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 2
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 3
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 4
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 5
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 6
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 7
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 8
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 9
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 10
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 11
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 12
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 13
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 14
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 15
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 16
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 17
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 18
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])
    **************************************************
    Time step: 19
    torch.Size([1, 10, 100])
    torch.Size([1, 10])
    torch.Size([1, 10])


`max_len`만큼의 for 문을 돌면서 모든 결과물의 모양을 확인했지만 만약 종료 조건(예를 들어 문장의 끝을 나타내는 end token 등)이 되면 중간에 생성을 그만둘 수도 있습니다.

### **양방향 및 여러 layer 사용**

이번엔 양방향 + 2개 이상의 layer를 쓸 때 얻을 수 있는 결과에 대해 알아봅니다.


```python
num_layers = 2
num_dirs = 2
dropout=0.1

gru = nn.GRU(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=True if num_dirs > 1 else False
)
```

Bidirectional이 되었고 layer의 개수가 $2$로 늘었기 때문에 hidden state의 shape도 `(4, B, d_h)`가 됩니다.


```python
# d_w: word embedding size, num_layers: layer의 개수, num_dirs: 방향의 개수
batch_emb = embedding(batch)  # (B, L, d_w)
h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h) = (4, B, d_h)

packed_batch = pack_padded_sequence(batch_emb.transpose(0, 1), batch_lens)

packed_outputs, h_n = gru(packed_batch, h_0)
print(packed_outputs)
print(packed_outputs[0].shape)
print(h_n.shape)
```

    PackedSequence(data=tensor([[ 0.0827,  0.1944,  0.0636,  ...,  0.0852,  0.1170,  0.0151],
            [-0.1174,  0.1121,  0.0577,  ...,  0.1434,  0.1489,  0.0837],
            [ 0.0769,  0.0100,  0.0431,  ..., -0.1292,  0.1496,  0.0155],
            ...,
            [ 0.1795,  0.1195,  0.1923,  ...,  0.1332,  0.0480, -0.0261],
            [-0.0340,  0.0286,  0.1525,  ...,  0.0929,  0.1965,  0.0478],
            [-0.1425,  0.0117,  0.1803,  ...,  0.0767,  0.0440, -0.0444]],
           grad_fn=<CatBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
             1,  1]), sorted_indices=None, unsorted_indices=None)
    torch.Size([123, 1024])
    torch.Size([4, 10, 512])



```python
outputs, output_lens = pad_packed_sequence(packed_outputs)

print(outputs.shape)  # (L, B, num_dirs*d_h)
print(output_lens)
```

    torch.Size([20, 10, 1024])
    tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])


각각의 결과물의 shape는 다음과 같습니다.

`outputs`: `(max_len, batch_size, num_dir * hidden_size)`  
`h_n`: `(num_layers*num_dirs, batch_size, hidden_size)`


```python
batch_size = h_n.shape[1]
print(h_n.view(num_layers, num_dirs, batch_size, hidden_size))
print(h_n.view(num_layers, num_dirs, batch_size, hidden_size).shape)
```

    tensor([[[[-1.9674e-01,  2.4627e-01, -1.4445e-01,  ...,  3.1748e-02,
                2.5783e-03, -9.9580e-02],
              [ 2.8595e-01, -4.0082e-02, -6.2906e-02,  ..., -3.6115e-02,
               -2.4756e-01,  1.3468e-01],
              [ 3.9420e-02,  1.3118e-01,  1.7063e-03,  ...,  6.0583e-02,
                1.0845e-02,  1.8837e-01],
              ...,
              [-3.0153e-01, -5.2889e-04,  9.4086e-02,  ..., -6.5434e-02,
               -3.1556e-01, -4.9354e-02],
              [ 1.9736e-01, -3.6229e-01,  2.2990e-01,  ...,  2.6293e-01,
               -2.7147e-01, -3.5277e-01],
              [-1.2331e-02, -1.7408e-01,  2.4146e-01,  ..., -4.2650e-01,
               -4.5176e-01, -1.2860e-01]],
    
             [[-1.5565e-01,  8.6950e-03,  2.1772e-01,  ...,  4.0799e-02,
                1.2468e-01, -2.1024e-01],
              [ 2.6892e-01,  3.8210e-03,  8.7415e-02,  ..., -1.8090e-01,
                1.0469e-01,  4.5651e-01],
              [-5.2029e-04,  1.6376e-01,  9.8803e-03,  ..., -1.5230e-01,
               -2.0754e-01, -5.8677e-01],
              ...,
              [ 1.1634e-01, -1.0958e-01, -2.6420e-02,  ...,  1.2079e-01,
               -1.2191e-01,  4.7661e-02],
              [-9.8796e-02,  5.2233e-01,  1.5400e-02,  ...,  4.7032e-01,
                2.9309e-01,  5.5460e-02],
              [ 1.5093e-01, -4.5746e-02, -5.5501e-02,  ...,  2.9151e-01,
                3.0409e-01,  5.8630e-03]]],
    
    
            [[[-1.4249e-01,  1.1747e-02,  1.8031e-01,  ..., -4.7487e-02,
                2.1114e-01,  5.6704e-02],
              [ 1.6090e-01,  1.2168e-01, -6.2232e-02,  ...,  1.7230e-01,
               -5.1712e-02,  2.7603e-02],
              [ 1.7945e-01,  1.1952e-01,  1.9232e-01,  ...,  2.0816e-01,
               -1.1695e-01,  1.6133e-01],
              ...,
              [-2.4357e-02, -1.8699e-02, -8.9615e-02,  ..., -4.6185e-02,
                8.4001e-02, -3.9337e-03],
              [ 7.7966e-02,  1.8958e-01, -2.4473e-02,  ..., -1.7179e-02,
               -2.8769e-01,  1.0724e-01],
              [ 3.0269e-01, -1.2059e-01, -1.2800e-01,  ..., -1.3279e-01,
                7.1996e-02,  5.9097e-02]],
    
             [[-7.2930e-02, -8.9310e-02, -4.6543e-02,  ...,  8.5154e-02,
                1.1700e-01,  1.5118e-02],
              [ 1.3183e-01, -1.1848e-01, -6.6658e-02,  ...,  1.4336e-01,
                1.4895e-01,  8.3670e-02],
              [-7.3289e-02,  1.9997e-02, -1.2237e-01,  ..., -1.2922e-01,
                1.4955e-01,  1.5539e-02],
              ...,
              [-3.8201e-02, -1.1898e-01,  3.5210e-02,  ..., -1.8293e-01,
                2.2528e-01, -1.5005e-01],
              [ 1.2466e-01,  8.0554e-02, -1.5651e-02,  ...,  2.4804e-01,
                9.9226e-02, -7.3880e-02],
              [-9.7185e-02,  2.0221e-01, -1.6147e-01,  ...,  4.5545e-02,
               -3.4185e-02,  8.7979e-03]]]], grad_fn=<ViewBackward>)
    torch.Size([2, 2, 10, 512])



```python

```
