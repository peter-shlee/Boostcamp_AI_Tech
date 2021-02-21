##**5. Seq2seq**
1. Encoder를 구현합니다.
2. Decoder를 구현합니다.
3. Seq2seq 모델을 구축하고 사용합니다.

### **필요 패키지 import**


```python
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import random
```

### **데이터 전처리**

`src_data`를 `trg_data`로 바꾸는 task를 수행하기 위한 sample data입니다.  
전체 단어 수는 $100$개이고 다음과 같이 pad token, start token, end token의 id도 정의합니다.


```python
vocab_size = 100
pad_id = 0
sos_id = 1
eos_id = 2

src_data = [
  [3, 77, 56, 26, 3, 55, 12, 36, 31],
  [58, 20, 65, 46, 26, 10, 76, 44],
  [58, 17, 8],
  [59],
  [29, 3, 52, 74, 73, 51, 39, 75, 19],
  [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93],
  [39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99, 5],
  [75, 34, 17, 3, 86, 88],
  [63, 39, 5, 35, 67, 56, 68, 89, 55, 66],
  [12, 40, 69, 39, 49]
]

trg_data = [
  [75, 13, 22, 77, 89, 21, 13, 86, 95],
  [79, 14, 91, 41, 32, 79, 88, 34, 8, 68, 32, 77, 58, 7, 9, 87],
  [85, 8, 50, 30],
  [47, 30],
  [8, 85, 87, 77, 47, 21, 23, 98, 83, 4, 47, 97, 40, 43, 70, 8, 65, 71, 69, 88],
  [32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18],
  [37, 14, 49, 24, 93, 37, 54, 51, 39, 84],
  [16, 98, 68, 57, 55, 46, 66, 85, 18],
  [20, 70, 14, 6, 58, 90, 30, 17, 91, 18, 90],
  [37, 93, 98, 13, 45, 28, 89, 72, 70]
]
```

각각의 데이터를 전처리합니다.


```python
trg_data = [[sos_id]+seq+[eos_id] for seq in tqdm(trg_data)]
```

    100%|██████████| 10/10 [00:00<00:00, 41040.16it/s]



```python
def padding(data, is_src=True):
  max_len = len(max(data, key=len))
  print(f"Maximum sequence length: {max_len}")

  valid_lens = []
  for i, seq in enumerate(tqdm(data)):
    valid_lens.append(len(seq))
    if len(seq) < max_len:
      data[i] = seq + [pad_id] * (max_len - len(seq))

  return data, valid_lens, max_len
```


```python
src_data, src_lens, src_max_len = padding(src_data)
trg_data, trg_lens, trg_max_len = padding(trg_data)
```

    100%|██████████| 10/10 [00:00<00:00, 22745.68it/s]
    100%|██████████| 10/10 [00:00<00:00, 24300.72it/s]

    Maximum sequence length: 15
    Maximum sequence length: 22


    



```python
# B: batch size, S_L: source maximum sequence length, T_L: target maximum sequence length
src_batch = torch.LongTensor(src_data)  # (B, S_L)
src_batch_lens = torch.LongTensor(src_lens)  # (B)
trg_batch = torch.LongTensor(trg_data)  # (B, T_L)
trg_batch_lens = torch.LongTensor(trg_lens)  # (B)

print(src_batch.shape)
print(src_batch_lens.shape)
print(trg_batch.shape)
print(trg_batch_lens.shape)
```

    torch.Size([10, 15])
    torch.Size([10])
    torch.Size([10, 22])
    torch.Size([10])


PackedSquence를 사용을 위해 source data를 기준으로 정렬합니다.


```python
src_batch_lens, sorted_idx = src_batch_lens.sort(descending=True)
src_batch = src_batch[sorted_idx]
trg_batch = trg_batch[sorted_idx]
trg_batch_lens = trg_batch_lens[sorted_idx]

print(src_batch)
print(src_batch_lens)
print(trg_batch)
print(trg_batch_lens)
```

    tensor([[39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99,  5],
            [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93,  0,  0,  0,  0],
            [63, 39,  5, 35, 67, 56, 68, 89, 55, 66,  0,  0,  0,  0,  0],
            [ 3, 77, 56, 26,  3, 55, 12, 36, 31,  0,  0,  0,  0,  0,  0],
            [29,  3, 52, 74, 73, 51, 39, 75, 19,  0,  0,  0,  0,  0,  0],
            [58, 20, 65, 46, 26, 10, 76, 44,  0,  0,  0,  0,  0,  0,  0],
            [75, 34, 17,  3, 86, 88,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [12, 40, 69, 39, 49,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [58, 17,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [59,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    tensor([15, 11, 10,  9,  9,  8,  6,  5,  3,  1])
    tensor([[ 1, 37, 14, 49, 24, 93, 37, 54, 51, 39, 84,  2,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0],
            [ 1, 32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18,  2,  0,  0,  0,  0,
              0,  0,  0,  0],
            [ 1, 20, 70, 14,  6, 58, 90, 30, 17, 91, 18, 90,  2,  0,  0,  0,  0,  0,
              0,  0,  0,  0],
            [ 1, 75, 13, 22, 77, 89, 21, 13, 86, 95,  2,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0],
            [ 1,  8, 85, 87, 77, 47, 21, 23, 98, 83,  4, 47, 97, 40, 43, 70,  8, 65,
             71, 69, 88,  2],
            [ 1, 79, 14, 91, 41, 32, 79, 88, 34,  8, 68, 32, 77, 58,  7,  9, 87,  2,
              0,  0,  0,  0],
            [ 1, 16, 98, 68, 57, 55, 46, 66, 85, 18,  2,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0],
            [ 1, 37, 93, 98, 13, 45, 28, 89, 72, 70,  2,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0],
            [ 1, 85,  8, 50, 30,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0],
            [ 1, 47, 30,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0]])
    tensor([12, 14, 13, 11, 22, 18, 11, 11,  6,  4])


### **Encoder 구현**


```python
embedding_size = 256
hidden_size = 512
num_layers = 2
num_dirs = 2
dropout = 0.1
```

Bidirectional GRU를 이용한 Encoder입니다.


*   `self.embedding`: word embedding layer.
*   `self.gru`: encoder 역할을 하는 Bi-GRU.
*   `self.linear`: 양/단방향 concat된 hidden state를 decoder의 hidden size에 맞게 linear transformation.



```python
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.gru = nn.GRU(
        input_size=embedding_size, 
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True if num_dirs > 1 else False,
        dropout=dropout
    )
    self.linear = nn.Linear(num_dirs * hidden_size, hidden_size)

  def forward(self, batch, batch_lens):  # batch: (B, S_L), batch_lens: (B)
    # d_w: word embedding size
    batch_emb = self.embedding(batch)  # (B, S_L, d_w)
    batch_emb = batch_emb.transpose(0, 1)  # (S_L, B, d_w)

    packed_input = pack_padded_sequence(batch_emb, batch_lens)

    h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers*num_dirs, B, d_h) = (4, B, d_h)
    packed_outputs, h_n = self.gru(packed_input, h_0)  # h_n: (4, B, d_h)
    outputs = pad_packed_sequence(packed_outputs)[0]  # outputs: (S_L, B, 2d_h)

    forward_hidden = h_n[-2, :, :]
    backward_hidden = h_n[-1, :, :]
    hidden = self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1)).unsqueeze(0)  # (1, B, d_h)

    return outputs, hidden
```


```python
encoder = Encoder()
```

### **Decoder 구현**

동일한 설정의 Bi-GRU로 만든 Decoder입니다.

*   `self.embedding`: word embedding layer.
*   `self.gru`: decoder 역할을 하는 Bi-GRU.
*   `self.output_layer`: decoder에서 나온 hidden state를 `vocab_size`로 linear transformation하는 layer.


```python
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.gru = nn.GRU(
        input_size=embedding_size, 
        hidden_size=hidden_size,
    )
    self.output_layer = nn.Linear(hidden_size, vocab_size)

  def forward(self, batch, hidden):  # batch: (B), hidden: (1, B, d_h)
    batch_emb = self.embedding(batch)  # (B, d_w)
    batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

    outputs, hidden = self.gru(batch_emb, hidden)  # outputs: (1, B, d_h), hidden: (1, B, d_h)
    
    # V: vocab size
    outputs = self.output_layer(outputs)  # (1, B, V)

    return outputs.squeeze(0), hidden
```


```python
decoder = Decoder()
```

### **Seq2seq 모델 구축**

생성한 encoder와 decoder를 합쳐 Seq2seq 모델을 구축합니다.


*   `self.encoder`: encoder.
*   `self.decoder`: decoder.



```python
class Seq2seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2seq, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
    # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L)

    _, hidden = self.encoder(src_batch, src_batch_lens)  # hidden: (1, B, d_h)

    input_ids = trg_batch[:, 0]  # (B)
    batch_size = src_batch.shape[0]
    outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

    for t in range(1, trg_max_len):
      decoder_outputs, hidden = self.decoder(input_ids, hidden)  # decoder_outputs: (B, V), hidden: (1, B, d_h)

      outputs[t] = decoder_outputs
      _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)

      input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

    return outputs
```


```python
seq2seq = Seq2seq(encoder, decoder)
```

### **모델 사용해보기**

학습 과정이라고 가정하고 모델에 input을 넣어봅니다.


```python
outputs = seq2seq(src_batch, src_batch_lens, trg_batch)

print(outputs)
print(outputs.shape)
```

    tensor([[[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
               0.0000e+00,  0.0000e+00],
             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
               0.0000e+00,  0.0000e+00],
             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
               0.0000e+00,  0.0000e+00],
             ...,
             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
               0.0000e+00,  0.0000e+00],
             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
               0.0000e+00,  0.0000e+00],
             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
               0.0000e+00,  0.0000e+00]],
    
            [[-7.6779e-02,  9.1174e-03,  9.1843e-02,  ..., -1.1557e-01,
               1.2939e-01,  2.8926e-02],
             [-9.2589e-02,  2.7958e-02,  9.7246e-02,  ..., -1.5671e-01,
               1.3749e-01,  2.8398e-02],
             [-5.6227e-02, -2.1124e-02,  9.1052e-02,  ..., -7.9320e-02,
               1.5984e-01,  4.7053e-02],
             ...,
             [-7.5632e-02,  1.1479e-03,  5.5590e-02,  ..., -1.3967e-01,
               1.7347e-01,  6.6512e-02],
             [-7.4429e-02, -2.7368e-02,  1.0590e-01,  ..., -1.1339e-01,
               1.6680e-01,  5.8511e-02],
             [-5.5577e-02,  4.6655e-03,  1.0566e-01,  ..., -1.1553e-01,
               1.3833e-01,  3.6371e-02]],
    
            [[-1.4969e-01,  9.8947e-02,  9.5138e-02,  ..., -1.3484e-01,
               5.1889e-02,  8.2625e-02],
             [-3.1474e-02,  1.7022e-01,  1.2204e-01,  ..., -1.6907e-01,
               5.2360e-02,  7.1318e-02],
             [ 7.5450e-02, -4.7441e-02, -1.2070e-02,  ...,  1.2345e-01,
               1.2105e-01, -7.6586e-02],
             ...,
             [-1.5773e-01,  9.1278e-02,  7.3640e-02,  ..., -1.5237e-01,
               6.5740e-02,  9.5734e-02],
             [-6.1886e-02, -4.4230e-03, -7.7863e-02,  ..., -9.8948e-02,
               1.7782e-01,  2.2267e-01],
             [-2.8301e-02, -3.3961e-02, -6.3985e-02,  ..., -9.6131e-02,
              -2.1989e-02,  2.9117e-01]],
    
            ...,
    
            [[ 2.0332e-02,  9.9809e-02,  1.2321e-03,  ...,  1.2803e-01,
              -9.0257e-02,  1.6155e-01],
             [-9.5852e-02,  8.3042e-02,  7.0774e-02,  ...,  1.6385e-01,
              -1.1734e-01,  9.8856e-02],
             [-1.0087e-01,  1.4002e-01,  1.4806e-01,  ...,  1.2824e-01,
              -1.4063e-01,  1.3920e-01],
             ...,
             [-2.9857e-02,  7.6422e-02,  7.7508e-02,  ...,  5.5725e-02,
              -9.5999e-02,  1.3610e-01],
             [-9.6485e-02,  8.5216e-02,  7.4623e-02,  ...,  1.6903e-01,
              -1.2486e-01,  9.9195e-02],
             [ 2.6710e-03,  1.1470e-01,  1.2682e-02,  ...,  1.2895e-01,
              -9.6135e-02,  1.5149e-01]],
    
            [[-1.7358e-01,  8.4905e-02,  1.2697e-01,  ...,  5.9132e-03,
              -9.6270e-02,  3.4440e-02],
             [-5.9522e-02, -8.6492e-04,  2.9375e-02,  ...,  1.8951e-01,
              -6.3578e-03,  7.5498e-02],
             [-6.8534e-02,  2.9085e-02,  7.2518e-02,  ...,  1.7379e-01,
              -2.8439e-02,  1.0569e-01],
             ...,
             [-2.7942e-02, -2.9895e-03,  3.3962e-02,  ...,  1.2409e-01,
               2.9254e-04,  9.7532e-02],
             [-5.9347e-02,  3.6594e-05,  3.0647e-02,  ...,  1.9284e-01,
              -9.6632e-03,  7.6258e-02],
             [-1.4047e-02,  2.2872e-02,  1.5629e-02,  ...,  1.7719e-01,
               5.9271e-03,  1.0661e-01]],
    
            [[-1.2378e-01,  1.2117e-01,  1.4247e-01,  ...,  1.0245e-01,
              -1.1476e-01,  1.2026e-01],
             [-8.8873e-02,  3.7576e-02,  6.9959e-02,  ...,  1.7593e-01,
              -8.5881e-02,  9.0971e-02],
             [-9.6943e-02,  5.9408e-02,  9.7414e-02,  ...,  1.6782e-01,
              -9.3794e-02,  1.1357e-01],
             ...,
             [-6.8361e-02,  4.3691e-02,  7.0832e-02,  ...,  1.3607e-01,
              -7.4270e-02,  9.2990e-02],
             [-8.8879e-02,  3.8660e-02,  7.1090e-02,  ...,  1.7780e-01,
              -8.8031e-02,  9.1736e-02],
             [-6.2233e-02,  5.4119e-02,  6.2029e-02,  ...,  1.6518e-01,
              -6.9187e-02,  1.1317e-01]]], grad_fn=<CopySlices>)
    torch.Size([22, 10, 100])


Language Modeling에 대한 loss 계산을 위해 shift한 target과 비교합니다.

![seq2seq-1.png](data:image/png;base64,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)


```python
loss_function = nn.CrossEntropyLoss()

preds = outputs[1:, :, :].transpose(0, 1)  # (B, T_L-1, V)
loss = loss_function(preds.contiguous().view(-1, vocab_size), trg_batch[:,1:].contiguous().view(-1, 1).squeeze(1))

print(loss)
```

    tensor(4.6300, grad_fn=<NllLossBackward>)


실제 inference에선 teacher forcing 없이 이전 결과만을 가지고 생성합니다.


```python
src_sent = [4, 10, 88, 46, 72, 34, 14, 51]
src_len = len(src_sent)

src_batch = torch.LongTensor(src_sent).unsqueeze(0)  # (1, L)
src_batch_lens = torch.LongTensor([src_len])  # (1)

_, hidden = seq2seq.encoder(src_batch, src_batch_lens)  # hidden: (1, 1, d_h)
```


```python
input_id = torch.LongTensor([sos_id]) # (1)
output = []

for t in range(1, trg_max_len):
  decoder_output, hidden = seq2seq.decoder(input_id, hidden)  # decoder_output: (1, V), hidden: (1, 1, d_h)

  _, top_id = torch.max(decoder_output, dim=-1)  # top_ids: (1)

  if top_id == eos_id:
    break
  else:
    output += top_id.tolist()
    input_id = top_id
```


```python
output
```




    [70,
     9,
     58,
     41,
     89,
     30,
     88,
     68,
     68,
     54,
     34,
     68,
     68,
     27,
     89,
     8,
     12,
     98,
     25,
     74,
     7]




```python

```
