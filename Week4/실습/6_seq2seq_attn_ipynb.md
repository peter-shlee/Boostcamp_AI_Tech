##**6. Seq2seq + Attention**
1. 여러 Attention 모듈을 구현합니다.
2. 기존 Seq2seq 모델과의 차이를 이해합니다.

### **필요 패키지 import**


```python
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

import torch
import random
```

### **데이터 전처리**

데이터 처리는 이전과 동일합니다.


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


```python
trg_data = [[sos_id]+seq+[eos_id] for seq in tqdm(trg_data)]
```

    100%|██████████| 10/10 [00:00<00:00, 25621.89it/s]



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

    100%|██████████| 10/10 [00:00<00:00, 16933.00it/s]
    100%|██████████| 10/10 [00:00<00:00, 13218.73it/s]

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

Encoder 역시 기존 Seq2seq 모델과 동일합니다.


```python
embedding_size = 256
hidden_size = 512
num_layers = 2
num_dirs = 2
dropout = 0.1
```


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
    outputs = torch.tanh(self.linear(outputs))  # (S_L, B, d_h)

    forward_hidden = h_n[-2, :, :]
    backward_hidden = h_n[-1, :, :]
    hidden = torch.tanh(self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1))).unsqueeze(0)  # (1, B, d_h)

    return outputs, hidden
```


```python
encoder = Encoder()
```

### **Dot-product Attention 구현**

우선 대표적인 attention 형태 중 하나인 Dot-product Attention은 다음과 같이 구현할 수 있습니다.




```python
class DotAttention(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
    query = decoder_hidden.squeeze(0)  # (B, d_h)
    key = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

    energy = torch.sum(torch.mul(key, query.unsqueeze(1)), dim=-1)  # (B, S_L)

    attn_scores = F.softmax(energy, dim=-1)  # (B, S_L)
    attn_values = torch.sum(torch.mul(encoder_outputs.transpose(0, 1), attn_scores.unsqueeze(2)), dim=1)  # (B, d_h)

    return attn_values, attn_scores
```


```python
dot_attn = DotAttention()
```

이제 이 attention 모듈을 가지는 Decoder 클래스를 구현하겠습니다.


```python
class Decoder(nn.Module):
  def __init__(self, attention):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.attention = attention
    self.rnn = nn.GRU(
        embedding_size,
        hidden_size
    )
    self.output_linear = nn.Linear(2*hidden_size, vocab_size)

  def forward(self, batch, encoder_outputs, hidden):  # batch: (B), encoder_outputs: (L, B, d_h), hidden: (1, B, d_h)  
    batch_emb = self.embedding(batch)  # (B, d_w)
    batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

    outputs, hidden = self.rnn(batch_emb, hidden)  # (1, B, d_h), (1, B, d_h)

    attn_values, attn_scores = self.attention(hidden, encoder_outputs)  # (B, d_h), (B, S_L)
    concat_outputs = torch.cat((outputs, attn_values.unsqueeze(0)), dim=-1)  # (1, B, 2d_h)

    return self.output_linear(concat_outputs).squeeze(0), hidden  # (B, V), (1, B, d_h)
```


```python
decoder = Decoder(dot_attn)
```

### **Seq2seq 모델 구축**

최종적으로 seq2seq 모델을 다음과 같이 구성할 수 있습니다.


```python
class Seq2seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2seq, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
    # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L)

    encoder_outputs, hidden = self.encoder(src_batch, src_batch_lens)  # encoder_outputs: (S_L, B, d_h), hidden: (1, B, d_h)

    input_ids = trg_batch[:, 0]  # (B)
    batch_size = src_batch.shape[0]
    outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

    for t in range(1, trg_max_len):
      decoder_outputs, hidden = self.decoder(input_ids, encoder_outputs, hidden)  # decoder_outputs: (B, V), hidden: (1, B, d_h)

      outputs[t] = decoder_outputs
      _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)

      input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

    return outputs
```


```python
seq2seq = Seq2seq(encoder, decoder)
```

### **모델 사용해보기**

만든 모델로 결과를 확인해보겠습니다.


```python
# V: vocab size
outputs = seq2seq(src_batch, src_batch_lens, trg_batch)  # (T_L, B, V)

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
    
            [[ 2.3249e-02,  6.9177e-02,  2.7970e-02,  ..., -2.4687e-02,
               5.7721e-03,  5.7630e-03],
             [ 7.7156e-02,  8.9973e-02,  2.1757e-02,  ..., -2.8352e-02,
               1.4128e-02, -1.7767e-02],
             [ 3.6274e-02,  9.5913e-02, -2.3567e-03,  ..., -5.4262e-02,
              -1.3508e-02, -1.8194e-02],
             ...,
             [ 5.0581e-02,  6.0535e-02,  3.4751e-02,  ..., -2.3565e-02,
              -2.2078e-02, -1.0750e-02],
             [ 3.3034e-02,  5.0459e-02,  3.0143e-02,  ..., -1.8930e-03,
               2.8245e-02,  3.3707e-02],
             [ 2.8714e-02,  6.4583e-02,  2.8263e-02,  ..., -1.7575e-02,
               2.3472e-02,  1.4847e-02]],
    
            [[-2.0859e-02, -2.6811e-02, -6.0540e-02,  ..., -4.3667e-02,
               1.2037e-01, -6.5637e-02],
             [ 2.6796e-02,  1.3532e-03, -7.5768e-02,  ..., -4.3350e-02,
               1.1759e-01, -9.6136e-02],
             [-4.4131e-03, -5.9813e-03, -8.1895e-02,  ..., -5.7079e-02,
               1.0227e-01, -8.5025e-02],
             ...,
             [ 1.9150e-03, -2.5682e-02, -5.5630e-02,  ..., -4.6846e-02,
               1.0036e-01, -8.4909e-02],
             [-1.3861e-02, -2.9709e-02, -5.8245e-02,  ..., -3.1691e-02,
               1.2217e-01, -5.7525e-02],
             [-1.5008e-02, -2.2953e-02, -6.2954e-02,  ..., -4.2928e-02,
               1.2646e-01, -6.6576e-02]],
    
            ...,
    
            [[ 1.6701e-02,  4.9486e-02,  1.3651e-01,  ..., -6.2899e-02,
              -3.5976e-01, -2.5687e-01],
             [ 3.1495e-04,  5.0625e-02, -6.5379e-02,  ..., -9.6058e-02,
              -1.4976e-01, -1.9011e-01],
             [ 3.0127e-02,  6.9062e-02,  1.1806e-01,  ..., -5.7982e-02,
              -3.7527e-01, -2.8398e-01],
             ...,
             [-2.6448e-02,  3.3830e-02, -4.2954e-02,  ..., -7.6129e-02,
              -1.8673e-01, -1.5521e-01],
             [-2.0950e-02,  4.1751e-02, -5.5490e-02,  ..., -4.8137e-02,
              -2.0792e-01, -1.8650e-01],
             [-3.1485e-02,  2.9965e-02, -4.6005e-02,  ..., -6.9904e-02,
              -1.9517e-01, -1.4995e-01]],
    
            [[ 6.4573e-02,  5.6403e-02, -3.8881e-02,  ..., -2.5162e-02,
              -2.2625e-01, -8.9272e-02],
             [ 1.1094e-01, -2.8034e-02,  6.7745e-03,  ..., -1.3461e-01,
              -5.3201e-02, -1.5250e-01],
             [ 8.0737e-02,  7.3825e-02, -5.6923e-02,  ..., -2.3022e-02,
              -2.3955e-01, -1.1367e-01],
             ...,
             [ 8.2463e-02, -5.1793e-02,  2.5790e-02,  ..., -1.1918e-01,
              -7.4070e-02, -1.2702e-01],
             [ 8.4015e-02, -4.1524e-02,  1.5407e-02,  ..., -1.0416e-01,
              -8.8347e-02, -1.4774e-01],
             [-3.6731e-02, -2.7467e-02, -1.0914e-01,  ...,  4.8974e-02,
              -1.7509e-01, -2.4518e-01]],
    
            [[-1.1726e-01,  1.3504e-02,  4.1177e-02,  ..., -6.3281e-02,
              -2.6244e-01, -1.9216e-01],
             [-9.2080e-02,  5.7005e-03,  5.0175e-02,  ..., -1.5292e-01,
              -1.4784e-01, -2.5637e-01],
             [-1.0325e-01,  3.0152e-02,  2.4652e-02,  ..., -5.9829e-02,
              -2.7359e-01, -2.1911e-01],
             ...,
             [-1.2769e-01, -1.7769e-02,  7.7363e-02,  ..., -1.4117e-01,
              -1.6162e-01, -2.3535e-01],
             [-1.2685e-01, -1.0231e-02,  6.7630e-02,  ..., -1.2738e-01,
              -1.6937e-01, -2.4995e-01],
             [-2.0170e-01, -1.0864e-02, -1.1293e-02,  ..., -3.3246e-02,
              -2.3974e-01, -2.7665e-01]]], grad_fn=<CopySlices>)
    torch.Size([22, 10, 100])



```python
sample_sent = [4, 10, 88, 46, 72, 34, 14, 51]
sample_len = len(sample_sent)

sample_batch = torch.LongTensor(sample_sent).unsqueeze(0)  # (1, L)
sample_batch_len = torch.LongTensor([sample_len])  # (1)

encoder_output, hidden = seq2seq.encoder(sample_batch, sample_batch_len)  # hidden: (4, 1, d_h)
```


```python
input_id = torch.LongTensor([sos_id]) # (1)
output = []

for t in range(1, trg_max_len):
  decoder_output, hidden = seq2seq.decoder(input_id, encoder_output, hidden)  # decoder_output: (1, V), hidden: (4, 1, d_h)

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




    [81,
     90,
     78,
     52,
     60,
     92,
     92,
     71,
     71,
     11,
     78,
     27,
     52,
     60,
     92,
     92,
     71,
     71,
     11,
     78,
     27]



### **Concat Attention 구현**

Bahdanau Attention이라고도 불리는 Concat Attention을 구현해보도록 하겠습니다.  


*   `self.w`: Concat한 query와 key 벡터를 1차적으로 linear transformation.
*   `self.v`: Attention logit 값을 계산.




```python
class ConcatAttention(nn.Module):
  def __init__(self):
    super().__init__()

    self.w = nn.Linear(2*hidden_size, hidden_size, bias=False)
    self.v = nn.Linear(hidden_size, 1, bias=False)

  def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
    src_max_len = encoder_outputs.shape[0]

    decoder_hidden = decoder_hidden.transpose(0, 1).repeat(1, src_max_len, 1)  # (B, S_L, d_h)
    encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

    concat_hiddens = torch.cat((decoder_hidden, encoder_outputs), dim=2)  # (B, S_L, 2d_h)
    energy = torch.tanh(self.w(concat_hiddens))  # (B, S_L, d_h)

    attn_scores = F.softmax(self.v(energy), dim=1)  # (B, S_L, 1)
    attn_values = torch.sum(torch.mul(encoder_outputs, attn_scores), dim=1)  # (B, d_h)

    return attn_values, attn_scores
```


```python
concat_attn = ConcatAttention()
```

마찬가지로 decoder를 마저 구현하겠습니다.


```python
class Decoder(nn.Module):
  def __init__(self, attention):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.attention = attention
    self.rnn = nn.GRU(
        embedding_size + hidden_size,
        hidden_size
    )
    self.output_linear = nn.Linear(hidden_size, vocab_size)

  def forward(self, batch, encoder_outputs, hidden):  # batch: (B), encoder_outputs: (S_L, B, d_h), hidden: (1, B, d_h)  
    batch_emb = self.embedding(batch)  # (B, d_w)
    batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

    attn_values, attn_scores = self.attention(hidden, encoder_outputs)  # (B, d_h), (B, S_L)

    concat_emb = torch.cat((batch_emb, attn_values.unsqueeze(0)), dim=-1)  # (1, B, d_w+d_h)

    outputs, hidden = self.rnn(concat_emb, hidden)  # (1, B, d_h), (1, B, d_h)

    return self.output_linear(outputs).squeeze(0), hidden  # (B, V), (1, B, d_h)
```


```python
decoder = Decoder(concat_attn)
```


```python
seq2seq = Seq2seq(encoder, decoder)
```


```python
outputs = seq2seq(src_batch, src_batch_lens, trg_batch)

print(outputs)
print(outputs.shape)
```

    tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
             ...,
             [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
    
            [[ 0.1540,  0.0182, -0.1775,  ...,  0.0435,  0.0766, -0.0979],
             [ 0.1224, -0.0289, -0.2057,  ..., -0.0018,  0.1062, -0.1322],
             [ 0.0965, -0.0296, -0.1581,  ...,  0.0691,  0.0547, -0.1677],
             ...,
             [ 0.0910, -0.0579, -0.1565,  ...,  0.0723,  0.0677, -0.1190],
             [ 0.1319, -0.0127, -0.1894,  ...,  0.0353,  0.0640, -0.1193],
             [ 0.1050, -0.0252, -0.1634,  ...,  0.0419,  0.1026, -0.1349]],
    
            [[-0.0605,  0.0448,  0.1414,  ...,  0.1516,  0.1022,  0.0233],
             [-0.0960,  0.0320,  0.1205,  ...,  0.1310,  0.1321, -0.0139],
             [-0.0946,  0.0236,  0.1392,  ...,  0.1725,  0.0947, -0.0187],
             ...,
             [-0.1125,  0.0132,  0.1511,  ...,  0.1747,  0.1039,  0.0090],
             [-0.0907,  0.0305,  0.1257,  ...,  0.1588,  0.1128,  0.0008],
             [-0.1037,  0.0256,  0.1367,  ...,  0.1585,  0.1284, -0.0121]],
    
            ...,
    
            [[ 0.1258,  0.0701, -0.0052,  ...,  0.0692,  0.1045,  0.1134],
             [ 0.0751,  0.1030, -0.0066,  ...,  0.0744,  0.1161,  0.1053],
             [ 0.1253,  0.0778, -0.0202,  ...,  0.1001,  0.1257,  0.1037],
             ...,
             [ 0.0997,  0.0930, -0.0118,  ...,  0.0998,  0.1174,  0.1159],
             [ 0.1026,  0.0965, -0.0167,  ...,  0.0950,  0.1321,  0.1188],
             [ 0.1067,  0.0964, -0.0147,  ...,  0.0937,  0.1317,  0.1154]],
    
            [[ 0.0716, -0.0691,  0.0022,  ...,  0.1879,  0.0180,  0.0645],
             [ 0.0296, -0.0324, -0.0064,  ...,  0.1987,  0.0357,  0.0503],
             [ 0.0674, -0.0548, -0.0120,  ...,  0.2192,  0.0411,  0.0523],
             ...,
             [ 0.0454, -0.0448, -0.0043,  ...,  0.2132,  0.0334,  0.0592],
             [ 0.0443, -0.0415, -0.0093,  ...,  0.2095,  0.0447,  0.0603],
             [ 0.0485, -0.0412, -0.0069,  ...,  0.2089,  0.0453,  0.0556]],
    
            [[ 0.0315, -0.0434, -0.0186,  ...,  0.0018,  0.2140,  0.1842],
             [-0.0081, -0.0103, -0.0211,  ...,  0.0171,  0.2322,  0.1672],
             [ 0.0237, -0.0313, -0.0312,  ...,  0.0388,  0.2354,  0.1787],
             ...,
             [ 0.0009, -0.0207, -0.0209,  ...,  0.0259,  0.2277,  0.1796],
             [-0.0034, -0.0191, -0.0272,  ...,  0.0253,  0.2416,  0.1798],
             [ 0.0018, -0.0193, -0.0248,  ...,  0.0240,  0.2410,  0.1753]]],
           grad_fn=<CopySlices>)
    torch.Size([22, 10, 100])



```python

```
