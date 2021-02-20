# Day 20 - Self-supervised Pre-training Models, Other Self-supervised Pre-training Models

* Transformer model and its self-attention block has become a general-purpose sequence (or set) encoder and decoder in recent NLP applications as well as in other areas.
* Training deeply stacked Transformer models via a self-supervised learning framework has significantly advanced various NLP tasks through transfer learning, e.g., BERT, GPT-3, XLNet, ALBERT, RoBERTa, Reformer, T5, ELECTRA...

## GPT-1

* 다양한 token을 이용해 하나의 model로 여러가지 task를 수행
* 문장의 끝에는 Extract token을 사용
* 두 문장을 이용하는 task의 경우 두 문장 사이를 Delim token으로 이어줌
![GPT-1](./img/day20/gpt-1.png)
* model을 pre-train 할 때에는 다음 단어를 예측하는 task로 학습을 진행함
* pre-train된 모델을 이용해 각각의 task에 맞게 fine-tuning
* inference 할때는 문장 끝의 Extract token의 encoding vector를 각 task에 맞는 output layer에 전달하는 방식으로 하나의 model로 여러가지 task를 수행함
  * Classification의 경우 Extract vector token의 encoding vector를 해당 문장이 긍정인지 부정인지 등을 구분하는데 사용
  * Entailment의 경우 Extract vector token의 encoding vector를 두 문장이 서로 모순되는지를 판단하는데 사용
* 새로운 task를 위한 model이 새롭게 필요하다면 encoder 부분은 기존에 학습된 pre-trained model을 그대로 가져다 쓰고, 위쪽에 새로운 task를 ㅜ이한 추가적인 layer 하나를 붙임
* 새롭게 학습할 때는 새로 추가된 layer를 중점적으로 학습하고, pre-train된 encoder는 learning rate를 작게 해서 fine-tuning만 함
* GPT-1을 이용한 fine-tuned model은 해당 task만을 위해 설계, 학습된 model들보다 더 좋은 성능을 보여줌

## BERT

* GPT-1은 다음 단어를 순차적으로 예측하는 task를 이용해 pre-train 되었다
* 하지만 이런 task는 단어의 전후 문맥을 보지 않고 예측을 하게 됨
* 문장의 앞쪽과 뒤쪽을 모두 고려해야만 더 정확한 예측을 할 수 있음
* BERT는 이러한 문제를 해결하기 위해, 순차적 단어 예측이 아닌, 문장 내 몇몇 단어를 mask한 뒤 앞뒤 문맥을 이용해 이를 예측하도록 설계함
* 이러한 model을 Masked Language Model (MLM) 이라고 함

### Masked Language Model (MLM)

* 문장 내 특정 percentage의 단어들을 mask token으로 치환 후, mask된 단어들이 무엇이었는지 맞추는 방식으로 학습

![BERT](./img/day20/bert1.png)

* GPT-1과 비슷하게 CLS, SEP token을 이용한다
  * CLS token은 문장의 맨 앞에 넣는다, SEP token은 두 문장을 연결할 때 사용한다
* MASK token의 encoding vector는 단어 예측에 사용된다
* CLS token의 encoding vector는 binary classification 등에 사용된다

* 적절한 비율
  * mask된 단어의 비율이 높아지면 예측이 필요한 정보가 충분히 제공되지 않아 문제가 됨
  * mask된 단어의 비율이 작아지면 학습 효율이 좋지 않음
  * BERT에서는 적절한 값으로 15%를 제시함
* 문제점
  * 하지만 여기에도 문제가 있음
  * model이 mask가 끼어있는 문장에 익숙해짐
  * 이런 model은 주제 분류 등의 mask가 등장하지 않는 task에서는 제대로 동작하지 않을 수 있음
* 해결 방안
  * mask 대상인 15%의 단어를 모두 mask하지 않는다
    * 이중 80%만 MASK token으로 바꾼다
    * 10%는 랜덤하게 다른 단어로 바꾼다
      * 다른 단어로 바뀐 것을 찾아서 원래대로 돌려놓는 task
    * 나머지 10%는 그대로 둔다
      * 원래 있던게 맞다고 소신있게 주장하는 능력을 학습

### Next Sentence Prediction (NSP)

* BERT는 word 단위 예측 뿐만 아니라 문장 단위 task에도 대항해야 한다
* 이를 위한 pre-training 기법으로 next sentence prediction을 사용한다
* 문장 2개를 뽑아 SEP 토큰으로 연결한다
* 연결된 문장이 이어지는 문장인지 아닌지를 판별하는 task로 학습한다
* 예측한 결과는 입력 맨 앞의 CLS token에 담기게 된다
* NSP task에도 mask를 적용하여 단어 예측과 함께 학습한다

![NSP](./img/day20/nsp1.png)

### 그 외 BERT의 특징

* WordPiece embedding - 단어를 더 잘게 쪼개 subword로 만들어 사용 (Day 19 과제 참고)
* Learned positional embedding - positional encoding에 사용하는 값을 미리 정의해 둔 값이 아니라 학습에 의해 결정된 값을 사용함
* Segment Embedding - 두 문장을 이었을 경우 기존 방식의 positional embedding으로는 정보를 제대로 표현하지 못함.
  
  ![Segment Embedding](./img/day20/segmentEmbedding.png)  

    위의 그림의 position embeddings를 보면 두 문장의 position imbedding이 쭉 이어져 있음. 이렇게 되면 이 input에 대한 정보를 온전히 표현하지 못함. 따라서 segment embedding을 이용하여 각 문장을 구분하는 embedding vector를 추가적으로 더해줌

## GPT-2

## GPT-3

## ALBERT

## ELECTRA
