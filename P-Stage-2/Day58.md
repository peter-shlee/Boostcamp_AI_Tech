# Day 58 - RoBERTa

## 오늘 해본 것

* RoBERTa 모델을 제대로 사용할 수 있도록 코드를 수정하였음
  * pre-train된 XLMRoberta에 맞춰 XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification을 사용하도록 수정했더니 코드가 정상적으로 작동됐다.
* RoBERTa는 KoBERT보다 무거운 model인듯 하다. 학습 시간도 오래 걸리고, model을 저장하려면 5GB 정도의 저장공간이 필요했다. 안그래도 서버 용량이 부족한데...
* RoBERTa를 이용해보니 accuracy가 77.2%가 나왔다. 서버를 제대로 사용하지 못해 다양한 시도는 해보지 못했다.

![accuracy](./img/day58Accuracy.png)

## 앞으로 할 일

* 앙상블하여 대회 마무리
