# Day 54 - KoBERT

## 오늘 해본 것

* baseline code는 ```bert-base-multilingual-cased``` model을 사용하고 있었음
* ```monologg/kobert```를 사용하도록 code를 일부 수정함
  ```Python
  MODEL_NAME = 'monologg/kobert'
  tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
  ```
* accuracy는 59.7%로 약간 상승함
  * 하지만 random seed initialization을 하지 않았기 때문에 kobert가 bert-base-multilingual model보다 성능이 좋다고는 말할 수 없음
* 피어세션에서 baseline code를 그대로 사용할 시 pre-train 되지 않은 transformer를 사용하게 된다는 것을 알게 됨
    ```Python
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
    ```
* 위와 같이 code를 수정하면 pre-trained model을 사용할 수 있음
* 오늘은 제출 횟수를 다 써서 결과는 내일 알 수 있다

  ![EDA](./img/day54Accuracy.png)