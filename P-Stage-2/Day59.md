# Day 59 - 앙상블

## 오늘 해본 것

* RoBERTa 모델에서 나온 예측 값 중 가장 accuracy가 높았던 것과, 이전에 KoBERT를 이용해 만들었던 예측 값 중에서 가장 점수가 높았던 것을 hard-voting 방식으로 앙상블했다. 79.1% 정도의 accuracy가 나왔다
* 더 많은 예측 값들을 앙상블 할수록 좋은 결과가 나온다던 이야기가 생각나서 급하게 KoElectra를 학습시켜 예측값을 만들고, 여기에 RoBERTa와 KoBERT에서 나온 결과물들 중 accuracy가 좀 낮았던 것들도 다 합쳐보았다. 그 결과 79.5%의 accuracy가 나왔다.
* 많이 합칠수록 결과가 좋아지는걸 보니 위의 세 모델에서 나온 결과물 수십개를 합쳐보면 어떤 결과가 나올지 궁금해졌다.
* RoBERTa, KoBERT, KoElectra를 이용해, random seed값을 바꿔가며 학습시키고, 그 결과물들을 모았다.
* 20개 정도의 예측 값들을 만들어 앙상블한 뒤 기대를 안고 제출했다. 하지만 accuracy는 79.4%가 나왔다 ㅠㅠ. 이게 이번 competition의 마지막 제출
* 최종 순위는 58/136 (이번엔 private leader board가 없음)

![accuracy](./img/day59Accuracy.png)

## 앞으로 할 일

* wrap up report 작성
