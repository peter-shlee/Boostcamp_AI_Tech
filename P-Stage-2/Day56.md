# Day 56 - K fold ensemble

## 오늘 해본 것

* 6개의 fold로 나누어뒀던 data를 이용해 6개의 KoBert model을 각각 학습시켰다
  * 최대 10번의 epoch을 돌면서 (batch size = 32) 가장 validation loss가 낮았던 epoch에서 저장한 model을 inference에 사용
  * 보통 4 epoch 정도에서 validation loss가 제일 낮았음 (이 때 validation accuracy도 70% ~ 73% 정도로 가장 높았음)
* 위에서 저장해 둔 model들을 이용해 inference 하고, 그 결과로 나온 soft label들을 이용해 soft 앙상블 하였음
* 이렇게 만든 결과물을 제출해보니 leader board 점수는 전보다 0.1% 오른 72.4%....
* 대회 막바지에 점수를 조금이라도 끌어 올리는데는 앙상블이 좋을 지 모르겠으나, 마감까지 여유가 조금 있는 지금 같은 상황에는 앙상블을 하는 것 보다는 model을 바꿔보거나, 데이터를 추가로 만드는 등의 일을 하는 것이 맞는것 같다.

![accuracy](./img/day56Accuracy.png)

## 앞으로 할 일

* 토론 게시판에 추가로 사용할 수 있는 dataset이 올라왔다. 이것을 이용해보자.
* RoBERTa 라는 model이 성능이 좋다고 한다. 이것도 한번 써보자.
* 이 외에 토론 게시판에 올라온 글들 읽어보고 써볼만 한 것들 적용해보기.
