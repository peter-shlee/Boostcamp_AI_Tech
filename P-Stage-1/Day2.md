# Day 2 - 

## 오늘의 목표

* augmentation으로 60대 이상 data, 마스크 미착용, 잘못 착용 data 늘리기 (random rotaion, 좌우 대칭 등 이용)
* 얼굴 주름 강조되도록 하는 data augmentation 찾기
* sample code 이용해 학습시킨 뒤, 제출

## 오늘 해본 것

* 기본으로 주어졌던 sample_submission code를 이용해 model을 학습시킨 뒤 제출해봄
* pre-trained inception_v3를 사용함
* backbone model의 fc layer의 output 수가 18이 되도록 바꾸었음
* backbone model도 같이 fine tuning 함
* epoch = 10, lr = 1e-3, 손실함수는 crossEntropy, optimizer는 Adam 사용
* 결과는 accuracy 약 7%, f1 score 0.03%으로 매우 좋지 않음...

## 앞으로 할일

* pre-trained model을 가져와 class 개수 18개로 맞춰 학습시켰다는 다른 분들은 점수가 매우 높다... 무엇이 문제인 것일까. 문제 파악하고 점수를 높여보자
* sample code로 model 학습 하는데 시간을 너무 많이 써서 augmentation 못했음, data augmentation 해서 전처리된 data 따로 저장하기
* 구상해뒀던 것처럼 성별, 나이, 마스크 착용 여부 따로 따로 3개의 model 이용하여 예측 후 ensemble 하는 것이 최종 목표