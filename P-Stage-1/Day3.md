# Day 3 - Model

## 오늘의 목표

* accuracy 70 넘기기
* augmentation 적용해보기
* 3가지 task로 나누어 train 해보기

## 오늘 해본 것

* 어제 하던 것을 이어서 했다. (sample submission code 변형하여 학습, 제출하기)
  * 어제는 accuracy가 7%로 매우 낮았음
  * model을 efficient net으로 바꿔봐도 결과는 동일 (7% -> 8%)
  * 생각해보니 class가 18개 이므로 7%는 그냥 아무렇게나 찍었을 때 나오는 점수임
* 내가 작성한 코드가 잘못되었다고 판단하고 어디가 잘못되었는지를 파악하기 위해 애썼다
  * 학습에 이용할 image와 정답 label이 잘 맞지 않을 것이라고 생각해, baseline code를 참고해 이 부분을 수정하였다
    * 내가 기존에 이용한 방법은 모든 image file 이름을 ```glob```을 이용해 구해오고, label은 train.csv 파일을 읽어와 이용하는 방식이었음. 이 과정에서 정렬 등에 문제가 생겨 학습 data와 정답으로 사용할 label이 매칭되지 않았던게 아닐까 추측해봄.
    * baseline code의 방식은 이미지 파일명을 이용해 이미지를 불러오고, 동일한 파일명에서 parsing하여 label을 구해내는 방식. 이 방식은 동일한 파일명에서 image와 label을 동시에 가져오므로 서로 어긋날 일이 없음.
* label 구하는 방식을 바꿔서 학습시켰더니 accuracy가 바로 70%를 넘겼다
  * 하지만 ai stages의 리더보드에는 새로운 점수가 반영되지 않음... (ai stages의 버그인듯)
* accuracy가 괜찮게 나오니 augmentation을 해봐야겠다 생각해서 ```torch.transforms```를 이용해 random affine, random rotation, random horizontal flip, center crop을 적용시켜 보았지만 10%대의 accuracy가 나왔다
  * 이후 그냥 augmentation 없이 10 epoch으로 학습 진행하며 6, 7, 10 번쨰 epoch에서 submission을 위한 결과를 출력해 봤다
  * 6, 7번째 epoch에서는 accuracy가 73%, 10번쨰 epoch에서 75%가 나옴
* augmentation 방법이 잘못된 듯 하다. 이 부분을 개선해봐야겠음.

## 앞으로 할일

* augmentation 제대로 적용해보기
* task 3개로 나눠서 학습해보기
