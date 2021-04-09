# Day 10 - Competition 종료, 마무리

## 앞으로 보완할 점

* 체계적인 baseline code를 작성해 실험하기 좋은 환경을 만들자
* 체계적으로 실험하자 (계획 세우고, 계획대로 실험하고, 실험 결과를 기록하자)
* 주먹 구구식으로 하지 말고, 의미 있는 실험을 하자.
* 학습 시작 전 우선 num workers 를 바꿔가며 실험해보며 최적의 workers 수를 정하고, numpy를 많이 사용하는 등의 최적화를 통해 train을 최대한 빨리 할 수 있는 환경을 만들자
* validation을 확실하게 하자. 일단 제출 한 뒤 leader board를 확인하는 방식으로 validation을 하는 것은 competition에서만 할 수 있는 방식이고, 매우 비효율적이다.
* 열심히 하자...

## 앞으로 사용해 보면 좋을 것 같은 방법들

* wandb 등을 사용해 train 과정을 시각화하자
* scheduler를 사용해보자
* 여러가지 loss를 합쳐서 사용해보자. ex) cross entropy와 f1 loss를 합쳐서 사용 (단순히 두 종류의 loss 값을 더하는 방식)
