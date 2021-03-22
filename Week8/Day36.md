# Day 36 - 가벼운 모델, 동전의 뒷면, 가장 적당하게

## 가벼운 모델

### 결정이란?

* 연역적 결정이란?
  * 이미 알고있는 것들을 근거로 새로운 결정을 내리는 것
  * 전제가 참일 때 결과도 참이 됨
  * 수학적인 사고방식
* 귀납적 결정이란?
  * 사실과 근거를 통해 가장 그럴듯한 결정을 내리는 것
  * 완벽한 결론까지 이르는 것은 아님
  * 결론이 참이 아닐 수도 있음

    (참고: https://opentutorials.org/module/3653/22098)

### 결정기 (decision making machine)

* machine learning model은 인간 대신 결정을 해주는 기계임
* 추천 시스템과 같은 비교적 정확도가 덜 중요한 결정 시스템도 있고, 환자 진단, 자율 주행 같은 정확도가 매우 중요한 결정 시스템도 있음
* 딥러닝을 통해 높은 정확도가 요구되는 task도 decision making machine에 맡길 수 있게 되었음

### 가벼운 결정기 (light weight decision making machine)

* ML model을 실제 서비스에 사용할 때는 inference가 빠를 수록 좋음
* 따라서 여러 방법을 사용해 ML model을 경량화 함
* ML model을 경량화 하면 서버에서가 아니라 edge device에서 inference를 할 수 있게 됨
* edge device에서 inference를 하면 서버 비용 절감, 네트워크로 정보가 전송되지 않아 보안성 향상, latency 감소 등의 장점이 있음

## 동전의 뒷면

* ML model을 이용해 서비스를 만드는 과정에서 ML modeling, training이 차지하는 비중은 매우 적다 (다른 중요한 과정이 매우 많다)
* AI model은 입력에 대한 정확한 출력을 보장하지 않음
* 같은 model을 똑같은 방법으로 학습시키더라도 초기 값 세팅에 따라 결과가 달라짐
* model의 accuracy만 중요한 것은 아님. 경량화, 비용 등 따져봐야 할 것들이 많음 (GPT-3를 edge device에서 사용할 순 없음)

## 가장 적당하게

* 우리는 우리가 사용할 수 있는 한정된 자원만 이용해서 최대의 효율을 내는 model을 만들기 위해 노력해야 함

### mini code

```Python
from scipy.spatial import distance

print(distance.euclidean([1, 0, 0], [0, 1, 0]))
print(distance.euclidean([1, 1, 0], [0, 1, 0]))

print(distance.hamming([1, 0, 0], [0, 1, 0]))
print(distance.hamming([1, 0, 0], [1, 1, 0]))
print(distance.hamming([1, 0, 0], [2, 0, 0]))
print(distance.hamming([1, 0, 0], [3, 0, 0]))

print(distance.cosine([1, 0, 0], [0, 1, 0]))
print(distance.cosine([100, 0, 0], [0, 1, 0]))
print(distance.cosine([1, 1, 0], [0, 1, 0]))
```

* euclidean, hamming, cosine distance의 차이는?
* 언제 어떤 거리를 사용해야 할까?

### Constraints

* 문제를 해결하는 데에는 고려해야 할 여러가지 constraints가 있다
* 이러한 constraints들에 대해 여러가지 결정을 내려가며 문제를 해결해야 한다

### Constraints in model compression

* cost: budget, security, connectivity, model size, stability, adaptability, inference time, training time, power consumption
* object: performance

* 위의 것들을 고려하여 model을 compression 해야 함