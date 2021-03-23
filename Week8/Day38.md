# Day 38 - 빠르게, 가지치기

## 빠르게

### Acceleration

* compression이 model에서 불필요한 부분들을 잘라내 model의 크기를 줄여 연산을 빠르게 하는 것이라면, acceleration은 연산 자체를 수행하는 속도를 빠르게 만드는 것
* 연산을 빠르게 처리하려면 software보다는 hardware 단에서 효율적인 처리가 필요함 (GPU를 이용한 병렬처리 등)
* software를 hardware에 맞춰 설계하면 더 좋은 성능을 낼 수 있다 (hardware-software co-design)
  
### deep learning compiler

* LLVM은 여러 언어와, 여러 플랫폼 사이의 통합된 중간언어를 이용해 필요한 컴파일러의 수를 줄여주는 개념
* MLIR은 LLVM 개념을 ML에 가져온 것 (sub-project of LLVM)
* 다른 컴파일 과정과 마찬가지로 Deep Learning 컴파일 과정에서도 여러가지 최적화가 이루어진다

    ![compiler](./img/Day38/compiler1.png)

## 가지치기 (Pruning)

### Pruning이란?

* 덜 중요한 parameter를 잘라내는 방식으로(가지치기) model을 경량화 한다
* parameter의 중요도를 판단할 때는 weighted sum을 사용한다
* pruning을 하면 inference speed가 빨라지고, 일반화 성능도 더 좋아질 수 있다
* 하지만 과도하게 pruning을 하는 경우 정보를 너무 많이 잃어 성능이 크게 떨어질 수 있다

    ![pruning](./img/Day38/pruning1.png)

    pruning algorithm

### Pruning의 종류

* pruning에도 여러 종류가 있음

    ![pruning](./img/Day38/pruning2.png)

