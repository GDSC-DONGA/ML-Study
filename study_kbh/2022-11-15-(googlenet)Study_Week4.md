# Intro

**GoogleNet을 공부하겠습니다.**  

* 참고 URL
  * [GoogleNet(Inception)](https://sotudy.tistory.com/13)
  * [CNN_Algorithm_-_GoogLeNet(Inception)](https://dlaguddnr.tistory.com/16#CNN_Algorithm_-_GoogLeNet(Inception))
  * [GoogleNet](https://oi.readthedocs.io/en/latest/computer_vision/cnn/googlenet.html)
* [GoogleNet 참고 깃](https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14)

<br>

## CNN

* **CNN(Convolutional Neural Network)** : `Convolution(합성곱)` 연산을 수행
  * 이미지관련 데이터에 효과적인 알고리즘
  * 보통 `Convolution` 연산을 통해 이미지 또는 `Feature map(출력)`의 Width와 Height의 크기는 줄이고 Channel은 늘리는 형태
    * `Convolution`할때 Channel은 동일해야함
* **MLP(Multi Layer Perceptron)가 아닌 CNN이 등장한 이유?**  
  => `MLP`는 이미지, 영상 처리할 때 문제점이 있었기 때문
  * 이미지는 1차원이 아닌 480x480처럼 다차원 형태인데, `Perceptron 모델`은 평탄화를 통해서 1차원 형태의 데이터를 기본적으로 사용**(데이터 손실)**
  * **`weight parameter` 의 개수가 `CNN`에 비해 많이 필요**
    * **Perceptron** - 784픽셀 입력에 10 출력이라면 784x10=7840개 weights이 필요
    * **Convolution** - 784픽셀이여도 [5x5x1]x64=1600개 weights로 표현 가능

<br>

## CNN - GoogleNet(Inception)

`GoogleNet`은 2014년 ILSVRC(이미지 인식 대회)에서 1등을 차지한 모델  
**이때 핵심은 네트워크의 깊이가 깊었다는 것**

* 하지만, 단순히 **네트워크 깊이(모델 size)**가 깊어지면 그만큼 훈련수가 증가하므로 연산이 급격히 증가
* `Inception`이라는 모듈로 구성된 GoogLeNet으로 **네트워크의 구조**적인 변화를 줘서 이를 해결
  * 따라서 `GoogLeNet`은 `Inception-v1`와 같은 모델
  * `Inception` 모듈 사용해서 `1x1 Convolution`으로 연산량을 대폭 줄임
* `GoogLeNet`은 총 9개의 **Inception Module**
* GoogLeNet은 Fully Connected 대신 **Global Average Pooling**
  * Feature Map들을 각각 평균 낸 것을 이어서 1차원 벡터




### Inception 모듈

- **1x1 Convolution → Parameter 수 감소 (AlexNet보다 12배 적음)**

  - 위에 CNN에서 언급했던 `Convolution` 연산은 보통 대부분 W, H를 줄이는 대신 Channel이 늘리는 형태를 취한다고 했는데,  
    `1x1 Convolution`을 사용해서 **Channel 감소 가능 -> 차원 줄어듬 -> Parameter수 감소 효과**

  - 아래는 A1, A2, A3, A4 Feature map을 B1, B2 Feature map으로 `1x1 Convolution`한 예시

    ​			<img src="..\images\2022-11-15-(googlenet)Study_Week4\image-20221116154616512.png" alt="image-20221116154616512"  />

    * 4개의 Neuron을 2개의 Neuron으로 Fully connected한 경우와 유사한 형태
    * `1x1 Convolution` 로 위와 같은 느낌으로 Parameter수 감소 효과

  - 또다른 예시 - `1x1 Convolution`

    <img src="..\images\2022-11-15-(googlenet)Study_Week4\image-20221116155228953.png" alt="image-20221116155228953"  />

    * 28x28x**192** 입력 -> 1x1x**192** 필터 **32**개 -> Convolution -> 28x28x**32** 출력
    * 위와 같이 `1x1 Convolution`를 활용해서 Channel의 증감이 가능

- **Convolution의 병렬적 사용**

  - 다양한 Feature(출력)를 추출하려는 목적

  - **초기 Inception의 구조(1)**

    <img src="..\images\2022-11-15-(googlenet)Study_Week4\image-20221116165255721.png" alt="image-20221116165255721"  />

  - **초기 Inception의 구조(2)**  
    Layer에 다른 크기를 가지는 Filter를 적용하여 다른 Scale의 Feature를 추출할 수 있게 한다.

    <img src="..\images\2022-11-15-(googlenet)Study_Week4\image-20221116165554658.png" alt="image-20221116165554658"  />

    - 위와 같이 연산량이 많아 문제(5x5필터 32개 적용)

  

  - **후기 Inception의 구조(1) - 1x1 Convolution 사용**  
    Bottleneck layer라고 부르기도 한다.

    <img src="..\images\2022-11-15-(googlenet)Study_Week4\image-20221116165857151.png" alt="image-20221116165857151"  />

    * 12.4M으로 연산량이 확 줄었다.
    * Feature map의 크기를 줄여 성능이 떨어질 수도 있다고 생각할 수 있는데, 그래서 적절한 개수의 `1x1 Convolution`을 사용하는 것이 중요

  - **후기 Inception의 구조(2) - 1x1 Convolution 사용**

    <img src="..\images\2022-11-15-(googlenet)Study_Week4\image-20221116170301271.png" alt="image-20221116170301271"  />

    * 다른 Layer와 다르게 Max pooling이 있는 Layer에서는 `1x1 Convolution` 보다 Max pooling을 먼저 실시한다.   
      그 이유는 Max pooling이 Feature map의 개수 (Channel 수)를 조정할 수 없기 때문이라고 한다.



### Auxiliary classifier(보조 분류기)

* GoogLeNet에 기존 CNN에서 사용하지 않았던 새로운 구조
* Vanishing gradient 문제 해결
  * **Vanishing gradient란?**  
    가중치를 훈련하는 역전파(back propagation) 과정에서 가중치를 업데이트하는 데 사용되는 gradient가 점점 작아져서 0이 돼버리는 것을 말합니다. 즉, 기울기 0

* 네트워크 중간에 두 개의 **보조 분류기(Auxiliary classifier)**를 달아서 해결
* 이 보조 분류기들은 훈련 시에만 활용되고 사용할 때는 제거해줍니다.



### 전체 구조

<img src="..\images\2022-11-15-(googlenet)Study_Week4\image-20221116193901123.png" alt="image-20221116193901123"  />

- 빨간 동그라미: Inception (위에 숫자: Feature map 수)
- 파란색 모듈: Convolutional layer
- 빨간색 모듈: Max pooling
- 노란색 모듈: Softmax layer
- 녹색 모듈: 기타 Function
- 참고로 4(b), 4(e)는 `보조 분류기` 



초반 부분엔 `Inception`을 사용하지 않고, 일반적인 CNN에서 보이는 Conv-Pool 구조를 가진다.  
초반에 `Inception`을 배치해도 효과가 없었기 때문이라고 한다.
