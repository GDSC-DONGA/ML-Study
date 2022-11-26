# Intro

**word embedding를 공부하겠습니다.**  

* 참고 URL
  * [Word Embedding](https://wikidocs.net/33520)
  * [Word2Vec-Youtube](https://www.youtube.com/watch?v=sY4YyacSsLc)
* 실습 참고 URL
  * [Word2Vec 실습](https://wikidocs.net/50739)

<br>

## Word Embedding

**워드 임베딩(Word Embedding)은 단어를 밀집벡터(dense vector)로 표현하는 방법**

* **희소 행렬**
  * 단어 10000개 가정, 강아지는 인덱스 4라고 가정
  * `강아지 = [0 0 0 0 1 0 0 0 0 ... 0 ]`  즉, 심각한 공간 낭비
  * 이를 워드 임베딩을 한다면?

* **밀집 행렬**
  * 밀집 표현의 차원을 128로 설정한다고 가정 -> 이때 모든값이 실수가 됨
  * `강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ...]` 즉, 공간 낭비 해결(밀집)



다음은 LSA, Word2Vec, FastText, Glove 등 여러 방법 중 **Word2Vec**만 간단히 소개하겠습니다.

<br>

## Word2Vec(워드투벡터)

`희소 표현`의 경우 대부분이 0으로 표현되므로 각 단어 벡터간 유의미한 유사성을 표현할 수 없다.  
대안으로 단어의 의미를 다차원 공간에 벡터화하는 방법을 사용 -> **분산 표현**

<img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221121002736086.png" alt="image-20221121002736086"  />

`분산 표현`을 이용해 단어 간 의미적 유사성을 벡터화하는 작업 -> **워드 임베딩**

* 차원이 줄어듬, 유사성 표현

<img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221121002758622.png" alt="image-20221121002758622"  />



### 분산 표현(Distributed Representation)

**비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다** 라는 가정을 토대로 진행 -> **분포 가설**

* 예로 `강아지` 란 단어는 귀엽다, 예쁘다, 애교 등의 단어와 함께 주로 등장하기 때문에  
  해당 텍스트를 벡터화한다면 해당 단어 벡터들은 유사한 벡터값을 가진다.
* 또다른 예로는,    
  She is a beautiful woman.
  She is an awesome woman.
  * `beautiful`, `awesome`은 해당 단어 주변에 분포한 단어가 유사하므로 두 단어의 뜻은 비슷할것이라고 가정하는 것이 분포 가설이다.
* 앞에서 워드 임베딩을 통해서 `강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ...]`  로 차원을 줄일수 있다고 소개했었다.
  * 방법 2가지 : CBOW, Skip-Gram



### CBOW(Continuous Bag of Words)

**주변에 있는 단어들을 입력으로 중간에 있는 단어들을 예측하는 방법**

* 윈도우(window) : 중심 단어의 앞, 뒤로 몇 개의 단어를 볼지 결정
* 아래 예시는 One-Hot Encoding으로 데이터 셋을 만드는 과정이며 window는 2이다.
  * One-Hot Encoding이란 text를 독립적인 벡터로 표현한것

<img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221120233715998.png" alt="image-20221120233715998"  />

* 중심 단어가 `sat` 인 경우를 간단히 도식화 했을때는 아래와 같다.

  <img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221120234103513.png" alt="image-20221120234103513"  />

* 좀 더 자세히 나타내기 위해 주변 단어인 `cat, on`  만 나타내서 보겠다(아래 그림)

  * 투사층의 크기 M=5이므로 CBOW를 수행하고 나면 각 단어의 차원은 5가 될 것
    * 7차원에서 5차원으로 줄게되는 것(사용자가 설정)
    * 왜냐하면, 최종적으로 W행렬의 각 행이 입력한 각 단어의 벡터가 되기 때문
    * 참고로 초기의 행렬 W는 랜덤 숫자로 이루어져 있고,
    * 아래 W'는 전치행렬은 아니라 랜덤한 값을 둘다 가져서 서로 다른 행렬
  * 식 : `X*W -> V, V*W' -> Z`  
    차원 : `(1x7) x (7x5) -> (1x5), (1x5) x (5x7) -> (1x7)`
    * 입력 벡터의 차원이 7이었다면, 나오는 벡터도 마찬가지라는 것을 보여준다.

  <img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221120234256625.png" alt="image-20221120234256625"  />

* 더욱 자세히 보겠다(아래 그림)

  * 위에서 구한 식 Z를 softmax(Z)를 하게되면, 0과 1사이 값으로 얻게되며

    * `softmax`는 0~1사이 값으로 출력해주는 활성화 함수

  * 이를 실제 정답값(중심 단어)인 y와 cross entropy를 통해서 GD Method 를 수행

    * `cross entropy`는 손실 함수(Loss함수)
    * `GD Method`는 위 손실 함수(오차)를 줄이는 목적으로 올바른 Weight(W행렬)을 찾아가는 방법

  * 즉, 역전파를 하면서 W, W' 행렬이 학습

    * 최종적으로 W나 W'행렬의 각 행들을 사용해서 각 단어의 임베딩 벡터를 사용

    <img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221121003913171.png" alt="image-20221121003913171"  />



### Skip-gram

**중간에 있는 단어들을 입력으로 주변 단어들을 예측하는 방법**

* CBOW와는 반대의 방법이기 때문에 자세한 설명은 하지 않겠다.
* 아래는 CBOW처럼 데이터셋을 하는 과정인데 조금 차이가 있다.

<img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221121005113743.png" alt="image-20221121005113743"  />

* 인공 신경망을 도식화해보면 아래와 같다.

<img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221121005209640.png" alt="image-20221121005209640"  />



**여러 논문에서 성능 비교를 진행했을 때 전반적으로 Skip-gram이 CBOW보다 성능이 좋다고 알려져 있다.**
