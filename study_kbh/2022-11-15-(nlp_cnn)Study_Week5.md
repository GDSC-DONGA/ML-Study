# Intro

**NLP - CNN을 공부하겠습니다.**  

* 참고 URL
  * [CNN](https://happy-jihye.github.io/nlp/nlp-4/)
* 실습
  * 해당 링크 참고 [CNN](https://happy-jihye.github.io/nlp/nlp-4/)


<br>

## CNN on text

CNN은 앞전에 공부했듯이 이미지에 강력한 알고리즘

* 그런데, 이미지는 너비x높이 처럼 2차원인데 Text는 1차원

  * 이때 embedding 방법으로 Text를 2차원의 vector로 봐서 해결
  * 즉, 단어의 수를 10개, embedding을 통해 차원을 100차원이라 한다면 10x100으로 표현 가능

* 다음으로, CNN에 활용되는 filter(=커널)의 경우?

  * filter차원은 [n x emb_dim]의 size를 가진 tensor이고, 이때 n은 연속된 단어의 수를 의미
  * n의 예로 uni-gram, bi-gram, tri-gram, 4-gram(순서대로 1,2,3,4 의미)라면?
    * 문장 : I want you love me
    * uni-gram : I, want, you, love, me
    * bi-gram : I want, want you, you love, love me
    * tri-gram : I want you, want you love, you love me
    * 4-gram : I want you love, want you love me
  * emb_dim은 말그대로 embedding을 통해 구한 차원이다.

* 아래는 I hate this film이란 문장을 embedding을 통해서 4x5 차원으로 표현한 행렬

  * filter는 bi-gram 사용한 모습이며 filter size는 [2x5]

  <img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221120220313686.png" alt="image-20221120220313686"  />

* 다음으로 Max-pooling을 적용한 모습

  * 빨간 네모에서 최대값을 적용한것으로 보면 된다.

  <img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221120220659649.png" alt="image-20221120220659649" style="zoom:80%;" />

* single vector(빨간색)에서 나온 최댓값(보라색)들을 하나의 linear vector으로 만든 후에 감정을 최종적으로 예측

  * linear vector의 모습은 아래 또 다른 예시를 통해서 관측 할 수 있다(맨마지막 이전 네모)

    * 2, 3, 4-filter를 사용한 모습

    <img src="..\images\2022-11-15-(nlp_cnn)Study_Week5\image-20221120221133834.png" alt="image-20221120221133834"  />

* 따라서 CNN으로 NLP적용이 가능하다.

  * 예시 : 3종류의 총 300개의 n-gram filter를 사용해서 적용하는 실습
    * 해당 링크 참고 [CNN](https://happy-jihye.github.io/nlp/nlp-4/)



