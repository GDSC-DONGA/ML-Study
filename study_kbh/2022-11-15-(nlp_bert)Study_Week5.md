# Intro

**NLP - BERT를 공부하겠습니다.**  

* 참고 URL
  * [BERT(실습 포함)](https://wikidocs.net/109251)
  * [BERT-Youtube](https://www.youtube.com/watch?v=30SvdoA6ApE&t=201)

<br>

## BERT(Bidirectional Encoder Representations from Transformers)

**트랜스포머 계열의 BERT, GPT, T5 등 다양한 사전 훈련된 언어 모델들 중 대표적인 BERT에 대해서 소개**

* Bidirectional(양방향), 따라서 양방향 인코더 구조를 가짐
* 트랜스포머는 BERT들어가기전 약간의 사전 지식이 필요



### 사전 훈련된 워드 임베딩

* 문제점 : 워드 임베딩 방법은 모두 하나의 단어가 하나의 벡터값으로 맵핑되므로, 문맥을 고려하지 못 하여 다의어나 동음이의어를 구분하지 못함
  * EX) '사과' 라는 단어는 사람에 사과하는 의미와 먹는 과일의 의미로도 사용
  * 그러나 임베딩 벡터는 '사과'라는 벡터에 하나의 벡터값을 맵핑하므로 이 두 가지 의미를 구분할 수 없음

* 이 한계는 사전 훈련된 언어 모델을 사용하므로서 극복할 수 있었으며 위에서 언급한 BERT 등이 이러한 문제의 해결책



### Transformer(트랜스포머)

**구글에서 만들었으며, 양방향 인코더와 Left to Right 방향 디코더로 구성**

* BERT는 GPT-1의 영향을 받음

  * GPT-1 : 트랜스포머의 Left to Right 방향 디코더 구조로 만든 자연어처리 모델

  * 문장을 데이터로 사용, 단어를 하나씩 읽어가면서 다음 단어를 예측하면서 학습을 하는 모델

    <img src="..\images\2022-11-15-(nlp_bert)Study_Week5\image-20221121123938304.png" alt="image-20221121123938304"  />

  * 기본적으로 다음 단어를 예측하는 학습을 하기 때문에 따로 Label을 만들 필요는 없음

  * 그런데, 한방향의 구조는 질의 응답을 할때 문맥의 이해가 굉장히 중요한데,  
    문맥 이해에 약하다고 한다.

  * 따라서 BERT는 양방향으로 구조를 했으며, 이 GPT-1의 영향을 받게된것이라 할 수 있다.



#### 트랜스포머의 작동방식 간략히

<img src="..\images\2022-11-15-(nlp_bert)Study_Week5\image-20221121144503134.png" alt="image-20221121144503134" style="zoom:80%;" />

* 입력값은 인코딩의 입력이 되며, 이는 `positional encoding`과 더해지게 된다.
  * `positional encoding` : 스칼라값이 아닌 벡터값으로 단어 벡터와 같은 차원을 지닌 벡터값  
    위치 벡터값이 같아지는 문제를 해결하기 위해 사용
* 이 값을 행렬 연산을 통해서 `attention vector`를 생성. 이때 모든 토큰을 한번에 계산
  * `attention vector` : 각각의 토큰들은 문장속의 모든 토큰들을 봄으로써 토큰의 의미를 정확히 모델에 전달  
    즉, 토큰의 의미를 구하기 위해서 사용
  * 예로 [text, message] 로 구현된 attention vector가 있다면 text와 message가 함께 있으므로  
    'text는 문자를 전송하다' 라는 의미를 얻을 수 있다.
* 이 벡터는 FC(Fully Connected)로 연결되는데 총 6번 반복
* 인코더는 모든 토큰을 한번에 계산하므로 왼쪽에서 오른쪽으로 하나씩 읽어가는 과정이 없다.
  * **양방향 인코더**


* 디코더는 왼쪽에서 오른쪽으로 하나씩 읽어가면서 순차적으로 출력값 생성
  * **Left to Right 방향 디코더**

<img src="..\images\2022-11-15-(nlp_bert)Study_Week5\image-20221121223636160.png" alt="image-20221121223636160" style="zoom: 67%;" />                <img src="..\images\2022-11-15-(nlp_bert)Study_Week5\image-20221121223705217.png" alt="image-20221121223705217" style="zoom: 80%;" />

* 이처럼 디코더는 순차적으로 출력값을 생성
  * 디코더는 인코더의 출력값과 이전 생성한 디코더의 출력값을 사용해서 현재 디코더의 출력값을 생성
  * 마찬가지로 attention vector 를 만들고, FC로 연결되는데 총 6번 반복
  * 스페셜 토큰인 End 토큰을 출력할때까지 디코더는 반복



### GPT vs BERT

 <img src="..\images\2022-11-15-(nlp_bert)Study_Week5\image-20221121225323337.png" alt="image-20221121225323337" style="zoom:80%;" />

* 왼쪽 GPT는 순차적으로 다음 단어를 예측하고 있는 반면
* 오른쪽 BERT는 순차적이 아닌 랜덤으로 예측할 단어를 mask로 가려두고, 이를 찾게끔 학습
* BERT도 따로 사용자가 label을 만들어줄 필요없이 기존 데이터로 알아서 학습이 가능



**만약 두 문장으로 입력값이 들어온 경우?(BERT)**

* `CLS 벡터`로 이 입력데이터 두 문장을 알려주고, `SEP 벡터`로 이 두 문장을 구분
* 그리고 앞서 트랜스포머에서 `입력토큰과 Postion encoding`이 더해진다고 했는데,  
  BERT는 `입력토큰과 Segment encoding, Position encoding`이 더해진다.
  * 또한, 토큰의 구분은 `WordPiece` 방법으로 구분한다고 함.
  * `Segment` : 두개의 문장 입력이면, 이 문장을 구별하기 위해 두 문장에 서로다른 번호를 줌
  * `Position` : 토큰들의 상대적 위치 정보를 알려줌



### BERT(pre-training -> fine tuning)

사전 학습된 BERT는 인터넷에서 쉽게 구할수 있다고 한다.  
따라서 개발자에게 중요한것은 `fine tuning`을 어떻게 할지 생각하는 것

* 예제 2개만 소개

  * 두 문장의 관계를 예측하는 형식
    * CLS로 이 입력데이터 두 문장을 알려주고 SEP로 이 두 문장을 구분한다고 했으므로  
      이 CLS를 두 문장의 관계를 나타내게 학습
  * 한 문장의 각 단어들(토큰) 구분하는 형식
    * 문장 한개를 입력받고 CLS토큰이 분류값 중 한개가 되도록 학습

  <img src="..\images\2022-11-15-(nlp_bert)Study_Week5\image-20221122004108044.png" alt="image-20221122004108044"  />



