# Intro

**NLP - Bag of Words를 공부하겠습니다.**  

* 참고 URL
  * [Bag of Words](https://bkshin.tistory.com/entry/NLP-5-Bag-of-Words-BOW)
  * [벡터화 2가지](https://bkshin.tistory.com/entry/NLP-6-%EC%B9%B4%EC%9A%B4%ED%84%B0-%EA%B8%B0%EB%B0%98-%EB%B2%A1%ED%84%B0%ED%99%94CountVectorizer%EC%99%80-TF-IDF-%EB%B2%A1%ED%84%B0%ED%99%94)
  

<br>

## Bag of Words(단어들의 가방)

**단어들의 문맥이나 순서를 무시하고, 단어들에 대해 빈도 값(frequency)을 부여해 feature(출력) 값을 만드는 모델**

* Bag of Words(BOW) 모델 : 문서 내 모든 단어를 한꺼번에 가방(Bag) 안에 넣은 뒤에 흔들어서 섞는다는 의미

  <img src="..\images\2022-11-15-(nlp_bow)Study_Week5\image-20221120190909087.png" alt="image-20221120190909087" style="zoom:80%;" />



### BOW 기반 feature 추출

1) John likes to watch movies. Mary likes movies too.

2) John also likes to watch football games.



**위의 두 문장을 예시로 토큰화하여 배열에 담아 준다.**

`[ "John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games" ]`



**각 토큰이 몇 번 등장하는지 빈도수를 배열에 담아 준다.** 

`[1, 2, 1, 1, 2, 1, 1, 0, 0, 0]` -> 1번 문장의 단어 빈도수

`[1, 1, 1, 1, 0, 0, 0, 1, 1, 1]` -> 2번 문장의 단어 빈도수



이 과정의 결론은,  
BOW 기반으로 피처를 만드는 것은 문서 내의 단어(더 정확히 말하면 토큰)를 숫자형 데이터로 바꾸는 것



### BOW 단점

**문맥 의미(Semantic Context) 반영 부족**

* 단어의 순서를 고려하지 않기 때문에 문맥적인 의미가 무시

**희소 행렬 문제**

* 항상 같은 단어들이 아니라 각 문서마다 서로 다른 단어로 구성되는 경우가 훨씬 많음
* 따라서 대부분의 칼럼이 0으로 채워지게 됨 -> `희소(Sparse)`
  * 이는 머신러닝의 성능을 낮추게 된다



### BOW Feature Vectorization(피처 벡터화)

**모든 문서의 모든 단어를 칼럼 형태로 나열하고 해당 단어에 빈도 값을 부여하는 것**

* M개 문서, N개 단어 -> `MxN 행렬`
* 피처 벡터화 두 가지 방식
  * `카운트 기반 벡터화(CountVectorizer)`
  * `TF-IDF(Term Frequency - Inverse Document Frequency) 기반 벡터화`

<br>

## CountVectorizer

**BOW모델에서 설명한 각 단어별로 빈도수 값을 부여하는 것 처럼, 빈도수를 Count로 부여하는 경우를 의미한다.**

* 기본적으로 2자리 이상의 문자에 대해서만 토큰으로 인식
  * `I`가 사라짐

```python
from sklearn.feature_extraction.text import CountVectorizer
data = ['I know you want my love. because you love me.']
vector = CountVectorizer()
print(vector.fit_transform(data).toarray()) # data의 각 단어 빈도수 기록 출력
print(vector.vocabulary_) # 각 단어와 함께 인덱스를 출력
```

```python
# 출력
[[1 1 2 1 2 1]]
{'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
```



* `불용어(Stop words)`를 지정해서 `I`를 없앨수도 있음
  * 사용자가 정의한 불용어를 제거

```python
data = ['I know you want my love. because you love me.']
vector = CountVectorizer(stop_words=["I"]) # 불용어 설정
print(vector.fit_transform(data).toarray()) # data의 각 단어 빈도수 기록 출력
print(vector.vocabulary_) # 각 단어와 함께 인덱스를 출력
```

```python
# 출력
[[1 1 2 1 2 1]]
{'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
```

<br>

## TF-IDF

**TF-IDF는 각 문서의 높은 빈도의 단어들에 높은 가중치를 주되, 모든 문서에서 자주 등장하는 단어에는 패널티를 주는 방식으로 값을 부여하는 방식이다.**

* A문서에서도 'The'가 가장 많이 등장하고, B문서에서도 'The'가 가장 많이 등장한다고 해서 두 문서가 유사한 문서라고 판단할 수는 없음  
  * **이런 문제를 보완하기 위해 TF-IDF 벡터화를 사용**
  * 문서의 양이 많을 경우 일반적으로 이 방식을 선호한다고 한다.
  * `'love', 'you'` 가 제일 많은 빈도수인데, 나머지가 더 가중치가 크게 나왔다.  
    즉, 다른 문장의 단어 빈도도 고려하여 해당 단어의 중요도를 보여준다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
data = [
    'you know I want your love',
    'I love you',
]
tfidfv = TfidfVectorizer().fit(data) # TF-IDF 클래스 사용
print(tfidfv.transform(data).toarray())
print(tfidfv.vocabulary_)
```

```python
# 출력
[[0.49922133 0.35520009 0.49922133 0.35520009 0.49922133] # 1번 문장
 [0.         0.70710678 0.         0.70710678 0.        ]] # 2번 문장
{'you': 3, 'know': 0, 'want': 2, 'your': 4, 'love': 1} # 데이터 : index
```

