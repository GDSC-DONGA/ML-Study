## Intro

**ML**에 들어가기 앞서 `Python`과 라이브러리`(Numpy, Matplotlib)`를 선행학습 합니다.

* `Numpy, Matplotlib` 순으로 정리하겠습니다.

<br>

## Numpy

**numpy라이브러리를 통해 복잡한 수치연산을 수행할 수 있다(선형대수 라이브러리)**

* 연산에 numpy를 이용, C언어로 구현
* 기본적으로 Numpy는 배열/행렬(array) 단위를 데이터 기본으로 구성
  * 1d, 2d, 3d(d는 차원으로써 dimention의 약어)에 기반한 배열 연산을 수행
* 매우 방대하기 때문에, 자세한 내용은 공식 문서를 참고



### 1. Init(데이터 선언)

* **Numpy의 배열과 행렬**

```python
data1 = [1,2,3,4,5] # list
arr1 = np.array(data1) # 1d array
arr4 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) # 2d array

np.zeros((3,3)) # 3x3 array - init : 0
np.ones((3,3)) # 3x3 array - init : 1
np.arange(0,30) # array([0, 1, 2, 3, 4... 27, 28, 29])
```

* **형변환**

```python
arr1 = np.array([1.0, 2.0, 3.0]) # list -> ndarray
arr2 = list(arr1) # ndarray -> list
```



### 2. Operation(데이터 연산)

**간단한 예시들**

* **수학 계산**
  * 4a + 3b = 23  
    3a + 2b = 16

```python
a = np.array([[4,3], [3,2]]) # 식
b = np.array([23, 16]) # 결과
x = np.linalg.solve(a, b)
print(x) # 위 식의 a, b 값
# output : [2. 5.]
```

* **논리 연산**

```python
arr1 = np.array([[True,True],[False, False]])
arr2 = np.array([[True,True],[True, False]])

# 논리합 계산하기(AND 연산)
# astype(int)란 True/False 값을 1/0로 변환
np.logical_and(arr1, arr2).astype(int) 
# output : array([[1, 1], [0, 0]])
```

* **날짜 계산**

```python
monthly_days = np.arange(0,30) # 0~29 array
base_date = np.datetime64('2021-03-01') # numpy의 datetime64 메소드 사용
random_date = base_date + np.random.choice(monthly_days) # 랜덤으로 날짜 반환
# output : 2021-03-05, 2021-03-08 ... 날짜의 day는 랜덤으로 출력
```

<br>

## Matplotlib

**matplotilib라이브러리를 통해 그래프를 시각화해서 보여줄 수 있다.**

* 좀더 진보된 라이브러리를 원한다면, seaborn 등을 참고하면 좋다.
* 그래프 종류가 다양한데, 몇개만 알아보겠다.



### 1. 그래프 기본(Line Plot)

* **라인 플롯(line plot)** : 간단한 선을 그리는 그래프

```python
import matplotlib.pyplot as plt
plt.title('Line Graph') # Title
x = [1, 2, 3] # x축
y = [2, 4, 8] # y축
plt.plot(x, y) # line plot
plt.show() # 그래프 보여줌
```

* **축(X, Y)**

```python
# X, Y축 Title 지정
plt.xlabel('X축') # X축 Label
plt.ylabel('Y축') # Y축 Label

# X축의 1, 2, 3... 표기처럼 그 부분 수정 방법
plt.xticks([1, 2, 3])
plt.yticks([3, 6, 9, 12])
```

* **범례(legend)**

```python
plt.legend() # 위치 기본값 : 'best'이며, 그래프 제일 빈공간에 자동 위치
plt.legend(loc='upper right') # 우측 상단 위치로
plt.legend(loc='lower right') # 우측 하단 위치로
plt.legend(loc=(0.7, 0.8)) # x, y직접 좌표 입력도 가능
```

* **스타일(Style)**

  * Marker

  ```python
  plt.plot(x, y, marker='o') # 데이터 존재하는 부분에 마커설정
  plt.plot(x, y, marker='o', linestyle='None') # 마커만 있고, 선X
  plt.plot(x, y, marker='o', markersize=20, markeredgecolor='red') # 마커 모서리 빨간색, 크기20
  plt.plot(x, y, marker='o', markersize=20, markeredgecolor='red', markerfacecolor='yellow') # 마커 내부 노란색
  ```

  * Line

  ```python
  plt.plot(x, y, linewidth=5) # 선두께 조정
  plt.plot(x, y, linestyle=':') # 선 모양 점선 형태
  plt.plot(x, y, color='pink') # 선 색깔 변경(기본값 : blue)
  
  # 한번에 하는 방식들도 존재(포맷방식)
  plt.plot(x, y, 'ro--') # color, marker, linestyle
  ```

  * Graph

  ```python
  plt.figure(figsize=(10, 5)) # 그래프 크기 : width:10, height:5
  plt.figure(facecolor='yellow') # 그래프 배경색 : yellow
  ```

* **텍스트(Text)**

  * 참고로 `enumerate()` 는 열거하는 함수
  * `plt.text`의 인자인 `x[idx], y[idx] + -0.3`은 텍스트가 들어갈 위치를 의미

  ```python
  plt.plot(x, y, marker='o')
  
  # idx - index, txt - value
  for idx, txt in enumerate(y):
      plt.text(x[idx], y[idx] + 0.3, txt, ha='center', color='blue')
  ```

* **여러 데이터(Lines)**

  * 여러 데이터를 넣어서 많은 선들을 그릴 수 있음

  ```python
  x = [1, 2, 3]
  y1 = [2, 4, 8] 
  y2 = [5, 1, 3] 
  y3 = [1, 2, 5] 
  
  plt.plot(x, y1)
  plt.plot(x, y2)
  plt.plot(x, y3)
  ```

  

### 2. 막대 그래프

* **바 차트(bar chart)** : 막대 그래프를 의미
* `plot`이 아닌 `bar` 메소드를 사용
  * 속성들 유사

```python
x = [1, 2, 3]
y1 = [2, 4, 8] 

plt.bar(x, y)
```



### 3. 원 그래프

* **파이 차트(pie chart)** : 원 그래프를 의미
* `pie` 메소드를 사용

```python
values = [30, 25, 20, 13, 10, 2]
labels = ['Python', 'Java', 'Javascript', 'C#', 'C/C++', 'ETC']

# labels속성을 통해 이름, autopct속성을 통해 퍼센트값, startangle속성을 통해 시작위치, counterclock속성을 통해 방향 조정
plt.pie(values, labels=labels, autopct='%.1f%%', startangle=90, counterclock=False)
plt.show()
# values = [1,1,1,1,1,1] 라면?? 똑같은 퍼센트로 균일하게 나뉘어져 나옴.
```



### 4. 산점도 그래프

* **스캐터 플롯(scatter plot)** : 산점도 그래프를 의미
* `scatter` 메소드를 사용

```python
X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
plt.title("Scatter Plot")
plt.scatter(X, Y)
plt.show()
```



### 5. 여러 그래프 출력

* 한번에 여러 그래프를 구성해서 출력하는 방법

* `subplots` 메소드를 사용
  * 아래의 예시는 `subplots`를 통해서 4개의 그래프를 생성
  * df(데이터프레임)는 그래프를 구성할 데이터들

```python
fig, axs = plt.subplots(2, 2, figsize=(15, 10)) # 2 x 2 에 해당하는 plot 들을 생성(4개 그래프 생성)
fig.suptitle('여러 그래프 넣기') # 전체 제목(fig를 통해)

# 첫 번째 그래프(axs를 통해)
axs[0, 0].bar(df['이름'], df['국어'], label='국어점수') # 데이터 설정
axs[0, 0].set_title('첫 번째 그래프') # 제목
axs[0, 0].legend() # 범례
axs[0, 0].set(xlabel='이름', ylabel='점수') # x, y 축 label
axs[0, 0].set_facecolor('lightyellow') # 전면 색
axs[0, 0].grid(linestyle='--', linewidth=0.5)

# 두 번째 그래프
axs[0, 1].plot(df['이름'], df['수학'], label='수학')
axs[0, 1].plot(df['이름'], df['영어'], label='영어')
axs[0, 1].legend()

# 세 번째 그래프
axs[1, 0].barh(df['이름'], df['키'])

# 네 번째 그래프
axs[1, 1].plot(df['이름'], df['사회'], color='green', alpha=0.5)
```

