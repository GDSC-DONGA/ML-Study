# Intro

**openCV를 공부하겠습니다.**  

**다만, 예전에 유튜브를 보고 정리했던 내용이 있어서 복습을 하고나서 그냥 정리했던 내용 복붙해서 올리겠습니다.**

<br>


# 환경 설정

Anaconda Prompt 에서 다음 명령 수행

> pip install opencv-python



```python
import cv2
cv2.__version__
```

'4.5.5'

<br>


# OpenCV (Computer Vision)

다양한 영상 (이미지) / 동영상 처리에 사용되는 오픈소스 라이브러리

* 색상 BGR로 사용한다는점 기억(보통 RGB니까)

* 크기 shape로 보면 세로x가로인점 기억(보통 가로x세로니까)

* 자세히 공부하고 싶다면 : `python opencv readthedocs` 검색후 공식문서 참고

<br>

# 1. 이미지 출력


예제 이미지 : https://pixabay.com/images/id-2083492/  

크기 : 640 x 390  

파일명 : img.jpg



```python
import cv2
img = cv2.imread('img.jpg') # 해당 경로의 파일 읽어오기
cv2.imshow('img', img) # img 라는 이름의 창에 img 를 표시
cv2.waitKey(0) # 지정된 시간(ms) 동안 사용자 키 입력 대기(위의 동작을 바로 종료하면 안되니까) 0은 무한정 기다림을 의미
cv2.destroyAllWindows() # 모든 창 닫기

# 출력 113은? key값 즉, 아스키코드값(q)이다. cv2.waitKey(0)에서 입력받은 key값을 의미!
```

113

<br>

## 읽기 옵션

1. cv2.IMREAD_COLOR : 컬러 이미지. 투명 영역은 무시 (기본값)

1. cv2.IMREAD_GRAYSCALE : 흑백 이미지

1. cv2.IMREAD_UNCHANGED : 투명 영역까지 포함



```python
import cv2
img_color = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('img.jpg', cv2.IMREAD_UNCHANGED)

# imshow를 통해서 이미지 3개를 띄움
cv2.imshow('img_color', img_color) 
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_unchanged', img_unchanged)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## Shape

이미지의 height, width, channel 정보



```python
import cv2
img = cv2.imread('img.jpg')
img.shape # 세로, 가로, Channel
```


(390, 640, 3)

<br>


# 2. 동영상 출력

* 이미지를 여러번 출력해서 영상처럼 출력하는 것

<br>

## 동영상 파일 출력

* cv2.VideoCapture('파일명') 을 이용

* waitKey에 1(ms)을 넣어서 아주 빠르게 영상이 실행되는것처럼 보인다


예제 동영상 : https://www.pexels.com/video/7515833/  

크기 : SD (360 x 640)  

파일명 : video.mp4



```python
import cv2
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened(): # 동영상 파일이 올바로 열렸는지?
    ret, frame = cap.read() # ret : 성공 여부,  frame : 받아온 이미지 (프레임)
    if not ret:
        print('더 이상 가져올 프레임이 없어요')
        break
        
    cv2.imshow('video', frame) # frame을 띄움
    
    if cv2.waitKey(1) == ord('q'): # ord함수이용해 'q'의 아스키코드 사용
        print('사용자 입력에 의해 종료합니다')
        break
        
cap.release() # 자원 해제
cv2.destroyAllWindows() # 모든 창 닫기
```


더 이상 가져올 프레임이 없어요

<br>

## 카메라 출력

* cv2.VideoCapture(0) 을 이용



```python
import cv2
cap = cv2.VideoCapture(0) # 0번째 카메라 장치 (Device ID)

if not cap.isOpened(): # 카메라가 잘 열리지 않은 경우
    exit() # 프로그램 종료
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    cv2.imshow('camera', frame)
    if cv2.waitKey(1) == ord('q'): # 사용자가 q 를 입력하면 
        break
        
cap.release()
cv2.destroyAllWindows()
```

<br>

# 3. 도형 그리기

<br>


## 빈 스케치북 만들기



```python
import cv2
import numpy as np

# 세로 480 x 가로 640, 3 Channel (RGB) 에 해당하는 스케치북 만들기
img = np.zeros((480, 640, 3), dtype=np.uint8) # (0,0,0)의 배열형식으로 세로:480x가로:640개 저장
# img[:] = (255, 255, 255) # 전체 공간을 흰 색으로 채우기(흰색의 BGR : 255,255,255 니까)
# print(img)
cv2.imshow('img', img) # 까만색의 이미지 창을 띄움(0,0,0으로 채웠기때문)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 일부 영역 색칠



```python
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)
img[100:200, 200:300] = (255, 255, 255)
# [세로 영역, 가로 영역]

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 직선

직선의 종류 (line type)



1. cv2.LINE_4 : 상하좌우 4 방향으로 연결된 선

1. cv2.LINE_8 : 대각선을 포함한 8 방향으로 연결된 선 (기본값)

1. cv2.LINE_AA : 부드러운 선 (anti-aliasing)

*  그림판은 LINE_8이고, 마우스 클릭후 선긋고 클릭을 땠을때 : LINE_4->LINE_8처럼 형식이 보이는걸 알 수 있다(2개의 차이점 확인)

*  LINE_AA는 투명도 같은걸 등등 활용해서 좀 더 부드러운 선이 그려지는 것이다.

*  line함수 이용



```python
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 255, 255) # BGR : Yellow, 색깔(BGR인점 기억!!)
THICKNESS = 3 # 두께

cv2.line(img, (50, 100), (400, 50), COLOR, THICKNESS, cv2.LINE_8)
# 그릴 위치, 시작 점(x,y), 끝 점(x,y), 색깔, 두께, 선 종류
cv2.line(img, (50, 200), (400, 150), COLOR, THICKNESS, cv2.LINE_4)
cv2.line(img, (50, 300), (400, 250), COLOR, THICKNESS, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 원

* circle함수 이용



```python
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (255, 255, 0) # BGR 옥색
RADIUS = 50 # 반지름
THICKNESS = 10 # 두께

cv2.circle(img, (200, 100), RADIUS, COLOR, THICKNESS, cv2.LINE_AA) # 속이 빈 원
# 그릴 위치, 원의 중심점, 반지름, 색깔, 두께, 선 종류
cv2.circle(img, (400, 100), RADIUS, COLOR, cv2.FILLED, cv2.LINE_AA) # 꽉 찬 원

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 사각형

* rectangle함수 이용



```python
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 255, 0) # BGR 초록색
THICKNESS = 3 # 두께

cv2.rectangle(img, (100, 100), (200, 200), COLOR, THICKNESS) # 속이 빈 사각형
# 그릴 위치, 왼쪽 위 좌표, 오른쪽 아래 좌표, 색깔, 두께
cv2.rectangle(img, (300, 100), (400, 300), COLOR, cv2.FILLED) # 꽉 찬 사각형

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 다각형

* polylines, fillPoly 함수 이용



```python
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 0, 255) # BGR 빨간색
THICKNESS = 3 # 두께

pts1 = np.array([[100, 100], [200, 100], [100, 200]])
pts2 = np.array([[200, 100], [300, 100], [300, 200]])

# cv2.polylines(img, [pts1], True, COLOR, THICKNESS, cv2.LINE_AA) # 삼각형 그려짐
# cv2.polylines(img, [pts2], True, COLOR, THICKNESS, cv2.LINE_AA) # 닫힘 True면 처음, 끝 좌표도 그어줌
cv2.polylines(img, [pts1, pts2], True, COLOR, THICKNESS, cv2.LINE_AA) # 속이 빈 다각형
# 그릴 위치, 그릴 좌표들, 닫힘 여부, 색깔, 두께, 선 종류

pts3 = np.array([[[100, 300], [200, 300], [100, 400]], [[200, 300], [300, 300], [300, 400]]])
cv2.fillPoly(img, pts3, COLOR, cv2.LINE_AA) # 꽉 찬 다각형(pts3은 초기화때 대괄호 한번더 감싸서 여기선 바로 pts3이 온것이다)
# 그릴 위치, 그릴 좌표들, 색깔, 선 종류

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 4. 텍스트

* putText 함수 이용

<br>


## OpenCV 에서 사용하는 글꼴 종류

1. cv2.FONT_HERSHEY_SIMPLEX : 보통 크기의 산 세리프(sans-serif) 글꼴

1. cv2.FONT_HERSHEY_PLAIN : 작은 크기의 산 세리프 글꼴

1. cv2.FONT_HERSHEY_SCRIPT_SIMPLEX : 필기체 스타일 글꼴

1. cv2.FONT_HERSHEY_TRIPLEX : 보통 크기의 세리프 글꼴

1. cv2.FONT_ITALIC : 기울임 (이탤릭체)



```python
import numpy as np
import cv2

img = np.zeros((480, 640, 3), dtype=np.uint8)

SCALE = 1 # 크기
COLOR = (255, 255, 255) # 흰색
THICKNESS = 1 # 두께

cv2.putText(img, "Nado Simplex", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께
cv2.putText(img, "Nado Plain", (20, 150), cv2.FONT_HERSHEY_PLAIN, SCALE, COLOR, THICKNESS)
cv2.putText(img, "Nado Script Simplex", (20, 250), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "Nado Triplex", (20, 350), cv2.FONT_HERSHEY_TRIPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "Nado Italic", (20, 450), cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC, SCALE, COLOR, THICKNESS)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 한글

* 그냥 한글을 쓰면 ???가 출력된다



```python
import numpy as np
import cv2

img = np.zeros((480, 640, 3), dtype=np.uint8)

SCALE = 1 # 크기
COLOR = (255, 255, 255) # 흰색
THICKNESS = 1 # 두께

cv2.putText(img, "나도코딩", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 한글 우회 방법

* PIL라이브러리 활용해 따로 text를 이미지화 해서 한글로 나타내는 함수를 만들어 사용



```python
import numpy as np
import cv2
# PIL (Python Image Library)
from PIL import ImageFont, ImageDraw, Image

def myPutText(src, text, pos, font_size, font_color): # PIL라이브러리를 활용해 만든 함수(그냥 사용하자)
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font, fill=font_color)
    return np.array(img_pil)

img = np.zeros((480, 640, 3), dtype=np.uint8)

FONT_SIZE = 30
COLOR = (255, 255, 255) # 흰색

# cv2.putText(img, "나도코딩", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께
img = myPutText(img, "나도코딩", (20, 50), FONT_SIZE, COLOR)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 5. 파일 저장

<br>


## 이미지 저장

* imwrite 함수 이용



```python
import cv2
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE) # 흑백으로 이미지 불러오기
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.imwrite('img_save.jpg', img)
print(result)
```


True



### 저장 포맷 (jpg, png)



```python
import cv2
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE) # 흑백으로 이미지 불러오기
cv2.imwrite('img_save.png', img) # png 형태로 저장
```


True

<br>

## 동영상 저장

* VideoWriter_fourcc로 코덱 정의

* CAP_PROP_FRAME_WIDTH,HEIGHT 프레임 크기 정의

* CAP_PROP_FPS 영상 속도 정의(FPS 정의)

* VideoWriter 통해서 저장

 * out.write 통해서 각 frame 저장(out=위의 VideoWriter객체)




```python
import cv2
cap = cv2.VideoCapture('video.mp4')

# 코덱 정의
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 'D', 'I', 'V', 'X'를 의미(*의 활용)

# 프레임 크기, FPS
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 정수값을 가지기 위해 round반올림 사용
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) * 2 # 영상 재생 속도가 2배

out = cv2.VideoWriter('output_fast.avi', fourcc, fps, (width, height))
# 저장 파일명, 코덱, FPS, 크기 (width, height)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    out.write(frame) # 영상 데이터만 저장 (소리 X)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break
        
out.release() # 자원 해제
cap.release()
cv2.destroyAllWindows()
```


```python
# codec = 'DIVX'
# print(codec)
# print(*codec)
# print([codec])
# print([*codec])
```

DIVX
D I V X
['DIVX']
['D', 'I', 'V', 'X']

<br>

# 6. 크기 조정

<br>


## 이미지


고정 크기로 설정



```python
import cv2
img = cv2.imread('img.jpg')
dst = cv2.resize(img, (400, 500)) # width, height 고정 크기

cv2.imshow('img', img)
cv2.imshow('resize', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

비율로 설정

* 사진이 찌그러지지 않고 비율을 유지한 상태로 크기 조절



```python
import cv2
img = cv2.imread('img.jpg')
dst = cv2.resize(img, None, fx=0.5, fy=0.5) # x, y 비율 정의 (0.5 배로 축소)

cv2.imshow('img', img)
cv2.imshow('resize', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



### 보간법

* 보간법이란 이미지를 줄이거나 키우거나 할때 보다 자연스럽게 하기 위함

1. cv2.INTER_AREA : 크기 줄일 때 사용

1. cv2.INTER_CUBIC : 크기 늘릴 때 사용 (속도 느림, 퀄리티 좋음)

1. cv2.INTER_LINEAR : 크기 늘릴 때 사용 (기본값)



보간법 적용하여 축소



```python
import cv2
img = cv2.imread('img.jpg')
dst = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) # x, y 비율 정의 (0.5 배로 축소)

cv2.imshow('img', img)
cv2.imshow('resize', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



보간법 적용하여 확대



```python
import cv2
img = cv2.imread('img.jpg')
dst = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) # x, y 비율 정의 (1.5 배로 확대)

cv2.imshow('img', img)
cv2.imshow('resize', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 동영상


고정 크기로 설정



```python
import cv2
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (400, 500))        
    cv2.imshow('video', frame_resized)
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
```



비율로 설정

* +보간법까지 이용하였음



```python
import cv2
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('video', frame_resized)
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
```

<br>

# 7. 이미지 자르기


영역을 잘라서 새로운 윈도우(창)에 표시



```python
import cv2
img = cv2.imread('img.jpg')
# img.shape # (390, 640, 3)

crop = img[100:200, 200:400] # 세로 기준 100 : 200 까지, 가로 기준 200 : 400 까지 자름

cv2.imshow('img', img) # 원본 이미지
cv2.imshow('crop', crop) # 잘린 이미지
cv2.waitKey(0)
cv2.destroyAllWindows()
```



영역을 잘라서 기존 윈도우에 표시



```python
import cv2
img = cv2.imread('img.jpg')

crop = img[100:200, 200:400] # 세로 기준 100 : 200 까지, 가로 기준 200 : 400 까지 자름
img[100:200, 400:600] = crop # 자른것을 다른위치에 보여주기 위해 좌표 좀 더 오른쪽에 넣음

cv2.imshow('img', img) # 원본 이미지
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 8. 이미지 대칭

* flip함수 이용

<br>


## 좌우 대칭

* 오른쪽 보는 사진이 왼쪽 보는 사진으로 변경



```python
import cv2
img = cv2.imread('img.jpg')
flip_horizontal = cv2.flip(img, 1) # flipCode > 0 : 좌우 대칭 Horizontal (1이 flipCode입력부분)

cv2.imshow('img', img)
cv2.imshow('flip_horizontal', flip_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 상하 대칭

* 사진이 매달려 있는 모습으로 변경



```python
import cv2
img = cv2.imread('img.jpg')
flip_vertical = cv2.flip(img, 0) # flipCode == 0 : 상하 대칭 Vertical

cv2.imshow('img', img)
cv2.imshow('flip_vertical', flip_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 상하좌우 대칭

* 회전한것처럼 변경 (시계방향으로)



```python
import cv2
img = cv2.imread('img.jpg')
flip_both = cv2.flip(img, -1) # flipCode < 0 : 상하좌우 대칭

cv2.imshow('img', img)
cv2.imshow('flip_both', flip_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 9. 이미지 회전

* rotate 함수이용

<br>


## 시계 방향 90도 회전



```python
import cv2
img = cv2.imread('img.jpg')

rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계 방향으로 90도 회전

cv2.imshow('img', img)
cv2.imshow('rotate_90', rotate_90)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 180도 회전



```python
import cv2
img = cv2.imread('img.jpg')

rotate_180 = cv2.rotate(img, cv2.ROTATE_180) # 180도 회전

cv2.imshow('img', img)
cv2.imshow('rotate_180', rotate_180)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 시계 반대 방향 90도 회전 (시계 방향 270도 회전)



```python
import cv2
img = cv2.imread('img.jpg')

rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 시계 반대 방향으로 90도

cv2.imshow('img', img)
cv2.imshow('rotate_270', rotate_270)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 10. 이미지 변형 (흑백)


이미지를 흑백으로 읽음



```python
import cv2
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



불러온 이미지를 흑백으로 변경

* cvtColor 함수 이용



```python
import cv2
img = cv2.imread('img.jpg')

dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img)
cv2.imshow('gray', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 11. 이미지 변형 (흐림)

<br>

## 가우시안 블러


커널 사이즈 변화에 따른 흐림



```python
import cv2
img = cv2.imread('img.jpg')

# (3, 3), (5, 5), (7, 7) 흐림 효과의 정도가 달라짐(이 값에 따라)
kernel_3 = cv2.GaussianBlur(img, (3, 3), 0) # 0은 자동으로 표준편차 선택한다는 의미
kernel_5 = cv2.GaussianBlur(img, (5, 5), 0)
kernel_7 = cv2.GaussianBlur(img, (7, 7), 0)

cv2.imshow('img', img)
cv2.imshow('kernel_3', kernel_3)
cv2.imshow('kernel_5', kernel_5)
cv2.imshow('kernel_7', kernel_7)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



표준 편차 변화에 따른 흐림

* 커널 사이즈 변화보다 더 많이 블러가 적용이 되는게 보인다.



```python
import cv2
img = cv2.imread('img.jpg')

sigma_1 = cv2.GaussianBlur(img, (0, 0), 1) # sigmaX - 가우시안 커널의 x 방향의 표준 편차
sigma_2 = cv2.GaussianBlur(img, (0, 0), 2)
sigma_3 = cv2.GaussianBlur(img, (0, 0), 3)

cv2.imshow('img', img)
cv2.imshow('sigma_1', sigma_1)
cv2.imshow('sigma_2', sigma_2)
cv2.imshow('sigma_3', sigma_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 12. 이미지 변형 (원근)

* 이미지 자르기같은 느낌이지만, 이부분이 활용하기 훨씬 좋다
* 원하는 4개의 지점을 행렬로 가져와서, 다시 그림으로 변환한다

<br>


## 사다리꼴 이미지 펼치기




```python
import cv2
import numpy as np

img = cv2.imread('newspaper.jpg')

width, height = 640, 240 # 가로 크기 640, 세로 크기 240 으로 결과물 출력

src = np.array([[511, 352], [1008, 345], [1122, 584], [455, 594]], dtype=np.float32) # Input 4개 지점
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32) # Output 4개 지점
# 좌상, 우상, 우하, 좌하 (시계 방향으로 4개의 지점 정의)

matrix = cv2.getPerspectiveTransform(src, dst) # Matrix 얻어옴
result = cv2.warpPerspective(img, matrix, (width, height)) # matrix 대로 변환을 함

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 회전된 이미지 올바로 세우기


예제 이미지 : https://pixabay.com/images/id-682332/  

크기 : 1280 x 1019  

파일명 : poker.jpg

* 기울어진 카드 이미지를 4개 지점(네모형태)을 행렬로 가져와서, 다시 그림으로 변환



```python
import cv2
import numpy as np

img = cv2.imread('poker.jpg')

width, height = 530, 710

src = np.array([[702, 143], [1133, 414], [726, 1007], [276, 700]], dtype=np.float32) # Input 4개 지점
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32) # Output 4개 지점
# 좌상, 우상, 우하, 좌하 (시계 방향으로 4 지점 정의)

matrix = cv2.getPerspectiveTransform(src, dst) # Matrix 얻어옴
result = cv2.warpPerspective(img, matrix, (width, height)) # matrix 대로 변환을 함

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 13. 이미지 변형 (이진화)

<br>


## Threshold

* Threshold(=임계값) : 기준
* threshold 함수 사용
  * 127을 임계값으로 설정하고, 최대값을 255로 설정한다면 
    * 127 보다 큰 색의 경우 255(흰색)로 변환
    * 127 보다 작은 색의 경우 0(검정색)으로 변환

* 색 참고
  * 검은색 : 0
  * 진한 회색 : 127

  * 밝은 회색 : 195

  * 흰색 : 255


예제 이미지 : https://www.pexels.com/photo/1029807/  

크기 : Small (640 x 853)  

파일명 : book.jpg



```python
import cv2
img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)

# 127보다 크면 255(흰색)로 작으면 0(검은색)로 변환
ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 

cv2.imshow('img', img)
cv2.imshow('binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src="..\images\2022-12-01-(opencv)Study_Week6\image-20230118183223526.png" alt="image-20230118183223526"  />

<br>

## Trackbar (값 변화에 따른 변형 확인)

* createTrackbar 함수 사용
  * 예로 127, 255 를 준 경우는 127~255 사이 중에서 선택한 값을 임계값으로 사용한다는 것
  * 따라서 threshold 함수에 위 값을 적용해주는 것

* 트랙바 : 말그대로 bar를 의미(좌우로 움직일 수 있는)



```python
import cv2

def empty(pos):
    # print(pos)
    pass

img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)

name = 'Trackbar'
cv2.namedWindow(name)

cv2.createTrackbar('threshold', name, 127, 255, empty) # bar 이름, 창의 이름, 초기값, 최대값, 이벤트 처리

while True:
    thresh = cv2.getTrackbarPos('threshold', name) # bar 이름, 창의 이름
    ret, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY) # 트랙바의 값을 통해 임계값을 수정하는 것
    
    if not ret:
        break
        
    cv2.imshow(name, binary)
    if cv2.waitKey(1) == ord('q'):
        break
        
cv2.destroyAllWindows()
```

<img src="..\images\2022-12-01-(opencv)Study_Week6\image-20230118183442275.png" alt="image-20230118183442275" style="zoom: 67%;" />

<br>

## Adaptive Threshold

이미지를 작은 영역으로 나누어서 임계치 적용

* adaptiveThreshold 함수 사용
  * 트랙바를 2개 만들어서 좀 더 값의 범위를 세분화 해서 임계값을 설정하는 것
  * 아래 코드는, 트랙바로 구한 2개의 임계값과 최대값인 255를 adaptiveThreshold 에 설정한 것

* 예로, 이미지에 글자를 읽고싶은데 햇빛, 조명 등등에 의해 이진화가 제대로 안되는경우 이를 사용
* 위의 책 사진을 참고하자면, 햇빛 부분은 밝고 그림자 지어진 부분은 어두워서 글자를 해석하고 싶지만 이진화가 하기 힘든 상황이다.
  * 이때 영역을 세분화해서 임계치를 적용하자는 아이디어




```python
import cv2

def empty(pos):
    # print(pos)
    pass

img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)

name = 'Trackbar'
cv2.namedWindow(name)

 # bar 이름, 창의 이름, 초기값, 최대값, 이벤트 처리
cv2.createTrackbar('block_size', name, 25, 100, empty) # 홀수만 가능, 1보다는 큰 값
cv2.createTrackbar('c', name, 3, 10, empty) # 일반적으로 양수의 값을 사용
 # bar 2개를('block_size', 'c') 조정함으로써 이미지 이진화가 훨씬(기존 Threshold보다) 잘된다는걸 볼 수 있음

while True:
    block_size = cv2.getTrackbarPos('block_size', name) # bar 이름, 창의 이름
    c = cv2.getTrackbarPos('c', name)
    
    # 위에서 block_size는 홀수 & 1보다 큰 값이라 가정했으므로 이를 처리
    if block_size <= 1: # 1 이하면 3 으로
        block_size = 3
        
    if block_size % 2 == 0: # 짝수이면 홀수로
        block_size += 1
    
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    
    cv2.imshow(name, binary)
    if cv2.waitKey(1) == ord('q'):
        break
        
cv2.destroyAllWindows()
```

<img src="..\images\2022-12-01-(opencv)Study_Week6\image-20230118184445275.png" alt="image-20230118184445275" style="zoom:67%;" />

<br>

## 오츠 알고리즘

Bimodal Image 에 사용하기 적합 (최적의 임계치를 자동으로 발견)

* cv2.THRESH_OTSU 를 사용
  * thresholde에 임계값을 -1로 설정하는 이유는 오츠로 어차피 구해지기 때문에 무시되는 값임

* book.jpg에는 별루였으니 Bimodal Image에 사용하는걸 추천
  * Bimodal Image 관련해서는 조금 딥한 얘기이므로 구글링 할 것
  * 공식 홈페이지에서는 히스토그램 상에서 2개의 피크를 이루는 형태의 이미지라고 하긴 함




```python
import cv2
img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)

ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # -1 부분은 어차피 무시되는 값이라 보기 쉽게 -1적음
print('otsu threshold ', ret) # 임계치가 100이라고 출력이 된것이다.

cv2.imshow('img', img)
cv2.imshow('binary', binary)
cv2.imshow('otsu', otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


otsu threshold  100.0

<br>

# 14. 이미지 변환 (팽창)

<br>


## 이미지를 확장하여 작은 구멍을 채움

흰색 영역의 외곽 픽셀 주변에 흰색을 추가

* dilate 함수 사용
  * 글자에 구멍이 있는데 이 글자를 팽창(키우기)시켜서 구멍을 매꿔나가는 형태

* dilate.png : 검정배경에 흰색글자이고 글자에 구멍(검정)만든 이미지



```python
import cv2
import numpy as np

kernel = np.ones((3, 3), dtype=np.uint8) # 3x3배열 (1로 채움)
# kernel

img = cv2.imread('dilate.png', cv2.IMREAD_GRAYSCALE)
dilate1 = cv2.dilate(img, kernel, iterations=1) # 반복 횟수(팽창1번)
dilate2 = cv2.dilate(img, kernel, iterations=2)
dilate3 = cv2.dilate(img, kernel, iterations=3) # 3번 팽창(글자가 커짐과 동시에 글자의 구멍이 매꿔짐)

cv2.imshow('gray', img)
cv2.imshow('dilate1', dilate1)
cv2.imshow('dilate2', dilate2)
cv2.imshow('dilate3', dilate3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 15. 이미지 변환 (침식)

* erode 함수 사용
  * 글자 주변의 배경에 구멍이 있는것이고, 글자의 색을 침식(줄이기)시켜서 배경 구멍을 매꿈
  * 즉, 글자색과 구멍색이 같은 상황인것이며, 침식 진행하면 글자 크기 줄어들고 배경 구멍 매꿈

* erode.png : 검정배경에 흰색글자이고 배경에 구멍(흰색)만든 이미지

<br>


## 이미지를 깎아서 노이즈 제거

흰색 영역의 외곽 픽셀을 검은색으로 변경



```python
import cv2
import numpy as np
kernel = np.ones((3, 3), dtype=np.uint8)

img = cv2.imread('erode.png', cv2.IMREAD_GRAYSCALE)
erode1 = cv2.erode(img, kernel, iterations=1) # 1회 반복
erode2 = cv2.erode(img, kernel, iterations=2) 
erode3 = cv2.erode(img, kernel, iterations=3) # 3회 반복(글자가 줄어듬과 동시에 배경 구멍도 매꿔짐)

cv2.imshow('gray', img)
cv2.imshow('erode1', erode1)
cv2.imshow('erode2', erode2)
cv2.imshow('erode3', erode3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 16. 이미지 변환 (열림 & 닫힘)

* 위에서 배운 침식, 팽창을 응용해서 구하는 방식이다.

<br>


## 열림 (Opening) : 침식 후 팽창. 깎아서 노이즈 제거 후 살 찌움

> dilate(erode(image))
>
> 즉, erode(침식) -> dilate(팽창) 순으로 진행하는 방식



```python
import cv2
import numpy as np
kernel = np.ones((3, 3), dtype=np.uint8)

img = cv2.imread('erode.png', cv2.IMREAD_GRAYSCALE)

erode = cv2.erode(img, kernel, iterations=3)
dilate = cv2.dilate(erode, kernel, iterations=3)

cv2.imshow('img', img)
cv2.imshow('erode', erode)
cv2.imshow('dilate', dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 닫힘 (Closing) : 팽창 후 침식. 구멍을 메운 후 다시 깎음

> erode(dilate(image))
>
> 즉, dilate(팽창) -> erode(침식) 순으로 진행하는 방식



```python
import cv2
import numpy as np
kernel = np.ones((3, 3), dtype=np.uint8)

img = cv2.imread('dilate.png', cv2.IMREAD_GRAYSCALE)

dilate = cv2.dilate(img, kernel, iterations=3)
erode = cv2.erode(dilate, kernel, iterations=3)

cv2.imshow('img', img)
cv2.imshow('dilate', dilate)
cv2.imshow('erode', erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

# 17. 이미지 검출 (경계선)

<br>

## Canny Edge Detection

* Canny 함수 사용
  * 아래 코드로 해석하자면, 이미지 픽셀의 기울기(변화량)이 200보다 크면 경계선을 긋는다.
  * 그리고 150보다 작으면 경계선을 긋지 않는다.

* 가장 많이쓰는 알고리즘


예제 이미지 : https://pixabay.com/images/id-1300089/  

크기 : 1280 x 904  

파일명 : snowman.png



```python
# 픽셀값이 서로 다른(예:빨강->노랑) 경우를 선을 그어서 경계선으로 보여줌
import cv2
img = cv2.imread('snowman.png')

canny = cv2.Canny(img, 150, 200) 
# 대상 이미지, minVal (하위임계값), maxVal (상위임계값)

cv2.imshow('img', img)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src="..\images\2022-12-01-(opencv)Study_Week6\image-20230118191846926.png" alt="image-20230118191846926" style="zoom:80%;" />



### 트랙바로 임계값을 설정 및 경계선

* 아래 그림은 임계값을 매우 작게 줬더니 발생한 형태이다.




```python
# 트랙바를 통해서 어떤 값에 경계선이 제일 잘 나타나는지 확인
import cv2

def empty(pos):
    pass

img = cv2.imread('snowman.png')

name = "Trackbar"
cv2.namedWindow(name)
cv2.createTrackbar('threshold1', name, 0, 255, empty) # minVal
cv2.createTrackbar('threshold2', name, 0, 255, empty) # maxVal

while True:
    threshold1 = cv2.getTrackbarPos('threshold1', name)
    threshold2 = cv2.getTrackbarPos('threshold2', name)
    
    canny = cv2.Canny(img, threshold1, threshold2)
    # 대상 이미지, minVal (하위임계값), maxVal (상위임계값)
    
    cv2.imshow('img', img)
    cv2.imshow(name, canny)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
```

<img src="..\images\2022-12-01-(opencv)Study_Week6\image-20230118191729538.png" alt="image-20230118191729538" style="zoom:80%;" />

<br>

# 18. 이미지 검출 (윤곽선)

<br>


## 윤곽선 (Contour) : 경계선을 연결한 선


예제 이미지 : https://pixabay.com/images/id-161404/  

크기 : 640 x 408  

파일명 : card.png

* findContours, drawContours 함수 사용

* 아래 코드 해석해 보자면,

* 사진 색 gray 계열로 변환 -> 오츠 알고리즘으로 이진화 -> 윤곽선 찾는 함수 사용
  * 윤곽선 찾는 함수에 임계값을 오츠 알고리즘으로 구한 임계값을 적용했음



```python
# 윤곽선을 위해서는 이미지를 먼저 바이너리 이미지화 시켜야 한다.(=이진화)
# 그후 윤곽선을 찾아보면 된다.
import cv2
img = cv2.imread('card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # 윤곽선 검출
# 윤곽선 정보, 계층 구조
# 이미지, 윤곽선 찾는 모드 (mode) : RETR_LIST , ...
# 윤곽선 찾을때 사용하는 근사치 방법 (method) : CHAIN_APPROX_NONE(모든 좌표만), CHAIN_APPROX_SIMPLE(꼭짓점 좌표만)

COLOR = (0, 200, 0) # 녹색
cv2.drawContours(target_img, contours, -1, COLOR, 2) # 윤곽선 그리기
# 대상 이미지, 윤곽선 정보, 인덱스 (-1 이면 전체), 색깔, 두께
# => 인덱스의 의미는 몇개의 윤곽선을 그릴지 의미

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('otsu', otsu)
cv2.imshow('contour', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```



### 윤곽선 찾기 모드

1. cv2.RETR_EXTERNAL : 가장 외곽의 윤곽선만 찾음(예로 카드의 내부그림 X, 네모난 외각만!)

1. cv2.RETR_LIST : 모든 윤곽선 찾음 (계층 정보 없음) 

1. cv2.RETR_TREE : 모든 윤곽선 찾음 (계층 정보(=hierarchy)를 트리 구조로 생성)



```python
import cv2
img = cv2.imread('card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(hierarchy)
# print(f'총 발견 갯수 : {len(contours)}')

COLOR = (0, 200, 0) # 녹색
cv2.drawContours(target_img, contours, -1, COLOR, 2)

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('otsu', otsu)
cv2.imshow('contour', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## 경계 사각형

윤곽선의 경계면을 둘러싸는 사각형

> boundingRect()



```python
import cv2
img = cv2.imread('card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

COLOR = (0, 200, 0) # 녹색

for cnt in contours: # contours는 x, y, width, height이 있다
    x, y, width, height = cv2.boundingRect(cnt)
    cv2.rectangle(target_img, (x, y), (x + width, y + height), COLOR, 2) # 사각형 그림

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('otsu', otsu)
cv2.imshow('contour', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src="..\images\2022-12-01-(opencv)Study_Week6\image-20230118192631395.png" alt="image-20230118192631395"  />

<br>

## 면적

면적이 설정한 값보다 큰 면적들만 윤곽선 경계에 사각형을 그리려고 할 때 이용

> contourArea()



```python
import cv2
img = cv2.imread('card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

COLOR = (0, 200, 0) # 녹색

for cnt in contours:
    if cv2.contourArea(cnt) > 25000: # 면적이 25000보다 크다면 사각형을 그리겠다
        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(target_img, (x, y), (x + width, y + height), COLOR, 2) # 사각형 그림

cv2.imshow('img', img)
cv2.imshow('contour', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src="..\images\2022-12-01-(opencv)Study_Week6\image-20230118192908881.png" alt="image-20230118192908881"  />
