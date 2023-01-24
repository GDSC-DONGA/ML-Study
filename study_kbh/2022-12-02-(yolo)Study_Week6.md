# Intro

**참고 문서**

- **[개념 + 실습](https://www.youtube.com/watch?v=fdWx3QV5n44)**
- **[with 라즈베리파이](http://daddynkidsmakers.blogspot.com/2019/01/blog-post.html)**

<br>

# 객체 탐지(Object Detection)

* 이미지에서 객체, bounding box(경계 상자)를 탐지
* `객체 탐지 알고리즘`
  * 입력 : 이미지
  * 출력 :  [bounding box, 객체 클래스]
  * bounding box에 대해 예측 클래스와의 confidence(신뢰도)를 최종 구함

<br>

## Bounding box

* 객체 전체를 포함하는 가장 작은 직사각형

<img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118215256983.png" alt="image-20230118215256983" style="zoom:80%;" />

<br>

## IOU(Intersection Over Union)

* Ground Truth(실제값) 과 모델이 예측한 값이 얼마나 겹치는지 나타냄
  * `IOU = 겹치는 부분/ 전체 부분`
* IOU 가 높을수록 예측이 잘된 것
  * 아래 사진 기준으로 보자면, 초록색은 실제 값이고 빨간색은 예측 값

<img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118215523848.png" alt="image-20230118215523848" style="zoom: 67%;" />

<img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118215556065.png" alt="image-20230118215556065"  />

<br>

## NMS(Non-Maximum Suppression, 비최댓값 억제)

* 국지적인 최대값을 찾아 그 값만 남기고 나머지 값은 모두 삭제하는 알고리즘
* 과정
  * 확률 기준으로 모든상자 내림차순 정렬
  * 상자들 각각 모든 상자와 IOU 계산
  * 임계값 넘는 상자는 제거

<img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118220103790.png" alt="image-20230118220103790" style="zoom:80%;" />

<br>

# 객체 탐지(Object Detection)의 역사

- RCNN (2013)
  - Rich feature hierarchies for accurate object detection and semantic segmentation (https://arxiv.org/abs/1311.2524)
  - 물체 검출에 사용된 기존 방식인 sliding window는 background를 검출하는 소요되는 시간이 많았는데, 이를 개선시킨 기법으로 Region Proposal 방식 제안
  - 매우 높은 Detection이 가능하지만, 복잡한 아키텍처 및 학습 프로세스로 인해 Detection 시간이 매우 오래 걸림
- SPP Net (2014)
  - Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition (https://arxiv.org/abs/1406.4729)
  - RCNN의 문제를 Selective search로 해결하려 했지만, bounding box의 크기가 제각각인 문제가 있어서 FC Input에 고정된 사이즈로 제공하기 위한 방법 제안
  - SPP은 RCNN에서 conv layer와 fc layer사이에 위치하여 서로 다른 feature map에 투영된 이미지를 고정된 값으로 풀링
  - SPP를 이용해 RCNN에 비해 실행시간을 매우 단축시킴
- Fast RCNN (2015)
  - Fast R-CNN (https://arxiv.org/abs/1504.08083)
  - SPP layer를 ROI pooling으로 바꿔서 7x7 layer 1개로 해결
  - SVM을 softmax로 대체하여 Classification 과 Regression Loss를 함께 반영한 Multi task Loss 사용
  - ROI Pooling을 이용해 SPP보다 간단하고, RCNN에 비해 수행시간을 많이 줄임
- Fater RCNN(2015)
  - Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (https://arxiv.org/abs/1506.01497)
  - RPN(Region proposal network) + Fast RCNN 방식
  - Selective Search를 대체하기 위한 Region Proposal Network구현
  - RPN도 학습시켜서 전체를 end-to-end로 학습 가능 (GPU사용 가능)
  - Region Proposal를 위해 Object가 있는지 없는지의 후보 Box인 Anchor Box 개념 사용
  - Anchor Box를 도입해 FastRCNN에 비해 정확도를 높이고 속도를 향상시킴
- SSD (2015)
  - SSD: Single Shot MultiBox Detector (https://arxiv.org/abs/1512.02325)
  - Faster-RCNN은 region proposal과 anchor box를 이용한 검출의 2단계를 걸치는 과정에서 시간이 필요해 real-time(20~30 fps)으로는 어려움
  - SSD는 Feature map의 size를 조정하고, 동시에 앵커박스를 같이 적용함으로써 1 shot으로 물체 검출이 가능
  - real-time으로 사용할 정도의 성능을 갖춤 (30~40 fps)
  - 작은 이미지의 경우에 잘 인식하지 못하는 경우가 생겨서 data augmentation을 통해 mAP를 63에서 74로 비약적으로 높임
- RetinaNet (2017)
  - Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
  - RetinaNet이전에는 1-shot detection과 2-shot detection의 차이가 극명하게 나뉘어 속도를 선택하면 정확도를 trade-off 할 수 밖에 없는 상황
  - RetinaNet은 Focal Loss라는 개념의 도입과 FPN 덕분에 기존 모델들보다 정확도도 높고 속도도 여타 1-shot detector와 비견되는 모델
  - Detection에선 검출하고 싶은 물체와 (foreground object) 검출할 필요가 없는 배경 물체들이 있는데 (background object) 배경 물체의 숫자가 매우 많을 경우 배경 Loss를 적게 하더라도 숫자에 압도되어 배경의 Loss의 총합을 학습해버림 (예를 들어, 숲을 배경으로 하는 사람을 검출해야하는데 배경의 나무가 100개나 되다보니 사람의 특징이 아닌 나무가 있는 배경을 학습해버림)
  - Focal Loss는 이런 문제를 기존의 crossentropy 함수에서 (1-sig)을 제곱하여 background object의 loss를 현저히 줄여버리는 방법으로 loss를 변동시켜 해결
  - Focal Loss를 통해 검출하고자 하는 물체와 관련이 없는 background object들은 학습에 영향을 주지 않게 되고, 학습의 다양성이 더 넓어짐 (작은 물체, 큰 물체에 구애받지 않고 검출할 수 있게됨)
  - 실제로 RetinaNet은 object proposal을 2000개나 실시하여 이를 확인
- Mask R-CNN (2018)
  - Mask R-CNN (https://arxiv.org/pdf/1703.06870.pdf)
- YOLO (2018)
  - YOLOv3: An Incremental Improvement (https://arxiv.org/abs/1804.02767)
  - YOLO는 v1, v2, v3의 순서로 발전하였는데, v1은 정확도가 너무 낮은 문제가 있었고 이 문제는 v2까지 이어짐
  - 엔지니어링적으로 보완한 v3는 v2보다 살짝 속도는 떨어지더라도 정확도를 대폭 높인 모델
  - RetinaNet과 마찬가지로 FPN을 도입해 정확도를 높임
  - RetinaNet에 비하면 정확도는 4mAP정도 떨어지지만, 속도는 더 빠르다는 장점
- RefineDet (2018)
  - Single-Shot Refinement Neural Network for Object Detection (https://arxiv.org/pdf/1711.06897.pdf)
- M2Det (2019)
  - M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network (https://arxiv.org/pdf/1811.04533.pdf)
- EfficientDet (2019)
  - EfficientDet: Scalable and Efficient Object Detection (https://arxiv.org/pdf/1911.09070v1.pdf)
- YOLOv4 (2020)
  - YOLOv4: Optimal Speed and Accuracy of Object Detection (https://arxiv.org/pdf/2004.10934v1.pdf)
  - YOLOv3에 비해 AP, FPS가 각각 10%, 12% 증가
  - YOLOv3와 다른 개발자인 AlexeyBochkousky가 발표
  - v3에서 다양한 딥러닝 기법(WRC, CSP ...) 등을 사용해 성능을 향상시킴
  - CSPNet 기반의 backbone(CSPDarkNet53)을 설계하여 사용
- YOLOv5 (2020)
  - YOLOv4에 비해 낮은 용량과 빠른 속도 (성능은 비슷)
  - YOLOv4와 같은 CSPNet 기반의 backbone을 설계하여 사용
  - YOLOv3를 PyTorch로 implementation한 GlennJocher가 발표
  - Darknet이 아닌 PyTorch 구현이기 때문에, 이전 버전들과 다르다고 할 수 있음
- 이후
  - 수 많은 YOLO 버전들이 탄생
  - Object Detection 분야의 논문들이 계속해서 나오고 있음

<br>

# YOLO (You Only Look Once)

* 객체 검출 알고리즘 중에서 빠른 축에 속함
* 파이썬, 텐서플로가 아닌 C++ 구현된 코드 기준 GPU 사용 시 초당 170 프레임(170 FPS)

<br>

## YOLO 아키텍처

* backbone model 기반
* Feature Extractor (특징 추출기) 라고도 함
  * 어떤 `특징 추출기 아키텍처`를 사용하냐에 따라 성능이 달라진다.
* YOLO는 자체 맞춤 아키텍쳐 사용
* 아래 그림을 보면, YOLOv3 부터는 Scale이 3개 추가가 되었다
  * small, medium, big 객체에 맞게 사용할 수 있다
  * 또한, 왼쪽의 Inputs -> Conv(행렬곱) 연산 등등 과정을 거쳐 나간다.

<img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118221617679.png" alt="image-20230118221617679"  />

<br>

## YOLO 계층 출력

* 마지막 계층 출력은 (w x h x M) 행렬

  * w, h는 당연히 width, height을 의미

  * M = B x (C+5)

    * B : 그리드 셀당 Bounding box(경계 상자) 개수
      * 즉, 이미지가 가지는 그리드 셀 한개당 Bounding box(경계 상자)의 개수를 의미
    * C : 클래스 개수
      * +5 는 클래스 개수만큼 뿐만아니라 5개를 더 예측하려는 의미

  * Objectness Score: 바운딩 박스에 객체가 포함되어 있을 확률

    <img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118222616505.png" alt="image-20230118222616505" style="zoom:80%;" />

<br>

## 앵커 박스(Anchor Box)

* YOLOv2부터 도입
* prior box 라고도 함
* 객체에 가장 근접한 Anchor Box를 맞추고 신경망을 사용해 Anchor Box의 크기를 조정하는 과정때문에 위 그림의 Box Co-ordinates 부분이 필요

<img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118223141644.png" alt="image-20230118223141644"  />



<img src="..\images\2022-12-02-(yolo)Study_Week6\image-20230118223211256.png" alt="image-20230118223211256"  />

<br>

## YOLOv3 모델

생략

<br>

## YOLOv5 모델 - Detect

- https://github.com/ultralytics/yolov5
  - 참고로 학습되어있는 모델 바로 Detect 하는 것
- [https://www.ultralytics.com](https://www.ultralytics.com/)



### 모델 다운로드

```python
# yolov5 클론해서 사용
%cd /content
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt

import torch
from IPython.display import Image, clear_output
```



### 추론(Inference)

```python
# 이미지 확인
!ls data/images
```

bus.jpg  zidane.jpg



```python
# detect.py 접근 및 yolov5s.pt 사용(이 파일이 학습된 모델임)
# data/images/ 경로의 이미지들을 detect 하는것
!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
```

... Results saved to **runs/detect/exp**



```python
# 이미지 확인 (탐지된 모습으로 이미지 나옴)
Image(filename='runs/detect/exp/bus.jpg', width=600)
```

<br>

## YOLOv5 모델 - Detect + 학습까지(포트홀 학습)

* **이번엔 처음부터 모델을 학습시켜서 detect 하는 과정을 진행**



### 데이터셋 다운로드

- 포트홀 데이터셋: https://public.roboflow.com/object-detection/pothole
  - 라벨링 과정도 되어있고, 학습할 수 있게 데이터셋이 구성되어있음
- 다운로드 코드를 가져온다.
- 예 : !curl -L "https://public.roboflow.com/ds/pmIZBbuQ6H?key=mUDHxmGP99" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

```python
# 앞에선 학습된 모델을 사용한거고(detect)
# 이번엔 직접 모델을 처음부터 학습시켜서 사용해보려고 한다.

# 데이터 다운
%mkdir /content/yolov5/pothole
%cd /content/yolov5/pothole
!curl -L "https://public.roboflow.com/ds/pmIZBbuQ6H?key=mUDHxmGP99" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```



```python
# 이미지 리스트로 변수에 저장하기(glob 라이브러리 활용)
from glob import glob

train_img_list = glob('/content/yolov5/pothole/train/images/*.jpg')
test_img_list = glob('/content/yolov5/pothole/test/images/*.jpg')
valid_img_list = glob('/content/yolov5/pothole/valid/images/*.jpg')
print(len(train_img_list), len(test_img_list), len(valid_img_list))
```



```python
# yaml 파일로 만들어보기 - 우선 txt로 저장(이미지 경로)
import yaml

with open('/content/yolov5/pothole/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list)+'\n')

with open('/content/yolov5/pothole/test.txt', 'w') as f:
  f.write('\n'.join(test_img_list)+'\n')

with open('/content/yolov5/pothole/val.txt', 'w') as f:
  f.write('\n'.join(valid_img_list)+'\n')
```



```python
# 아래에서 사용할 함수 만들기
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
  with open(line, 'w') as f:
    f.write(cell.format(**globals()))
```



```python
# 기본적으로 데이터셋에 있는 data.yaml 파일 열어보기
%cat /content/yolov5/pothole/data.yaml
```

train: ../train/images val: ../valid/images nc: 1 names: ['pothole']



```python
# yaml 파일을 수정하기
# 네임 클래스는 1개만 있는 상황
# 위에서 만든 writetemplate 함수 활용

%%writetemplate /content/yolov5/pothole/data.yaml

train: ./pothole/train/images
test: ./pothole/test/images
val: ./pothole/valid/images

nc: 1
names: ['pothole']
```



```python
%cat /content/yolov5/pothole/data.yaml
```

train: ./pothole/train/images test: ./pothole/test/images val: ./pothole/valid/images nc: 1 names: ['pothole']



### 모델 구성

```python
import yaml

# 위에 저장한 data.yaml 파일의 "nc"(넘버클래스) 정보를 가져와 저장
with open("/content/yolov5/pothole/data.yaml", "r") as stream:
  num_classes = str(yaml.safe_load(stream)['nc'])

# 실제 yolov5 모델의 스몰 모델을 가져와 출력(확인해보는 것)
%cat /content/yolov5/models/yolov5s.yaml
```



```python
# 확인결과 "nc"가 너무 많다. 수정이 필요
# nc의 값을 위에서 구한 num_classes 로 변경

%%writetemplate /content/yolov5/models/custom_yolov5s.yaml

# Parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```



### 학습(Training)

- `img`: 입력 이미지 크기 정의
- `batch`: 배치 크기 결정
- `epochs`: 학습 기간 개수 정의
- `data`: yaml 파일 경로
- `cfg`: 모델 구성 지정
- `weights`: 가중치에 대한 경로 지정
  - weights은? 이어서 학습할 pt가 있다면 넣어서 더 학습을 시켜준다(last.pt같은)
    - 그런데, 구글링 해보니 내용이 바뀌거나 이미지 추가한다던지 등등 했을때는 기존 pt 모델을 이어서 추가로 학습이 안된다고 하는것 같다. 그래서 새로 학습을 시켜야한다고 했던것 같다.
    - 그래도 확실한 정보인지는 잘 모르겠다.
- `resume`: 중간에 끊긴 학습을 이어서 할 수 있다고 한다.
  - 이 속성을 이용하면 반복 학습을 하다가 문제가 발생해 끊긴 부분부터 다시 학습을 이어서 할 수 있다고 한다.
- `name`: 결과 이름
- `nosave`: 최종 체크포인트만 저장
- `cache`: 빠른 학습을 위한 이미지 캐시

```python
# 이제 학습을 해보려 함
# train.py 사용 - data.yaml, custom_yolov5s.yaml 이용

%%time
%cd /content/yolov5/
!python train.py --img 640 --batch 32 --epochs 100 --data ./pothole/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name pothole_results --cache
```



```python
# 텐서보드로 결과 확인
%load_ext tensorboard
%tensorboard --logdir runs
```



```python
# 결과는 runs에 들어가있음 (weights 폴더에 pt 있음)
!ls /content/yolov5/runs/train/pothole_results/
```

confusion_matrix.png				     results.png events.out.tfevents.1674553354.727f061f06c0.12154.0  train_batch0.jpg F1_curve.png					     train_batch1.jpg hyp.yaml					     train_batch2.jpg labels_correlogram.jpg				     val_batch0_labels.jpg labels.jpg					     val_batch0_pred.jpg opt.yaml					     val_batch1_labels.jpg P_curve.png					     val_batch1_pred.jpg PR_curve.png					     val_batch2_labels.jpg R_curve.png					     val_batch2_pred.jpg results.csv					     weights



```python
# 결과 그래프로 보여줌
Image(filename='/content/yolov5/runs/train/pothole_results/results.png', width=1000)
# 결과를 훈련사진으로 보여줌
Image(filename='/content/yolov5/runs/train/pothole_results/train_batch0.jpg', width=1000)
# 결과를 검증(validation)사진으로 보여줌
Image(filename='/content/yolov5/runs/train/pothole_results/val_batch0_labels.jpg', width=1000)
```



### 검증(Validation)

```python
# train 한것에 검증한 결과를 나타냄
!python val.py --weights runs/train/pothole_results/weights/best.pt --data ./pothole/data.yaml --img 640 --iou 0.65 --half
```

... Results saved to **runs/val/exp**



```python
# test 데이터들로 검증한 결과를 나타냄 (--task test)
!python val.py --weights runs/train/pothole_results/weights/best.pt --data ./pothole/data.yaml --img 640 --task test
```

... Results saved to **runs/val/exp2**



### 추론(Inference)

```python
%ls runs/train/pothole_results/weights
```

best.pt  last.pt



```python
# 테스트 이미지 전체를 detect 수행 해보는 것
!python detect.py --weights runs/train/pothole_results/weights/best.pt --img 640 --conf 0.4 --source ./pothole/test/images
```



```python
# 랜덤으로 detect한 이미지 출력
import glob
import random
from IPython.display import Image, display

image_name = random.choice(glob.glob('/content/yolov5/runs/detect/exp2/*.jpg'))
display(Image(filename=image_name))
```



### 모델 내보내기

```python
# 학습한 best.pt 파일을 내보내서 사용 가능.
# 구글 드라이브에 현재 했던것 옮겨두겠음.

from google.colab import drive
drive.mount('/content/drive')
```



```python
# 복제
%mkdir /content/drive/My\ Drive/pothole
%cp /content/yolov5/runs/train/pothole_results/weights/best.pt /content/drive/My\ Drive/pothole
```

<br>

## 라벨링 방법(LabelImg 툴)

https://velog.io/@kimsoohyun/YOLO-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%9D%BC%EB%B2%A8%EB%A7%81%EC%9D%84-%EC%9C%84%ED%95%9C-labelImg-%EC%82%AC%EC%9A%A9%EB%B2%95



**참고로 폴더 구조는 labels, images 폴더로 나눠서 txt, jpg 파일을 분리해줘야 한다.**

<br>

# 라즈베리파이4 + YOLOv5 (with docker)

라즈베리파이4 에서 YOLOv5 세팅을 한 후 직접 여기서 모델을 학습할 수도 있지만,   

구글링 결과 PC에서 미리 YOLOv5로 학습한 pt형식의 모델을 usb로 라즈베리파이에 바로 옮겨서  
해당 학습된 모델을 바로 사용할 수 있다는것을 알게 되었다.



아래에 참고 문서들을 확인!!

* 라즈베리파이와 딥러닝 사용하는 흐름 참고 : **[with 라즈베리파이](http://daddynkidsmakers.blogspot.com/2019/01/blog-post.html)**

* 실제로 할거면 이 문서 참고 : **[라즈베리파이4 + YOLOv5](https://prod.velog.io/@seven800/%EB%9D%BC%EC%A6%88%EB%B2%A0%EB%A6%AC%ED%8C%8C%EC%9D%B44-Yolov5-%ED%99%98%EA%B2%BD-%EC%84%B8%ED%8C%85with-docker)**

