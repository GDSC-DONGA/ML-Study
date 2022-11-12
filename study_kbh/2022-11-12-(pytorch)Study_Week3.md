---

---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>

# Intro

**파이토치 기본을 공부하겠습니다.**  

* 참고 URL
  * [파이토치 기본](https://tutorials.pytorch.kr/beginner/basics/intro.html)
* `Colab`을 사용하겠습니다.

<br>

# 텐서(TENSOR)

텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조입니다. PyTorch에서는 텐서를 사용하여 모델의 입력(input)과 출력(output), 그리고 모델의 매개변수들을 부호화(encode)합니다.



```python
import torch
import numpy as np
```

<br>

## 텐서(tensor) 초기화



```python
# 데이터로부터 직접 생성
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)
```

<pre>
tensor([[1, 2],
        [3, 4]])
</pre>

```python
# NumPy 배열로부터 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
```

<pre>
tensor([[1, 2],
        [3, 4]])
</pre>

```python
# 다른 텐서로부터 생성
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮음
print(f"Random Tensor: \n {x_rand} \n")
```

<pre>
Ones Tensor: 
 tensor([[1, 1],
        [1, 1]]) 

Random Tensor: 
 tensor([[0.5171, 0.1379],
        [0.1828, 0.7331]]) 

</pre>

```python
# 차원 적용 가능
shape = (2,3,) # 2x3 차원
rand_tensor = torch.rand(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
```

<pre>
Random Tensor: 
 tensor([[0.6872, 0.4439, 0.2150],
        [0.8149, 0.2264, 0.2461]]) 

</pre>

<br>

## 텐서의 연산(Operation)

[URL](https://pytorch.org/docs/stable/torch.html) 해당 사이트에서 연산들 확인가능  



```python
# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```


```python
# 텐서 합치기
tensor = torch.ones(4, 4) # 4x4차원
tensor[:,1] = 0 # 모든 row, 1 column = 0
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

<pre>
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
</pre>

```python
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T # 행렬곱을 위해 차원 맞추려고 T로 전치행렬 수행
y2 = tensor.matmul(tensor.T) # 위와 같은 결과
y3 = torch.rand_like(y1) # 임의로 아무값으로 행렬 생성
torch.matmul(tensor, tensor.T, out=y3) # 위와 같은 결과
print(y1) # y1, y2, y3 전부 같은 값

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor # 차원 꼭 맞아야 함
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1) # z1, z2, z3 전부 같은 값
```

<pre>
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
</pre>
<br>

## NumPy 변환(Bridge)

CPU 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경됩니다.



```python
t = torch.ones(5)
n = t.numpy()

# 같은 메모리 공유하기 때문에 텐서의 변경이 NumPy에 반영
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

<pre>
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
</pre>

```python
n = np.ones(5)
t = torch.from_numpy(n)

# NumPy의 변경도 텐서에 반영
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

<pre>
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
</pre>
<br>

# DATASET과 DATALOADER

데이터셋 코드를 모델 학습 코드로부터 분리하는 것이 이상적입니다.   

PyTorch는 `torch.utils.data.DataLoader` 와 `torch.utils.data.Dataset` 의 두 가지 데이터 기본 요소를 제공합니다.  

Dataset 은 샘플과 정답(label)을 저장하고, DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌉니다.

<br>

## 데이터셋 불러오기

TorchVision 에서 [Fashion-MNIST](https://github.com/zalandoresearch/) 데이터셋을 불러오는 예제  

* `root` : 저장 경로

* `train` : train/test 여부 지정

* `download=True` : root 에 데이터가 없는 경우 인터넷에서 다운로드.

* `transform` 과 `target_transform` : 특징(feature)과 정답(label) 변형(transform)을 지정

<br>

## 파일에서 사용자 정의 데이터셋 만들기

사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 합니다

* `__init__` : 객체 생성될 때 한 번만 실행

* `__len__` : 데이터셋의 샘플 개수 반환

* `__getitem__` : 인덱스에 해당하는 데이터 반환

<br>

## DataLoader로 학습용 데이터 준비하기

`DataLoader`는 복잡한 `Dataset`의 과정을 추상화한 순회 가능한 객체



```python
# EX)
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

<br>

## DataLoader를 통해 순회하기(iterate)

`DataLoader` 에 데이터셋을 불러온 뒤에는 필요에 따라 데이터셋을 순회(iterate)할 수 있습니다.



```python
train_features, train_labels = next(iter(train_dataloader)) # 순회
```

<br>

# 변형(TRANSFORM)

변형(transform) 을 해서 데이터를 조작하고 학습에 적합하게 만들어야 합니다.

* [`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html) 모듈이 몇가지 변형을 제공

* ToTensor

  * PIL Image나 NumPy `ndarray` 를 `FloatTensor` 로 변환하고, 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)

* Lambda 변형

  * 정수를 원-핫으로 부호화된 텐서로 바꾸는 함수를 정의



```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

<br>

# 신경망 모델 구성하기

신경망은 데이터에 대한 연산을 수행하는 계층(layer)/모듈(module)로 구성

* [`torch.nn`](https://pytorch.org/docs/stable/nn.html) 는 신경망을 구성하는데 필요한 모든 구성 요소를 제공

* **FashionMNIST 데이터셋의 이미지들을 분류하는 신경망을 만드는 과정을 나타내겠습니다**



```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

<br>

## 학습을 위한 장치 얻기

가능한 경우 GPU와 같은 하드웨어 가속기에서 모델을 학습하려고 합니다. `torch.cuda` 를 사용할 수 있는지 확인하고 그렇지 않으면 CPU를 계속 사용합니다.

* Colab을 사용한다면, Edit > Notebook Settings 에서 GPU를 할당할 수 있습니다.



```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

<pre>
Using cpu device
</pre>
<br>

## 클래스 정의하기

* 신경망 모델을 nn.Module 의 하위클래스로 정의  

* `__init__` 에서 신경망 계층들을 초기화  

* `nn.Module` 을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산들을 구현



```python
class NeuralNetwork(nn.Module): # nn.Module 상속
    def __init__(self): # 계층
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # 2D -> 1D
        self.linear_relu_stack = nn.Sequential( # 순서를 가짐
            nn.Linear(28*28, 512), # 선형 변환
            nn.ReLU(), # 활성화 함수
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): # 연산
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) # 객체 생성 및 device로 이동
print(model)

X = torch.rand(1, 28, 28, device=device) # 28x28 크기 이미지 3개
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) # 예측 확률을 나타내도록 [0, 1] 범위로 비례하여 조정(scale)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}") # 예측 확률
```

<pre>
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
Predicted class: tensor([7])
</pre>
<br>

# `TORCH.AUTOGRAD`를 사용한 자동 미분

신경망을 학습할 때 가장 자주 사용되는 알고리즘은 `역전파`

* `torch.autograd` 라는 자동 미분 엔진이 내장



```python
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) # weight
b = torch.randn(3, requires_grad=True) # bias
z = torch.matmul(x, w)+b # perceptron
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # loss 함수
```


```python
# 변화도(Gradient) 계산 - 미분값
loss.backward()
print(w.grad)
print(b.grad)
```

<pre>
tensor([[0.1557, 0.2715, 0.0344],
        [0.1557, 0.2715, 0.0344],
        [0.1557, 0.2715, 0.0344],
        [0.1557, 0.2715, 0.0344],
        [0.1557, 0.2715, 0.0344]])
tensor([0.1557, 0.2715, 0.0344])
</pre>

```python
# 변화도 추적 멈추기(순전파 연산만 필요한 경우 추적 필요없으니까)
print(z.requires_grad) # true면 추적 지원
with torch.no_grad(): # 추적 멈추기
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

<pre>
True
False
</pre>
<br>

# 모델 매개변수 최적화하기

이제 모델과 데이터가 준비되었으니, 데이터에 매개변수를 최적화하여 모델을 학습하고, 검증하고, 테스트할 차례


**중요한 흐름**  

모델을 학습하는 과정은 반복적인 과정을 거칩니다(에폭(epoch)이라고 부르는)  

각 반복 단계에서 모델은 출력을 추측하고,  

추측과 정답 사이의 오류(손실(loss))를 계산하고,   

매개변수에 대한 오류의 도함수(derivative)를 수집한 뒤, 경사하강법을 사용하여 이 파라미터들을 최적화(optimize)

<br>

## 하이퍼파라미터(Hyperparameter)

모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수입니다.  

* 에폭(epoch) - 데이터셋 전체를 반복하는 횟수

* 배치 크기(batch size) - 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수

* 학습률(learning rate) - 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.

<br>

## 최적화 단계(Optimization Loop)

최적화 단계의 각 반복(iteration)을 **에폭**  

* 학습 단계(train loop) - 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴합니다.

* 검증/테스트 단계(validation/test loop) - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)합니다.

<br>

## 손실 함수(loss function)

획득한 결과와 실제 값 사이의 틀린 정도(degree of dissimilarity)를 측정하며, 학습 중에 이 값을 최소화하려고 합니다.   

주어진 데이터 샘플을 입력으로 계산한 예측과 정답(label)을 비교하여 손실(loss)을 계산합니다.

* [`nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)(평균 제곱 오차(MSE; Mean Square Error)) : 회귀 문제에 사용 

* [`nn.NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)(음의 로그 우도(Negative Log Likelihood)) : 분류에 사용 

* [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) : `nn.LogSoftmax`와 `nn.NLLLoss`를 합침

<br>

## 옵티마이저(Optimizer)

최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정입니다.  

* 대표적 예 : 경사하강법



```python
# 매개변수, 학습률로 초기화(SGD란 확률적 경사하강법)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

<br>

# 모델 저장하고 불러오기

저장하기나 불러오기를 통해 모델의 상태를 유지(persist)하고 모델의 예측을 실행하는 방법을 알아보겠습니다.



```python
import torch
import torchvision.models as models
```

<br>

## 모델 가중치 저장하고 불러오기

* `state_dict` : 학습한 매개변수를 내부 상태 사전에 저장

* `torch.save` : 이 상태 값들을 저장

* `load_state_dict` : 매개변수들 불러오기



```python
# 저장
model = models.vgg16(pretrained=True) # pretrained=True : 기본 가중치 불러오기
torch.save(model.state_dict(), 'model_weights.pth')

# 불러오기
model = models.vgg16() # 기본 가중치 필요 X(불러올거라서)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

<br>

## 모델의 형태를 포함하여 저장하고 불러오기

모델의 가중치를 불러올 때, 신경망의 구조를 정의하기 위해 모델 클래스를 먼저 생성(instantiate)해야 했습니다.(실제로 위에서 객체 생성후 저장, 불러오기 했음)  



이것을 안하는 방법으로 클래스의 구조를 모델과 함께 저장하는 방법이 있습니다.

* 코드 확인



```python
# 저장
torch.save(model, 'model.pth')

# 불러오기
model = torch.load('model.pth')
```
