# Intro

**MLP(Multi-Layer Perceptron)을 공부하겠습니다.**

* 학교 수업 중 MLP 실습을 최근에 경험했기 때문에 해당 코드를 분석하며 정리하는식으로 진행하겠습니다.

<br>

# MLP(Multi-Layer Perceptron)

Perceptron을 이용한 MNIST 손글씨 데이터셋 분류



```python
# 폴더 경로
datasetPath = "./drive/MyDrive/2022-2-semester-AI/dataset"
parameterPath = "./drive/MyDrive/2022-2-semester-AI/parameters"
```

<br>

## 패키지 선언



```python
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
```

<br>

## Dataset 선언



```python
# Training dataset 다운로드
mnist_train = dataset.MNIST(root = datasetPath, 
                            train = True,
                            transform = transform.ToTensor(),
                            download = True)
# Testing dataset 다운로드
mnist_test = dataset.MNIST(root = datasetPath, # 경로
                            train = False, # train의 유무로 구분(train/test 인지)
                            transform = transform.ToTensor(),
                            download = True)
```

<br>

## MNIST 데이터셋 형상 확인



```python
import matplotlib.pyplot as plt
print(len(mnist_train)) # training dataset 개수 확인

first_data = mnist_train[0] # [0] = img, [1] = 정답
print(first_data[0].shape) # 첫번째 data의 형상 확인
print(first_data[1]) # 첫번째 data의 정답 확인

plt.imshow(first_data[0][0,:,:], cmap='gray')
plt.show()
```

<pre>
60000
torch.Size([1, 28, 28])
5
</pre>

<br>

## 2D->1D 평탄화(전처리)



```python
first_img = first_data[0]
print(first_img.shape)

first_img = first_img.view(-1, 28*28) # 이미지 평탄화 수행 2D->1D
print(first_img.shape)
```

<pre>
torch.Size([1, 28, 28])
torch.Size([1, 784])
</pre>
<br>

## MLP 모델 정의

SLP에서 이부분만 수정



```python
class MLP(nn.Module): # 2-layer
  def __init__(self): # 신경망 계층 초기화
    super(MLP, self).__init__()
    # 입력은 784개(img크기), 출력은 10개(숫자 10까지 있으니까)
    # 중간 히든 노드는 100개로 지정
    self.fc1 = nn.Linear(in_features=784, out_features=100)
    self.fc2 = nn.Linear(in_features=100, out_features=10)
    self.sigmoid = nn.Sigmoid() # 활성화 함수

  def forward(self, x): # 연산
    x = x.view(-1, 28*28) # 이미지 평탄화
    y = self.sigmoid(self.fc1(x))
    y = self.fc2(y)
    return y
```

<br>

## Hyper-parameter 지정



```python
batch_size = 100
learning_rate = 0.1
training_epochs = 15
loss_function = nn.CrossEntropyLoss() # Loss 함수
network = MLP() # 신경망을 MLP로 사용
optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate) # GD방법 사용
# 파라미터 및 학습률 적용

# Batch 단위 학습을 위해 DataLoader 함수 사용
data_loader = DataLoader(dataset=mnist_train,
                         batch_size = batch_size,
                         shuffle=True,
                         drop_last=True)
```

<br>

## Perceptron 학습 반복



```python
for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = len(data_loader)

  for img, label in data_loader:

    pred = network(img) # 입력 이미지에 대해 forward pass

    loss = loss_function(pred, label) # 예측 값, 정답을 이용해 loss 계산
    optimizer.zero_grad() # gradient 초기화
    loss.backward() # 모든 weight에 대해 편미분 계산(backpropagation)
    optimizer.step() # 파라미터 업데이트

    avg_cost += loss / total_batch

  print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))

print('Learning finished')
```

<pre>
Epoch: 1 Loss = 1.162427
Epoch: 2 Loss = 0.449319
Epoch: 3 Loss = 0.359580
Epoch: 4 Loss = 0.322798
Epoch: 5 Loss = 0.299738
Epoch: 6 Loss = 0.282503
Epoch: 7 Loss = 0.268211
Epoch: 8 Loss = 0.255711
Epoch: 9 Loss = 0.244562
Epoch: 10 Loss = 0.234365
Epoch: 11 Loss = 0.224941
Epoch: 12 Loss = 0.216165
Epoch: 13 Loss = 0.208207
Epoch: 14 Loss = 0.200568
Epoch: 15 Loss = 0.193664
Learning finished
</pre>

```python
# Weight parameter 저장
torch.save(network.state_dict(), parameterPath+"slp_mnist.pth")
```


```python
# 저장된 weight parameter 불러오기(예시)
new_network = MLP()
new_network.load_state_dict(torch.load(parameterPath+"slp_mnist.pth"))
```

<pre>
keys matched successfully
</pre>


<br>

## 분류 성능 확인



```python
# MNIST test dataset 분류 성능 확인
with torch.no_grad(): # test에서는 기울기 계산 제외
  img_test = mnist_test.data.float()
  label_test = mnist_test.targets

  prediction = network(img_test) # 전체 test data를 한번에 계산

  correct_prediction = torch.argmax(prediction, 1) == label_test # 예측값이 가장 높은 숫자(0~9)와 정답데이터가 일치한 지 확인 
  accuracy = correct_prediction.float().mean()
  print('Accuracy:', accuracy.item()) # 정답률
```

<pre>
Accuracy: 0.9434000253677368
</pre>