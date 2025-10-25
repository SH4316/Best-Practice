# 4주차: 딥러닝 기초

## 강의 목표
- 신경망의 기본 구조와 원리 이해
- 역전파 알고리즘의 수학적 원리와 구현 방법 습득
- 활성화 함수와 손실 함수의 종류와 특성 학습
- PyTorch를 이용한 기본 신경망 구현 능력 배양
- 딥러닝 모델 훈련과 평가 과정 이해

## 이론 강의 (90분)

### 1. 신경망의 기본 구조 (25분)

#### 퍼셉트론에서 딥러닝까지
**단층 퍼셉트론(Single-Layer Perceptron)**
- 역사: 1957년 프랭크 로젠블라트 제안
- 구조: 입력 → 가중치 합 → 활성화 함수 → 출력
- 수학적 표현: $y = f(\sum_{i=1}^{n} w_i x_i + b)$
- 한계: 선형 분리 가능한 문제만 해결 가능 (XOR 문제)

**다층 퍼셉트론(Multi-Layer Perceptron, MLP)**
- 구조: 입력층 → 은닉층(들) → 출력층
- 표현력: 보편 근사 정리(Universal Approximation Theorem)
- LLM과의 연결: 트랜스포머는 복잡한 피드포워드 네트워크의 일종

#### 신경망의 구성 요소

**뉴런(Neuron) 또는 노드(Node)**
- 입력: 이전 층의 출력
- 가중치(Weights): 입력의 중요도 조절
- 편향(Bias): 활성화 임계값 조절
- 출력: 활성화 함수를 통한 최종 출력

**층(Layers)**
- 입력층(Input Layer): 원본 데이터 수신
- 은닉층(Hidden Layer): 특성 추출 및 변환
- 출력층(Output Layer): 최종 예측 결과 생성
- 깊이(Depth): 은닉층의 수, 딥러닝의 "딥" 의미

**연결성(Connectivity)**
- 완전 연결(Fully Connected): 모든 노드가 다음 층의 모든 노드와 연결
- 희소 연결(Sparse Connectivity): 일부 노드만 연결 (CNN 등)
- 순환 연결(Recurrent Connections): 시간적 순서 정보 처리 (RNN)

### 2. 활성화 함수 (20분)

#### 활성화 함수의 역할과 필요성
- 비선형성 도입: 선형 변환의 연속은 선형 변환
- 표현력 향상: 복잡한 함수 근사 가능
- 그래디언트 흐름 제어: 역전파 시 신호 전달

#### 주요 활성화 함수

**시그모이드(Sigmoid)**
- 수식: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- 출력 범위: (0, 1)
- 특징: 확률 해석 용이, 미분 가능
- 단점: 그래디언트 소실 문제, 출력이 0 중심이 아님
- LLM 적용: 초기 신경망, 현재는 제한적으로 사용

**하이퍼볼릭 탄젠트(Tanh)**
- 수식: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- 출력 범위: (-1, 1)
- 특징: 0 중심 출력, 시그모이드보다 그래디언트 소실 적음
- 단점: 여전히 그래디언트 소실 문제 존재

**ReLU (Rectified Linear Unit)**
- 수식: $\text{ReLU}(x) = \max(0, x)$
- 출력 범위: [0, ∞)
- 특징: 계산 효율성, 그래디언트 소실 문제 완화
- 단점: 죽은 뉴런 문제(Dead Neurons), 출력이 0 중심이 아님
- LLM 적용: 트랜스포머의 기본 활성화 함수

**Leaky ReLU**
- 수식: $\text{LeakyReLU}(x) = \max(\alpha x, x)$ (일반적으로 $\alpha = 0.01$)
- 특징: 죽은 뉴런 문제 완화
- LLM 적용: 일부 트랜스포머 변형에서 사용

**GELU (Gaussian Error Linear Unit)**
- 수식: $\text{GELU}(x) = x \cdot \Phi(x)$ (Φ는 표준 정규 분포의 CDF)
- 특징: 부드러운 비선형성, 성능 우수
- LLM 적용: BERT, GPT 등 최신 LLM에서 주로 사용

**Swish**
- 수식: $\text{Swish}(x) = x \cdot \sigma(\beta x)$ (β는 학습 가능한 파라미터)
- 특징: 자기 게이팅(Self-gating), 성능 우수
- LLM 적용: 일부 최신 모델에서 사용

### 3. 손실 함수 (20분)

#### 손실 함수의 역할
- 모델 성능 측정: 예측과 실제 값의 차이
- 최적화 목표: 손실 함수 최소화
- 그래디언트 계산: 역전파를 위한 기울기 제공

#### 회귀 문제용 손실 함수

**평균 제곱 오차(Mean Squared Error, MSE)**
- 수식: $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- 특징: 큰 오차에 민감, 미분 가능
- LLM 적용: 연속값 예측 헤드

**평균 절대 오차(Mean Absolute Error, MAE)**
- 수식: $L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- 특징: 이상값에 강건, 미분 불가능한 지점 존재

**Huber 손실**
- 수식: $L = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$
- 특징: MSE와 MAE의 장점 결합

#### 분류 문제용 손실 함수

**이진 교차 엔트로피(Binary Cross-Entropy)**
- 수식: $L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$
- 특징: 확률론적 해석, 그래디언트 안정성
- LLM 적용: 다음 단어 예측의 기본

**범주형 교차 엔트로피(Categorical Cross-Entropy)**
- 수식: $L = -\sum_{i=1}^{n}\sum_{j=1}^{c} y_{ij}\log\hat{y}_{ij}$
- 특징: 다중 클래스 분류용
- LLM 적용: 어휘 집합에 대한 다음 단어 예측

**Kullback-Leibler 발산(KL Divergence)**
- 수식: $D_{KL}(P||Q) = \sum_{i} P(i)\log\frac{P(i)}{Q(i)}$
- 특징: 확률 분포 간 차이 측정
- LLM 적용: 지식 증류, 정규화

### 4. 역전파 알고리즘 (25분)

#### 역전파의 수학적 원리
**연쇄 법칙(Chain Rule)**
- 기본 아이디어: 합성 함수의 도함수는 각 함수 도함수의 곱
- 다변수 연쇄 법칙: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$
- LLM에서의 중요성: 수백만 개 파라미터의 효율적 그래디언트 계산

**계산 그래프(Computational Graph)**
- 정의: 연산을 노드, 데이터를 엣지로 표현하는 그래프
- 순전파(Forward Pass): 입력에서 출력으로의 계산
- 역전파(Backward Pass): 출력에서 입력으로의 그래디언트 전파

#### 역전파 알고리즘의 단계
1. **순전파**: 입력 데이터를 네트워크에 통과시켜 출력 계산
2. **손실 계산**: 예측과 실제 값의 차이로 손실 함수 계산
3. **역전파**: 출력층에서 입력층으로 그래디언트 전파
4. **가중치 업데이트**: 계산된 그래디언트로 가중치 조정

#### 그래디언트 소실과 폭주 문제
**그래디언트 소실(Vanishing Gradients)**
- 원인: 연쇄 법칙에서 0보다 작은 값의 반복 곱셈
- 영향: 깊은 층의 가중치가 제대로 학습되지 않음
- 해결책: ReLU 활성화 함수, 잔차 연결, 배치 정규화

**그래디언트 폭주(Exploding Gradients)**
- 원인: 연쇄 법칙에서 1보다 큰 값의 반복 곱셈
- 영향: 가중치 업데이트가 불안정해짐
- 해결책: 그래디언트 클리핑, 가중치 초기화 기법

## 실습 세션 (90분)

### 1. PyTorch 기초와 신경망 구현 (30분)

#### PyTorch 텐서 기본 조작
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 텐서 생성
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("텐서 x:\n", x)
print("텐서 크기:", x.shape)
print("텐서 타입:", x.dtype)

# 텐서 연산
y = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
print("행렬 곱셈:\n", torch.matmul(x, y.T))
print("원소별 곱셈:\n", x * y)

# 자동 미분
x.requires_grad_(True)
z = torch.sum(x**2)
z.backward()
print("x에 대한 z의 그래디언트:\n", x.grad)
```

#### 간단한 신경망 구현
```python
# 간단한 신경망 클래스 정의
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 모델 생성
input_size = 10
hidden_size = 20
output_size = 1
model = SimpleNet(input_size, hidden_size, output_size)
print("모델 구조:")
print(model)

# 모델 파라미터 확인
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 2. 신경망 훈련 과정 구현 (30분)

#### 데이터 생성과 전처리
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_redundant=0, n_classes=2, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 텐서로 변환
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

print(f"훈련 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")
```

#### 훈련 루프 구현
```python
# 모델, 손실 함수, 옵티마이저 정의
model = SimpleNet(input_size=10, hidden_size=20, output_size=1)
criterion = nn.BCEWithLogitsLoss()  # 이진 분류용 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 기록 저장
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 훈련 루프
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    # 미니배치 훈련
    model.train()
    epoch_train_loss = 0
    correct_train = 0
    total_train = 0
    
    # 간단한 배치 처리 (실제로는 DataLoader 사용)
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # 순전파
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        
        # 정확도 계산
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()
    
    # 평가
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        predicted_test = (torch.sigmoid(test_outputs) > 0.5).float()
        test_accuracy = (predicted_test == y_test).float().mean()
    
    # 기록 저장
    train_loss = epoch_train_loss / (len(X_train) / batch_size)
    train_accuracy = correct_train / total_train
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss.item())
    test_accuracies.append(test_accuracy.item())
    
    # 10 에포크마다 출력
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
```

#### 훈련 과정 시각화
```python
# 손실과 정확도 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 3. 다양한 활성화 함수와 손실 함수 실험 (30분)

#### 활성화 함수 비교
```python
# 다양한 활성화 함수 정의
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    return torch.tanh(x)

def relu(x):
    return torch.max(torch.zeros_like(x), x)

def leaky_relu(x, alpha=0.01):
    return torch.max(alpha * x, x)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))

# 활성화 함수 시각화
x = torch.linspace(-5, 5, 100)
activations = {
    'Sigmoid': sigmoid(x),
    'Tanh': tanh(x),
    'ReLU': relu(x),
    'Leaky ReLU': leaky_relu(x),
    'GELU': gelu(x)
}

plt.figure(figsize=(12, 8))
for i, (name, y) in enumerate(activations.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(x.numpy(), y.numpy())
    plt.title(name)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 활성화 함수에 따른 그래디언트 비교
```python
# 활성화 함수의 도함수
def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_grad(x):
    return 1 - torch.pow(tanh(x), 2)

def relu_grad(x):
    return (x > 0).float()

def leaky_relu_grad(x, alpha=0.01):
    grad = torch.ones_like(x)
    grad[x < 0] = alpha
    return grad

def gelu_grad(x):
    # 근사적인 도함수
    return 0.5 * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3)))) + \
           0.5 * x * (1 - torch.pow(torch.tanh(torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))), 2)) * \
           torch.sqrt(2 / torch.pi) * (1 + 0.134145 * torch.pow(x, 2))

# 그래디언트 시각화
gradients = {
    'Sigmoid Grad': sigmoid_grad(x),
    'Tanh Grad': tanh_grad(x),
    'ReLU Grad': relu_grad(x),
    'Leaky ReLU Grad': leaky_relu_grad(x),
    'GELU Grad': gelu_grad(x)
}

plt.figure(figsize=(12, 8))
for i, (name, y) in enumerate(gradients.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(x.numpy(), y.numpy())
    plt.title(name)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 손실 함수 비교
```python
# 다양한 손실 함수 정의
def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)

# 손실 함수 시각화
y_true = torch.zeros(100)
y_pred = torch.linspace(-3, 3, 100)

losses = {
    'MSE': mse_loss(y_true, y_pred),
    'MAE': mae_loss(y_true, y_pred),
    'Huber': huber_loss(y_true, y_pred)
}

plt.figure(figsize=(10, 6))
for name, loss in losses.items():
    plt.plot(y_pred.numpy(), loss.numpy(), label=name)

plt.xlabel('Prediction Error')
plt.ylabel('Loss')
plt.title('Comparison of Loss Functions')
plt.legend()
plt.grid(True)
plt.show()
```

## 과제

### 1. 신경망 구조 과제
- 다양한 은닉층 크기에 따른 성능 비교
- 깊이(층 수)에 따른 성능 변화 분석
- 과적합과 과소적합 사례 분석

### 2. 활성화 함수 과제
- 다양한 활성화 함수를 사용한 신경망 성능 비교
- 그래디언트 소실 문제 시뮬레이션
- Xavier/He 초기화와 활성화 함수의 관계 분석

### 3. 최적화 과제
- 다양한 옵티마이저(SGD, Adam, RMSprop) 성능 비교
- 학습률 스케줄링 효과 분석
- 배치 크기에 따른 훈련 속도와 성능 비교

## 추가 학습 자료

### 온라인 강의
- [DeepLearning.AI Specialization](https://www.coursera.org/specializations/deep-learning)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### 교재
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron
- "Python Deep Learning" by Valentino Zocca, Gianmario Spacagna, Daniel Slater

### 온라인 자료
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Distill.pub](https://distill.pub/) - 딥러닝 개념의 시각적 설명