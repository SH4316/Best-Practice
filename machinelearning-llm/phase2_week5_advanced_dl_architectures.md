# 5주차: 심화 딥러닝 아키텍처

## 강의 목표
- 합성곱 신경망(CNN)의 원리와 구조 이해
- 순환 신경망(RNN)과 LSTM의 시퀀스 처리 메커니즘 습득
- 어텐션 메커니즘의 기본 원리와 구현 방법 학습
- 각 아키텍처의 LLM에서의 응용 사례 파악
- PyTorch를 이용한 고급 딥러닝 모델 구현 능력 배양

## 이론 강의 (90분)

### 1. 합성곱 신경망 (CNN) (30분)

#### CNN의 기본 원리
**합성곱 연산(Convolution Operation)**
- 정의: 필터(커널)를 입력 데이터에 슬라이딩하며 적용하는 연산
- 수학적 표현: $(f * g)[t] = \sum_{a=-\infty}^{\infty} f[a]g[t-a]$
- 2D 합성곱: $(I * K)[i,j] = \sum_{m}\sum_{n} I[m,n]K[i-m,j-n]$
- 특징: 가중치 공유, 국소적 연결, 변위 불변성

**핵심 구성 요소**
- **합성곱 층(Convolutional Layer)**: 특징 맵 생성
- **활성화 함수(Activation Function)**: 비선형성 도입
- **풀링 층(Pooling Layer)**: 공간 크기 감소, 주요 특징 추출
- **완전 연결 층(Fully Connected Layer)**: 최종 분류

#### CNN 아키텍처 발전
**LeNet-5 (1998)**
- 구조: 합성곱-풀링-합성곱-풀링-완전 연결
- 특징: 현대 CNN의 기초 마련
- 응용: 손글씨 숫자 인식

**AlexNet (2012)**
- 혁신: 더 깊은 네트워크, ReLU 활성화 함수, 드롭아웃
- 성과: ImageNet 대회에서 압도적 성능
- 영향: 딥러닝 부흥의 촉매

**VGGNet (2014)**
- 특징: 작은 필터(3x3)만 사용, 규칙적 구조
- 장점: 단순함과 깊이의 효과 입증
- 단점: 많은 파라미터, 높은 계산 비용

**ResNet (2015)**
- 혁신: 잔차 학습(Residual Learning)
- 구조: $F(x) + x$ 형태의 숏컷 연결
- 효과: 매우 깊은 네트워크(152층 이상) 학습 가능

#### CNN의 LLM 응용
**멀티모달 LLM**
- 텍스트-이미지 이해: CLIP, DALL-E
- 비전-언어 모델: ViT(Vision Transformer)와의 결합
- 응용: 이미지 캡셔닝, 시각적 질문 답변

**1D CNN for Text**
- 텍스트 처리: 단어 시퀀스를 1D 신호로 처리
- 특징 추출: n-그램 특징 자동 학습
- 장점: 병렬 처리, 계산 효율성

### 2. 순환 신경망 (RNN)과 LSTM (30분)

#### RNN의 기본 원리
**순환 구조**
- 아이디어: 시퀀스 정보를 숨겨진 상태에 저장
- 수학적 표현: $h_t = f(W_h h_{t-1} + W_x x_t + b)$
- 출력: $y_t = g(W_y h_t + c)$
- 특징: 가변 길이 시퀀스 처리, 시간적 의존성 모델링

**RNN의 한계**
- 그래디언트 소실/폭주: 장기 의존성 학습 어려움
- 계산 순차성: 병렬 처리 어려움
- 정보 병목: 고정 크기 은닉 상태

#### LSTM (Long Short-Term Memory)
**게이트 메커니즘**
- 망각 게이트(Forget Gate): $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- 입력 게이트(Input Gate): $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- 후보 값: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- 셀 상태 업데이트: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
- 출력 게이트(Output Gate): $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- 은닉 상태: $h_t = o_t * \tanh(C_t)$

**LSTM의 장점**
- 장기 의존성 학습: 셀 상태를 통한 정보 흐름 제어
- 그래디언트 흐름 안정화: 델타-루프 덕분
- 선택적 정보 저장/삭제: 게이트 메커니즘

#### GRU (Gated Recurrent Unit)
- 단순화된 LSTM: 망각 게이트와 입력 게이트 결합
- 수식: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ (업데이트 게이트)
- 수식: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$ (리셋 게이트)
- 수식: $\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])$
- 수식: $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$
- 장점: 더 적은 파라미터, 더 빠른 학습

#### RNN의 LLM 응용
**초기 언어 모델**
- Char-RNN: 문자 수준 언어 모델링
- Word-RNN: 단어 수준 언어 모델링
- 한계: 장기 문맥 이해 어려움

**현대적 응용**
- 시퀀스-투-시퀀스: 번역, 요약
- 어텐션과 결합: 성능 향상
- 트랜스포머로의 발전: RNN의 한계 극복

### 3. 어텐션 메커니즘 (30분)

#### 어텐션의 기본 아이디어
**동적 가중치 할당**
- 원리: 입력의 중요한 부분에 더 많은 주의 집중
- 유비: 인간의 시각적 주의 메커니즘
- 수학적 표현: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

**핵심 구성 요소**
- **쿼리(Query)**: 현재 상태에서의 질문
- **키(Key)**: 입력의 검색 가능한 표현
- **값(Value)**: 입력의 실제 내용
- **어텐션 가중치**: 쿼리와 키의 유사도

#### 어텐션의 종류

**내용 기반 어텐션(Content-Based Attention)**
- 원리: 쿼리와 키의 내용 유사도 계산
- 수식: $\alpha_{ij} = \frac{\exp(s_i^T k_j)}{\sum_{k=1}^{n} \exp(s_i^T k_k)}$
- 응용: 번역, 이미지 캡셔닝

**위치 기반 어텐션(Location-Based Attention)**
- 원리: 위치 정보를 직접 활용
- 특징: 순차적 처리 강조
- 응용: 음성 인식, 시계열 예측

**셀프 어텐션(Self-Attention)**
- 원리: 동일한 시퀀스 내 요소 간의 관계 모델링
- 수식: $\text{SelfAttention}(X) = \text{softmax}\left(\frac{XW_Q (XW_K)^T}{\sqrt{d_k}}\right) XW_V$
- 중요성: 트랜스포머의 핵심 메커니즘

#### 어텐션의 발전과 응용
**Bahdanau 어텐션 (2014)**
- 혁신: RNN 기반 번역 모델에 어텐션 도입
- 효과: 고정 길이 벡터의 한계 극복
- 구조: 양방향 RNN 인코더 + 어텐션 디코더

**Luong 어텐션 (2015)**
- 개선: 다양한 점수 함수 제안
- 종류: 점곱, 일반, 연결 방식
- 효과: 계산 효율성과 성능 향상

**멀티-헤드 어텐션 (Multi-Head Attention)**
- 아이디어: 여러 어텐션 메커니즘 병렬 실행
- 수식: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$
- 효과: 다양한 표현 학습, 성능 향상

#### 어텐션의 LLM 응용
**트랜스포머 아키텍처**
- 핵심: 셀프 어텐션과 멀티-헤드 어텐션
- 장점: 병렬 처리, 장거리 의존성 모델링
- 영향: 현대 LLM의 기반 기술

**효율적 어텐션 변형**
- Sparse Attention: 계산 복잡도 감소
- Linear Attention: 선형 시간 복잡도
- Reformer: 역해싱을 통한 효율화

## 실습 세션 (90분)

### 1. CNN 구현과 실험 (30분)

#### 기본 CNN 구현
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# MNIST 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 모델, 손실 함수, 옵티마이저
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 함수
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return train_loss, accuracy

# 테스트 함수
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

# 훈련 실행
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(1, 11):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
```

#### CNN 필터 시각화
```python
# 합성곱 필터 시각화
def visualize_filters(model):
    # 첫 번째 합성곱 층의 필터 가져오기
    filters = model.conv1.weight.data.clone()
    
    # 필터 정규화
    filters = filters - filters.min()
    filters = filters / filters.max()
    
    # 필터 시각화
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < filters.size(0):
            ax.imshow(filters[i][0].numpy(), cmap='gray')
        ax.axis('off')
    plt.suptitle('Convolutional Filters')
    plt.show()

# 특성 맵 시각화
def visualize_feature_maps(model, data):
    model.eval()
    with torch.no_grad():
        # 첫 번째 합성곱 층 통과
        x = model.conv1(data)
        x = torch.relu(x)
        
    # 특성 맵 시각화
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < x.size(1):
            ax.imshow(x[0][i].numpy(), cmap='gray')
        ax.axis('off')
    plt.suptitle('Feature Maps')
    plt.show()

# 시각화 실행
sample_data, _ = next(iter(test_loader))
visualize_filters(model)
visualize_feature_maps(model, sample_data[:1])
```

### 2. RNN과 LSTM 구현 (30분)

#### 기본 RNN 구현
```python
# RNN 모델 정의
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 초기 은닉 상태
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # RNN 순전파
        out, _ = self.rnn(x, h0)
        
        # 마지막 시간 스텝의 출력
        out = self.fc(out[:, -1, :])
        return out

# LSTM 모델 정의
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 초기 은닉 상태와 셀 상태
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시간 스텝의 출력
        out = self.fc(out[:, -1, :])
        return out

# 합성 데이터 생성
def generate_sequence_data(n_samples=1000, seq_length=20):
    X = torch.randn(n_samples, seq_length, 1)
    y = torch.sum(X, dim=1)  # 시퀀스의 합을 타겟으로
    return X, y

# 데이터 생성
X_train, y_train = generate_sequence_data(800, 20)
X_test, y_test = generate_sequence_data(200, 20)

# 모델 훈련 함수
def train_sequence_model(model, X_train, y_train, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    return losses

# RNN과 LSTM 비교
input_size, hidden_size, output_size = 1, 32, 1
rnn_model = SimpleRNN(input_size, hidden_size, output_size)
lstm_model = SimpleLSTM(input_size, hidden_size, output_size)

print("Training RNN:")
rnn_losses = train_sequence_model(rnn_model, X_train, y_train)

print("\nTraining LSTM:")
lstm_losses = train_sequence_model(lstm_model, X_train, y_train)

# 손실 곡선 비교
plt.figure(figsize=(10, 6))
plt.plot(rnn_losses, label='RNN')
plt.plot(lstm_losses, label='LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNN vs LSTM Training Loss')
plt.legend()
plt.grid(True)
plt.show()
```

#### LSTM 게이트 메커니즘 시각화
```python
# LSTM 게이트 활성화 시각화
def visualize_lstm_gates(model, X):
    model.eval()
    with torch.no_grad():
        # LSTM 층 접근
        lstm_layer = model.lstm
        h0 = torch.zeros(1, X.size(0), model.hidden_size)
        c0 = torch.zeros(1, X.size(0), model.hidden_size)
        
        # 각 타임스텝에서의 게이트 값 저장
        forget_gates = []
        input_gates = []
        output_gates = []
        cell_states = []
        
        # 수동으로 LSTM 순전파
        for t in range(X.size(1)):
            x_t = X[:, t, :]
            
            # 게이트 계산 (단순화된 버전)
            gates = torch.cat([h0.squeeze(0), x_t], dim=1)
            
            # 실제 LSTM 게이트 계산은 더 복잡하지만, 개념적 시각화를 위해 단순화
            forget_gate = torch.sigmoid(gates[:, :model.hidden_size])
            input_gate = torch.sigmoid(gates[:, model.hidden_size:2*model.hidden_size])
            output_gate = torch.sigmoid(gates[:, 2*model.hidden_size:3*model.hidden_size])
            
            forget_gates.append(forget_gate.mean().item())
            input_gates.append(input_gate.mean().item())
            output_gates.append(output_gate.mean().item())
            
            # 다음 타임스텝을 위해 실제 LSTM 사용
            _, (h0, c0) = lstm_layer(x_t.unsqueeze(1), (h0, c0))
            cell_states.append(c0.mean().item())
    
    # 게이트 활성화 시각화
    time_steps = range(len(forget_gates))
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, forget_gates, 'r-', label='Forget Gate')
    plt.ylabel('Activation')
    plt.title('LSTM Gate Activations Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time_steps, input_gates, 'g-', label='Input Gate')
    plt.ylabel('Activation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, output_gates, 'b-', label='Output Gate')
    plt.ylabel('Activation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(time_steps, cell_states, 'k-', label='Cell State')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# LSTM 게이트 시각화
visualize_lstm_gates(lstm_model, X_test[:1])
```

### 3. 어텐션 메커니즘 구현 (30분)

#### 기본 어텐션 구현
```python
# 기본 어텐션 메커니즘 구현
class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # hidden을 [batch_size, seq_len, hidden_size]로 확장
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 에너지 계산
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # 어텐션 스코어 계산
        attention_scores = torch.sum(self.v * energy, dim=2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 컨텍스트 벡터 계산
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights

# 어텐션을 사용한 시퀀스-투-시퀀스 모델
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqWithAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # 인코더
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 어텐션
        self.attention = SimpleAttention(hidden_size)
        
        # 디코더
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, src):
        # 인코딩
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # 디코더 입력 (인코더의 마지막 은닉 상태)
        decoder_input = hidden[-1].unsqueeze(1)
        
        # 어텐션 적용
        context, attention_weights = self.attention(hidden[-1], encoder_outputs)
        
        # 디코딩
        output, (hidden, cell) = self.decoder(decoder_input, (hidden.unsqueeze(0), cell))
        
        # 컨텍스트와 디코더 출력 결합
        combined = torch.cat((output.squeeze(1), context), dim=1)
        output = self.fc(combined)
        
        return output, attention_weights

# 셀프 어텐션 구현
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 헤드 수로 분할
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # 선형 변환
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # 어텐션 스코어 계산
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # 어텐션 적용
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        
        return self.fc_out(out)

# 어텐션 가중치 시각화
def visualize_attention(attention_weights, source_tokens, target_tokens):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights.cpu().numpy(), cmap='viridis')
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title('Attention Weights')
    plt.colorbar()
    
    # 축 레이블 설정
    plt.xticks(range(len(source_tokens)), source_tokens, rotation=45)
    plt.yticks(range(len(target_tokens)), target_tokens)
    
    plt.show()

# 간단한 시퀀스 데이터로 어텐션 테스트
seq_length = 10
embed_size = 64
heads = 4

# 랜덤 입력 데이터
x = torch.randn(1, seq_length, embed_size)

# 셀프 어텐션 레이어
self_attention = SelfAttention(embed_size, heads)
output = self_attention(x, x, x, mask=None)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

#### 멀티-헤드 어텐션 시각화
```python
# 멀티-헤드 어텐션 분석
def analyze_multihead_attention(model, input_seq):
    model.eval()
    with torch.no_grad():
        # 각 헤드의 어텐션 가중치 추출 (단순화된 예시)
        batch_size, seq_len, embed_size = input_seq.shape
        
        # 쿼리, 키, 값 생성
        queries = input_seq.reshape(batch_size, seq_len, model.heads, model.head_dim)
        keys = input_seq.reshape(batch_size, seq_len, model.heads, model.head_dim)
        
        # 각 헤드별 어텐션 스코어 계산
        head_attentions = []
        for h in range(model.heads):
            q = queries[:, :, h, :]  # [batch_size, seq_len, head_dim]
            k = keys[:, :, h, :]     # [batch_size, seq_len, head_dim]
            
            # 어텐션 스코어
            scores = torch.matmul(q, k.transpose(-2, -1)) / (model.head_dim ** 0.5)
            attention_weights = torch.softmax(scores, dim=-1)
            head_attentions.append(attention_weights[0].cpu().numpy())  # 첫 번째 배치
        
        # 헤드별 어텐션 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for i, ax in enumerate(axes.flat):
            if i < len(head_attentions):
                im = ax.imshow(head_attentions[i], cmap='viridis')
                ax.set_title(f'Head {i+1}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im, ax=ax)
        
        plt.suptitle('Multi-Head Attention Weights')
        plt.tight_layout()
        plt.show()

# 멀티-헤드 어텐션 분석
analyze_multihead_attention(self_attention, x)
```

## 과제

### 1. CNN 과제
- CIFAR-10 데이터셋에 대한 ResNet 구현
- 다양한 합성곱 필터 크기와 스트라이드 실험
- 전이 학습(Transfer Learning) 적용 및 성능 비교

### 2. RNN/LSTM 과제
- 시계열 데이터 예측을 위한 LSTM 모델 구현
- 다양한 시퀀스 길이에 대한 모델 성능 분석
- GRU와 LSTM의 성능과 계산 효율성 비교

### 3. 어텐션 과제
- 시퀀스-투-시퀀스 모델에 어텐션 메커니즘 적용
- 셀프 어텐션과 일반 어텐션의 차이점 분석
- 어텐션 가중치 시각화와 해석

## 추가 학습 자료

### 온라인 강의
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [DeepLearning.AI Specialization](https://www.coursera.org/specializations/deep-learning)

### 논문
- "Gradient-Based Learning Applied to Document Recognition" (LeNet-5)
- "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Long Short-Term Memory" (LSTM)
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau Attention)

### 온라인 자료
- [Distill.pub - Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)