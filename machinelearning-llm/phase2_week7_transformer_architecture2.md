# 7주차: 트랜스포머 아키텍처 (2)

## 강의 목표
- 위치 인코딩의 원리와 다양한 구현 방법 이해
- 인코더-디코더 구조의 동작 원리와 정보 흐름 파악
- 포지션 와이즈 피드포워드 네트워크의 역할과 구조 습득
- 트랜스포머의 전체 아키텍처와 계산 복잡도 분석
- PyTorch를 이용한 완전한 트랜스포머 모델 구현 능력 배양

## 이론 강의 (90분)

### 1. 위치 인코딩 (25분)

#### 위치 정보의 필요성
**셀프 어텐션의 한계**
- 순서 무시: 순수 셀프 어텐션은 입력 순서 정보 무시
- 순열 불변성: 입력의 순서를 바꿔도 동일한 출력
- 문제: "고양이가 쥐를 잡았다"와 "쥐가 고양이를 잡았다" 구분 불가

**위치 정보의 중요성**
- 문법적 관계: 주어-동사-목적어 순서
- 의미적 관계: 단어 순서에 따른 의미 변화
- 시퀀스 모델링: 시간적/공간적 순서 정보

#### 위치 인코딩의 종류

**절대적 위치 인코딩(Absolute Positional Encoding)**
- 아이디어: 각 위치에 고유한 벡터 할당
- 원본 논문 방식: 사인/코사인 함수 사용
- 수식: $PE_{(pos,2i)} = \sin(\frac{pos}{10000^{2i/d_{model}}})$
- 수식: $PE_{(pos,2i+1)} = \cos(\frac{pos}{10000^{2i/d_{model}}})$

**상대적 위치 인코딩(Relative Positional Encoding)**
- 아이디어: 토큰 간의 상대적 거리 정보 사용
- 장점: 시퀀스 길이에 대한 일반화 능력
- 종류: T5, DeBERTa 등에서 사용

**학습 가능한 위치 인코딩**
- 아이디어: 위치 인코딩을 학습 가능한 파라미터로 구현
- 장점: 데이터에 적응적인 위치 표현 학습
- 단점: 긴 시퀀스에 대한 일반화 한계

#### 사인/코사인 위치 인코딩의 수학적 원리
**주파수 분할**
- 저주파수: 먼 위치 간의 관계 모델링
- 고주파수: 가까운 위치 간의 미세한 차이 모델링
- 기하급수적 감소: $10000^{2i/d_{model}}$

**선형 관계 보존**
- 특징: $PE_{pos+k}$는 $PE_{pos}$의 선형 함수로 표현 가능
- 의미: 상대적 위치 관계가 모델이 학습하기 쉬움
- 수학적 증명: 삼각함수의 덧셈 정리 활용

**고유성과 일관성**
- 고유성: 각 위치에 고유한 인코딩 벡터
- 일관성: 비슷한 위치는 비슷한 인코딩
- 확장성: 훈련된 최대 길이보다 긴 시퀀스도 처리 가능

#### 위치 인코딩의 구현 방법
**원본 논문 구현**
```python
def positional_encoding(max_len, d_model):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)
```

**학습 가능한 위치 인코딩**
```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        return self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
```

### 2. 인코더-디코더 구조 (30분)

#### 인코더 구조
**입력 처리**
- 단어 임베딩: $X \in \mathbb{R}^{n \times d_{model}}$
- 위치 인코딩 추가: $X + PE$
- 드롭아웃 적용: 정규화 효과

**셀프 어텐션 레이어**
- 멀티-헤드 셀프 어텐션: 입력 시퀀스 내 관계 모델링
- 잔차 연결: $x + \text{MultiHead}(x, x, x)$
- 레이어 정규화: $\text{LayerNorm}(x + \text{MultiHead}(x, x, x))$

**피드포워드 네트워크**
- 위치별 독립 처리: 각 토큰에 동일한 변환 적용
- 잔차 연결: $x + \text{FFN}(x)$
- 레이어 정규화: $\text{LayerNorm}(x + \text{FFN}(x))$

**스택 구조**
- N개의 동일한 레이어 쌓기: $6$레이어가 원본 논문에서 사용
- 점진적 특성 추출: 하위 레이어는 문법적, 상위 레이어는 의미적 특성
- 출력: 인코딩된 문맥 표현 $Z \in \mathbb{R}^{n \times d_{model}}$

#### 디코더 구조
**입력 처리**
- 출력 임베딩: 이전에 생성된 토큰들
- 위치 인코딩 추가: 순서 정보 보존
- 드롭아웃 적용

**마스크드 셀프 어텐션**
- 목적: 현재 위치 이후의 토큰 참조 방지
- 마스크: 미래 위치에 대한 어텐션 가중치를 $-\infty$로 설정
- 자기 회귀: 한 번에 하나의 토큰씩 생성

**인코더-디코더 어텐션**
- 목적: 인코더 출력과 디코더 입력 간 관계 모델링
- 쿼리: 디코더의 현재 상태
- 키/값: 인코더의 출력 표현
- 정보 흐름: 소스 시퀀스에서 타겟 시퀀스로 정보 전달

**피드포워드 네트워크**
- 인코더와 동일한 구조
- 특성 변환: 디코딩된 표현의 비선형 변환

#### 인코더-디코더 정보 흐름
**순전파 과정**
1. 인코더: 소스 시퀀스 → 문맥 표현
2. 디코더: 이전 출력 + 문맥 표현 → 다음 토큰 예측
3. 반복: 시퀀스 끝까지 토큰 생성

**어텐션 메커니즘의 역할**
- 셀프 어텐션: 각 시퀀스 내부 관계 모델링
- 인코더-디코더 어텐션: 두 시퀀스 간 정렬 학습
- 계층적 처리: 다양한 수준의 문맥 정보 추출

**잔차 연결과 정규화**
- 그래디언트 흐름: 깊은 네트워크에서의 안정적 학습
- 정보 보존: 원본 정보와 변환된 정보의 결합
- 학습 안정성: 수렴 속도 향상

### 3. 포지션 와이즈 피드포워드 네트워크 (20분)

#### 구조와 동작 원리
**기본 구조**
- 두 개의 선형 변환: $d_{model} \rightarrow d_{ff} \rightarrow d_{model}$
- 활성화 함수: ReLU 또는 GELU
- 위치별 적용: 각 토큰에 독립적으로 동일한 변환 적용

**수학적 표현**
- 첫 번째 선형: $\text{FFN}_1(x) = xW_1 + b_1$
- 활성화 함수: $\text{FFN}_2(x) = \text{ReLU}(\text{FFN}_1(x))$
- 두 번째 선형: $\text{FFN}_3(x) = \text{FFN}_2(x)W_2 + b_2$
- 최종 출력: $\text{FFN}(x) = \text{FFN}_3(x)$

**차원 확장의 역할**
- 표현력 향상: 고차원 공간에서의 비선형 변환
- 특성 분리: 다양한 특성을 독립적으로 처리
- 계산 효율성: $d_{ff} = 4 \times d_{model}$이 일반적

#### 피드포워드 네트워크의 역할
**특성 변환**
- 어텐션 후 처리: 어텐션 가중합 결과의 비선형 변환
- 특성 추출: 고차원 특성 공간에서의 패턴 학습
- 표현 풍부화: 단순한 가중합을 넘어선 복잡한 관계 모델링

**모델 용량 증가**
- 파라미터 증가: 대부분의 파라미터가 피드포워드 네트워크에 집중
- 표현력 향상: 더 복잡한 함수 근사 가능
- 학습 안정성: 어텐션과 피드포워드의 상호 보완

**계산 복잡도**
- 시간 복잡도: $O(n \times d_{model} \times d_{ff})$
- 공간 복잡도: $O(n \times d_{ff})$ (중간 계산)
- 병렬화: 모든 위치에서 동시에 계산 가능

#### 활성화 함수의 선택
**ReLU (Rectified Linear Unit)**
- 장점: 계산 효율성, 그래디언트 소실 문제 완화
- 단점: 죽은 뉴런 문제, 0 중심 출력 아님
- 사용: 원본 트랜스포머에서 사용

**GELU (Gaussian Error Linear Unit)**
- 수식: $\text{GELU}(x) = x \cdot \Phi(x)$
- 장점: 부드러운 비선형성, 성능 우수
- 단점: 계산 복잡도 높음
- 사용: BERT, GPT 등 최신 모델에서 사용

**Swish**
- 수식: $\text{Swish}(x) = x \cdot \sigma(\beta x)$
- 특징: 자기 게이팅, 성능 우수
- 적용: 일부 최신 모델에서 사용

### 4. 트랜스포머 전체 아키텍처 (15분)

#### 완전한 트랜스포머 모델
**입력/출력 처리**
- 소스 임베딩: 단어 인덱스 → 벡터
- 타겟 임베딩: 이전 출력 → 벡터
- 위치 인코딩: 순서 정보 추가
- 드롭아웃: 정규화

**인코더 스택**
- N개의 인코더 레이어: 원본에서 6개
- 각 레이어 구조: 셀프 어텐션 + 피드포워드
- 출력: 인코딩된 소스 표현

**디코더 스택**
- N개의 디코더 레이어: 원본에서 6개
- 각 레이어 구조: 마스크드 셀프 어텐션 + 인코더-디코더 어텐션 + 피드포워드
- 출력: 다음 토큰 예측

**최종 출력 레이어**
- 선형 변환: $d_{model} \rightarrow \text{vocab\_size}$
- 소프트맥스: 확률 분포로 변환
- 손실 계산: 교차 엔트로피 손실

#### 계산 복잡도 분석
**시간 복잡도**
- 셀프 어텐션: $O(n^2 \cdot d_{model})$
- 피드포워드: $O(n \cdot d_{model} \cdot d_{ff})$
- 전체 모델: $O(n^2 \cdot d_{model} \cdot L)$ (L: 레이어 수)

**공간 복잡도**
- 어텐션 행렬: $O(n^2)$
- 중간 활성화: $O(n \cdot d_{model} \cdot L)$
- 파라미터: $O(d_{model}^2 \cdot L)$

**RNN과의 비교**
- 병렬화: 트랜스포머 $O(1)$, RNN $O(n)$
- 장거리 의존성: 트랜스포머 $O(1)$, RNN $O(n)$
- 순차 처리: 트랜스포머 불필요, RNN 필수

#### 트랜스포머의 변형과 발전
**아키텍처 변형**
- 인코더 전용: BERT, RoBERTa
- 디코더 전용: GPT, OPT
- 인코더-디코더: T5, BART

**효율성 개선**
- 메모리 효율성: Reformer, Performer
- 계산 효율성: Longformer, BigBird
- 파라미터 효율성: ALBERT, DistilBERT

**규모 확장**
- 더 깊은 네트워크: Deep Transformer
- 더 넓은 네트워크: Wide Transformer
- 혼합 전문가: Switch Transformer

## 실습 세션 (90분)

### 1. 위치 인코딩 구현과 분석 (30분)

#### 다양한 위치 인코딩 구현
```python
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        return self.dropout(x + pos_emb)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 상대적 위치 임베딩
        self.relative_positions = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model)
        )
        
    def forward(self, seq_len):
        # 상대적 위치 인덱스 생성
        positions = torch.arange(seq_len, device=self.relative_positions.device)
        relative_pos = positions[:, None] - positions[None, :]
        relative_pos = relative_pos + self.max_len - 1
        
        return self.relative_positions[relative_pos]

# 위치 인코딩 비교
d_model = 512
max_len = 100
seq_len = 50

# 사인/코사인 위치 인코딩
sinusoidal_pe = PositionalEncoding(d_model, max_len)
sinusoidal_matrix = sinusoidal_pe.pe[:seq_len, :].detach().numpy()

# 학습 가능한 위치 인코딩
learned_pe = LearnedPositionalEncoding(d_model, max_len)
dummy_input = torch.randn(1, seq_len, d_model)
learned_output = learned_pe(dummy_input)
learned_matrix = learned_output[0, :, :].detach().numpy()

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 사인/코사인 위치 인코딩 (차원별)
axes[0,0].imshow(sinusoidal_matrix.T, cmap='viridis', aspect='auto')
axes[0,0].set_title('Sinusoidal Positional Encoding')
axes[0,0].set_xlabel('Position')
axes[0,0].set_ylabel('Dimension')

# 학습 가능한 위치 인코딩 (차원별)
axes[0,1].imshow(learned_matrix.T, cmap='viridis', aspect='auto')
axes[0,1].set_title('Learned Positional Encoding')
axes[0,1].set_xlabel('Position')
axes[0,1].set_ylabel('Dimension')

# 특정 차원의 위치별 값
sample_dims = [0, 64, 128, 256]
for dim in sample_dims:
    if dim < d_model:
        axes[1,0].plot(sinusoidal_matrix[:, dim], label=f'Dim {dim}')
axes[1,0].set_title('Sinusoidal PE - Sample Dimensions')
axes[1,0].set_xlabel('Position')
axes[1,0].set_ylabel('Value')
axes[1,0].legend()

# 학습 가능한 위치 인코딩의 특정 차원
for dim in sample_dims:
    if dim < d_model:
        axes[1,1].plot(learned_matrix[:, dim], label=f'Dim {dim}')
axes[1,1].set_title('Learned PE - Sample Dimensions')
axes[1,1].set_xlabel('Position')
axes[1,1].set_ylabel('Value')
axes[1,1].legend()

plt.tight_layout()
plt.show()
```

#### 위치 인코딩의 특성 분석
```python
def analyze_positional_encoding_properties():
    """위치 인코딩의 수학적 특성 분석"""
    
    d_model = 512
    max_len = 1000
    
    # 위치 인코딩 생성
    pe = PositionalEncoding(d_model, max_len)
    pe_matrix = pe.pe[:max_len, :].detach().numpy()
    
    # 1. 상대적 위치 관계 분석
    def get_relative_encoding(pos1, pos2):
        return pe_matrix[pos1] - pe_matrix[pos2]
    
    # 다양한 거리에 대한 상대적 인코딩
    distances = [1, 2, 5, 10, 20, 50]
    base_pos = 100
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 상대적 위치 인코딩 시각화
    for i, dist in enumerate(distances):
        if i < 4:
            rel_encoding = get_relative_encoding(base_pos, base_pos + dist)
            axes[i//2, i%2].plot(rel_encoding[:100])  # 처음 100차원만 표시
            axes[i//2, i%2].set_title(f'Relative Encoding: Distance {dist}')
            axes[i//2, i%2].set_xlabel('Dimension')
            axes[i//2, i%2].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
    
    # 2. 주파수 분석
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 저주파수 차원 (짝수)
    low_freq_dims = pe_matrix[:, :10]  # 처음 10개 차원
    axes[0].imshow(low_freq_dims.T, cmap='viridis', aspect='auto')
    axes[0].set_title('Low Frequency Dimensions (Even)')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    
    # 고주파수 차원 (홀수)
    high_freq_dims = pe_matrix[:, 10:20]  # 다음 10개 차원
    axes[1].imshow(high_freq_dims.T, cmap='viridis', aspect='auto')
    axes[1].set_title('High Frequency Dimensions (Odd)')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Dimension')
    
    plt.tight_layout()
    plt.show()
    
    # 3. 내적 분석
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 위치 간 코사인 유사도 행렬
    sample_positions = list(range(0, 200, 10))  # 0, 10, 20, ..., 190
    similarity_matrix = np.zeros((len(sample_positions), len(sample_positions)))
    
    for i, pos1 in enumerate(sample_positions):
        for j, pos2 in enumerate(sample_positions):
            similarity_matrix[i, j] = cosine_similarity(pe_matrix[pos1], pe_matrix[pos2])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Positional Encoding Cosine Similarity Matrix')
    plt.xlabel('Position Index')
    plt.ylabel('Position Index')
    plt.xticks(range(len(sample_positions)), sample_positions)
    plt.yticks(range(len(sample_positions)), sample_positions)
    plt.show()

# 위치 인코딩 특성 분석
analyze_positional_encoding_properties()
```

### 2. 인코더-디코더 구조 구현 (30분)

#### 완전한 트랜스포머 인코더
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # 멀티-헤드 어텐션
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 피드포워드 네트워크
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 레이어 정규화와 드롭아웃
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 활성화 함수
        self.activation = nn.GELU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 셀프 어텐션
        src2, attn_weights = self.self_attn(
            src, src, src, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 피드포워드
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            encoder_layer for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        all_attn_weights = []
        
        for mod in self.layers:
            output, attn_weights = mod(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask
            )
            all_attn_weights.append(attn_weights)
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output, all_attn_weights
```

#### 완전한 트랜스포머 디코더
```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # 마스크드 멀티-헤드 어텐션
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 인코더-디코더 어텐션
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 피드포워드 네트워크
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 레이어 정규화와 드롭아웃
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # 활성화 함수
        self.activation = nn.GELU()
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        # 마스크드 셀프 어텐션
        tgt2, self_attn_weights = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 인코더-디코더 어텐션
        tgt2, cross_attn_weights = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 피드포워드
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, self_attn_weights, cross_attn_weights

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            decoder_layer for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        output = tgt
        all_self_attn_weights = []
        all_cross_attn_weights = []
        
        for mod in self.layers:
            output, self_attn_weights, cross_attn_weights = mod(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output, all_self_attn_weights, all_cross_attn_weights
```

#### 마스킹 함수 구현
```python
def generate_square_subsequent_mask(sz):
    """미래 토큰을 마스킹하는 삼각형 마스크 생성"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_padding_mask(seq, pad_idx):
    """패딩 토큰을 마스킹"""
    return (seq == pad_idx)

# 마스크 시각화
def visualize_masks():
    seq_len = 10
    
    # 미래 마스크
    future_mask = generate_square_subsequent_mask(seq_len)
    
    # 패딩 마스크 예시
    seq = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])  # 마지막 7개는 패딩
    padding_mask = create_padding_mask(seq, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 미래 마스크
    sns.heatmap(future_mask.numpy(), cmap='viridis', ax=axes[0])
    axes[0].set_title('Future Mask (Causal Mask)')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # 패딩 마스크
    sns.heatmap(padding_mask.numpy(), cmap='viridis', ax=axes[1])
    axes[1].set_title('Padding Mask')
    axes[1].set_xlabel('Sequence Position')
    axes[1].set_ylabel('Batch')
    
    plt.tight_layout()
    plt.show()

# 마스크 시각화
visualize_masks()
```

### 3. 완전한 트랜스포머 모델 구현 (30분)

#### 전체 트랜스포머 모델
```python
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 임베딩 레이어
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 인코더와 디코더
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 출력 레이어
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # 파라미터 초기화
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                memory_mask=None, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        # 소스와 타겟 임베딩
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # 위치 인코딩 추가
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)
        
        # 인코더와 디코더 통과
        memory, encoder_attn_weights = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        
        output, decoder_self_attn_weights, decoder_cross_attn_weights = self.decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 최종 출력
        output = self.generator(output)
        
        return output, {
            'encoder_attention': encoder_attn_weights,
            'decoder_self_attention': decoder_self_attn_weights,
            'decoder_cross_attention': decoder_cross_attn_weights
        }

# 간단한 번역 데이터로 테스트
def test_transformer():
    # 모델 파라미터
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    
    # 모델 생성
    model = TransformerModel(
        src_vocab_size, tgt_vocab_size, d_model, nhead,
        num_encoder_layers, num_decoder_layers
    )
    
    # 가짜 데이터
    batch_size = 2
    src_seq_len = 20
    tgt_seq_len = 15
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 마스크 생성
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    
    # 순전파
    output, attention_weights = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"소스 크기: {src.shape}")
    print(f"타겟 크기: {tgt.shape}")
    print(f"출력 크기: {output.shape}")
    print(f"인코더 어텐션 레이어 수: {len(attention_weights['encoder_attention'])}")
    print(f"디코더 셀프 어텐션 레이어 수: {len(attention_weights['decoder_self_attention'])}")
    print(f"디코더 교차 어텐션 레이어 수: {len(attention_weights['decoder_cross_attention'])}")
    
    return model, output, attention_weights

# 트랜스포머 모델 테스트
model, output, attention_weights = test_transformer()
```

#### 어텐션 가중치 시각화
```python
def visualize_transformer_attention(attention_weights, src_tokens, tgt_tokens):
    """트랜스포머의 다양한 어텐션 가중치 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 인코더 어텐션 (첫 번째 레이어, 첫 번째 헤드)
    encoder_attn = attention_weights['encoder_attention'][0][0, 0].detach().numpy()
    sns.heatmap(encoder_attn, xticklabels=src_tokens, yticklabels=src_tokens, 
               cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Encoder Self-Attention (Layer 1, Head 1)')
    
    # 디코더 셀프 어텐션 (첫 번째 레이어, 첫 번째 헤드)
    decoder_self_attn = attention_weights['decoder_self_attention'][0][0, 0].detach().numpy()
    sns.heatmap(decoder_self_attn, xticklabels=tgt_tokens, yticklabels=tgt_tokens,
               cmap='viridis', ax=axes[0,1])
    axes[0,1].set_title('Decoder Self-Attention (Layer 1, Head 1)')
    
    # 디코더 교차 어텐션 (첫 번째 레이어, 첫 번째 헤드)
    decoder_cross_attn = attention_weights['decoder_cross_attention'][0][0, 0].detach().numpy()
    sns.heatmap(decoder_cross_attn, xticklabels=src_tokens, yticklabels=tgt_tokens,
               cmap='viridis', ax=axes[0,2])
    axes[0,2].set_title('Decoder Cross-Attention (Layer 1, Head 1)')
    
    # 인코더 어텐션 (마지막 레이어, 첫 번째 헤드)
    encoder_attn_last = attention_weights['encoder_attention'][-1][0, 0].detach().numpy()
    sns.heatmap(encoder_attn_last, xticklabels=src_tokens, yticklabels=src_tokens,
               cmap='viridis', ax=axes[1,0])
    axes[1,0].set_title('Encoder Self-Attention (Last Layer, Head 1)')
    
    # 디코더 셀프 어텐션 (마지막 레이어, 첫 번째 헤드)
    decoder_self_attn_last = attention_weights['decoder_self_attention'][-1][0, 0].detach().numpy()
    sns.heatmap(decoder_self_attn_last, xticklabels=tgt_tokens, yticklabels=tgt_tokens,
               cmap='viridis', ax=axes[1,1])
    axes[1,1].set_title('Decoder Self-Attention (Last Layer, Head 1)')
    
    # 디코더 교차 어텐션 (마지막 레이어, 첫 번째 헤드)
    decoder_cross_attn_last = attention_weights['decoder_cross_attention'][-1][0, 0].detach().numpy()
    sns.heatmap(decoder_cross_attn_last, xticklabels=src_tokens, yticklabels=tgt_tokens,
               cmap='viridis', ax=axes[1,2])
    axes[1,2].set_title('Decoder Cross-Attention (Last Layer, Head 1)')
    
    plt.tight_layout()
    plt.show()

# 예시 토큰으로 어텐션 시각화
src_tokens = [f'src_{i}' for i in range(20)]
tgt_tokens = [f'tgt_{i}' for i in range(15)]
visualize_transformer_attention(attention_weights, src_tokens, tgt_tokens)
```

#### 모델 파라미터 분석
```python
def analyze_model_parameters(model):
    """모델 파라미터 분석"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능 파라미터 수: {trainable_params:,}")
    
    # 레이어별 파라미터 분석
    layer_params = {}
    for name, param in model.named_parameters():
        layer_name = '.'.join(name.split('.')[:2])
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += param.numel()
    
    # 파라미터 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 레이어별 파라미터 수
    layers = list(layer_params.keys())
    params = list(layer_params.values())
    
    axes[0].barh(layers, params)
    axes[0].set_xlabel('Number of Parameters')
    axes[0].set_title('Parameters by Layer')
    axes[0].set_xscale('log')
    
    # 파라미터 분포 파이 차트
    other_params = sum(params[5:])  # 작은 레이어들을 '기타'로 그룹화
    pie_data = params[:5] + [other_params]
    pie_labels = layers[:5] + ['Other']
    
    axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
    axes[1].set_title('Parameter Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return layer_params

# 모델 파라미터 분석
layer_params = analyze_model_parameters(model)
```

## 과제

### 1. 위치 인코딩 과제
- 다양한 위치 인코딩 방법의 성능 비교 실험
- 긴 시퀀스에 대한 일반화 능력 분석
- 위치 인코딩의 초기화 방법에 따른 성능 차이 연구

### 2. 인코더-디코더 구조 과제
- 인코더와 디코더 레이어 수에 따른 성능 분석
- 다양한 마스킹 기법의 효과 비교
- 인코더-디코더 어텐션의 정보 흐름 분석

### 3. 트랜스포머 아키텍처 과제
- 레이어 수와 모델 너비의 트레이드오프 분석
- 다양한 활성화 함수(ReLU, GELU, Swish) 비교
- 잔차 연결과 레이어 정규화의 효과 정량적 분석

## 추가 학습 자료

### 논문
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Universal Transformer" (Dehghani et al., 2019)
- "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (Dai et al., 2019)

### 온라인 자료
- [Harvard's Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Transformer Explained](https://peterbloem.nl/blog/transformers)

### 구현 참고
- [PyTorch Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [fairseq](https://github.com/pytorch/fairseq)