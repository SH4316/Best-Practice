# 6주차: 트랜스포머 아키텍처 (1)

## 강의 목표
- "Attention Is All You Need" 논문의 핵심 아이디어 이해
- 셀프 어텐션 메커니즘의 수학적 원리와 구현 방법 습득
- 멀티-헤드 어텐션의 동작 원리와 장점 파악
- 트랜스포머 아키텍처의 기본 구조와 혁신성 이해
- PyTorch를 이용한 셀프 어텐션과 멀티-헤드 어텐션 구현 능력 배양

## 이론 강의 (90분)

### 1. "Attention Is All You Need" 논문 심층 분석 (25분)

#### 논문의 배경과 동기
**RNN/CNN의 한계**
- 순차적 처리: 병렬화 어려움, 계산 효율성 저하
- 장거리 의존성: 정보 손실, 그래디언트 소실
- 고정된 문맥 길이: 긴 시퀀스 처리 어려움

**어텐션의 잠재력**
- 순차적 제약 없음: 모든 위치 간 직접 연결 가능
- 장거리 의존성 모델링: 거리에 무관한 관계 학습
- 병렬 처리 가능: 계산 효율성 향상

#### 트랜스포머의 핵심 혁신
**순환/합성곱 없는 아키텍처**
- 완전한 어텐션 기반: 순전파를 어텐션으로만 구성
- 위치 정보 처리: 위치 인코딩으로 순서 정보 보존
- 병렬화 가능: 모든 위치 동시 처리

**인코더-디코더 구조**
- 인코더: 입력 시퀀스를 문맥 표현으로 변환
- 디코더: 문맥 표현을 기반으로 출력 시퀀스 생성
- 인코더-디코더 어텐션: 두 모듈 간 정보 교환

#### 논문의 실험 결과
**번역 성능**
- WMT 2014 영어-독일어 번역: BLEU 28.4 (최고 기록)
- WMT 2014 영어-불어 번역: BLEU 41.0 (최고 기록)
- 훈련 효율성: 더 적은 계산 비용으로 더 높은 성능

**계산 효율성**
- 병렬화: GPU 활용 극대화
- 훈련 속도: RNN/LSTM보다 훨씬 빠른 수렴
- 확장성: 모델 크기와 데이터 크기에 따른 성능 향상

### 2. 셀프 어텐션 메커니즘 (30분)

#### 셀프 어텐션의 기본 원리
**정의와 목적**
- 정의: 동일한 시퀀스 내 요소 간의 관계 모델링
- 목적: 각 위치가 다른 모든 위치의 정보를 활용하여 표현 학습
- 장점: 장거리 의존성 효과적 모델링, 병렬 처리 가능

**수학적 표현**
- 쿼리(Query): 현재 위치의 질문 벡터 $Q = XW_Q$
- 키(Key): 모든 위치의 검색 가능한 표현 $K = XW_K$
- 값(Value): 모든 위치의 실제 내용 $V = XW_V$
- 어텐션 스코어: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

#### 스케일드 닷 프로덕트 어텐션
**점곱 어텐션의 문제**
- 큰 내적 값: 소프트맥스 함수의 그래디언트 소실
- 차원 의존성: 벡터 차원이 커질수록 내적 값 증가
- 수치적 불안정성: 극단적인 확률 분포

**스케일링의 필요성**
- 제안: $\sqrt{d_k}$로 나누어 내적 값 스케일링
- 효과: 그래디언트 안정화, 학습 안정성 향상
- 이유: 내적의 분산이 $d_k$에 비례하기 때문

**수학적 정당화**
- 가정: $q_i, k_j$가 평균 0, 분산 1인 독립 확률 변수
- 내적의 기댓값: $E[q_i \cdot k_j] = 0$
- 내적의 분산: $\text{Var}(q_i \cdot k_j) = d_k$
- 표준화: $\frac{q_i \cdot k_j}{\sqrt{d_k}}$로 분산 1로 조정

#### 셀프 어텐션의 계산 과정
**1단계: 쿼리, 키, 값 생성**
- 입력: $X \in \mathbb{R}^{n \times d_{model}}$
- 가중치 행렬: $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}$
- 변환: $Q = XW_Q$, $K = XW_K$, $V = XW_V$

**2단계: 어텐션 스코어 계산**
- 행렬 곱: $QK^T \in \mathbb{R}^{n \times n}$
- 스케일링: $\frac{QK^T}{\sqrt{d_k}}$
- 의미: 각 위치 쌍 간의 유사도

**3단계: 소프트맥스 정규화**
- 적용: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$
- 결과: 각 행의 합이 1인 확률 행렬
- 해석: 각 위치가 다른 위치에 할당할 주의 가중치

**4단계: 값의 가중합**
- 계산: $\text{softmax}(\cdot) \cdot V$
- 결과: $n \times d_k$ 크기의 출력 행렬
- 의미: 모든 위치 정보의 가중 평균

#### 셀프 어텐션의 직관적 이해
**단어 유사성 예시**
- 문장: "The cat sat on the mat"
- "cat"에 대한 어텐션: "sat", "mat"에 높은 가중치
- 문맥적 관계: 동사와 명사의 의존 관계 반영

**위치 정보 처리**
- 순서 정보: 셀프 어텐션 자체는 위치 정보 무시
- 해결책: 위치 인코딩으로 위치 정보 추가
- 효과: 순서에 따른 다른 어텐션 패턴 학습

### 3. 멀티-헤드 어텐션 (35분)

#### 멀티-헤드 어텐션의 동기
**단일 어텐션의 한계**
- 표현력 제한: 하나의 표현 공간만 학습
- 관계 다양성: 다양한 종류의 단어 관계 존재
- 특성 분리: 문법적, 의미적 관계를 동시에 모델링 어려움

**다중 표현의 아이디어**
- 인간의 인지: 다양한 관점에서 정보 처리
- 앙상블 효과: 여러 모델의 결합으로 성능 향상
- 특성 분리: 각 헤드가 특정 유형의 관계 전담

#### 멀티-헤드 어텐션의 구조
**헤드 분할**
- 입력: $d_{model}$ 차원의 벡터
- 분할: $h$개의 헤드, 각 헤드는 $d_k = d_{model}/h$ 차원
- 가중치: $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d_{model} \times d_k}$

**병렬 어텐션 계산**
- 각 헤드: $\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$
- 병렬 처리: 모든 헤드 동시 계산 가능
- 독립성: 각 헤드가 독립적인 어텐션 패턴 학습

**결합과 출력**
- 연결: $\text{Concat}(\text{head}_1, ..., \text{head}_h)$
- 투영: $W^O \in \mathbb{R}^{hd_k \times d_{model}}$
- 최종 출력: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

#### 멀티-헤드 어텐션의 수학적 표현
**행렬 형태의 효율적 구현**
- 입력 행렬: $X \in \mathbb{R}^{n \times d_{model}}$
- 가중치 행렬: $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}$
- 3D 텐서로 재구성: $Q, K, V \in \mathbb{R}^{n \times h \times d_k}$

**계산 최적화**
- 배치 행렬 곱: 한 번에 모든 헤드 계산
- 메모리 효율성: 중간 결과 재사용
- 병렬화: GPU 연산 극대화

#### 멀티-헤드 어텐션의 직관적 이해
**헤드별 특성화**
- 문법 헤드: 주어-동사, 수식-명사 관계
- 의미 헤드: 동의어, 반의어 관계
- 위치 헤드: 인접 단어, 원거리 단어 관계

**실제 예시**
- 문장: "The teacher who taught math won the award"
- 헤드 1: "teacher"와 "taught" 간의 주어-동사 관계
- 헤드 2: "math"와 "taught" 간의 목적어-동사 관계
- 헤드 3: "teacher"와 "award" 간의 원거리 의미 관계

#### 멀티-헤드 어텐션의 장점
**표현력 향상**
- 다양한 관계: 여러 종류의 단어 관계 동시 모델링
- 특성 분리: 각 헤드가 특정 관계 유형 전담
- 조합 효과: 여러 관계의 결합으로 풍부한 표현

**학습 안정성**
- 다중 경로: 그래디언트 흐름의 다양성
- 정규화 효과: 여러 헤드의 평균화로 과적합 방지
- 안정적 수렴: 단일 헤드보다 안정적인 학습

**해석 가능성**
- 헤드 분석: 각 헤드가 학습한 관계 유형 분석 가능
- 시각화: 어텐션 가중치 패턴 시각화
- 디버깅: 특정 헤드의 문제 진단 용이

## 실습 세션 (90분)

### 1. 셀프 어텐션 구현 (30분)

#### 기본 셀프 어텐션 구현
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        # 쿼리, 키, 값 선형 변환
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # 출력 투영
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask=None):
        # 입력 크기: [batch_size, seq_len, embed_size]
        batch_size = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 입력을 헤드 수로 분할
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = query.reshape(batch_size, query_len, self.heads, self.head_dim)
        
        # 선형 변환
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # 어텐션 스코어 계산: (batch_size, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # 마스킹 (옵션)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # 스케일링과 소프트맥스
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=3)
        
        # 어텐션 적용: (batch_size, heads, query_len, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        
        # 헤드 결합: (batch_size, query_len, heads * head_dim)
        out = out.reshape(batch_size, query_len, self.heads * self.head_dim)
        
        # 최종 선형 변환
        out = self.fc_out(out)
        
        return out, attention

# 셀프 어텐션 테스트
embed_size = 64
heads = 4
seq_len = 10
batch_size = 2

# 랜덤 입력 데이터
x = torch.randn(batch_size, seq_len, embed_size)

# 셀프 어텐션 레이어
self_attention = SelfAttention(embed_size, heads)
output, attention_weights = self_attention(x, x, x)

print(f"입력 크기: {x.shape}")
print(f"출력 크기: {output.shape}")
print(f"어텐션 가중치 크기: {attention_weights.shape}")
```

#### 어텐션 가중치 시각화
```python
def visualize_attention(attention_weights, tokens, heads_to_show=None):
    """
    어텐션 가중치 시각화 함수
    
    Args:
        attention_weights: (batch_size, heads, query_len, key_len)
        tokens: 시퀀스 토큰 리스트
        heads_to_show: 표시할 헤드 인덱스 리스트 (None이면 모든 헤드)
    """
    batch_size, num_heads, query_len, key_len = attention_weights.shape
    
    if heads_to_show is None:
        heads_to_show = list(range(min(num_heads, 4)))  # 최대 4개 헤드만 표시
    
    fig, axes = plt.subplots(len(heads_to_show), 1, figsize=(10, 3*len(heads_to_show)))
    
    if len(heads_to_show) == 1:
        axes = [axes]
    
    for i, head_idx in enumerate(heads_to_show):
        if head_idx < num_heads:
            # 첫 번째 배치의 특정 헤드 어텐션 가중치
            attn_data = attention_weights[0, head_idx].detach().numpy()
            
            sns.heatmap(attn_data, 
                       xticklabels=tokens, 
                       yticklabels=tokens,
                       cmap='viridis', 
                       ax=axes[i])
            axes[i].set_title(f'Head {head_idx + 1} Attention Weights')
    
    plt.tight_layout()
    plt.show()

# 예시 문장으로 어텐션 시각화
sentence = "the cat sat on the mat"
tokens = sentence.split()

# 간단한 임베딩 (실제로는 학습된 임베딩 사용)
word_embeddings = torch.randn(len(tokens), embed_size)
word_embeddings = word_embeddings.unsqueeze(0)  # 배치 차원 추가

# 셀프 어텐션 적용
output, attention_weights = self_attention(word_embeddings, word_embeddings, word_embeddings)

# 어텐션 가중치 시각화
visualize_attention(attention_weights, tokens)
```

#### 스케일링 효과 분석
```python
def analyze_scaling_effect(embed_size, seq_len=10):
    """스케일링 효과 분석"""
    
    # 랜덤 쿼리와 키 생성
    query = torch.randn(1, seq_len, embed_size)
    key = torch.randn(1, seq_len, embed_size)
    
    # 기본 어텐션 스코어
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 스케일링된 어텐션 스코어
    scaled_scores = scores / math.sqrt(embed_size)
    
    # 소프트맥스 적용
    attention = torch.softmax(scores, dim=-1)
    scaled_attention = torch.softmax(scaled_scores, dim=-1)
    
    # 통계 비교
    print(f"임베딩 크기: {embed_size}")
    print(f"어텐션 스코어 - 평균: {scores.mean().item():.4f}, 표준편차: {scores.std().item():.4f}")
    print(f"스케일링된 스코어 - 평균: {scaled_scores.mean().item():.4f}, 표준편차: {scaled_scores.std().item():.4f}")
    print(f"어텐션 가중치 - 최대: {attention.max().item():.4f}, 최소: {attention.min().item():.4f}")
    print(f"스케일링된 가중치 - 최대: {scaled_attention.max().item():.4f}, 최소: {scaled_attention.min().item():.4f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 원본 스코어
    sns.heatmap(scores[0].detach().numpy(), ax=axes[0,0], cmap='viridis')
    axes[0,0].set_title('Original Attention Scores')
    
    # 스케일링된 스코어
    sns.heatmap(scaled_scores[0].detach().numpy(), ax=axes[0,1], cmap='viridis')
    axes[0,1].set_title('Scaled Attention Scores')
    
    # 원본 어텐션 가중치
    sns.heatmap(attention[0].detach().numpy(), ax=axes[1,0], cmap='viridis')
    axes[1,0].set_title('Original Attention Weights')
    
    # 스케일링된 어텐션 가중치
    sns.heatmap(scaled_attention[0].detach().numpy(), ax=axes[1,1], cmap='viridis')
    axes[1,1].set_title('Scaled Attention Weights')
    
    plt.tight_layout()
    plt.show()

# 다양한 임베딩 크기에 대한 스케일링 효과 분석
for size in [16, 64, 256]:
    analyze_scaling_effect(size)
```

### 2. 멀티-헤드 어텐션 구현 (30분)

#### 멀티-헤드 어텐션 상세 구현
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size
        
        # 쿼리, 키, 값, 출력 투영
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 스케일링 상수
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 선형 투영
        Q = self.q_proj(query)  # (batch_size, seq_len, embed_size)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 헤드 분할: (batch_size, seq_len, heads, head_dim)
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 마스킹
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        
        # 소프트맥스와 드롭아웃
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 어텐션 적용
        context = torch.matmul(attention_weights, V)
        
        # 헤드 결합
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_size)
        
        # 최종 투영
        output = self.out_proj(context)
        
        return output, attention_weights

# 멀티-헤드 어텐션 테스트
multihead_attn = MultiHeadAttention(embed_size, heads)
output, attention = multihead_attn(x, x, x)

print(f"입력 크기: {x.shape}")
print(f"출력 크기: {output.shape}")
print(f"어텐션 가중치 크기: {attention.shape}")
```

#### 헤드별 특성 분석
```python
def analyze_head_characteristics(multihead_attn, input_data, tokens):
    """각 헤드의 특성 분석"""
    
    multihead_attn.eval()
    with torch.no_grad():
        # 중간 계산을 위해 forward 과정 수동 구현
        batch_size = input_data.size(0)
        
        # 선형 투영
        Q = multihead_attn.q_proj(input_data)
        K = multihead_attn.k_proj(input_data)
        V = multihead_attn.v_proj(input_data)
        
        # 헤드 분할
        Q = Q.view(batch_size, -1, multihead_attn.heads, multihead_attn.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, multihead_attn.heads, multihead_attn.head_dim).transpose(1, 2)
        
        # 각 헤드별 어텐션 스코어 계산
        head_scores = []
        for h in range(multihead_attn.heads):
            head_Q = Q[:, h, :, :]  # (batch_size, seq_len, head_dim)
            head_K = K[:, h, :, :]
            scores = torch.matmul(head_Q, head_K.transpose(-2, -1)) / multihead_attn.scale
            head_scores.append(scores[0].detach().numpy())  # 첫 번째 배치
        
        # 헤드별 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for h in range(min(multihead_attn.heads, 4)):
            sns.heatmap(head_scores[h], 
                       xticklabels=tokens, 
                       yticklabels=tokens,
                       cmap='viridis', 
                       ax=axes[h])
            axes[h].set_title(f'Head {h+1} Raw Scores')
        
        plt.tight_layout()
        plt.show()
        
        # 헤드별 통계 분석
        print("헤드별 통계:")
        for h in range(multihead_attn.heads):
            scores = head_scores[h]
            print(f"Head {h+1}: 평균={scores.mean():.4f}, 표준편차={scores.std():.4f}, "
                  f"최대={scores.max():.4f}, 최소={scores.min():.4f}")

# 헤드 특성 분석
analyze_head_characteristics(multihead_attn, x, tokens)
```

#### 헤드 수에 따른 성능 비교
```python
def compare_head_performance(embed_size, seq_len, head_counts):
    """헤드 수에 따른 성능 비교"""
    
    input_data = torch.randn(1, seq_len, embed_size)
    
    results = {}
    
    for heads in head_counts:
        if embed_size % heads != 0:
            continue
            
        # 멀티-헤드 어텐션 생성
        mha = MultiHeadAttention(embed_size, heads)
        
        # 순전파
        with torch.no_grad():
            output, attention = mha(input_data, input_data, input_data)
        
        # 메트릭 계산
        output_variance = output.var().item()
        attention_entropy = -torch.sum(attention * torch.log(attention + 1e-9), dim=-1).mean().item()
        
        results[heads] = {
            'output_variance': output_variance,
            'attention_entropy': attention_entropy,
            'max_attention': attention.max().item(),
            'min_attention': attention.min().item()
        }
    
    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    head_counts_list = list(results.keys())
    
    # 출력 분산
    variances = [results[h]['output_variance'] for h in head_counts_list]
    axes[0,0].plot(head_counts_list, variances, 'o-')
    axes[0,0].set_xlabel('Number of Heads')
    axes[0,0].set_ylabel('Output Variance')
    axes[0,0].set_title('Output Variance vs Number of Heads')
    axes[0,0].grid(True)
    
    # 어텐션 엔트로피
    entropies = [results[h]['attention_entropy'] for h in head_counts_list]
    axes[0,1].plot(head_counts_list, entropies, 'o-')
    axes[0,1].set_xlabel('Number of Heads')
    axes[0,1].set_ylabel('Attention Entropy')
    axes[0,1].set_title('Attention Entropy vs Number of Heads')
    axes[0,1].grid(True)
    
    # 최대 어텐션
    max_attentions = [results[h]['max_attention'] for h in head_counts_list]
    axes[1,0].plot(head_counts_list, max_attentions, 'o-')
    axes[1,0].set_xlabel('Number of Heads')
    axes[1,0].set_ylabel('Max Attention')
    axes[1,0].set_title('Max Attention vs Number of Heads')
    axes[1,0].grid(True)
    
    # 최소 어텐션
    min_attentions = [results[h]['min_attention'] for h in head_counts_list]
    axes[1,1].plot(head_counts_list, min_attentions, 'o-')
    axes[1,1].set_xlabel('Number of Heads')
    axes[1,1].set_ylabel('Min Attention')
    axes[1,1].set_title('Min Attention vs Number of Heads')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# 헤드 수에 따른 성능 비교
head_counts = [1, 2, 4, 8, 16]
results = compare_head_performance(embed_size, seq_len, head_counts)
```

### 3. 트랜스포머 기본 블록 구현 (30분)

#### 트랜스포머 인코더 블록
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        # 멀티-헤드 어텐션
        self.attention = MultiHeadAttention(embed_size, heads, dropout)
        
        # 피드포워드 네트워크
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        # 멀티-헤드 어텐션 + 잔차 연결 + 레이어 정규화
        attention, attention_weights = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        
        # 피드포워드 + 잔차 연결 + 레이어 정규화
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out, attention_weights

# 트랜스포머 블록 테스트
transformer_block = TransformerBlock(
    embed_size=embed_size, 
    heads=heads, 
    dropout=0.1, 
    forward_expansion=4
)

output, attention = transformer_block(x, x, x, mask=None)
print(f"트랜스포머 블록 출력 크기: {output.shape}")
print(f"어텐션 가중치 크기: {attention.shape}")
```

#### 간단한 트랜스포머 인코더
```python
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(SimpleTransformerEncoder, self).__init__()
        
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        batch_size, seq_length = x.shape
        
        # 위치 인덱스
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        
        # 단어 임베딩 + 위치 임베딩
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        # 트랜스포머 블록 통과
        attention_weights = []
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
            attention_weights.append(attention)
        
        return out, attention_weights

# 간단한 트랜스포머 인코더 테스트
vocab_size = 1000
max_length = 50
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = SimpleTransformerEncoder(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_layers=num_layers,
    heads=heads,
    device=device,
    forward_expansion=4,
    dropout=0.1,
    max_length=max_length
)

# 랜덤 입력 (단어 인덱스)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
output, all_attention_weights = encoder(input_ids, mask=None)

print(f"입력 ID 크기: {input_ids.shape}")
print(f"인코더 출력 크기: {output.shape}")
print(f"어텐션 가중치 레이어 수: {len(all_attention_weights)}")
print(f"각 레이어 어텐션 가중치 크기: {all_attention_weights[0].shape}")
```

#### 트랜스포머 블록 내부 동작 시각화
```python
def visualize_transformer_block(transformer_block, input_data, tokens):
    """트랜스포머 블록 내부 동작 시각화"""
    
    transformer_block.eval()
    with torch.no_grad():
        # 1단계: 멀티-헤드 어텐션
        attention, attention_weights = transformer_block.attention(input_data, input_data, input_data, None)
        
        # 2단계: 첫 번째 잔차 연결과 정규화
        x1 = transformer_block.norm1(attention + input_data)
        
        # 3단계: 피드포워드 네트워크
        ff_output = transformer_block.feed_forward(x1)
        
        # 4단계: 두 번째 잔차 연결과 정규화
        final_output = transformer_block.norm2(ff_output + x1)
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 입력 데이터
        input_norm = torch.norm(input_data[0], dim=1).detach().numpy()
        axes[0,0].plot(input_norm)
        axes[0,0].set_title('Input Vector Norms')
        axes[0,0].set_xlabel('Token Position')
        axes[0,0].set_ylabel('L2 Norm')
        
        # 어텐션 출력
        attention_norm = torch.norm(attention[0], dim=1).detach().numpy()
        axes[0,1].plot(attention_norm)
        axes[0,1].set_title('Attention Output Norms')
        axes[0,1].set_xlabel('Token Position')
        axes[0,1].set_ylabel('L2 Norm')
        
        # 첫 번째 정규화 후
        x1_norm = torch.norm(x1[0], dim=1).detach().numpy()
        axes[0,2].plot(x1_norm)
        axes[0,2].set_title('After First Norm + Residual')
        axes[0,2].set_xlabel('Token Position')
        axes[0,2].set_ylabel('L2 Norm')
        
        # 피드포워드 출력
        ff_norm = torch.norm(ff_output[0], dim=1).detach().numpy()
        axes[1,0].plot(ff_norm)
        axes[1,0].set_title('Feed Forward Output Norms')
        axes[1,0].set_xlabel('Token Position')
        axes[1,0].set_ylabel('L2 Norm')
        
        # 최종 출력
        final_norm = torch.norm(final_output[0], dim=1).detach().numpy()
        axes[1,1].plot(final_norm)
        axes[1,1].set_title('Final Output Norms')
        axes[1,1].set_xlabel('Token Position')
        axes[1,1].set_ylabel('L2 Norm')
        
        # 어텐션 가중치 (첫 번째 헤드)
        sns.heatmap(attention_weights[0, 0].detach().numpy(), 
                   xticklabels=tokens, yticklabels=tokens,
                   cmap='viridis', ax=axes[1,2])
        axes[1,2].set_title('Head 1 Attention Weights')
        
        plt.tight_layout()
        plt.show()

# 트랜스포머 블록 시각화
visualize_transformer_block(transformer_block, x, tokens)
```

## 과제

### 1. 셀프 어텐션 과제
- 다양한 스케일링 기법(예: 학습 가능한 스케일링) 실험
- 어텐션 가중치의 통계적 특성 분석
- 마스킹 기법의 효과 비교(패딩 마스크, 미래 마스크)

### 2. 멀티-헤드 어텐션 과제
- 헤드 수가 모델 성능에 미치는 영향 체계적 분석
- 헤드별 특성화 유도 방법 실험
- 계산 복잡도와 성능의 트레이드오프 분석

### 3. 트랜스포머 블록 과제
- 잔차 연결과 레이어 정규화의 효과 분석
- 다양한 활성화 함수(ReLU, GELU, Swish) 비교
- 드롭아웃 위치와 강도에 따른 성능 변화 분석

## 추가 학습 자료

### 논문
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (Dai et al., 2019)
- "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)

### 온라인 자료
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Transformer Circuits](https://transformer-circuits.pub/)
- [Harvard's Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

### 구현 참고
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [fairseq](https://github.com/pytorch/fairseq)
- [OpenAI GPT Implementation](https://github.com/openai/gpt-2)