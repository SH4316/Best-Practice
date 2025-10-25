# 9주차: LLM 훈련 기초

## 강의 목표
- LLM 훈련의 전체 파이프라인과 핵심 개념 이해
- 토크나이저의 종류와 동작 원리 습득
- 언어 모델링 목적 함수(Causal LM, Masked LM)의 특징 파악
- 분산 훈련 기법과 최적화 전략 이해
- 대규모 데이터 처리와 훈련 효율화 방법 학습

## 이론 강의 (90분)

### 1. LLM 훈련 파이프라인 개요 (25분)

#### LLM 훈련의 단계
**데이터 수집과 전처리**
- 대규모 텍스트 데이터 수집: 웹, 책, 논문, 코드 등
- 데이터 정제: 중복 제거, 저품질 데이터 필터링, 개인정보 제거
- 토큰화: 텍스트를 모델이 이해할 수 있는 단위로 분할
- 데이터 포맷팅: 훈련에 적합한 형태로 변환

**모델 아키텍처 설계**
- 모델 크기 결정: 파라미터 수, 레이어 수, 임베딩 차원
- 트랜스포머 변형: 표준 트랜스포머 vs 효율적 변형
- 특수 토큰: PAD, BOS, EOS, UNK 등
- 위치 인코딩: 절대적 vs 상대적 위치 인코딩

**훈련 실행과 모니터링**
- 손실 함수 정의: 교차 엔트로피, 레이블 스무딩
- 옵티마이저 선택: Adam, AdamW 등
- 학습률 스케줄링: 웜업, 코사인 감쇠 등
- 체크포인팅과 모델 저장

#### LLM 훈련의 특수성
**대규모 데이터 처리**
- 데이터 양: 수백억에서 수조 개의 토큰
- 저장 공간: 테라바이트 단위의 데이터 저장
- I/O 병목: 데이터 로딩이 훈련 속도 제한
- 데이터 분산: 여러 머신에 데이터 분산 저장

**계산 자원 요구**
- GPU 메모리: 수십에서 수백 GB의 VRAM 필요
- 훈련 시간: 수주에서 수개월의 장기 훈련
- 통신 비용: 분산 훈련 시 네트워크 통신 오버헤드
- 전력 소비: 대규모 훈련의 높은 에너지 소비

**훈련 안정성**
- 그래디언트 폭주: 대규모 모델의 불안정한 훈련
- 수치적 안정성: FP16 혼합 정밀도 훈련의 도전
- 하이퍼파라미터 민감성: 작은 변화가 큰 성능 차이
- 재현성: 대규모 훈련의 재현성 확보 어려움

### 2. 토크나이저 (30분)

#### 토크나이저의 역할과 중요성
**텍스트-숫자 변환**
- 역할: 인간이 이해하는 텍스트를 모델이 처리하는 숫자 시퀀스로 변환
- 중요성: 토크나이저의 품질이 모델 성능에 직접적 영향
- 트레이드오프: 어휘 크기 vs 시퀀스 길이 vs OOV 문제

**토크나이저의 평가 지표**
- 어휘覆盖率: 실제 텍스트의 얼마나 많은 부분을 커버하는가
- 시퀀스 길이: 평균 토큰 수는 얼마인가
- OOV 비율: 알 수 없는 단어의 비율은 얼마인가
- 언어적 의미: 토큰이 의미적으로 의미 있는 단위인가

#### 주요 토크나이저 알고리즘

**BPE (Byte Pair Encoding)**
- 원리: 가장 빈번한 문자 쌍을 반복적으로 병합
- 과정:
  1. 초기: 각 문자를 개별 토큰으로 분리
  2. 반복: 가장 빈번한 인접 문자 쌍을 새 토큰으로 병합
  3. 종료: 목표 어휘 크기에 도달할 때까지 반복
- 장점: 데이터 기반, 하위 단어(subword) 단위 처리
- 단점: 병합 순서에 따른 결과 불일치, 언어적 의미 무시

**WordPiece**
- 원리: 언어 모델 손실을 최소화하는 토큰 병합
- 과정:
  1. 초기: 문자 단위 토큰화
  2. 평가: 각 가능한 병합의 언어 모델 손실 계산
  3. 선택: 손실을 가장 많이 줄이는 병합 선택
  4. 반복: 목표 어휘 크기까지 반복
- 장점: 통계적 최적화, BPE보다 언어적 의미 고려
- 단점: 계산 복잡도 높음, 사전 훈련된 언어 모델 필요

**SentencePiece**
- 원리: 언어 독립적인 토크나이징 프레임워크
- 특징:
  - 공백을 특수 문자(▁)로 처리
  - BPE, WordPiece, Unigram 언어 모델 지원
  - 유니코드 직접 처리
- 장점: 언어 독립성, 일관된 토크나이징
- 단점: 복잡성, 하이퍼파라미터 튜닝 필요

**Unigram Language Model**
- 원리: 단어 분포의 확률 모델 기반 토크나이징
- 과정:
  1. 초기: 큰 어휘로 시작
  2. 반복: 손실 증가가 가장 적은 토큰 제거
  3. 종료: 목표 어휘 크기에 도달
- 장점: 통계적 최적화, 빠른 토큰화
- 단점: 초기 어휘 선택의 중요성

#### 토크나이저의 실제 적용

**GPT 계열 토크나이저**
- BPE 기반: GPT-2/3는 BPE 변형 사용
- 어휘 크기: GPT-2 (50K), GPT-3 (50K+256)
- 특징: UTF-8 바이트 수준 처리, 공백 특수 처리

**BERT 계열 토크나이저**
- WordPiece 기반: BERT, RoBERTa 등
- 어휘 크기: 30K 토큰
- 특징: WordPiece 최적화, 대소문자 구분

**T5 계열 토크나이저**
- SentencePiece 기반: Unigram 언어 모델
- 어휘 크기: 32K 토큰
- 특징: 공백 처리, 다국어 지원

#### 토크나이저 선택 고려사항
**언어 특성**
- 분리 언어: 한국어, 일본어 등은 형태소 분석 고려
- 교착 언어: 독일어, 핀란드어 등은 합성어 처리
- 고립 언어: 영어, 중국어 등은 단어 경계 명확

**도메인 특성**
- 기술 도메인: 전문 용어 처리
- 대화 도메인: 구어체 특성 반영
- 코드 도메인: 프로그래밍 언어 문법 고려

**계산 효율성**
- 어휘 크기: 메모리 사용량과 직접적 관계
- 토큰 길이: 시퀀스 길이와 계산 복잡도
- 인코딩 속도: 실시간 응용의 중요한 요소

### 3. 언어 모델링 목적 함수 (20분)

#### Causal Language Modeling (CLM)
**정의와 목적**
- 정의: 다음 토큰 예측을 통한 언어 모델링
- 목적: $P(w_t | w_1, w_2, ..., w_{t-1})$ 모델링
- 응용: 텍스트 생성, 대화 시스템, 코드 생성

**수학적 표현**
- 손실 함수: $L = -\sum_{t=1}^{T} \log P(w_t | w_{<t})$
- 마스킹: 미래 토큰 정보 누출 방지
- 자기 회귀: 한 토큰씩 순차적 생성

**장점과 단점**
- 장점: 자연스러운 텍스트 생성, 순차적 의존성 모델링
- 단점: 병렬화 제한, 장거리 의존성 학습 어려움

**실제 적용**
- GPT 계열: GPT, GPT-2, GPT-3, GPT-4
- OPT: 메타의 오픈 소스 LLM
- BLOOM: 다국어 대규모 언어 모델

#### Masked Language Modeling (MLM)
**정의와 목적**
- 정의: 마스킹된 토큰 예측을 통한 언어 모델링
- 목적: $P(w_{\text{masked}} | w_{\text{context}})$ 모델링
- 응용: 이해 중심 모델, 사전 훈련

**수학적 표현**
- 마스킹: 입력의 일부 토큰을 [MASK]로 대체
- 손실 함수: $L = -\sum_{i \in \text{masked}} \log P(w_i | w_{\text{context}})$
- 마스킹 전략: 15% 마스킹 (80% [MASK], 10% 원본, 10% 랜덤)

**장점과 단점**
- 장점: 양방향 문맥 이해, 효율적인 훈련
- 단점: 생성 능력 제한, 마스킹과 실제 생성의 불일치

**실제 적용**
- BERT 계열: BERT, RoBERTa, ALBERT
- ELECTRA: 대체 토큰 감지 모델
- DeBERTa: 분해된 어텐션 기반 BERT

#### 기타 언어 모델링 목적 함수
**Permutation Language Modeling (PLM)**
- XLNet: 입력 순서를 무작위로 순열하여 순방향-역방향 문맥 모두 활용
- 장점: 양방향 문맥 + 자기 회귀 생성 능력
- 단점: 복잡성, 계산 비용

**Span Prediction**
- SpanBERT: 연속적인 토큰 범위 예측
- 장점: 더 긴 범위의 의존성 모델링
- 응용: 질문 답변, 엔티티 인식

**Denoising Autoencoding**
- T5: 다양한 노이징 기법으로 입력 손상 후 복원
- 노이징: 토큰 삭제, 마스킹, 문장 순서 섞기 등
- 장점: 다양한 언어 이해 능력 학습

### 4. 분산 훈련과 최적화 (15분)

#### 데이터 병렬화
**기본 원리**
- 아이디어: 데이터를 여러 GPU로 분할, 각 GPU에서 동일한 모델 복제
- 과정:
  1. 데이터 배치를 여러 GPU로 분할
  2. 각 GPU에서 독립적으로 순전파와 역전파
  3. 그래디언트 수집과 평균화
  4. 평균화된 그래디언트로 모델 파라미터 업데이트
- 효과: 배치 크기 효과적으로 증가

**장점과 한계**
- 장점: 구현 단순, 거의 선형적 확장
- 한계: 각 GPU가 전체 모델 저장 필요, 메모리 제한

#### 모델 병렬화
**Pipeline Parallelism**
- 아이디어: 모델 레이어를 여러 GPU로 분할
- 과정:
  1. 레이어를 그룹으로 나누어 여러 GPU에 분배
  2. 순차적으로 레이어 그룹 통과
  3. 마이크로 배치로 파이프라인 효율화
- 효과: 큰 모델을 여러 GPU에 분산 저장

**Tensor Parallelism**
- 아이디어: 텐서 연산을 여러 GPU로 분할
- 과정:
  1. 큰 행렬 곱셈을 여러 GPU로 분할
  2. 각 GPU에서 부분 연산 수행
  3. 결과 통합
- 효과: 메모리 사용량 감소, 큰 행렬 연산 가능

#### 혼합 정밀도 훈련
**FP16 혼합 정밀도**
- 원리: 순전파는 FP16, 그래디언트 축적은 FP32
- 장점: 메모리 사용량 절반, 계산 속도 향상
- 단점: 수치적 불안정성, 언더플로우 위험

**동적 손실 스케일링**
- 원리: 그래디언트 언더플로우 방지를 위해 동적 손실 스케일링
- 과정:
  1. 손실에 스케일 인자 곱하기
  2. FP16으로 역전파
  3. 그래디언트를 FP32로 변환 후 스케일 인자로 나누기
  4. 스케일 인자 동적 조절

#### 최적화 기법
**메모리 최적화**
- 그래디언트 체크포인팅: 중간 활성화 재계산으로 메모리 절약
- 활성화 오프로딩: CPU로 활성화 이동으로 GPU 메모리 절약
- 제로 최적화: 그래디언트와 파라미터 분할 저장

**통신 최적화**
- 그래디언트 압축: 통신량 감소
- 비동기 통신: 계산과 통신 중첩
- 토폴로지 최적화: GPU 간 통신 경로 최적화

## 실습 세션 (90분)

### 1. 토크나이저 구현과 비교 (30분)

#### BPE 토크나이저 구현
```python
import re
import collections
import json
from typing import List, Dict, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token
        self.merges = []  # merge rules
        
    def train(self, texts: List[str]):
        """BPE 훈련"""
        # 초기 문자 수집
        vocab = collections.Counter()
        for text in texts:
            # 공백으로 단어 분리 후 문자 단위로 분리
            words = text.split()
            for word in words:
                chars = list(word)
                chars.append('</w>')  # 단어 끝 표시
                vocab[' '.join(chars)] += 1
        
        # 초기 어휘 (문자)
        tokens = set()
        for word in vocab:
            tokens.update(word.split())
        
        # BPE 병합 반복
        num_merges = self.vocab_size - len(tokens)
        for i in range(num_merges):
            # 가장 빈번한 문자 쌍 찾기
            pairs = collections.defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j+1])] += freq
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # 어휘 업데이트
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = self._apply_merge(word, best_pair)
                new_vocab[new_word] = freq
            vocab = new_vocab
        
        # 최종 어휘 구축
        self._build_vocab(vocab)
    
    def _apply_merge(self, word: str, pair: Tuple[str, str]) -> str:
        """병합 규칙 적용"""
        first, second = pair
        parts = word.split()
        i = 0
        while i < len(parts) - 1:
            if parts[i] == first and parts[i+1] == second:
                parts = parts[:i] + [first + second] + parts[i+2:]
            else:
                i += 1
        return ' '.join(parts)
    
    def _build_vocab(self, vocab: Dict[str, int]):
        """어휘 구축"""
        tokens = set()
        for word in vocab:
            tokens.update(word.split())
        
        # 특수 토큰 추가
        tokens.add('<pad>')
        tokens.add('<unk>')
        tokens.add('<bos>')
        tokens.add('<eos>')
        
        self.vocab = {token: i for i, token in enumerate(sorted(tokens))}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID로 인코딩"""
        # 전처리
        words = text.lower().split()
        
        token_ids = []
        for word in words:
            # 문자 단위로 시작
            chars = list(word)
            chars.append('</w>')
            
            # BPE 병합 적용
            while len(chars) > 1:
                pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                
                # 병합 가능한 쌍 찾기
                merge_candidates = []
                for pair in pairs:
                    if pair in self.merges:
                        # 병합 우선순위 (먼저 학습된 것 우선)
                        priority = self.merges.index(pair)
                        merge_candidates.append((priority, pair, chars.index(pair[0])))
                
                if not merge_candidates:
                    break
                
                # 가장 높은 우선순위의 병합 적용
                _, best_pair, start_idx = min(merge_candidates)
                chars = chars[:start_idx] + [best_pair[0] + best_pair[1]] + chars[start_idx+2:]
            
            # 토큰 ID로 변환
            for token in chars:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.vocab['<unk>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        tokens = [self.inverse_vocab.get(id, '<unk>') for id in token_ids]
        
        # </w> 제거와 공백 처리
        text = ''
        for token in tokens:
            if token == '</w>':
                text += ' '
            elif token not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                text += token
        
        return text.strip()

# BPE 토크나이저 훈련
sample_texts = [
    "hello world",
    "hello there",
    "world wide web",
    "machine learning is fun",
    "deep learning models"
]

bpe_tokenizer = BPETokenizer(vocab_size=50)
bpe_tokenizer.train(sample_texts)

# 테스트
test_text = "hello deep learning world"
encoded = bpe_tokenizer.encode(test_text)
decoded = bpe_tokenizer.decode(encoded)

print(f"원본: {test_text}")
print(f"인코딩: {encoded}")
print(f"디코딩: {decoded}")
print(f"어휘 크기: {len(bpe_tokenizer.vocab)}")
print(f"병합 규칙: {bpe_tokenizer.merges[:5]}")  # 처음 5개 규칙
```

#### 다양한 토크나이저 비교
```python
# Hugging Face 토크나이저 비교
from transformers import AutoTokenizer

def compare_tokenizers(text: str, model_names: List[str]):
    """다양한 토크나이저 비교"""
    
    results = {}
    
    for model_name in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 토큰화
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            
            # 통계
            results[model_name] = {
                'tokens': tokens,
                'token_ids': token_ids,
                'vocab_size': tokenizer.vocab_size,
                'num_tokens': len(tokens),
                'compression_ratio': len(text.split()) / len(tokens)
            }
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    return results

# 비교할 모델들
models_to_compare = [
    'gpt2',           # BPE
    'bert-base-uncased',  # WordPiece
    't5-base',        # SentencePiece (Unigram)
    'facebook/opt-125m'   # BPE 변형
]

test_text = "Large language models are transforming artificial intelligence."
results = compare_tokenizers(test_text, models_to_compare)

# 결과 비교
print(f"원본 텍스트: {test_text}")
print(f"원본 단어 수: {len(test_text.split())}")
print()

for model_name, result in results.items():
    print(f"모델: {model_name}")
    print(f"어휘 크기: {result['vocab_size']}")
    print(f"토큰 수: {result['num_tokens']}")
    print(f"압축률: {result['compression_ratio']:.3f}")
    print(f"토큰: {result['tokens']}")
    print(f"토큰 ID: {result['token_ids']}")
    print()

# 시각화
import matplotlib.pyplot as plt

model_names = list(results.keys())
vocab_sizes = [results[name]['vocab_size'] for name in model_names]
token_counts = [results[name]['num_tokens'] for name in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 어휘 크기 비교
ax1.bar(model_names, vocab_sizes)
ax1.set_title('Vocabulary Size Comparison')
ax1.set_ylabel('Vocabulary Size')
ax1.tick_params(axis='x', rotation=45)

# 토큰 수 비교
ax2.bar(model_names, token_counts)
ax2.set_title('Token Count Comparison')
ax2.set_ylabel('Number of Tokens')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### 2. 언어 모델링 목적 함수 구현 (30분)

#### Causal Language Modeling 구현
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int = 2048):
        super(CausalLanguageModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 임베딩
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1000, d_model)  # 최대 길이 1000
        
        # 트랜스포머 디코더 (Causal LM은 디코더만 필요)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 출력 레이어
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 임베딩
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        
        # 미래 마스크 생성 (Causal LM)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 트랜스포머 통과
        x = self.transformer(x, memory=x, tgt_mask=mask)
        
        # 출력
        logits = self.fc_out(x)
        
        return logits
    
    def generate(self, start_tokens: torch.Tensor, max_length: int, 
                temperature: float = 1.0, top_k: int = None):
        """텍스트 생성"""
        self.eval()
        
        with torch.no_grad():
            current_tokens = start_tokens.clone()
            
            for _ in range(max_length):
                # 다음 토큰 예측
                logits = self.forward(current_tokens)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k 필터링
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 다음 토큰 추가
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                # EOS 토큰 확인
                if next_token.item() == 2:  # EOS 토큰 ID
                    break
        
        return current_tokens

# Causal LM 훈련 예시
def train_causal_lm():
    # 간단한 데이터
    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "birds fly in the sky",
        "fish swim in the water"
    ]
    
    # 간단한 토크나이저 (단어 수준)
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    vocab_size = len(vocab)
    inverse_vocab = {v: k for k, v in vocab.items()}
    
    # 데이터 준비
    data = []
    for sentence in sentences:
        tokens = [vocab['<bos>']] + [vocab.get(word, vocab['<unk>']) for word in sentence.split()] + [vocab['<eos>']]
        data.append(tokens)
    
    # 모델
    model = CausalLanguageModel(vocab_size, d_model=128, nhead=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    # 훈련
    model.train()
    for epoch in range(100):
        total_loss = 0
        
        for tokens in data:
            inputs = torch.tensor(tokens[:-1]).unsqueeze(0)  # 마지막 토큰 제외
            targets = torch.tensor(tokens[1:]).unsqueeze(0)   # 첫 토큰 제외
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.squeeze(0), targets.squeeze(0))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(data):.4f}')
    
    return model, vocab, inverse_vocab

# 훈련 실행
model, vocab, inverse_vocab = train_causal_lm()

# 텍스트 생성
start_text = "the cat"
start_tokens = [vocab['<bos>']] + [vocab.get(word, vocab['<unk>']) for word in start_text.split()]
input_tensor = torch.tensor(start_tokens).unsqueeze(0)

generated = model.generate(input_tensor, max_length=10, temperature=0.8)
generated_tokens = generated.squeeze(0).tolist()

# 결과 디코딩
generated_text = ' '.join([inverse_vocab[token] for token in generated_tokens if token not in [vocab['<bos>'], vocab['<eos>'], vocab['<pad>']]])
print(f"생성된 텍스트: {generated_text}")
```

#### Masked Language Modeling 구현
```python
class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int = 2048):
        super(MaskedLanguageModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 임베딩
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1000, d_model)
        
        # 트랜스포머 인코더 (MLM은 인코더 사용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 레이어
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        # 임베딩
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        
        # 트랜스포머 통과
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # 출력
        logits = self.fc_out(x)
        
        return logits
    
    def mask_tokens(self, input_ids: torch.Tensor, mask_token_id: int, 
                  mask_prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLM 마스킹 적용"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mask_prob)
        
        # [MASK] 토큰으로 대체 (80%)
        mask_indices = torch.bernoulli(probability_matrix).bool()
        labels[~mask_indices] = -100  # 손실 계산에서 제외
        
        # 80% [MASK], 10% 원본, 10% 랜덤
        random_indices = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & mask_indices
        original_indices = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & mask_indices & ~random_indices
        
        input_ids[mask_indices & ~random_indices & ~original_indices] = mask_token_id
        input_ids[random_indices] = torch.randint(1, self.vocab_size, input_ids.shape).to(input_ids.device)
        
        return input_ids, labels

# MLM 훈련 예시
def train_mlm():
    # 간단한 데이터
    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "birds fly in the sky",
        "fish swim in the water"
    ]
    
    # 어휘
    vocab = {'<pad>': 0, '<unk>': 1, '<mask>': 2}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    vocab_size = len(vocab)
    
    # 데이터 준비
    data = []
    for sentence in sentences:
        tokens = [vocab.get(word, vocab['<unk>']) for word in sentence.split()]
        data.append(tokens)
    
    # 모델
    model = MaskedLanguageModel(vocab_size, d_model=128, nhead=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 훈련
    model.train()
    for epoch in range(100):
        total_loss = 0
        
        for tokens in data:
            input_tensor = torch.tensor(tokens).unsqueeze(0)
            
            # MLM 마스킹
            masked_input, labels = model.mask_tokens(input_tensor, vocab['<mask>'])
            
            optimizer.zero_grad()
            logits = model(masked_input)
            loss = criterion(logits.squeeze(0), labels.squeeze(0))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(data):.4f}')
    
    return model, vocab

# 훈련 실행
mlm_model, vocab = train_mlm()

# 마스킹된 단어 예측 테스트
test_sentence = "the cat sat on the mat"
test_tokens = [vocab.get(word, vocab['<unk>']) for word in test_sentence.split()]
input_tensor = torch.tensor(test_tokens).unsqueeze(0)

# 마스킹 적용
masked_input, _ = mlm_model.mask_tokens(input_tensor, vocab['<mask>'])
mlm_model.eval()

with torch.no_grad():
    logits = mlm_model(masked_input)
    predictions = torch.argmax(logits, dim=-1)

# 결과 분석
inverse_vocab = {v: k for k, v in vocab.items()}
original_tokens = input_tensor.squeeze(0).tolist()
masked_tokens = masked_input.squeeze(0).tolist()
predicted_tokens = predictions.squeeze(0).tolist()

print("원본:     ", [inverse_vocab[t] for t in original_tokens])
print("마스킹:   ", [inverse_vocab[t] for t in masked_tokens])
print("예측:     ", [inverse_vocab[t] for t in predicted_tokens])

# 마스킹된 위치만 표시
for i, (orig, masked, pred) in enumerate(zip(original_tokens, masked_tokens, predicted_tokens)):
    if masked == vocab['<mask>']:
        print(f"위치 {i}: {inverse_vocab[orig]} -> {inverse_vocab[pred]}")
```

### 3. 분산 훈련 시뮬레이션 (30분)

#### 데이터 병렬화 시뮬레이션
```python
def simulate_data_parallelism():
    """데이터 병렬화 시뮬레이션"""
    
    # 가상의 모델 파라미터
    model_params = torch.randn(1000)  # 1000개 파라미터
    
    # 가상의 데이터 배치
    batch_size = 32
    data = torch.randn(batch_size, 100)  # 32개 샘플, 각 100차원
    
    # GPU 수
    num_gpus = 4
    
    # 데이터 분할
    data_splits = torch.chunk(data, num_gpus, dim=0)
    print(f"원본 배치 크기: {data.shape}")
    for i, split in enumerate(data_splits):
        print(f"GPU {i} 데이터 크기: {split.shape}")
    
    # 각 GPU에서의 순전파 시뮬레이션
    gpu_gradients = []
    for i, gpu_data in enumerate(data_splits):
        # 각 GPU에서 동일한 모델 복제
        gpu_output = torch.matmul(gpu_data, model_params.unsqueeze(0).T)
        gpu_loss = gpu_output.mean()
        
        # 역전파
        gpu_grad = torch.autograd.grad(gpu_loss, model_params, retain_graph=True)[0]
        gpu_gradients.append(gpu_grad)
        
        print(f"GPU {i} 그래디언트 크기: {gpu_grad.shape}")
    
    # 그래디언트 평균화
    averaged_grad = torch.stack(gpu_gradients).mean(dim=0)
    print(f"평균화된 그래디언트 크기: {averaged_grad.shape}")
    
    # 파라미터 업데이트
    learning_rate = 0.001
    updated_params = model_params - learning_rate * averaged_grad
    
    print(f"파라미터 업데이트 완료")
    
    return updated_params

# 데이터 병렬화 시뮬레이션
updated_params = simulate_data_parallelism()
```

#### 메모리 최적화 기법 비교
```python
def compare_memory_optimization():
    """메모리 최적화 기법 비교"""
    
    # 가상의 큰 모델
    batch_size = 8
    seq_len = 512
    d_model = 1024
    
    # 일반적인 순전파 (메모리 많이 사용)
    def standard_forward():
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        # 여러 레이어 통과 (모든 활성화 저장)
        activations = []
        for i in range(12):  # 12 레이어
            x = torch.nn.Linear(d_model, d_model)(x)
            x = torch.nn.ReLU()(x)
            activations.append(x)  # 모든 활성화 저장
        
        loss = x.mean()
        loss.backward()
        
        return len(activations) * x.numel() * 4  # 메모리 사용량 (bytes)
    
    # 그래디언트 체크포인팅 (메모리 절약)
    def gradient_checkpointing_forward():
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        def create_layer(layer_idx):
            def layer_fn(x):
                x = torch.nn.Linear(d_model, d_model)(x)
                x = torch.nn.ReLU()(x)
                return x
            return layer_fn
        
        # 체크포인팅 사용 (중간 활성화 저장 안 함)
        from torch.utils.checkpoint import checkpoint
        
        x = checkpoint(create_layer(0), x)
        x = checkpoint(create_layer(1), x)
        # ... 실제로는 루프로 구현
        
        loss = x.mean()
        loss.backward()
        
        # 체크포인팅 시 메모리 사용량은 훨씬 적음
        return 2 * x.numel() * 4  # 입력과 출력만 저장
    
    # 메모리 사용량 비교
    standard_memory = standard_forward()
    checkpoint_memory = gradient_checkpointing_forward()
    
    print(f"표준 순전파 메모리: {standard_memory / 1024**2:.2f} MB")
    print(f"체크포인팅 메모리: {checkpoint_memory / 1024**2:.2f} MB")
    print(f"메모리 절약률: {(1 - checkpoint_memory/standard_memory) * 100:.1f}%")
    
    # 시각화
    import matplotlib.pyplot as plt
    
    methods = ['Standard', 'Gradient Checkpointing']
    memory_usage = [standard_memory / 1024**2, checkpoint_memory / 1024**2]
    
    plt.figure(figsize=(8, 5))
    plt.bar(methods, memory_usage)
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Optimization Comparison')
    plt.show()

# 메모리 최적화 비교
compare_memory_optimization()
```

#### 혼합 정밀도 훈련 시뮬레이션
```python
def simulate_mixed_precision_training():
    """혼합 정밀도 훈련 시뮬레이션"""
    
    # 가상의 훈련 데이터
    x = torch.randn(32, 100, requires_grad=True)
    y = torch.randn(32, 10)
    
    # 모델
    model = torch.nn.Linear(100, 10)
    
    # FP32 훈련
    def fp32_training():
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 순전파
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item(), [p.dtype for p in model.parameters()]
    
    # FP16 혼합 정밀도 훈련
    def fp16_training():
        from torch.cuda.amp import autocast, GradScaler
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()
        
        # autocast 컨텍스트 내에서 순전파
        with autocast():
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
        
        # 스케일된 역전파
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        return loss.item(), [p.dtype for p in model.parameters()]
    
    # 비교
    fp32_loss, fp32_dtypes = fp32_training()
    fp16_loss, fp16_dtypes = fp16_training()
    
    print(f"FP32 손실: {fp32_loss:.6f}")
    print(f"FP32 파라미터 타입: {fp32_dtypes[0]}")
    print(f"FP16 손실: {fp16_loss:.6f}")
    print(f"FP16 파라미터 타입: {fp16_dtypes[0]}")
    
    # 메모리 사용량 비교 (이론적)
    fp32_memory = 32 * sum(p.numel() for p in model.parameters())
    fp16_memory = 16 * sum(p.numel() for p in model.parameters())
    
    print(f"FP32 메모리 사용량: {fp32_memory / 1024**2:.2f} MB")
    print(f"FP16 메모리 사용량: {fp16_memory / 1024**2:.2f} MB")
    print(f"메모리 절약률: {(1 - fp16_memory/fp32_memory) * 100:.1f}%")
    
    return {
        'fp32_loss': fp32_loss,
        'fp16_loss': fp16_loss,
        'memory_savings': (1 - fp16_memory/fp32_memory) * 100
    }

# 혼합 정밀도 훈련 시뮬레이션
results = simulate_mixed_precision_training()
```

## 과제

### 1. 토크나이저 과제
- 다양한 토크나이저(BPE, WordPiece, SentencePiece) 구현
- 한국어 텍스트에 대한 토크나이저 성능 비교
- 어휘 크기에 따른 모델 성능 분석

### 2. 언어 모델링 과제
- Causal LM과 MLM의 성능 특성 비교
- 다양한 마스킹 전략의 효과 분석
- 언어 모델링 목적 함수에 따른 생성 품질 비교

### 3. 훈련 최적화 과제
- 데이터 병렬화와 모델 병렬화의 효율성 비교
- 메모리 최적화 기법의 성능 측정
- 혼합 정밀도 훈련의 안정성 분석

## 추가 학습 자료

### 논문
- "Neural Machine Translation and Sequence-to-Sequence Models" (Sutskever et al., 2014)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "GPT-3: Language Models are Few-Shot Learners" (Brown et al., 2020)

### 온라인 자료
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers)
- [The Illustrated BERT, RoBERTa, etc.](http://jalammar.github.io/)
- [Efficient Transformers](https://huggingface.co/docs/transformers/performance)

### 구현 참고
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [FairScale](https://github.com/facebookresearch/fairscale)