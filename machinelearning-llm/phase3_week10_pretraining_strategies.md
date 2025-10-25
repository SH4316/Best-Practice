# 10주차: 사전 훈련 전략

## 강의 목표
- 대규모 데이터 수집과 전처리 방법 이해
- 다양한 사전 훈련 목적 함수의 특징과 장단점 파악
- 학습률 스케줄링과 최적화 기법 습득
- 분산 훈련 인프라와 하드웨어 요구사항 이해
- 사전 훈련의 비용 효율성과 성능 최적화 방법 학습

## 이론 강의 (90분)

### 1. 대규모 데이터 수집과 전처리 (25분)

#### 데이터 소스와 종류
**웹 데이터**
- Common Crawl: 웹 전체의 대규모 크롤링 데이터
- 특징: 방대한 양, 다양한 주제, 노이즈 많음
- 처리: 중복 제거, 품질 필터링, 개인정보 제거
- 양: 수십 TB의 원시 텍스트

**도서 데이터**
- Google Books: 수백만 권의 디지털화된 도서
- Project Gutenberg: 저작권 만료된 고전 문학
- 특징: 높은 품질, 문법적 정확성, 도메인 편향
- 양: 수백 GB에서 수 TB

**코드 데이터**
- GitHub: 오픈 소스 코드 저장소
- 특징: 구조화된 형식, 기술 용어, 다국어
- 처리: 주석 제거, 코드 블록 분리, 라이선스 확인
- 양: 수백 GB의 코드

**학술 데이터**
- arXiv: 과학 논문 프린트 서버
- PubMed: 생의학 논문 데이터베이스
- 특징: 전문성, 최신 연구, 형식적 구조
- 양: 수십 GB의 논문

#### 데이터 전처리 파이프라인
**기본 전처리**
- 텍스트 정규화: 유니코드 정규화, 소문자 변환
- 특수 문자 처리: HTML 태그 제거, 특수 문자 정리
- 언어 식별: 언어 감지, 언어별 분리
- 품질 필터링: 매우 짧은/긴 텍스트 제거

**중복 제거**
- 정확 중복: 해시 기반 중복 제거
- 근사 중복: MinHash, LSH 기반 근사 중복 탐지
- 문서 수준 중복: 전체 문서 단위 중복 제거
- 문장 수준 중복: 문장 단위 중복 제거

**개인정보 제거**
- 식별자: 이름, 이메일, 전화번호, 주소 등
- 방법: 정규표현식, 개인정보 인식 모델
- 수준: 완전 제거 vs 마스킹
- 법적 고려사항: GDPR, CCPA 등 개인정보 보호법

**토큰화 준비**
- 문장 분리: 문장 경계 탐지
- 단어 분리: 공백, 구두점 기반 분리
- 특수 처리: 숫자, URL, 이메일 등 특수 패턴
- 언어별 처리: 형태소 분석, 단어 분리 규칙

#### 데이터 품질 평가
**정량적 지표**
- 어휘 크기: 고유 단어/토큰 수
- 시퀀스 길이 분포: 평균, 중앙값, 최대/최소 길이
- 언어 비율: 다국어 데이터의 언어별 비율
- 중복률: 중복된 내용의 비율

**정성적 지표**
- 문법적 정확성: 문법 오류 비율
- 내용 일관성: 주제 일관성, 논리적 흐름
- 도메인 균형: 다양한 주제의 균형성
- 편향성 분석: 성별, 인종, 문화적 편향

### 2. 사전 훈련 목적 함수 (30분)

#### Causal Language Modeling (CLM)
**원리와 특징**
- 목표: $P(w_t | w_1, ..., w_{t-1})$ 최대화
- 특징: 자연스러운 텍스트 생성, 순차적 의존성 모델링
- 마스킹: 미래 토큰 정보 차단 (삼각형 마스크)
- 손실: 교차 엔트로피, 자기 회귀 손실

**장점**
- 생성 능력: 자연스러운 텍스트 생성
- 순차적 처리: 시간적 순서 정보 보존
- 추론 효율성: 한 토큰씩 순차적 생성 가능

**단점**
- 병렬화 제한: 순차적 처리로 인한 병렬화 어려움
- 양방향 문맥 부재: 미래 정보 활용 불가
- 장거리 의존성: 긴 시퀀스에서 정보 손실

**적용 모델**
- GPT 계열: GPT, GPT-2, GPT-3, GPT-4
- OPT: 메타의 오픈 소스 LLM
- BLOOM: 다국어 대규모 모델

#### Masked Language Modeling (MLM)
**원리와 특징**
- 목표: $P(w_{\text{masked}} | w_{\text{context}})$ 최대화
- 특징: 양방향 문맥 이해, 효율적 훈련
- 마스킹: 입력의 일부 토큰을 [MASK]로 대체
- 손실: 마스킹된 위치에서만 계산

**마스킹 전략**
- 80% [MASK]: 실제 마스킹 토큰으로 대체
- 10% 원본: 원본 토큰 유지 (모델이 올바르게 예측하도록)
- 10% 랜덤: 랜덤 토큰으로 대체 (어휘 전체 활용)

**장점**
- 양방향 문맥: 전체 문맥 정보 활용
- 훈련 효율성: 병렬 처리 가능
- 안정성: CLM보다 안정적인 훈련

**단점**
- 생성 능력 제한: 직접적인 텍스트 생성 어려움
- 사전-훈련/미세조정 불일치: 훈련과 사용 목적 차이
- 마스킹 의존성: 마스킹 전략에 성능 의존

**적용 모델**
- BERT 계열: BERT, RoBERTa, ALBERT
- ELECTRA: 대체 토큰 감지 모델
- DeBERTa: 분해된 어텐션 기반 BERT

#### 혼합 목적 함수
**Span Prediction**
- 원리: 연속적인 토큰 범위 예측
- 구현: 여러 토큰을 하나의 [MASK]로 대체
- 장점: 더 긴 범위의 의존성 모델링
- 적용: SpanBERT, ERNIE

**Permutation Language Modeling**
- 원리: 입력 순서를 무작위로 순열
- 구현: XLNet의 순열 언어 모델링
- 장점: 양방향 문맥 + 자기 회귀 생성 능력
- 단점: 복잡성, 계산 비용

**Denoising Autoencoding**
- 원리: 다양한 노이징으로 입력 손상 후 복원
- 노이징: 토큰 삭제, 문장 순서 섞기, 문서 회전
- 장점: 다양한 언어 이해 능력
- 적용: T5, BART

#### 목적 함수 선택 가이드
**생성 중심 애플리케이션**
- 추천: Causal Language Modeling
- 이유: 자연스러운 텍스트 생성 능력
- 예시: 대화 시스템, 창작, 코드 생성

**이해 중심 애플리케이션**
- 추천: Masked Language Modeling
- 이유: 양방향 문맥 이해 능력
- 예시: 분류, 질문 답변, 정보 추출

**다목적 애플리케이션**
- 추천: 혼합 목적 함수
- 이유: 생성과 이해 능력 모두 필요
- 예시: 번역, 요약, 다중 작업 모델

### 3. 학습률 스케줄링과 최적화 (20분)

#### 학습률 스케줄링의 중요성
**훈련 안정성**
- 초기 학습률: 너무 높으면 발산, 너무 낮으면 수렴 느림
- 학습률 감쇠: 점진적 감쇠로 안정적 수렴 유도
- 적응적 조절: 훈련 진행에 따른 동적 조절

**최적점 탐색**
- 복잡한 손실 경면: LLM의 손실 함수는 매우 복잡
- 국소 최적점: 단순한 최적화 기법의 한계
- 학습률 스케줄링: 더 나은 최적점 탐색 지원

#### 주요 학습률 스케줄러

**Cosine Annealing with Warmup**
- 원리: 웜업 후 코사인 함수로 감쇠
- 수식: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{\pi \cdot \min(t, T_{warmup})}{T_{warmup}}))$
- 특징: 부드러운 감쇠, 안정적인 웜업
- 적용: GPT-3, BERT 등 대부분의 LLM

**Linear Warmup with Decay**
- 원리: 선형 웜업 후 선형 감쇠
- 특징: 단순함, 예측 가능성
- 한계: 급격한 감쇠, 불안정성 가능

**Inverse Square Root Decay**
- 원리: $\eta_t = \eta_0 / \sqrt{1 + \beta t}$
- 특징: 초기에는 빠른 감쇠, 후기에는 느린 감쇠
- 적용: 초기 트랜스포머 모델

**Cyclical Learning Rates**
- 원리: 학습률을 주기적으로 변화
- 특징: 국소 최적점 탈출 용이
- 종류: Triangular, CLR, 1cycle

#### 최적화 기법
**AdamW**
- 개선: Adam에 가중치 감쇠 분리
- 장점: 더 나은 일반화 성능
- 수식: $w_{t+1} = w_t - \eta (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda w_t)$
- 하이퍼파라미터: $\beta_1, \beta_2, \epsilon, \lambda$

**Lion**
- 원리: 간단한 두 단계 업데이트
- 특징: 메모리 효율성, Adam과 유사한 성능
- 수식: $u_t = \text{sign}(\nabla f(w_t))$, $w_{t+1} = w_t - \eta (u_t + \beta \nabla f(w_t))$
- 장점: 적은 메모리, 빠른 계산

**Adafactor**
- 원리: 메모리 효율적인 Adam 변형
- 특징: 대규모 모델을 위한 메모리 최적화
- 적용: T5, 대규모 트랜스포머

#### 하이퍼파라미터 튜닝
**학습률**
- 초기값: 1e-4 ~ 1e-3 범위에서 시작
- 조정: 손실 곡선 관찰하며 조정
- 스케줄링: 웜업 스텝, 총 스텝 수 튜닝

**배치 크기**
- 메모리 제약: GPU 메모리에 따른 최대 배치 크기
- 안정성: 너무 작으면 불안정, 너무 크면 메모리 부족
- 그래디언트 축적: 작은 배치에서 그래디언트 축적 고려

**가중치 감쇠**
- 값: 0.01 ~ 0.1 범위
- 효과: 과적합 방지, 일반화 향상
- 조정: 검증 성능에 따라 조정

### 4. 분산 훈련 인프라 (15분)

#### 하드웨어 요구사항
**GPU 요구사항**
- 메모리: 80GB+ VRAM (대규모 모델)
- 대역폭: 1TB/s+ (데이터 전송 속도)
- 연결: NVLink/NVSwitch (다 GPU 간 고속 통신)
- 예시: NVIDIA A100 (80GB), H100 (80GB)

**CPU 요구사항**
- 코어 수: 64+ 코어 (데이터 로딩 병렬화)
- 메모리: 1TB+ (대규모 데이터 버퍼)
- 저장: 수십 TB의 고속 SSD
- 네트워크: 100Gbps+ (데이터 전송)

**시스템 아키텍처**
- 단일 노드: 8x A100 GPU, 1TB CPU 메모리
- 다중 노드: 수백~수천 GPU 클러스터
- 스토리지: 분산 파일 시스템 (Lustre, GPFS)
- 네트워크: InfiniBand, 고속 이더넷

#### 분산 훈련 프레임워크
**DeepSpeed**
- 특징: Microsoft의 분산 훈련 프레임워크
- 기능: ZeRO, 3D 병렬화, 혼합 정밀도
- 장점: 메모리 효율성, 사용 용이성
- 적용: BLOOM, Megatron-LM

**Megatron-LM**
- 특징: NVIDIA의 대규모 언어 모델 훈련
- 기능: 텐서 병렬화, 파이프라인 병렬화
- 장점: 최고의 성능, NVIDIA 하드웨어 최적화
- 적용: GPT-3, MT-NLG

**FairScale**
- 특징: Facebook의 효율적 분산 훈련
- 기능: 체크포인팅, 오프로딩, 동적 모양
- 장점: 메모리 최적화, 유연성
- 적용: OPT, LLaMA

#### 통신 최적화
**그래디언트 압축**
- 원리: 그래디언트 양자화로 통신량 감소
- 방법: 8비트, 16비트 양자화
- 효과: 통신 대역폭 절약
- 단점: 정확성 손실

**비동기 통신**
- 원리: 계산과 통신 중첩
- 구현: NCCL, Gloo, MPI
- 효과: 통신 대기 시간 감소
- 도전: 구현 복잡성

**통신-계산 중첩**
- 파이프라인 병렬화: 계산과 통신 중첩
- 그래디언트 축적: 통신 빈도 감소
- 동적 스케줄링: 실시간 통신 패턴 최적화

#### 비용 효율화
**클라우드 vs 온프레미스**
- 클라우드: 유연성, 선결제, 관리 편의성
- 온프레미스: 장기적 비용 효율, 제어권
- 하이브리드: 기본 부하 온프레미스, 피크 부하 클라우드

**스팟 인스턴스**
- 원리: 미사용 컴퓨팅 자원 할인
- 특징: 최대 90% 할인, 단기 사용 가능
- 도전: 예측 불가능성, 중단 위험
- 전략: 자동 스케줄링, 다중 리전

**모델 병렬화**
- 데이터 병렬화: 여러 모델 동시 훈련
- 하이퍼파라미터 탐색: 여러 설정 동시 실험
- 앙상블: 여러 모델 결과 결합
- 효과: 자원 활용 극대화

## 실습 세션 (90분)

### 1. 대규모 데이터 전처리 (30분)

#### 데이터 전처리 파이프라인 구현
```python
import re
import json
import hashlib
from typing import List, Dict, Set
import multiprocessing as mp
from collections import defaultdict

class LargeDataProcessor:
    def __init__(self, min_length=10, max_length=1000, dedup_threshold=0.95):
        self.min_length = min_length
        self.max_length = max_length
        self.dedup_threshold = dedup_threshold
        self.vocab = defaultdict(int)
        self.stats = {
            'total_docs': 0,
            'processed_docs': 0,
            'duplicates_removed': 0,
            'too_short': 0,
            'too_long': 0,
            'language_filtered': 0
        }
    
    def clean_text(self, text: str) -> str:
        """텍스트 정규화"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # URL 제거
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # 이메일 제거
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        
        # 여러 공백을 단일 공백으로
        text = re.sub(r'\s+', ' ', text)
        
        # 양 끝 공백 제거
        text = text.strip()
        
        return text.lower()
    
    def detect_language(self, text: str) -> str:
        """간단한 언어 감지 (실제로는 더 정교한 라이브러리 사용)"""
        # 한국어 자모 확인
        korean_chars = len(re.findall(r'[가-힣]', text))
        # 일본어 문자 확인
        japanese_chars = len(re.findall(r'[ひらがなカタカ]', text))
        # 중국어 문자 확인
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 영문자 확인
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = korean_chars + japanese_chars + chinese_chars + english_chars
        
        if total_chars == 0:
            return 'unknown'
        
        # 가장 많은 문자 언어로 결정
        lang_counts = {
            'korean': korean_chars,
            'japanese': japanese_chars,
            'chinese': chinese_chars,
            'english': english_chars
        }
        
        return max(lang_counts, key=lang_counts.get)
    
    def compute_hash(self, text: str) -> str:
        """문서 해시 계산"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def is_similar(self, text1: str, text2: str) -> bool:
        """근사 중복 확인 (간단한 Jaccard 유사도)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return False
        
        similarity = intersection / union
        return similarity > self.dedup_threshold
    
    def process_document(self, doc: Dict) -> Dict:
        """단일 문서 처리"""
        self.stats['total_docs'] += 1
        
        # 텍스트 정규화
        text = self.clean_text(doc.get('text', ''))
        
        # 길이 필터링
        if len(text.split()) < self.min_length:
            self.stats['too_short'] += 1
            return None
        
        if len(text.split()) > self.max_length:
            self.stats['too_long'] += 1
            return None
        
        # 언어 필터링 (영어만 유지)
        lang = self.detect_language(text)
        if lang != 'english':
            self.stats['language_filtered'] += 1
            return None
        
        # 해시 계산
        doc_hash = self.compute_hash(text)
        
        processed_doc = {
            'text': text,
            'hash': doc_hash,
            'language': lang,
            'word_count': len(text.split())
        }
        
        self.stats['processed_docs'] += 1
        
        # 어휘 통계
        for word in text.split():
            self.vocab[word] += 1
        
        return processed_doc
    
    def remove_duplicates(self, docs: List[Dict]) -> List[Dict]:
        """중복 제거"""
        seen_hashes = set()
        unique_docs = []
        
        for doc in docs:
            doc_hash = doc['hash']
            
            if doc_hash in seen_hashes:
                self.stats['duplicates_removed'] += 1
                continue
            
            # 근사 중복 확인
            is_duplicate = False
            for existing_doc in unique_docs[-100:]:  # 최근 100개만 확인
                if self.is_similar(doc['text'], existing_doc['text']):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                self.stats['duplicates_removed'] += 1
                continue
            
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)
        
        return unique_docs
    
    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """배치 처리"""
        processed_docs = []
        
        for doc in batch:
            processed = self.process_document(doc)
            if processed:
                processed_docs.append(processed)
        
        # 중복 제거
        unique_docs = self.remove_duplicates(processed_docs)
        
        return unique_docs
    
    def print_stats(self):
        """처리 통계 출력"""
        print("=== 데이터 처리 통계 ===")
        for key, value in self.stats.items():
            print(f"{key}: {value}")
        
        print(f"어휘 크기: {len(self.vocab)}")
        print(f"처리율: {self.stats['processed_docs']/self.stats['total_docs']*100:.2f}%")

# 가상의 대규모 데이터 생성
def generate_sample_data(num_docs=10000):
    """샘플 데이터 생성"""
    import random
    
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large amounts of data.",
        "Natural language processing has advanced significantly.",
        "Transformers revolutionized the field of NLP.",
        "Large language models can generate human-like text.",
        "Computer vision tasks include image classification and object detection.",
        "Reinforcement learning learns through trial and error.",
        "Neural networks consist of layers of interconnected nodes.",
        "Optimization algorithms adjust model parameters to minimize loss.",
        "Regularization techniques prevent overfitting in deep models."
    ]
    
    data = []
    for i in range(num_docs):
        text = random.choice(sample_texts)
        # 약간의 변형 추가
        if random.random() < 0.3:
            text += f" Document {i}."
        
        data.append({'id': i, 'text': text})
    
    return data

# 데이터 처리기 테스트
processor = LargeDataProcessor(min_length=5, max_length=100)
sample_data = generate_sample_data(1000)

# 배치 처리
batch_size = 100
processed_data = []

for i in range(0, len(sample_data), batch_size):
    batch = sample_data[i:i+batch_size]
    processed_batch = processor.process_batch(batch)
    processed_data.extend(processed_batch)
    
    if (i // batch_size) % 10 == 0:
        print(f"처리된 배치: {i//batch_size + 1}/{len(sample_data)//batch_size}")

# 통계 출력
processor.print_stats()

# 처리된 데이터 샘플 출력
print("\n=== 처리된 데이터 샘플 ===")
for i, doc in enumerate(processed_data[:5]):
    print(f"문서 {i+1}:")
    print(f"  텍스트: {doc['text']}")
    print(f"  언어: {doc['language']}")
    print(f"  단어 수: {doc['word_count']}")
    print(f"  해시: {doc['hash']}")
    print()
```

#### 어휘 분석과 시각화
```python
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_vocabulary(vocab: Dict[str, int], top_k: int = 20):
    """어휘 분석"""
    
    # 기본 통계
    total_tokens = sum(vocab.values())
    unique_tokens = len(vocab)
    
    # 빈도 분포
    token_counts = list(vocab.values())
    token_counts.sort(reverse=True)
    
    # 상위 토큰
    top_tokens = list(vocab.items())[:top_k]
    
    # 길이 분포
    token_lengths = [len(token) for token in vocab.keys()]
    
    print(f"=== 어휘 분석 ===")
    print(f"총 토큰 수: {total_tokens:,}")
    print(f"고유 토큰 수: {unique_tokens:,}")
    print(f"타입-토큰 비율: {total_tokens/unique_tokens:.2f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 상위 토큰 빈도
    tokens, counts = zip(*top_tokens)
    axes[0,0].bar(range(len(tokens)), counts)
    axes[0,0].set_xticks(range(len(tokens)))
    axes[0,0].set_xticklabels(tokens, rotation=45, ha='right')
    axes[0,0].set_title(f'Top {top_k} Tokens')
    axes[0,0].set_ylabel('Frequency')
    
    # 빈도 분포 (로그 스케일)
    axes[0,1].plot(range(len(token_counts)), token_counts)
    axes[0,1].set_yscale('log')
    axes[0,1].set_title('Token Frequency Distribution (Log Scale)')
    axes[0,1].set_xlabel('Token Rank')
    axes[0,1].set_ylabel('Frequency')
    
    # 토큰 길이 분포
    axes[1,0].hist(token_lengths, bins=20, alpha=0.7)
    axes[1,0].set_title('Token Length Distribution')
    axes[1,0].set_xlabel('Token Length')
    axes[1,0].set_ylabel('Count')
    
    # 누적 빈도
    cumulative_freq = np.cumsum(token_counts) / total_tokens
    axes[1,1].plot(range(len(cumulative_freq)), cumulative_freq)
    axes[1,1].set_title('Cumulative Frequency')
    axes[1,1].set_xlabel('Token Rank')
    axes[1,1].set_ylabel('Cumulative Frequency')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'type_token_ratio': total_tokens/unique_tokens,
        'top_tokens': top_tokens
    }

# 어휘 분석
vocab_stats = analyze_vocabulary(processor.vocab)
```

### 2. 학습률 스케줄링 구현 (30분)

#### 다양한 학습률 스케줄러 구현
```python
import torch
import torch.optim as optim
import math
import matplotlib.pyplot as plt

class LearningRateScheduler:
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.lr_history = []
    
    def step(self):
        """학습률 업데이트"""
        self.current_step += 1
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.lr_history.append(lr)
        return lr
    
    def get_lr(self) -> float:
        """현재 학습률 계산"""
        raise NotImplementedError

class CosineAnnealingWarmupScheduler(LearningRateScheduler):
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int, 
                 max_lr: float = 1e-3, min_lr: float = 0.0):
        super().__init__(optimizer, warmup_steps, total_steps)
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # 웜업 단계
            return self.max_lr * self.current_step / self.warmup_steps
        else:
            # 코사인 감쇠 단계
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

class LinearWarmupDecayScheduler(LearningRateScheduler):
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int,
                 max_lr: float = 1e-3, min_lr: float = 0.0):
        super().__init__(optimizer, warmup_steps, total_steps)
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # 선형 웜업
            return self.max_lr * self.current_step / self.warmup_steps
        else:
            # 선형 감쇠
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.current_step - self.warmup_steps) / decay_steps
            return self.max_lr * (1 - progress) + self.min_lr * progress

class InverseSqrtScheduler(LearningRateScheduler):
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int,
                 max_lr: float = 1e-3, warmup_init_lr: float = 1e-7):
        super().__init__(optimizer, warmup_steps, total_steps)
        self.max_lr = max_lr
        self.warmup_init_lr = warmup_init_lr
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # 웜업 단계
            return self.warmup_init_lr + (self.max_lr - self.warmup_init_lr) * self.current_step / self.warmup_steps
        else:
            # 역제곱근 감쇠
            decay_factor = self.current_step / self.warmup_steps
            return self.max_lr / math.sqrt(decay_factor)

# 학습률 스케줄러 비교
def compare_schedulers():
    """다양한 학습률 스케줄러 비교"""
    
    # 가상의 옵티마이저
    model = torch.nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 파라미터
    total_steps = 1000
    warmup_steps = 100
    
    # 스케줄러 생성
    schedulers = {
        'Cosine': CosineAnnealingWarmupScheduler(optimizer, warmup_steps, total_steps),
        'Linear': LinearWarmupDecayScheduler(optimizer, warmup_steps, total_steps),
        'InvSqrt': InverseSqrtScheduler(optimizer, warmup_steps, total_steps)
    }
    
    # 학습률 기록
    lr_histories = {}
    
    for name, scheduler in schedulers.items():
        # 옵티마이저 초기화
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler.optimizer = optimizer
        scheduler.current_step = 0
        scheduler.lr_history = []
        
        # 시뮬레이션
        for step in range(total_steps):
            lr = scheduler.step()
        
        lr_histories[name] = scheduler.lr_history
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    for name, lr_history in lr_histories.items():
        plt.plot(lr_history, label=name, linewidth=2)
    
    plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.7, label='Warmup End')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    return lr_histories

# 학습률 스케줄러 비교
lr_histories = compare_schedulers()
```

#### 최적화 기법 비교
```python
def compare_optimizers():
    """다양한 최적화 기법 비교"""
    
    # 간단한 회귀 문제
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # 모델
    model = torch.nn.Linear(10, 1)
    criterion = torch.nn.MSELoss()
    
    # 최적화 기법
    optimizers = {
        'Adam': optim.Adam(model.parameters(), lr=1e-3),
        'AdamW': optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01),
        'SGD': optim.SGD(model.parameters(), lr=1e-3, momentum=0.9),
        'RMSprop': optim.RMSprop(model.parameters(), lr=1e-3)
    }
    
    # 훈련 기록
    loss_histories = {name: [] for name in optimizers.keys()}
    
    # 훈련
    epochs = 100
    
    for name, optimizer in optimizers.items():
        # 모델 초기화
        model = torch.nn.Linear(10, 1)
        optimizer = type(optimizer)(model.parameters(), lr=1e-3)
        if hasattr(optimizer, 'param_groups'):
            for param_group in optimizer.param_groups:
                if 'weight_decay' in param_group:
                    param_group['weight_decay'] = 0.01 if 'AdamW' in name else 0.0
        
        losses = []
        
        for epoch in range(epochs):
            # 순전파
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        loss_histories[name] = losses
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    for name, losses in loss_histories.items():
        plt.plot(losses, label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    return loss_histories

# 최적화 기법 비교
loss_histories = compare_optimizers()
```

### 3. 분산 훈련 시뮬레이션 (30분)

#### 데이터 병렬화 시뮬레이션
```python
def simulate_distributed_training():
    """분산 훈련 시뮬레이션"""
    
    # 가상의 대규모 모델
    model_size = 1_000_000_000  # 10억 파라미터
    batch_size = 32
    seq_length = 2048
    
    # GPU 설정
    num_gpus = 8
    gpu_memory = 80 * 1024**3  # 80GB in bytes
    
    # 데이터 분할
    samples_per_gpu = batch_size // num_gpus
    print(f"전체 배치 크기: {batch_size}")
    print(f"GPU당 샘플 수: {samples_per_gpu}")
    
    # 메모리 사용량 계산
    # 모델 파라미터 (FP32)
    param_memory = model_size * 4  # 4 bytes per parameter
    # 그래디언트 (FP32)
    grad_memory = model_size * 4
    # 활성화 (FP32, 간단한 추정)
    activation_memory = samples_per_gpu * seq_length * 1024 * 4  # 1024은 hidden size 가정
    
    total_memory_per_gpu = param_memory + grad_memory + activation_memory
    
    print(f"\n=== 메모리 사용량 분석 ===")
    print(f"모델 파라미터: {param_memory / 1024**3:.2f} GB")
    print(f"그래디언트: {grad_memory / 1024**3:.2f} GB")
    print(f"활성화: {activation_memory / 1024**3:.2f} GB")
    print(f"총 메모리: {total_memory_per_gpu / 1024**3:.2f} GB")
    print(f"GPU 메모리: {gpu_memory / 1024**3:.2f} GB")
    print(f"메모리 부족: {(total_memory_per_gpu - gpu_memory) / 1024**3:.2f} GB")
    
    # 그래디언트 압축 효과
    compression_ratios = [1.0, 0.5, 0.25, 0.125]  # 압축 없음, 2비트, 4비트, 8비트
    
    print(f"\n=== 그래디언트 압축 효과 ===")
    for ratio in compression_ratios:
        compressed_grad_memory = grad_memory * ratio
        total_compressed = param_memory + compressed_grad_memory + activation_memory
        
        print(f"압축률 {ratio}: {total_compressed / 1024**3:.2f} GB "
              f"({(1 - total_compressed/total_memory_per_gpu) * 100:.1f}% 절약)")
    
    # 통신량 계산
    print(f"\n=== 통신량 분석 ===")
    # 각 스텝에서 모든 그래디언트 전송
    comm_per_step = model_size * 4  # FP32 그래디언트
    # 압축 시
    compressed_comm_per_step = model_size * 2  # FP16 그래디언트
    
    print(f"통신량 (FP32): {comm_per_step / 1024**2:.2f} MB per step")
    print(f"통신량 (FP16): {compressed_comm_per_step / 1024**2:.2f} MB per step")
    print(f"통신 절약: {(1 - compressed_comm_per_step/comm_per_step) * 100:.1f}%")
    
    return {
        'memory_per_gpu': total_memory_per_gpu,
        'gpu_memory': gpu_memory,
        'memory_shortage': total_memory_per_gpu - gpu_memory,
        'compression_savings': [1 - ratio for ratio in compression_ratios[1:]],
        'communication_savings': 1 - compressed_comm_per_step/comm_per_step
    }

# 분산 훈련 시뮬레이션
distributed_analysis = simulate_distributed_training()
```

#### 훈련 비용 분석
```python
def analyze_training_costs():
    """훈련 비용 분석"""
    
    # 모델 설정
    model_sizes = [1e9, 10e9, 100e9]  # 1B, 10B, 100B 파라미터
    tokens_per_parameter = 20  # 파라미터당 훈련 토큰 수
    
    # 하드웨어 비용 (월간)
    gpu_costs = {
        'A100_80GB': 3000,  # 월 $3,000
        'H100_80GB': 5000,  # 월 $5,000
    }
    
    # 클라우드 비용 (시간당)
    cloud_costs = {
        'aws_p4d.24xlarge': 32.77,  # $32.77/hour
        'gcp_a2-highgpu-8g': 38.40,  # $38.40/hour
    }
    
    print("=== 훈련 비용 분석 ===")
    
    for model_size in model_sizes:
        total_tokens = int(model_size * tokens_per_parameter)
        
        # 훈련 시간 추정 (매우 단순한 추정)
        # 1B 파라미터 모델이 1M 토큰/초 처리한다고 가정
        tokens_per_second = 1e6 * (model_size / 1e9)  # 모델 크기에 비례
        training_seconds = total_tokens / tokens_per_second
        training_hours = training_seconds / 3600
        training_days = training_hours / 24
        
        print(f"\n모델 크기: {model_size/1e9:.0f}B 파라미터")
        print(f"총 토큰 수: {total_tokens:,}")
        print(f"예상 훈련 시간: {training_days:.1f} 일")
        
        # 온프레미스 비용
        gpu_months = training_days / 30
        for gpu_name, monthly_cost in gpu_costs.items():
            total_cost = gpu_months * monthly_cost
            print(f"  {gpu_name} (온프레미스): ${total_cost:,.0f}")
        
        # 클라우드 비용
        for instance_name, hourly_cost in cloud_costs.items():
            total_cost = training_hours * hourly_cost
            print(f"  {instance_name} (클라우드): ${total_cost:,.0f}")
    
    # 비용 효율화 전략
    print(f"\n=== 비용 효율화 전략 ===")
    print("1. 스팟 인스턴스 활용: 최대 90% 비용 절약")
    print("2. 하이브리드 접근: 기본 부하는 온프레미스, 피크는 클라우드")
    print("3. 모델 병렬화: 여러 모델 동시 훈련으로 자원 활용 극대화")
    print("4. 효율적 훈련: 그래디언트 체크포인팅, 혼합 정밀도 등")
    print("5. 모델 공유: 오픈 소스로 공개하여 커뮤니티 기여 유도")

# 훈련 비용 분석
analyze_training_costs()
```

## 과제

### 1. 데이터 전처리 과제
- 대규모 텍스트 데이터 전처리 파이프라인 구현
- 다양한 중복 제거 알고리즘의 성능 비교
- 언어별 전처리 전략 개발과 비교

### 2. 사전 훈련 목적 함수 과제
- Causal LM과 MLM의 성능 특성 비교 실험
- 다양한 마스킹 전략의 효과 분석
- 혼합 목적 함수 설계와 구현

### 3. 최적화 전략 과제
- 다양한 학습률 스케줄러의 성능 비교
- 최적화 기법에 따른 수렴 속도 분석
- 분산 훈련 환경에서의 메모리 최적화

## 추가 학습 자료

### 논문
- "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
- "Chinchilla: Training Language Models with Compute-Optimal Scaling" (Hoffmann et al., 2022)

### 온라인 자료
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/)

### 하드웨어 가이드
- [NVIDIA DGX Systems](https://www.nvidia.com/en-us/data-center/dgx-systems/)
- [Cloud GPU Pricing Comparison](https://cloud-gpu.com/)
- [MLPerf Training Benchmark](https://mlcommons.org/en/training/)