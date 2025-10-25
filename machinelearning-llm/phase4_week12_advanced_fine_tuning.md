# 12주차: 고급 미세조정 기법

## 강의 목표
- 다중 작업 학습(Multi-task Learning)의 원리와 LLM 적용 방법 이해
- 도메인 적응(Domain Adaptation) 기법과 전략 습득
- 지식 증류(Knowledge Distillation)의 메커니즘과 구현 방법 학습
- 매개변수 효율적 미세조정(PEFT)의 고급 기법과 최적화 전략 파악
- 실제 LLM 미세조정 프로젝트 수행 능력 배양

## 이론 강의 (90분)

### 1. 다중 작업 학습 (Multi-task Learning) (25분)

#### 다중 작업 학습의 기본 원리
**정의와 목적**
- 정의: 여러 관련 작업을 동시에 학습하여 공통 표현 학습
- 목적: 작업 간 지식 공유를 통한 일반화 성능 향상
- 특징: 공통 표현 학습, 작업 특화 헤드, 정규화 효과

**수학적 표현**
- 공통 표현: $f_{\text{shared}}(x; \theta_{\text{shared}})$
- 작업 특화: $f_i(x) = g_i(f_{\text{shared}}(x); \theta_i)$
- 전체 손실: $L = \sum_{i=1}^{N} \alpha_i L_i$
- 가중치: $\sum_{i=1}^{N} \alpha_i = 1$

#### LLM에서의 다중 작업 학습
**다중 작업 아키텍처**
- 공통 트랜스포머 인코더: 모든 작업이 공유하는 인코더
- 작업별 디코더: 각 작업에 특화된 디코더
- 공유 임베딩: 어휘 공유 또는 부분 공유
- 어댑터 레이어: 작업별 적응 레이어

**작업 유형과 조합**
- 생성 작업: 텍스트 생성, 요약, 번역
- 이해 작업: 분류, 질문 답변, 엔티티 인식
- 혼합 작업: 생성과 이해의 결합
- 계층적 작업: 세분화된 작업 계층 구조

#### 다중 작업 학습의 장단점
**장점**
- 일반화 향상: 여러 작업에서 학습한 공통 패턴
- 데이터 효율성: 여러 데이터셋을 동시에 활용
- 정규화 효과: 과적합 방지
- 전이 학습: 한 작업에서 학습한 지식이 다른 작업에 도움

**단점**
- 부정적 전이: 한 작업에 대한 성능 저하 가능성
- 최적화 복잡성: 여러 손실 함수의 균형 최적화
- 데이터 불균형: 작업별 데이터 양 차이 문제
- 하이퍼파라미터 튜닝: 더 복잡한 튜닝 공간

#### 다중 작업 학습의 실제 적용
**T5 (Text-to-Text Transfer Transformer)**
- 아키텍처: 인코더-디코더 구조의 다중 작업 학습
- 작업: 번역, 요약, 질문 답변, 분류 등
- 통일 인터페이스: 모든 작업을 텍스트-텍스트 변환으로 통일
- 성과: 여러 벤치마크에서 최고 수준의 성능

**GLaM (Generalist Language Model)**
- 아키텍처: 다중 작업을 위한 GPT 계열 모델
- 작업: 자연어 이해, 추론, 수학 등
- 동적 작업 전환: 단일 모델에서 여러 작업 수행
- 특징: 작업별 프롬프트를 통한 동적 전환

**mT5 (multilingual T5)**
- 아키텍처: 다국어 다중 작업 학습
- 언어 간 전이: 한 언어에서 학습한 지식이 다른 언어에 전이
- 공유 표현: 언어 간 공통 표현 학습
- 성과: 저자원 언어에서도 높은 성능

### 2. 도메인 적응 (Domain Adaptation) (25분)

#### 도메인 적응의 기본 원리
**정의와 목적**
- 정의: 특정 도메인 데이터로 사전 훈련된 모델을 적응
- 목적: 도메인 특화 지식과 표현 학습
- 특징: 사전 지식 활용, 도메인 특화, 적은 데이터로도 좋은 성능

**도메인 이동의 종류**
- 순차적 적응: 소스 도메인 → 타겟 도메인
- 다중 도메인 적응: 여러 도메인에 동시 적응
- 점진적 적응: 새로운 도메인 데이터로 점진적 적응
- 온라인 적응: 실시간으로 도메인 변화에 적응

#### 도메인 적응 기법

**미세조정 기반 적응**
- 전체 미세조정: 도메인 데이터로 전체 모델 미세조정
- 계층적 미세조정: 상위 레이어만 미세조정
- 어댑터 기반 미세조정: 도메인별 어댑터 추가
- 프롬프트 엔지니어링: 도메인 특화 프롬프트 설계

**지식 증류 기반 적응**
- 도메인 교사: 큰 일반 모델 → 작은 도메인 모델
- 증류 손실: $L_{\text{distill}} = \text{KL}(p_{\text{teacher}} || p_{\text{student}})$
- 특징: 도메인 지식 보존, 모델 크기 감소
- 적용: 의료, 법률, 금융 등 특수 도메인

**지속적 학습 기반 적응**
- 점진적 학습: 새로운 도메인 데이터로 점진적 학습
- 기억 재생: 이전 지식을 잊지 않으면서 새로운 지식 학습
- 정규화: 기존 지식과 새로운 지식의 균형
- 도전: 치명적 망각 문제

#### 도메인 적응의 평가와 최적화
**도메인 특화 평가 지표**
- 도메인 내 성능: 타겟 도메인에서의 성능
- 도메인 간 전이: 다른 도메인으로의 전이 성능
- 도메인 특화도: 도메인 특화 표현의 학습 정도
- 일반화 유지: 일반 지식의 보존 정도

**도메인 적응 최적화 전략**
- 학습률 조절: 도메인별 학습률 동적 조절
- 데이터 균형화: 도메인별 데이터 가중치 조절
- 정규화 강도: 도메인별 정규화 강도 조절
- 어댑터 크기: 도메인별 어댑터 크기 최적화

#### 도메인 적응의 실제 적용
**BioBERT (생물의학 도메인)**
- 기반 모델: BERT
- 적응 데이터: 생물의학 논문과 텍스트
- 특징: 생물의학 용어와 개념 학습
- 성과: 생물의학 태스크에서 최고 수준의 성능

**LegalBERT (법률 도메인)**
- 기반 모델: BERT
- 적응 데이터: 법률 문서와 판례
- 특징: 법률 용어와 논리 구조 학습
- 성과: 법률 태스크에서 높은 성능

**FinBERT (금융 도메인)**
- 기반 모델: BERT
- 적응 데이터: 금융 뉴스와 보고서
- 특징: 금융 용어와 개념 학습
- 성과: 금융 감성 분석에서 높은 성능

### 3. 지식 증류 (Knowledge Distillation) (20분)

#### 지식 증류의 기본 원리
**정의와 목적**
- 정의: 큰 교사 모델의 지식을 작은 학생 모델로 전이
- 목적: 모델 크기 감소와 추론 속도 향상
- 특징: 지식 압축, 효율성 증가, 성능 유지

**지식 증류의 수학적 원리**
- 교사 출력: $y = f_{\text{teacher}}(x; \theta_{\text{teacher}})$
- 학생 출력: $\hat{y} = f_{\text{student}}(x; \theta_{\text{student}})$
- 증류 손실: $L_{\text{distill}} = \|f_{\text{teacher}}(x) - f_{\text{student}}(x)\|^2$
- 전체 손실: $L = L_{\text{task}} + \lambda L_{\text{distill}}$

#### 지식 증류의 종류

**응답 기반 증류(Response-based Distillation)**
- 원리: 교사의 최종 출력을 학생이 모방
- 방법: 교사의 로짓을 소프트 타겟으로 사용
- 손실: $L = \text{KL}(p_{\text{teacher}} || p_{\text{student}})$
- 특징: 구현 단순, 직접적 지식 전이

**특성 기반 증류(Feature-based Distillation)**
- 원리: 교사의 중간 특성을 학생이 모방
- 방법: 교사의 특성 맵을 학생이 학습
- 손실: $L = \|f_{\text{teacher}}^{\text{feat}}(x) - f_{\text{student}}^{\text{feat}}(x)\|^2$
- 특징: 더 풍부한 지식 전이, 내부 표현 학습

**관계 기반 증류(Relation-based Distillation)**
- 원리: 데이터 포인트 간 관계 구조를 증류
- 방법: 교사의 유사도 행렬을 학생이 모방
- 손실: $L = \|S_{\text{teacher}} - S_{\text{student}}\|_F^2$
- 특징: 데이터 구조 정보 보존, 클러스터링 효과

#### 고급 지식 증류 기법
**다단계 증류(Multi-stage Distillation)**
- 원리: 여러 단계에 걸쳐 점진적 증류
- 과정: 큰 교사 → 중간 학생 → 작은 학생
- 특징: 더 큰 압축률, 점진적 지식 정제
- 적용: 매우 큰 모델의 효율적 압축

**앙상블 증류(Ensemble Distillation)**
- 원리: 여러 교사 모델의 앙상블 지식을 학생에게 전이
- 방법: 교사들의 평균 또는 가중합을 타겟으로 사용
- 특징: 더 안정적인 지식, 다양한 관점 통합
- 적용: 여러 전문가 모델의 지식 통합

**자기 증류(Self-Distillation)**
- 원리: 동일한 모델의 다른 버전 간 증류
- 방법: 더 깊은 모델을 얕은 모델에게 증류
- 특징: 모델 압축 없이 성능 향상
- 적용: 단일 모델의 자기 개선

#### 지식 증류의 실제 적용
**DistilBERT**
- 교사 모델: BERT-base
- 학생 모델: 더 작은 BERT 변형
- 증류 방법: 응답 기반 + 특성 기반 증류
- 성과: BERT-base의 97% 성능을 40% 크기로 달성

**TinyBERT**
- 교사 모델: BERT-base
- 학생 모델: 4층 작은 BERT
- 증류 방법: 다단계 증류 + 지식 증류
- 성과: BERT-base의 96% 성능을 7.5배 작은 크기로 달성

**MiniLM**
- 교사 모델: GPT-2
- 학생 모델: 더 작은 LSTM 기반 모델
- 증류 방법: 응답 기반 증류
- 성과: GPT-2의 90% 성능을 10배 작은 크기로 달성

### 4. 고급 PEFT (Parameter-Efficient Fine-Tuning) (20분)

#### 고급 LoRA 변형
**QLoRA (Quantized LoRA)**
- 원리: 양자화된 기본 모델 + LoRA 적응
- 방법: 4비트 양자화된 기본 모델에 LoRA 적용
- 특징: 매우 적은 메모리, 거의 원본 성능
- 장점: 소비자 GPU에서도 대규모 모델 미세조정 가능

**DoRA (Weight-Decomposed Low-Rank Adaptation)**
- 원리: 가중치를 방향과 크기로 분해하여 LoRA 적용
- 방법: $W = \text{diag}(s) \cdot V$, $W' = W + \text{diag}(s') \cdot V'$
- 특징: 더 풍부한 표현, 안정적인 학습
- 장점: 더 나은 표현력, 안정적인 수렴

**LoRA+ (LoRA Plus)**
- 원리: 여러 LoRA 어댑터의 결합
- 방법: $W' = W + \sum_{i} B_i A_i$
- 특징: 더 풍부한 적응, 다양한 특성 학습
- 장점: 더 나은 성능, 유연한 적응

#### 고급 어댑터 기법
**(IA)^3 (Infused Adapter by Inhibiting and Amplifying)**
- 원리: 어댑터의 활성화와 억제 메커니즘
- 방법: 게이트된 어댑터, 활성화 함수 제어
- 특징: 동적 특성 선택, 효율적 계산
- 장점: 더 나은 성능, 적은 파라미터

**Compacter (Composition of Adapters)**
- 원리: 여러 어댑터의 조합으로 복잡한 함수 근사
- 방법: $f(x) = \sum_{i} \sigma_i(x) \cdot \text{adapter}_i(x)$
- 특징: 복잡한 함수 근사, 동적 조합
- 장점: 더 나은 표현력, 복잡한 관계 모델링

**AdapterDrop**
- 원리: 훈련 중 무작위 어댑터 드롭아웃
- 방법: 각 순전파에서 일부 어댑터 비활성화
- 특징: 앙상블 효과, 과적합 방지
- 장점: 더 나은 일반화, 안정적인 학습

#### 고급 프롬프트 튜닝
**P-Tuning v2**
- 원리: 입력 임베딩 공간에서의 직접 최적화
- 방법: 학습 가능한 프롬프트 임베딩 + 프롬프트 인코딩
- 특징: 더 효율적인 최적화, 빠른 수렴
- 장점: 매우 적은 파라미터, 빠른 전환

**Prefix-Tuning with Reparameterization**
- 원리: 재매개변수화를 통한 프리픽스 튜닝 최적화
- 방법: $P = W_P \cdot \text{embed}(P_{\text{raw}})$
- 특징: 더 안정적인 최적화, 더 나은 표현
- 장점: 더 나은 성능, 안정적인 학습

**Multi-Modal Prompt Tuning**
- 원리: 여러 모달리티에 대한 프롬프트 튜닝
- 방법: 텍스트, 이미지, 오디오 등에 대한 프롬프트 임베딩
- 특징: 다중 모달리티 통합, 멀티모달 이해
- 장점: 다양한 입력 타입 처리, 통합된 표현 학습

## 실습 세션 (90분)

### 1. 다중 작업 학습 구현 (30분)

#### 다중 작업 LLM 구현
```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json

class MultiTaskLLM(nn.Module):
    def __init__(self, base_model_name, tasks, shared_layers=6):
        super(MultiTaskLLM, self).__init__()
        
        self.tasks = tasks
        self.shared_layers = shared_layers
        
        # 공유 인코더 (사전 훈련된 모델)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 공유 인코더 레이어 분리
        self.shared_encoder = nn.ModuleList([
            self.base_model.transformer.h[i] for i in range(shared_layers)
        ])
        
        # 작업별 어댑터
        self.task_adapters = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.base_model.config.hidden_size)
            ) for task in tasks
        })
        
        # 작업별 출력 헤드
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(self.base_model.config.hidden_size, self.tokenizer.vocab_size)
            for task in tasks
        })
    
    def forward(self, input_ids, task_id, attention_mask=None):
        # 토큰 임베딩
        embeddings = self.base_model.transformer.wte(input_ids)
        
        # 공유 인코더 통과
        hidden_states = embeddings
        for layer in self.shared_encoder:
            hidden_states = layer(hidden_states)[0]
        
        # 작업별 어댑터 적용
        task_name = self.tasks[task_id]
        adapted_states = self.task_adapters[task_name](hidden_states)
        
        # 작업별 출력 헤드
        logits = self.task_heads[task_name](adapted_states)
        
        return logits

# 다중 작업 데이터셋
class MultiTaskDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 작업 ID 매핑
        self.task_to_id = {task: i for i, task in enumerate(self.data.keys())}
        self.id_to_task = {i: task for task, i in self.task_to_id.items()}
    
    def __len__(self):
        return sum(len(task_data) for task_data in self.data.values())
    
    def __getitem__(self, idx):
        # 작업과 샘플 인덱스 찾기
        task_counts = [len(task_data) for task_data in self.data.values()]
        cumulative_counts = [sum(task_counts[:i+1]) for i in range(len(task_counts))]
        
        task_id = 0
        for i, count in enumerate(cumulative_counts):
            if idx < count:
                task_id = i
                break
        
        # 해당 작업 내에서의 상대적 인덱스
        if task_id > 0:
            prev_count = cumulative_counts[task_id-1]
        else:
            prev_count = 0
        
        sample_idx = idx - prev_count
        task_name = self.id_to_task[task_id]
        sample = self.data[task_name][sample_idx]
        
        # 토큰화
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'task_id': task_id
        }

# 다중 작업 데이터 생성
def create_multi_task_data():
    """다중 작업 데이터 생성"""
    
    tasks = ['generation', 'classification', 'summarization']
    data = {task: [] for task in tasks}
    
    # 생성 작업 데이터
    for i in range(50):
        data['generation'].append({
            'text': f"This is a sample generation text number {i+1}."
        })
    
    # 분류 작업 데이터
    labels = ['positive', 'negative', 'neutral']
    for i in range(50):
        label = labels[i % len(labels)]
        data['classification'].append({
            'text': f"This is a {label} sentiment text number {i+1}."
        })
    
    # 요약 작업 데이터
    for i in range(50):
        long_text = "This is a very long text that needs to be summarized. " * 20
        data['summarization'].append({
            'text': f"{long_text} Summary number {i+1}."
        })
    
    return data

# 다중 작업 데이터 생성
multi_task_data = create_multi_task_data()
with open('multi_task_data.json', 'w', encoding='utf-8') as f:
    json.dump(multi_task_data, f, ensure_ascii=False, indent=2)

# 다중 작업 모델 훈련
def train_multi_task_model():
    """다중 작업 모델 훈련"""
    
    # 모델 초기화
    tasks = list(multi_task_data.keys())
    model = MultiTaskLLM('microsoft/DialoGPT-medium', tasks)
    
    # 데이터셋
    dataset = MultiTaskDataset('multi_task_data.json', model.tokenizer)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    # 훈련 루프
    model.train()
    for epoch in range(10):
        total_loss = 0
        num_batches = 0
        
        for batch_idx in range(0, len(dataset), 32):
            batch = []
            for i in range(batch_idx, min(batch_idx + 32, len(dataset))):
                batch.append(dataset[i])
            
            if not batch:
                continue
            
            # 배치 데이터 준비
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            task_ids = torch.tensor([item['task_id'] for item in batch])
            
            # 순전파
            optimizer.zero_grad()
            logits = model(input_ids, task_ids, attention_mask)
            
            # 손실 계산 (작업별로)
            loss = 0
            for i, task_id in enumerate(task_ids):
                task_logits = logits[i:i+1]  # 각 샘플에 대한 해당 작업 로짓
                task_labels = labels[i]
                task_loss = criterion(task_logits, task_labels)
                loss += task_loss
            
            loss = loss / len(task_ids)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    return model

# 다중 작업 모델 훈련
multi_task_model = train_multi_task_model()
```

#### 다중 작업 모델 평가
```python
def evaluate_multi_task_model(model, test_data_path):
    """다중 작업 모델 평가"""
    
    # 테스트 데이터셋
    test_dataset = MultiTaskDataset(test_data_path, model.tokenizer)
    
    # 작업별 평가
    task_results = {}
    
    for task_id, task_name in enumerate(model.tasks):
        task_samples = [sample for sample in test_dataset if sample['task_id'] == task_id]
        
        if not task_samples:
            continue
        
        # 배치 평가
        batch_size = 16
        total_loss = 0
        num_batches = 0
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(task_samples), batch_size):
                batch = task_samples[i:i+batch_size]
                
                if not batch:
                    continue
                
                # 배치 데이터 준비
                input_ids = torch.stack([item['input_ids'] for item in batch])
                attention_mask = torch.stack([item['attention_mask'] for item in batch])
                labels = torch.stack([item['labels'] for item in batch])
                task_ids = torch.tensor([task_id] * len(batch))
                
                # 순전파
                logits = model(input_ids, task_ids, attention_mask)
                
                # 손실 계산
                loss = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)(
                    logits.squeeze(), labels.squeeze()
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        task_results[task_name] = {
            'loss': avg_loss,
            'perplexity': perplexity.item(),
            'num_samples': len(task_samples)
        }
    
    return task_results

# 다중 작업 모델 평가
test_multi_task_data = create_multi_task_data()  # 실제로는 별도 테스트 데이터
with open('test_multi_task_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_multi_task_data, f, ensure_ascii=False, indent=2)

evaluation_results = evaluate_multi_task_model(multi_task_model, 'test_multi_task_data.json')

# 결과 출력
print("=== 다중 작업 모델 평가 결과 ===")
for task_name, results in evaluation_results.items():
    print(f"Task: {task_name}")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.4f}")
    print(f"  Samples: {results['num_samples']}")
    print()
```

### 2. 도메인 적응 구현 (30분)

#### 도메인 특화 어댑터 구현
```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class DomainAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_size, dropout=0.1):
        super(DomainAdapter, self).__init__()
        
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        down = self.down_project(x)
        act = self.activation(down)
        up = self.up_project(act)
        output = self.dropout(up)
        
        # 잔차 연결
        return x + output

class DomainAdaptedLLM(nn.Module):
    def __init__(self, base_model_name, domains):
        super(DomainAdaptedLLM, self).__init__()
        
        self.domains = domains
        
        # 기본 모델
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 도메인별 어댑터
        self.domain_adapters = nn.ModuleDict({
            domain: DomainAdapter(
                self.base_model.config.hidden_size, 
                adapter_size=64
            ) for domain in domains
        })
        
        # 도메인 분류기
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, len(domains)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input_ids, domain_id=None, attention_mask=None):
        # 기본 모델 순전파
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 마지막 레이어
        
        # 도메인 어댑터 적용
        if domain_id is not None:
            domain_name = self.domains[domain_id]
            adapted_states = self.domain_adapters[domain_name](hidden_states)
        else:
            # 도메인 자동 분류
            domain_logits = self.domain_classifier(hidden_states.mean(dim=1))
            domain_id = torch.argmax(domain_logits, dim=-1)
            domain_name = self.domains[domain_id]
            
            # 가장 가능성 높은 도메인 어댑터 적용
            adapted_states = self.domain_adapters[domain_name](hidden_states)
            
            # 도메인 분류 결과 저장
            self.predicted_domain_id = domain_id
        
        # 출력 레이어
        logits = self.base_model.lm_head(adapted_states)
        
        return logits

# 도메인 적응 데이터 생성
def create_domain_adaptation_data():
    """도메인 적응 데이터 생성"""
    
    domains = ['general', 'medical', 'legal', 'technical']
    data = {domain: [] for domain in domains}
    
    # 일반 도메인 데이터
    for i in range(100):
        data['general'].append({
            'text': f"This is a general domain text sample number {i+1}."
        })
    
    # 의료 도메인 데이터
    medical_terms = ['diagnosis', 'treatment', 'symptom', 'patient', 'medication', 'therapy', 'disease', 'clinical', 'prescription', 'diagnosis']
    for i in range(100):
        term1 = medical_terms[i % len(medical_terms)]
        term2 = medical_terms[(i+1) % len(medical_terms)]
        data['medical'].append({
            'text': f"The patient presents with {term1} and requires {term2}."
        })
    
    # 법률 도메인 데이터
    legal_terms = ['contract', 'liability', 'plaintiff', 'defendant', 'jurisdiction', 'statute', 'precedent', 'litigation', 'settlement', 'verdict']
    for i in range(100):
        term1 = legal_terms[i % len(legal_terms)]
        term2 = legal_terms[(i+1) % len(legal_terms)]
        data['legal'].append({
            'text': f"The {term1} files a lawsuit against the {term2} in this jurisdiction."
        })
    
    # 기술 도메인 데이터
    tech_terms = ['algorithm', 'implementation', 'optimization', 'performance', 'database', 'interface', 'framework', 'library', 'function', 'variable', 'method']
    for i in range(100):
        term1 = tech_terms[i % len(tech_terms)]
        term2 = tech_terms[(i+1) % len(tech_terms)]
        data['technical'].append({
            'text': f"The {term1} improves the {term2} performance in our system."
        })
    
    return data

# 도메인 적응 데이터 생성
domain_data = create_domain_adaptation_data()
with open('domain_adaptation_data.json', 'w', encoding='utf-8') as f:
    json.dump(domain_data, f, ensure_ascii=False, indent=2)

# 도메인 적응 모델 훈련
def train_domain_adapted_model():
    """도메인 적응 모델 훈련"""
    
    # 모델 초기화
    domains = list(domain_data.keys())
    model = DomainAdaptedLLM('microsoft/DialoGPT-medium', domains)
    
    # 통합 데이터셋
    all_data = []
    domain_ids = []
    
    for domain_id, (domain_name, domain_samples) in enumerate(domain_data.items()):
        for sample in domain_samples:
            all_data.append(sample['text'])
            domain_ids.append(domain_id)
    
    # 토큰화
    encodings = model.tokenizer(
        all_data,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 훈련 준비
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = input_ids.clone()
    domain_ids = torch.tensor(domain_ids)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    # 훈련 루프
    model.train()
    for epoch in range(10):
        total_loss = 0
        
        # 배치 처리
        for i in range(0, len(input_ids), 32):
            batch_input_ids = input_ids[i:i+32]
            batch_attention_mask = attention_mask[i:i+32]
            batch_labels = labels[i:i+32]
            batch_domain_ids = domain_ids[i:i+32]
            
            optimizer.zero_grad()
            
            # 순전파
            logits = model(batch_input_ids, batch_domain_ids, batch_attention_mask)
            
            # 손실 계산
            loss = criterion(logits.view(-1, logits.size(-1)), batch_labels.view(-1))
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(input_ids) // 32)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return model

# 도메인 적응 모델 훈련
domain_adapted_model = train_domain_adapted_model()
```

#### 도메인 적응 모델 평가
```python
def evaluate_domain_adapted_model(model, test_data_path):
    """도메인 적응 모델 평가"""
    
    # 테스트 데이터셋
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 도메인별 평가
    domain_results = {}
    
    for domain_id, (domain_name, domain_samples) in enumerate(test_data.items()):
        if not domain_samples:
            continue
        
        # 토큰화
        texts = [sample['text'] for sample in domain_samples]
        encodings = model.tokenizer(
            texts,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        domain_ids = torch.tensor([domain_id] * len(texts))
        
        # 평가
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, domain_ids, attention_mask)
            
            # 손실 계산
            loss = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)(
                logits.view(-1, logits.size(-1)), 
                input_ids.view(-1)
            )
            
            # 도메인 분류 정확도
            domain_logits = model.domain_classifier(
                model.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].mean(dim=1)
            predicted_domain_ids = torch.argmax(domain_logits, dim=-1)
            domain_accuracy = (predicted_domain_ids == domain_ids).float().mean().item()
        
        domain_results[domain_name] = {
            'loss': loss.item(),
            'domain_accuracy': domain_accuracy,
            'num_samples': len(texts)
        }
    
    return domain_results

# 도메인 적응 모델 평가
test_domain_data = create_domain_adaptation_data()  # 실제로는 별도 테스트 데이터
with open('test_domain_adaptation_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_domain_data, f, ensure_ascii=False, indent=2)

domain_evaluation_results = evaluate_domain_adapted_model(domain_adapted_model, 'test_domain_adaptation_data.json')

# 결과 출력
print("=== 도메인 적응 모델 평가 결과 ===")
for domain_name, results in domain_evaluation_results.items():
    print(f"Domain: {domain_name}")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Domain Accuracy: {results['domain_accuracy']:.4f}")
    print(f"  Samples: {results['num_samples']}")
    print()
```

### 3. 지식 증류 구현 (30분)

#### 응답 기반 지식 증류 구현
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # 온도 스케일링
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 증류 손실
        distill_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 학생 손실
        student_loss = F.cross_entropy(student_logits, labels)
        
        # 결합 손실
        loss = self.alpha * distill_loss + (1.0 - self.alpha) * student_loss
        
        return loss

class KnowledgeDistillation:
    def __init__(self, teacher_model_name, student_model_name):
        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        
        # 증류 손실
        self.distillation_loss = DistillationLoss()
        
        # 옵티마이저
        self.teacher_optimizer = optim.Adam(self.teacher_model.parameters(), lr=1e-5)
        self.student_optimizer = optim.Adam(self.student_model.parameters(), lr=1e-4)
    
    def train(self, train_data, epochs=5):
        """지식 증류 훈련"""
        
        # 데이터 준비
        texts = [sample['text'] for sample in train_data]
        encodings = self.tokenizer(
            texts,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = input_ids.clone()
        
        # 훈련 루프
        for epoch in range(epochs):
            self.teacher_model.train()
            self.student_model.train()
            
            total_loss = 0
            
            # 배치 처리
            for i in range(0, len(input_ids), 16):
                batch_input_ids = input_ids[i:i+16]
                batch_attention_mask = attention_mask[i:i+16]
                batch_labels = labels[i:i+16]
                
                # 교사 순전파
                self.teacher_optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = self.teacher_model(
                        batch_input_ids, 
                        attention_mask=batch_attention_mask
                    ).logits
                
                # 학생 순전파
                self.student_optimizer.zero_grad()
                student_logits = self.student_model(
                    batch_input_ids, 
                    attention_mask=batch_attention_mask
                ).logits
                
                # 증류 손실 계산
                loss = self.distillation_loss(
                    student_logits, teacher_logits, batch_labels
                )
                
                # 역전파
                loss.backward()
                self.student_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(input_ids) // 16)
            print(f"Epoch {epoch+1}, Distillation Loss: {avg_loss:.4f}")
        
        return self.student_model

# 지식 증류 데이터 생성
def create_distillation_data():
    """지식 증류 데이터 생성"""
    
    data = []
    for i in range(200):
        data.append({
            'text': f"This is a sample text for knowledge distillation number {i+1}."
        })
    
    return data

# 지식 증류 훈련
distillation_data = create_distillation_data()
with open('distillation_data.json', 'w', encoding='utf-8') as f:
    json.dump(distillation_data, f, ensure_ascii=False, indent=2)

# 지식 증류 실행
distillation = KnowledgeDistillation(
    'microsoft/DialoGPT-medium',  # 교사
    'microsoft/DialoGPT-small'    # 학생
)

student_model = distillation.train(distillation_data)
student_model.save_pretrained('./distilled_model')
```

#### 지식 증류 모델 평가
```python
def evaluate_distillation(teacher_model, student_model, test_data):
    """지식 증류 모델 평가"""
    
    # 테스트 데이터 준비
    texts = [sample['text'] for sample in test_data]
    encodings = teacher_model.tokenizer(
        texts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = input_ids.clone()
    
    # 평가
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        # 교사 평가
        teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits
        teacher_loss = F.cross_entropy(teacher_logits.view(-1, teacher_logits.size(-1)), labels.view(-1))
        teacher_perplexity = torch.exp(teacher_loss)
        
        # 학생 평가
        student_logits = student_model(input_ids, attention_mask=attention_mask).logits
        student_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        student_perplexity = torch.exp(student_loss)
        
        # 증류 품질
        kl_div = F.kl_div(
            F.log_softmax(student_logits / 4.0, dim=-1),
            F.softmax(teacher_logits / 4.0, dim=-1),
            reduction='batchmean'
        ) * (4.0 ** 2)
    
    results = {
        'teacher_loss': teacher_loss.item(),
        'teacher_perplexity': teacher_perplexity.item(),
        'student_loss': student_loss.item(),
        'student_perplexity': student_perplexity.item(),
        'kl_divergence': kl_div.item(),
        'num_samples': len(texts)
    }
    
    return results

# 지식 증류 평가
test_distillation_data = create_distillation_data()  # 실제로는 별도 테스트 데이터
with open('test_distillation_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_distillation_data, f, ensure_ascii=False, indent=2)

# 평가 실행
teacher_model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
student_model = AutoModelForCausalLM.from_pretrained('./distilled_model')

distillation_results = evaluate_distillation(teacher_model, student_model, test_distillation_data)

# 결과 출력
print("=== 지식 증류 평가 결과 ===")
print(f"Teacher Loss: {distillation_results['teacher_loss']:.4f}")
print(f"Teacher Perplexity: {distillation_results['teacher_perplexity']:.4f}")
print(f"Student Loss: {distillation_results['student_loss']:.4f}")
print(f"Student Perplexity: {distillation_results['student_perplexity']:.4f}")
print(f"KL Divergence: {distillation_results['kl_divergence']:.4f}")
print(f"Samples: {distillation_results['num_samples']}")
```

### 4. 고급 PEFT 구현 (30분)

#### QLoRA 구현
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

class QLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16, lora_dropout=0.1):
        super(QLoRALayer, self).__init__()
        
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # LoRA 파라미터
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 가중치 초기화
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # LoRA 경로
        lora_output = self.lora_B(self.lora_dropout(self.lora_A(x)))
        
        # 스케일링된 LoRA 출력
        return self.scaling * lora_output

class QLoRAModel(nn.Module):
    def __init__(self, base_model_name, rank=8, lora_alpha=16):
        super(QLoRAModel, self).__init__()
        
        # 4비트 양자화된 기본 모델
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            device_map='auto'
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # QLoRA 레이어
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        # 모든 선형 레이어에 QLoRA 적용
        self.q_lora_layers = nn.ModuleDict()
        self.k_lora_layers = nn.ModuleDict()
        self.v_lora_layers = nn.ModuleDict()
        self.o_lora_layers = nn.ModuleDict()
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                in_features = module.in_features
                out_features = module.out_features
                
                # QLoRA 레이어 생성
                self.q_lora_layers[name] = QLoRALayer(in_features, out_features, rank, lora_alpha)
                self.k_lora_layers[name] = QLoRALayer(in_features, out_features, rank, lora_alpha)
                self.v_lora_layers[name] = QLoRALayer(in_features, out_features, rank, lora_alpha)
                self.o_lora_layers[name] = QLoRALayer(in_features, out_features, rank, lora_alpha)
                
                # 원본 가중치 저장
                self.q_lora_layers[name].weight = module.weight
                self.k_lora_layers[name].weight = module.weight
                self.v_lora_layers[name].weight = module.weight
                self.o_lora_layers[name].weight = module.weight
    
    def forward(self, input_ids, attention_mask=None):
        # 기본 모델 순전파
        outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        
        # QLoRA 적용
        # 각 트랜스포머 블록에 대해
        for i, block in enumerate(self.base_model.transformer.h):
            # 어텐션 레이어
            q_proj_name = f'transformer.h.{i}.attn.q_proj'
            k_proj_name = f'transformer.h.{i}.attn.k_proj'
            v_proj_name = f'transformer.h.{i}.attn.v_proj'
            o_proj_name = f'transformer.h.{i}.attn.o_proj'
            
            if q_proj_name in self.q_lora_layers:
                block.attn.q_proj = self.q_lora_layers[q_proj_name]
            if k_proj_name in self.k_lora_layers:
                block.attn.k_proj = self.k_lora_layers[k_proj_name]
            if v_proj_name in self.v_lora_layers:
                block.attn.v_proj = self.v_lora_layers[v_proj_name]
            if o_proj_name in self.o_lora_layers:
                block.attn.o_proj = self.o_lora_layers[o_proj_name]
        
        # 수정된 모델로 다시 순전파
        outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask
        )
        
        return outputs.logits

# QLoRA 모델 훈련
def train_qlora_model():
    """QLoRA 모델 훈련"""
    
    # 모델 초기화
    model = QLoRAModel('microsoft/DialoGPT-medium', rank=8, lora_alpha=16)
    
    # 훈련 데이터
    train_data = create_distillation_data()  # 재사용
    texts = [sample['text'] for sample in train_data]
    encodings = model.tokenizer(
        texts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = input_ids.clone()
    
    # 옵티마이저 (QLoRA 파라미터만)
    optimizer = optim.Adam([
        p for n, p in model.named_parameters() 
        if 'lora' in n.lower()
    ], lr=1e-4)
    
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    # 훈련 루프
    model.train()
    for epoch in range(5):
        total_loss = 0
        
        # 배치 처리
        for i in range(0, len(input_ids), 16):
            batch_input_ids = input_ids[i:i+16]
            batch_attention_mask = attention_mask[i:i+16]
            batch_labels = labels[i:i+16]
            
            optimizer.zero_grad()
            
            # 순전파
            logits = model(batch_input_ids, batch_attention_mask)
            
            # 손실 계산
            loss = criterion(logits.view(-1, logits.size(-1)), batch_labels.view(-1))
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(input_ids) // 16)
        print(f"Epoch {epoch+1}, QLoRA Loss: {avg_loss:.4f}")
    
    return model

# QLoRA 모델 훈련
qlora_model = train_qlora_model()
qlora_model.save_pretrained('./qlora_model')
```

#### QLoRA 모델 평가
```python
def evaluate_qlora_model(model, test_data):
    """QLoRA 모델 평가"""
    
    # 테스트 데이터 준비
    texts = [sample['text'] for sample in test_data]
    encodings = model.tokenizer(
        texts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = input_ids.clone()
    
    # 평가
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss)
    
    # QLoRA 파라미터 수 계산
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n.lower())
    total_params = sum(p.numel() for n, p in model.named_parameters())
    
    results = {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'lora_parameters': lora_params,
        'total_parameters': total_params,
        'lora_ratio': lora_params / total_params * 100,
        'num_samples': len(texts)
    }
    
    return results

# QLoRA 모델 평가
test_qlora_data = create_distillation_data()  # 재사용
with open('test_qlora_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_qlora_data, f, ensure_ascii=False, indent=2)

qlora_results = evaluate_qlora_model(qlora_model, test_qlora_data)

# 결과 출력
print("=== QLoRA 모델 평가 결과 ===")
print(f"Loss: {qlora_results['loss']:.4f}")
print(f"Perplexity: {qlora_results['perplexity']:.4f}")
print(f"LoRA Parameters: {qlora_results['lora_parameters']:,}")
print(f"Total Parameters: {qlora_results['total_parameters']:,}")
print(f"LoRA Ratio: {qlora_results['lora_ratio']:.2f}%")
print(f"Samples: {qlora_results['num_samples']}")
```

## 과제

### 1. 다중 작업 학습 과제
- 3개 이상의 다른 작업에 대한 다중 작업 LLM 구현
- 작업 간 전이 학습 효과 분석
- 작업별 데이터 불균형 문제 해결 방안 연구

### 2. 도메인 적응 과제
- 3개 이상의 다른 도메인에 대한 적응 실험
- 도메인 분류기와 어댑터의 성능 비교
- 도메인 간 전이 학습 효과 측정

### 3. 지식 증류 과제
- 다양한 증류 방법(응답, 특성, 관계 기반) 구현과 비교
- 다단계 증류 실험
- 증류 효율성과 성능 품질의 트레이드오프 분석

### 4. 고급 PEFT 과제
- QLoRA, DoRA, LoRA+ 등 고급 PEFT 방법 구현과 비교
- PEFT 방법의 조합 실험
- 메모리 사용량과 성능의 정량적 분석

## 추가 학습 자료

### 논문
- "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al., 2018)
- "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)
- "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

### 온라인 자료
- [Hugging Face PEFT Library Documentation](https://huggingface.co/docs/peft/)
- [Multi-Task Learning Tutorial](https://pytorch.org/tutorials/intermediate/multi_task_learning_tutorial.html)
- [Knowledge Distillation Tutorial](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)

### 구현 참고
- [AdapterHub](https://github.com/adapter-hub/adapter-transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)