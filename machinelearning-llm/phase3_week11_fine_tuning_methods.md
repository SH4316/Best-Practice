# 11주차: 미세조정 방법론

## 강의 목표
- 지도 미세조정(SFT)의 원리와 구현 방법 이해
- 인간 피드백을 통한 강화 학습(RLHF)의 동작 원리와 단계 습득
- 매개변수 효율적 미세조정(PEFT)의 종류와 장단점 파악
- 다양한 미세조정 방법의 적용 사례와 성능 특성 학습
- 실제 LLM 미세조정 프로젝트 수행 능력 배양

## 이론 강의 (90분)

### 1. 지도 미세조정 (Supervised Fine-Tuning, SFT) (25분)

#### SFT의 기본 원리
**정의와 목적**
- 정의: 레이블된 데이터를 이용한 사전 훈련된 모델의 미세조정
- 목적: 특정 작업이나 도메인에 모델을 적응시키기
- 특징: 지도 학습의 안정성과 사전 훈련 모델의 지식 결합

**데이터 요구사항**
- 고품질 지시-응답 쌍: (입력, 출력) 형태의 데이터
- 다양성: 다양한 유형의 지시와 응답 포함
- 양: 수천에서 수만 개의 지시-응답 쌍 필요
- 품질: 인간이 생성한 고품질 데이터 선호

**훈련 과정**
1. **데이터 준비**: 지시-응답 쌍을 (입력, 목표) 형태로 변환
2. **모델 로드**: 사전 훈련된 LLM 로드
3. **미세조정**: 지시 데이터로 모델 훈련
4. **평가**: 별도의 평가 데이터로 성능 측정

#### SFT의 장단점
**장점**
- 안정성: 지도 학습의 안정적인 훈련 과정
- 효율성: 사전 훈련 지식 활용으로 적은 데이터로도 좋은 성능
- 단순성: 표준 지도 학습 파이프라인 적용 가능
- 재현성: 결정론적 과정으로 재현성 높음

**단점**
- 데이터 의존성: 고품질 지시-응답 데이터에 크게 의존
- 과적합 위험: 특정 데이터에 과적합될 수 있음
- 창의성 제한: 훈련 데이터에 있는 패턴만 학습
- 비용: 고품질 데이터 생성에 많은 비용 소요

#### SFT의 실제 적용
**InstructGPT**
- 데이터: 인간이 작성한 지시-응답 쌍
- 방법: GPT-3 모델을 지시 데이터로 미세조정
- 결과: 더 나은 지시 따르기 능력

**ChatGPT 초기 버전**
- 데이터: 대화 형식의 지시-응답 데이터
- 방법: 대화 형식에 맞춰 미세조정
- 결과: 대화 능력 향상

**도메인 특화 모델**
- 의료 LLM: 의료 지시-응답 데이터로 미세조정
- 법률 LLM: 법률 지시-응답 데이터로 미세조정
- 코드 LLM: 코드 생성 지시-응답 데이터로 미세조정

### 2. 인간 피드백을 통한 강화 학습 (RLHF) (30분)

#### RLHF의 기본 개념
**정의와 목적**
- 정의: 인간 피드백을 보상 신호로 활용한 강화 학습
- 목적: 인간의 선호도에 맞춰 모델 출력 조정
- 특징: 지도 학습과 강화 학습의 결합

**핵심 구성 요소**
1. **지시 미세조정 모델(SFT Model)**: 기본 LLM
2. **보상 모델(Reward Model)**: 인간 선호도 예측
3. **강화 학습 알고리즘**: PPO (Proximal Policy Optimization)

#### RLHF의 3단계 과정

**1단계: 지도 미세조정 (SFT)**
- 목적: 기본 지시 따르기 능력 확보
- 데이터: 지시-응답 쌍 데이터
- 결과: SFT 모델 (RLHF의 초기 모델)

**2단계: 보상 모델 훈련**
- 목적: 인간 선호도를 예측하는 보상 모델 학습
- 데이터: 모델 출력 쌍에 대한 인간 선호도 레이블
- 방법: 순위 데이터로 보상 모델 훈련

**보상 모델 훈련 과정**
1. **응답 생성**: SFT 모델로 여러 응답 생성
2. **인간 평가**: 생성된 응답의 선호도 순위 매김
3. **보상 모델 훈련**: 순위 데이터로 보상 모델 학습

**3단계: 강화 학습 미세조정**
- 목적: 보상 모델을 이용해 LLM을 인간 선호도에 맞춰 조정
- 알고리즘: PPO (Proximal Policy Optimization)
- 보상: 보상 모델이 제공하는 보상 신호

**PPO 알고리즘의 핵심**
- 정책 그래디언트: $\nabla_\theta J(\theta)$
- KL 발산 제약: $\text{KL}[\pi_\theta | \pi_{\theta_{old}}] \leq \delta$
- 클리핑: 그래디언트 클리핑으로 안정성 확보
- 적응적 크리핑: 신뢰 구간을 벗어나는 그래디언트 클리핑

#### RLHF의 장단점
**장점**
- 인간 선호도 반영: 수치적 성능뿐만 아니라 인간 선호도 고려
- 출력 품질 향상: 더 유용하고 안전한 출력 생성
- 미세 조정: 세밀한 출력 조정 가능
- 안전성: 유해한 출력 감소

**단점**
- 복잡성: 3단계 과정으로 구현 복잡
- 불안정성: 강화 학습의 불안정성 문제
- 비용: 인간 평가에 많은 비용 소요
- 편향성: 평가자의 편향이 모델에 반영될 수 있음

#### RLHF의 실제 적용
**InstructGPT**
- 과정: SFT → 보상 모델 훈련 → PPO 미세조정
- 결과: 인간 선호도에 맞춘 출력 생성
- 영향: ChatGPT의 기술적 기반

**ChatGPT**
- 발전: InstructGPT에서 더 발전된 RLHF 적용
- 특징: 대화 능력, 안전성, 유용성 향상
- 영향: 대화형 AI의 표준으로 자리 잡음

**Claude**
- 차별점: Constitutional AI 접근법
- 특징: 헌법에 기반한 자기 수정 메커니즘
- 안전성: 더 강력한 안전장치 내장

### 3. 매개변수 효율적 미세조정 (PEFT) (20분)

#### PEFT의 기본 원리
**정의와 목적**
- 정의: 소수의 매개변수만 미세조정하고 대부분은 고정
- 목적: 대규모 모델의 효율적인 미세조정
- 특징: 적은 저장 공간과 계산 비용으로 미세조정 가능

**핵심 아이디어**
- 저차원 적응: 고차원 파라미터 공간을 저차원으로 투영
- 적응층 추가: 원본 모델에 작은 적응층 추가
- 효율성: 전체 파라미터의 일부만 업데이트

#### 주요 PEFT 방법론

**LoRA (Low-Rank Adaptation)**
- 원리: 가중치 업데이트를 저차원 행렬로 근사
- 수식: $W' = W + BA$ (B는 $d \times r$, A는 $r \times k$)
- 특징: $r \ll \min(d,k)$로 매우 적은 파라미터 추가
- 장점: 메모리 효율성, 빠른 전환, 원본 모델 보존

**QLoRA (Quantized LoRA)**
- 원리: LoRA와 양자화 결합
- 방법: 4비트 양자화된 기본 모델 + LoRA 적응
- 특징: 더 적은 메모리 사용, 거의 원본 성능
- 장점: 소비자 GPU에서도 대규모 모델 미세조정 가능

**어댑터 레이어 (Adapter Layers)**
- 원리: 트랜스포머 블록 사이에 작은 신경망 레이어 추가
- 구조: 입력 → 어댑터 → 출력
- 특징: 원본 모델 구조 변경 없음
- 장점: 유연성, 다양한 작업에 적응 가능

**프리픽스 튜닝 (Prefix Tuning)**
- 원리: 입력 시퀀스 앞에 학습 가능한 프리픽스 추가
- 방법: 프리픽스 토큰을 최적화하여 모델 동작 제어
- 특징: 모델 파라미터 직접 수정 없음
- 장점: 매우 적은 추가 파라미터

**프롬프트 튜닝 (Prompt Tuning)**
- 원리: 입력 임베딩 공간에서 직접 최적화
- 방법: 소프트 프롬프를 학습 가능한 파라미터로 표현
- 특징: 입력 공간에서의 미세조정
- 장점: 가장 적은 파라미터 추가

#### PEFT의 장단점
**장점**
- 메모리 효율성: 전체 모델의 일부만 저장
- 계산 효율성: 적은 파라미터만 업데이트
- 빠른 전환: 여러 작업 간 빠른 모델 전환
- 원본 모델 보존: 사전 훈련 지식 손실 최소화

**단점**
- 성능 한계: 전체 미세조정보다 성능이 낮을 수 있음
- 복잡성: 다양한 PEFT 방법 이해 필요
- 하이퍼파라미터: 최적의 PEFT 설정 찾기 어려움
- 호환성: 모든 PEFT 방법이 모든 모델과 호환되지 않음

#### PEFT의 실제 적용
**Alpaca**
- 방법: LLaMA 모델에 LoRA 적용
- 데이터: 52K 지시-응답 쌍
- 결과: 오픈 소스 ChatGPT 유사 모델

**Vicuna**
- 방법: LLaMA에 LoRA 적용, 사용자 대화 데이터로 훈련
- 특징: 대화 능력, 친절한 응답
- 영향: 오픈 소스 커뮤니티에 큰 영향

**GPT-4 All Tools**
- 방법: 다양한 PEFT 방법 조합
- 특징: 여러 도구 사용 능력
- 결과: 다양한 작업에 특화된 모델

### 4. 미세조정 방법론 비교 (15분)

#### 성능 특성 비교
**전체 미세조정 (Full Fine-Tuning)**
- 성능: 가장 높은 성능 잠재력
- 비용: 높은 메모리와 계산 비용
- 유연성: 모델의 모든 부분 수정 가능
- 적용: 전문 도메인, 최고 성능 요구

**PEFT (LoRA, QLoRA 등)**
- 성능: 전체 미세조정보다 약간 낮은 성능
- 비용: 매우 낮은 메모리와 계산 비용
- 유연성: 제한된 수정만 가능
- 적용: 다양한 작업, 자원 제한 환경

**RLHF**
- 성능: 인간 선호도에 맞춘 출력 품질
- 비용: 매우 높은 훈련 비용
- 유연성: 세밀한 출력 조정 가능
- 적용: 대화형 AI, 안전성 중요 애플리케이션

#### 선택 가이드라인
**자원 고려**
- GPU 메모리: 16GB 미만 → PEFT, 32GB+ → 전체 미세조정
- 저장 공간: 제한된 경우 → PEFT
- 계산 예산: 제한된 경우 → PEFT

**데이터 고려**
- 적은 데이터 (<1K): PEFT 또는 전체 미세조정
- 중간 데이터 (1K-10K): 전체 미세조정
- 큰 데이터 (>10K): 전체 미세조정 또는 RLHF

**작업 특성 고려**
- 단일 작업: PEFT로 충분
- 다중 작업: 여러 PEFT 모델 또는 전체 미세조정
- 도메인 특화: 전체 미세조정으로 더 나은 성능

**품질 요구사항**
- 최고 성능: 전체 미세조정
- 빠른 전환: PEFT
- 인간 선호도: RLHF

## 실습 세션 (90분)

### 1. 지도 미세조정 구현 (30분)

#### SFT 데이터 준비와 모델 미세조정
```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 데이터 포맷팅
        self.instructions = []
        self.inputs = []
        self.outputs = []
        
        for item in self.data:
            if 'instruction' in item and 'output' in item:
                instruction = item['instruction']
                input_text = item.get('input', '')
                output_text = item['output']
                
                # 지시와 입력 결합
                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
                
                self.instructions.append(prompt)
                self.outputs.append(output_text)
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        output = self.outputs[idx]
        
        # 입력 토큰화
        input_encoding = self.tokenizer(
            instruction,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 출력 토큰화
        output_encoding = self.tokenizer(
            output,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze()
        }

def create_sample_instruction_data(output_path, num_samples=100):
    """샘플 지시 데이터 생성"""
    import random
    
    templates = [
        "Translate the following text to {target_lang}: {input_text}",
        "Summarize the following text: {input_text}",
        "Answer the following question: {input_text}",
        "Complete the following sentence: {input_text}",
        "Generate a {genre} story about: {input_text}"
    ]
    
    target_languages = ['French', 'German', 'Spanish', 'Japanese', 'Korean']
    genres = ['science fiction', 'fantasy', 'mystery', 'romance', 'thriller']
    
    sample_inputs = [
        "The weather is nice today.",
        "Machine learning is a subset of artificial intelligence.",
        "The cat sat on the mat.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question."
    ]
    
    data = []
    
    for i in range(num_samples):
        template = random.choice(templates)
        input_text = random.choice(sample_inputs)
        
        if '{target_lang}' in template:
            target_lang = random.choice(target_languages)
            prompt = template.format(target_lang=target_lang, input_text=input_text)
            output = f"[Translation to {target_lang}] {input_text}"
        elif '{genre}' in template:
            genre = random.choice(genres)
            prompt = template.format(genre=genre, input_text=input_text)
            output = f"[{genre} story] Once upon a time, {input_text.lower()}."
        else:
            prompt = template.format(input_text=input_text)
            output = f"[Response] {input_text} This is a sample response."
        
        data.append({
            'instruction': prompt,
            'input': '',
            'output': output
        })
    
    # 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return data

# 샘플 데이터 생성
sample_data = create_sample_instruction_data('sample_instructions.json', 50)

# 토크나이저와 데이터셋 로드
model_name = "microsoft/DialoGPT-medium"  # 작은 모델로 시작
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = InstructionDataset('sample_instructions.json', tokenizer)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(model_name)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./sft_results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
)

# 트레이너 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # 실제로는 별도 평가 데이터 사용
)

# 미세조정 훈련
print("SFT 미세조정 시작...")
trainer.train()
print("SFT 미세조정 완료")

# 미세조정된 모델 저장
model.save_pretrained('./sft_model')
tokenizer.save_pretrained('./sft_model')
```

#### SFT 모델 평가
```python
def evaluate_sft_model(model_path, tokenizer_path, test_instructions):
    """SFT 모델 평가"""
    
    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    results = []
    
    for i, instruction in enumerate(test_instructions):
        # 입력 토큰화
        inputs = tokenizer(instruction, return_tensors='pt')
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            'instruction': instruction,
            'generated': generated_text
        })
        
        print(f"지시 {i+1}:")
        print(f"  입력: {instruction}")
        print(f"  출력: {generated_text}")
        print()
    
    return results

# 테스트 지시
test_instructions = [
    "Translate to French: Hello, how are you?",
    "Summarize: Machine learning enables computers to learn from data without being explicitly programmed.",
    "What is the capital of France?",
    "Complete: The quick brown fox jumps over"
]

# SFT 모델 평가
sft_results = evaluate_sft_model('./sft_model', './sft_model', test_instructions)
```

### 2. RLHF 기본 구현 (30분)

#### 보상 모델 훈련
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super(RewardModel, self).__init__()
        
        # 기본 모델 로드 (시퀀스 분류 모델)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1  # 회귀 출력 (보상 점수)
        )
        
    def forward(self, chosen_input_ids, rejected_input_ids, attention_mask=None):
        # 선호된 응답과 거부된 응답 처리
        chosen_outputs = self.base_model(
            input_ids=chosen_input_ids,
            attention_mask=attention_mask
        )
        
        rejected_outputs = self.base_model(
            input_ids=rejected_input_ids,
            attention_mask=attention_mask
        )
        
        # 보상 점수 추출
        chosen_reward = chosen_outputs.logits.squeeze(-1)
        rejected_reward = rejected_outputs.logits.squeeze(-1)
        
        return chosen_reward, rejected_reward

def create_preference_data(output_path, num_samples=50):
    """선호도 데이터 생성"""
    import random
    
    data = []
    
    for i in range(num_samples):
        # 랜덤 지시
        instruction = f"Instruction {i+1}: What is the capital of {random.choice(['France', 'Germany', 'Japan', 'Korea'])}?"
        
        # 두 개의 다른 응답
        good_answer = f"The capital is {random.choice(['Paris', 'Berlin', 'Tokyo', 'Seoul'])}."
        bad_answer = f"The capital is {random.choice(['London', 'Madrid', 'Beijing', 'Busan'])}."
        
        # 무작위 순서
        if random.random() < 0.5:
            chosen, rejected = good_answer, bad_answer
            chosen_score, rejected_score = 1.0, 0.0
        else:
            chosen, rejected = bad_answer, good_answer
            chosen_score, rejected_score = 0.0, 1.0
        
        data.append({
            'instruction': instruction,
            'chosen': chosen,
            'rejected': rejected,
            'chosen_score': chosen_score,
            'rejected_score': rejected_score
        })
    
    # 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return data

class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 지시와 응답 결합
        instruction = item['instruction']
        chosen_response = item['chosen']
        rejected_response = item['rejected']
        
        # 입력 포맷팅
        chosen_text = f"{instruction}\n\n{chosen_response}"
        rejected_text = f"{instruction}\n\n{rejected_response}"
        
        # 토큰화
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(),
            'attention_mask': chosen_encoding['attention_mask'].squeeze()
        }

# 선호도 데이터 생성
preference_data = create_preference_data('sample_preferences.json', 30)

# 보상 모델 훈련
def train_reward_model():
    """보상 모델 훈련"""
    
    # 데이터셋과 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = PreferenceDataset('sample_preferences.json', tokenizer)
    reward_model = RewardModel(model_name)
    
    # 옵티마이저
    optimizer = optim.Adam(reward_model.parameters(), lr=1e-5)
    
    # 훈련 루프
    reward_model.train()
    for epoch in range(5):
        total_loss = 0
        
        for batch in dataset:
            optimizer.zero_grad()
            
            chosen_reward, rejected_reward = reward_model(
                batch['chosen_input_ids'].unsqueeze(0),
                batch['rejected_input_ids'].unsqueeze(0),
                batch['attention_mask'].unsqueeze(0)
            )
            
            # 손실: 선호된 응답이 더 높은 보상을 받도록
            loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")
    
    return reward_model

# 보상 모델 훈련
reward_model = train_reward_model()
reward_model.save_pretrained('./reward_model')
```

#### PPO 기반 강화 학습
```python
class PPOTrainer:
    def __init__(self, model, reward_model, tokenizer, ref_model, kl_coef=0.1):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.ref_model = ref_model  # 참조 모델 (KL 발산 계산용)
        self.kl_coef = kl_coef
        
        # PPO 하이퍼파라미터
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.gamma = 1.0
        self.lam = 0.95
        self.learning_rate = 1e-6
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
    
    def generate_responses(self, instructions, max_length=100):
        """지시에 대한 응답 생성"""
        
        responses = []
        logprobs = []
        ref_logprobs = []
        
        self.model.eval()
        self.ref_model.eval()
        
        with torch.no_grad():
            for instruction in instructions:
                # 입력 토큰화
                inputs = self.tokenizer(instruction, return_tensors='pt')
                
                # 응답 생성
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=1.0,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # 생성된 토큰과 로그 확률
                generated_ids = outputs.sequences[0]
                generated_scores = outputs.scores[0]
                
                # 로그 확률 계산
                log_probs = F.log_softmax(generated_scores, dim=-1)
                
                # 참조 모델의 로그 확률
                ref_outputs = self.ref_model(**inputs)
                ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
                
                # 응답 디코딩
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                responses.append(response)
                logprobs.append(log_probs)
                ref_logprobs.append(ref_log_probs)
        
        return responses, logprobs, ref_logprobs
    
    def compute_rewards(self, instructions, responses):
        """보상 계산"""
        
        rewards = []
        
        self.reward_model.eval()
        with torch.no_grad():
            for instruction, response in zip(instructions, responses):
                # 입력과 응답 결합
                text = f"{instruction}\n\n{response}"
                
                # 토큰화
                inputs = self.tokenizer(text, return_tensors='pt')
                
                # 보상 점수 계산
                outputs = self.reward_model(**inputs)
                reward = outputs.logits.squeeze(-1).item()
                
                rewards.append(reward)
        
        return rewards
    
    def compute_advantages(self, rewards):
        """어드밴티지 계산"""
        
        rewards = torch.tensor(rewards)
        
        # 기준선 계산
        with torch.no_grad():
            values = rewards  # 단순화: 보상 자체를 값으로 사용
            advantages = rewards - values
        
        return advantages
    
    def ppo_step(self, instructions, responses, logprobs, ref_logprobs, rewards, advantages):
        """PPO 단계 수행"""
        
        self.model.train()
        
        # 배치로 변환
        for i, (instruction, response, logprob, ref_logprob, reward, advantage) in enumerate(
            zip(instructions, responses, logprobs, ref_logprobs, rewards, advantages)
        ):
            # 입력 토큰화
            inputs = self.tokenizer(instruction, return_tensors='pt')
            
            # 현재 정책의 로그 확률
            current_outputs = self.model(**inputs)
            current_logprob = F.log_softmax(current_outputs.logits, dim=-1)
            
            # 확률 비율
            ratio = torch.exp(current_logprob - logprob)
            
            # 클리핑된 확률 비율
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # PPO 목적 함수
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            
            # KL 발산 페널티
            kl_penalty = -self.kl_coef * (current_logprob - ref_logprob)
            
            # 엔트로피 페널티 (다양성 촉진)
            entropy_penalty = -self.entropy_coef * current_logprob
            
            # 전체 손실
            total_loss = policy_loss + kl_penalty + entropy_penalty
            
            # 역전파
            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()
            
            print(f"응답 {i+1}:")
            print(f"  보상: {reward:.4f}")
            print(f"  어드밴티지: {advantage:.4f}")
            print(f"  손실: {total_loss.mean().item():.4f}")
            print()

# PPO 훈련 (단순화된 예시)
def train_ppo_model():
    """PPO 모델 훈련"""
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    reward_model = RewardModel('./reward_model')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # PPO 트레이너
    ppo_trainer = PPOTrainer(model, reward_model, tokenizer, ref_model)
    
    # 훈련 지시
    train_instructions = [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "Write a short poem about spring.",
        "How do you make a chocolate cake?"
    ]
    
    # PPO 훈련 루프
    for iteration in range(3):  # 실제로는 더 많은 반복
        print(f"\n=== PPO 반복 {iteration+1} ===")
        
        # 응답 생성
        responses, logprobs, ref_logprobs = ppo_trainer.generate_responses(train_instructions)
        
        # 보상 계산
        rewards = ppo_trainer.compute_rewards(train_instructions, responses)
        
        # 어드밴티지 계산
        advantages = ppo_trainer.compute_advantages(rewards)
        
        # PPO 업데이트
        ppo_trainer.ppo_step(
            train_instructions, responses, logprobs, ref_logprobs, rewards, advantages
        )
    
    return model

# PPO 훈련
ppo_model = train_ppo_model()
ppo_model.save_pretrained('./ppo_model')
```

### 3. PEFT (LoRA) 구현 (30분)

#### LoRA 구현과 미세조정
```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super(LoRALayer, self).__init__()
        
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # LoRA 파라미터
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 스케일링 팩터
        self.scaling = alpha / rank
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # 원본 출력
        original_output = x  # 실제로는 W @ x
        
        # LoRA 경로
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        
        # 스케일링된 LoRA 출력 추가
        return original_output + self.scaling * lora_output

def apply_lora_to_model(model, rank=8, alpha=16):
    """모델에 LoRA 적용"""
    
    # 모든 선형 레이어에 LoRA 적용
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 특정 레이어만 LoRA 적용 (예: q_proj, v_proj)
            if any(proj_name in name for proj_name in ['q_proj', 'v_proj', 'k_proj', 'o_proj']):
                in_features = module.in_features
                out_features = module.out_features
                
                # LoRA 레이어로 교체
                lora_layer = LoRALayer(in_features, out_features, rank, alpha)
                
                # 원본 가중치 저장
                lora_layer.weight = module.weight
                
                # 모듈 교체
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
                
                setattr(parent_module, child_name, lora_layer)
    
    return model

def create_sample_lora_data(output_path, num_samples=50):
    """LoRA 미세조정용 샘플 데이터 생성"""
    import random
    
    data = []
    
    for i in range(num_samples):
        # 다양한 작업 유형
        task_types = [
            'translation',
            'summarization',
            'question_answering',
            'code_generation',
            'creative_writing'
        ]
        
        task_type = random.choice(task_types)
        
        if task_type == 'translation':
            source_lang = random.choice(['English', 'French', 'German'])
            target_lang = random.choice(['French', 'German', 'English'])
            source_text = f"Sample text {i+1} in {source_lang}."
            target_text = f"[Translation to {target_lang}] Sample text {i+1} in {target_lang}."
            instruction = f"Translate from {source_lang} to {target_lang}: {source_text}"
        
        elif task_type == 'summarization':
            source_text = f"This is a long text about topic {i+1}. " * 20  # 반복으로 긴 텍스트 생성
            summary = f"[Summary] This is a summary of topic {i+1}."
            instruction = f"Summarize the following text: {source_text}"
        
        elif task_type == 'question_answering':
            question = f"What is the answer to question {i+1}?"
            answer = f"[Answer] The answer to question {i+1} is 42."
            instruction = f"{question}"
        
        elif task_type == 'code_generation':
            description = f"Create a function that does task {i+1}."
            code = f"[Code] def task_{i+1}():\n    # Implementation\n    return result"
            instruction = f"{description}"
        
        else:  # creative_writing
            prompt = f"Write a story about {random.choice(['a dragon', 'a robot', 'a wizard'])}."
            story = f"[Story] Once upon a time, there was a {random.choice(['dragon', 'robot', 'wizard'])}..."
            instruction = prompt
        
        data.append({
            'instruction': instruction,
            'input': '',
            'output': target_text if task_type != 'creative_writing' else story
        })
    
    # 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return data

# LoRA 미세조정 훈련
def train_lora_model():
    """LoRA 모델 미세조정"""
    
    # 기본 모델과 토크나이저 로드
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,  # alpha
        target_modules=["q_proj", "v_proj"],  # 적용할 모듈
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # PEFT 모델 생성
    peft_model = get_peft_model(base_model, lora_config)
    
    # 미세조정 데이터 생성
    lora_data = create_sample_lora_data('lora_instructions.json', 100)
    dataset = InstructionDataset('lora_instructions.json', tokenizer)
    
    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir='./lora_results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # 실제로는 별도 평가 데이터 사용
    )
    
    # LoRA 미세조정 훈련
    print("LoRA 미세조정 시작...")
    trainer.train()
    print("LoRA 미세조정 완료")
    
    return peft_model, tokenizer

# LoRA 미세조정 훈련
lora_model, lora_tokenizer = train_lora_model()

# LoRA 모델 저장
lora_model.save_pretrained('./lora_model')
lora_tokenizer.save_pretrained('./lora_model')

# LoRA 파라미터 수 확인
def count_lora_parameters(model):
    """LoRA 파라미터 수 계산"""
    
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora' in name.lower():
            lora_params += param.numel()
    
    print(f"전체 파라미터: {total_params:,}")
    print(f"LoRA 파라미터: {lora_params:,}")
    print(f"LoRA 비율: {lora_params/total_params*100:.2f}%")
    
    return lora_params, total_params

# LoRA 파라미터 분석
lora_params, total_params = count_lora_parameters(lora_model)
```

#### LoRA 모델 평가와 비교
```python
def compare_models(base_model_path, lora_model_path, test_instructions):
    """기본 모델과 LoRA 모델 비교"""
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # 기본 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_model.eval()
    
    # LoRA 모델 로드
    from peft import PeftModel
    lora_model = PeftModel.from_pretrained(base_model_path, lora_model_path)
    lora_model.eval()
    
    results = []
    
    for i, instruction in enumerate(test_instructions):
        # 입력 토큰화
        inputs = tokenizer(instruction, return_tensors='pt')
        
        # 기본 모델 생성
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # LoRA 모델 생성
        with torch.no_grad():
            lora_outputs = lora_model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        lora_text = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        
        results.append({
            'instruction': instruction,
            'base_output': base_text,
            'lora_output': lora_text
        })
        
        print(f"지시 {i+1}:")
        print(f"  입력: {instruction}")
        print(f"  기본 모델: {base_text}")
        print(f"  LoRA 모델: {lora_text}")
        print()
    
    return results

# 모델 비교
comparison_results = compare_models(model_name, './lora_model', test_instructions)
```

## 과제

### 1. 지도 미세조정 과제
- 다양한 도메인의 지시 데이터로 SFT 수행
- 데이터 크기와 품질이 미세조정 성능에 미치는 영향 분석
- 다양한 모델 크기에 대한 SFT 성능 비교

### 2. RLHF 과제
- 간단한 보상 모델 훈련 구현
- PPO 알고리즘의 핵심 요소 구현
- RLHF와 SFT의 성능 특성 비교

### 3. PEFT 과제
- 다양한 PEFT 방법(LoRA, 어댑터 등) 구현과 비교
- 랭크(rank)와 알파(alpha) 파라미터의 영향 분석
- PEFT와 전체 미세조정의 성능-비용 트레이드오프 분석

## 추가 학습 자료

### 논문
- "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

### 온라인 자료
- [Hugging Face PEFT Library](https://huggingface.co/docs/peft/)
- [OpenAI's RLHF Blog Post](https://openai.com/research/rlhf/)
- [Stanford's Human Preference Learning](https://cs.stanford.edu/~pliang/software/)

### 구현 참고
- [TRL (Transformer Reinforcement Learning)](https://github.com/lvwerra/trl)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [ChatGPT Implementation](https://github.com/openai/chatgpt-research)