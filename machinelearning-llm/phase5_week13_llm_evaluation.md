# 13주차: LLM 평가 방법론

## 강의 목표
- LLM 평가의 필요성과 도전 과제 이해
- 자동 평가 지표의 종류와 특징 파악
- 인간 평가 방법과 설계 원리 습득
- 편향성과 공정성 평가의 중요성 인지
- 실제 LLM 평가 실험 수행 능력 배양

## 이론 강의 (90분)

### 1. LLM 평가의 필요성과 도전 과제 (25분)

#### LLM 평가의 필요성
**성능 측정**
- 객관적 성능: 수치적 지표를 통한 객관적 비교
- 모델 개선: 약점 파악과 개선 방향 설정
- 기술 발전: 평가 방법의 발전이 모델 발전을 이끔

**신뢰성 확보**
- 사용자 신뢰: 일관된 성능으로 사용자 신뢰 확보
- 안전성 검증: 유해한 출력 감지와 방지
- 상업적 가치: 실제 애플리케이션에서의 성능 검증

**연구 진전**
- 기술적 한계 파악: 현재 기술의 한계와 개선점 식별
- 방향성 제시: 연구 방향과 목표 설정
- 커뮤니티 발전: 공통 평가 기준을 통한 커뮤니티 발전

#### LLM 평가의 도전 과제
**다차원성**
- 다양한 능력: 생성, 이해, 추론, 창의성 등
- 상충 관계: 한 능력의 향상이 다른 능력의 저하를 유발
- 균형 평가: 다양한 능력의 균형 있는 평가 필요

**주관성**
- 품질의 주관성: 출력 품질의 주관적 평가 어려움
- 평가자 편향: 평가자의 주관적 편향 문제
- 일관성 유지: 평가 기준의 일관성 유지 어려움

**맥락 의존성**
- 상황에 따른 성능: 다양한 상황에서의 성능 차이
- 일반화 능력: 특정 상황에서의 성능이 일반화 능력을 의미하지 않음
- 평가 환경: 평가 환경이 실제 사용 환경과의 차이

**계산 복잡성**
- 대규모 평가: 수많은 샘플과 다양한 시나리오 평가 필요
- 자원 요구: 평가에 많은 계산 자원과 시간 소요
- 비용 효율성: 효율적인 평가 방법의 필요성

### 2. 자동 평가 지표 (30분)

#### 생성 능력 평가 지표
**언어 모델링 지표**
- Perplexity (PPL): $PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i)\right)$
- 특징: 모델이 테스트 시퀀스를 얼마나 잘 예측하는지 측정
- 한계: 낮은 PPL이 항상 더 나은 생성 품질을 의미하지는 않음

**BLEU (Bilingual Evaluation Understudy)**
- 정의: $BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$
- 구성 요소:
  - 정밀도(Precision): 생성된 n-그램이 참조에 나타나는 비율
  - 재현률(Recall): 참조 n-그램이 생성에 나타나는 비율
  - 브레버티 페널티(Brevity Penalty): 너무 짧은 생성에 대한 페널티
- 용도: 기계 번역 평가, 텍스트 생성 평가

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- 정의: ROUGE-N: n-그램 재현률, ROUGE-L: 가장 긴 공통 부분열
- 특징: 요약 품질 평가에 효과적
- 종류: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-W, ROUGE-S

**자기 유사도(Self-Similarity)**
- 정의: 생성된 텍스트 내부의 유사성 측정
- 목적: 반복적인 생성 감지
- 측정: n-그램 중복도, 자기 BLEU 등

#### 이해 능력 평가 지표
**질문 답변 평가**
- EM (Exact Match): 정확히 일치하는 답변의 비율
- F1 Score: 정밀도와 재현률의 조화 평균
- 특징: 사실적 답변의 정확성 평가

**분류 평가**
- 정확도(Accuracy): 올바르게 분류한 샘플의 비율
- 정밀도(Precision): 양성으로 예측한 것 중 실제 양성의 비율
- 재현률(Recall): 실제 양성 중 양성으로 예측한 것의 비율
- F1 Score: 정밀도와 재현률의 조화 평균

**엔티티 인식 평가**
- 정확도: 올바르게 인식한 엔티티의 비율
- 재현률: 실제 엔티티 중 인식한 엔티티의 비율
- F1 Score: 정밀도와 재현률의 조화 평균
- 엔티티 유형별 평가: 사람, 장소, 조직 등

#### 추론 능력 평가 지표
**논리적 추론 평가**
- 정확성: 논리적 결론의 정확성
- 일관성: 추론 과정의 일관성
- 완전성: 필요한 모든 추론 단계 포함

**수학적 추론 평가**
- 정확성: 수학적 문제 해결의 정확성
- 단계별 점수: 문제 해결 과정의 각 단계 평가
- 최종 답변 점수: 최종 답변의 정확성

**상식 추론 평가**
- 사실적 정확성: 상식적 질문에 대한 사실적 답변
- 맥락 이해: 질문의 맥락을 이해하고 답변하는 능력
- 일관성: 답변의 내부적 일관성

#### 종합 평가 지표
**HELM (Holistic Evaluation of Language Models)**
- 다차원적 평가: 생성, 이해, 추론 등 다양한 능력 평가
- 통합 점수: 여러 지표의 통합적인 점수 계산
- 순위표: 모델별 순위표와 상세 분석

**GLUE (General Language Understanding Evaluation)**
- 다중 작업: 9개의 다른 자연어 이해 작업
- 종합 점수: 모든 작업에서의 평균 성능
- 표준 벤치마크: 널리 사용되는 표준 평가 벤치마크

**SuperGLUE**
- GLUE 확장: 더 어려운 작업과 더 많은 데이터
- 인간 기준선: 인간 성능과의 비교
- 도전 과제: 현재 기술의 한계 규명

### 3. 인간 평가 방법 (20분)

#### 인간 평가의 필요성
**자동 평가의 한계**
- 미묘적 측면: 창의성, 미묘적 품질 등 자동 평가 어려움
- 주관적 품질: 유머, 일관성 등 주관적 품질 평가 필요
- 맥락 적절성: 상황에 맞는 출력의 적절성 평가 필요

**인간 선호도 반영**
- 사용자 만족도: 실제 사용자의 만족도 반영
- 실용성: 실제 애플리케이션에서의 유용성 평가
- 사회적 가치: 사회적 규범과 가치관 반영

#### 인간 평가 설계 원리
**평가자 선정**
- 전문성: 해당 도메인의 전문가 선정
- 다양성: 다양한 배경과 관점의 평가자 선정
- 훈련: 평가 기준과 방법에 대한 일관된 훈련

**평가 기준**
- 명확한 기준: 명확하고 구체적인 평가 기준 수립
- 다차원적 기준: 여러 차원에서의 평가 기준
- 예시 기반: 구체적인 예시와 기준 제시

**평가 절차**
- 표준화된 절차: 일관된 평가 절차 수립
- 이중 맹검: 여러 평가자의 독립적 평가
- 피드백 메커니즘: 평가자 간 피드백과 조정 메커니즘

#### 인간 평가 방법
**직접 비교 평가**
- A/B 테스트: 두 모델의 출력을 직접 비교
- 순위 평가: 여러 모델을 순위 매김
- 선호도 조사: 특정 출력에 대한 선호도 조사

**상대적 평가**
- 기준 모델 비교: 기준 모델과의 상대적 성능 평가
- 개선 정도: 이전 버전과의 개선 정도 평가
- 경쟁 모델 비교: 경쟁 모델과의 상대적 성능 평가

**절대적 평가**
- 등급 평가: 미리 정의된 등급(예: 1-5점)으로 평가
- 기준 충족 여부: 특정 기준 충족 여부 평가
- 다차원적 평가: 여러 차원에서의 독립적 평가

#### 인간 평가의 도전 과제
**주관성과 일관성**
- 평가자 간 차이: 평가자 간의 주관적 차이 문제
- 일관성 유지: 시간과 평가자 간의 일관성 유지 어려움
- 편향성: 평가자의 개인적 편향 문제

**비용과 확장성**
- 높은 비용: 인간 평가의 높은 비용 문제
- 확장성 제한: 대규모 평가의 확장성 제한
- 시간 소요: 평가에 많은 시간 소요

**통계적 유의성**
- 충분한 샘플: 통계적 유의성을 위한 충분한 샘플 필요
- 신뢰 구간: 평가 결과의 신뢰 구간 계산
- 효과 크기: 작은 효과를 통계적으로 유의미하게 감지 어려움

### 4. 편향성과 공정성 평가 (15분)

#### 편향성의 종류
**성별 편향**
- 정의: 특정 성별에 대한 선호나 차별
- 측정: 성별 고정관념, 직업적 고정관념 등
- 예시: "의사는 남성이다"와 같은 성별 고정관념

**인종/민족 편향**
- 정의: 특정 인종이나 민족에 대한 선호나 차별
- 측정: 인종적 고정관념, 차별적 표현 등
- 예시: 특정 인종에 대한 부정적 묘사

**문화 편향**
- 정의: 특정 문화에 대한 선호나 차별
- 측정: 문화적 고정관념, 문화 중심주의 등
- 예시: 특정 문화를 중심으로 한 표현

**연령 편향**
- 정의: 특정 연령대에 대한 선호나 차별
- 측정: 연령적 고정관념, 세대 차별 등
- 예시: "노인은 기술을 사용하지 못한다"와 같은 연령 고정관념

#### 편향성 측정 방법
**명시적 편향 측정**
- 키워드 기반: 특정 키워드의 사용 빈도 측정
- 문맥 분석: 편향적 표현의 문맥적 분석
- 감성 분석: 특정 집단에 대한 감성 분석

**암묵적 편향 측정**
- 연관성 분석: 특정 개념과 편향적 개념 간의 연관성 분석
- 임베딩 공간 분석: 임베딩 공간에서의 편향적 클러스터링 분석
- 생성 텍스트 분석: 생성된 텍스트에서의 편향적 패턴 분석

**통계적 편향 측정**
- 집단별 성능 비교: 다른 집단에 대한 모델 성능 비교
- 표준 편차 측정: 기준 집단과의 표준 편차 측정
- 분산 분석: 집단별 예측 분산 분석

#### 공정성 평가 방법
**대표성 평가**
- 데이터셋 대표성: 훈련 데이터의 다양한 집단 대표성 평가
- 모델 출력 대표성: 모델 출력의 다양한 집단 대표성 평가
- 격차 분석: 데이터셋과 모델 출력 간의 대표성 격차 분석

**공정성 평가**
- 동일 조건 하 성능: 다른 집단에 대한 동일 조건 하 성능 평가
- 기회 균등성: 다른 집단에 대한 기회 균등성 평가
- 결과 균등성: 다른 집단에 대한 결과 균등성 평가

**투명성 평가**
- 평가 과정 투명성: 평가 과정과 기준의 투명성 평가
- 결정 과정 투명성: 모델 결정 과정의 투명성 평가
- 이의 제시: 편향성이나 불공정성 발견 시 이의 제시

#### 편향성 완화 방법
**데이터 다양성**
- 균형잡힌 데이터: 다양한 집단을 균형있게 포함하는 데이터
- 과소표현 집단 강화: 과소표현된 집단의 데이터 강화
- 편향적 데이터 제거: 편향적 표현을 포함하는 데이터 제거

**알고리즘 수정**
- 편향 감지: 편향적 출력을 감지하는 알고리즘
- 편향 완화: 편향적 출력을 완화하는 알고리즘
- 공정성 제약: 공정성을 보장하는 알고리즘 제약

**지속적 모니터링**
- 정기적 평가: 편향성에 대한 정기적 평가 수행
- 피드백 루프: 사용자 피드백을 통한 편향성 개선
- 투명성 보고: 편향성 평가 결과의 투명한 보고

## 실습 세션 (90분)

### 1. 자동 평가 지표 구현 (30분)

#### BLEU 점수 계산
```python
import math
from collections import Counter
import re
from typing import List, Tuple

def get_ngrams(sequence: List[str], n: int) -> Counter:
    """n-그램 추출"""
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngrams.append(tuple(sequence[i:i+n]))
    return Counter(ngrams)

def calculate_bleu(reference: List[str], candidate: List[str], max_n: int = 4) -> float:
    """BLEU 점수 계산"""
    
    # 참조와 후보 문장의 n-그램
    reference_counts = [get_ngrams(ref, n) for n in range(1, max_n+1)]
    candidate_counts = [get_ngrams(candidate, n) for n in range(1, max_n+1)]
    
    # 클리핑된 정밀도 계산
    clipped_counts = []
    for n in range(max_n):
        ref_count = reference_counts[n]
        cand_count = candidate_counts[n]
        
        # 각 n-그램에 대해 참조에서의 최대 빈도로 클리핑
        clipped = Counter()
        for ngram in cand_count:
            clipped[ngram] = min(cand_count[ngram], ref_count.get(ngram, 0))
        
        clipped_counts.append(clipped)
    
    # 정밀도 계산
    precisions = []
    for n in range(max_n):
        if sum(clipped_counts[n].values()) == 0:
            precisions.append(0.0)
        else:
            precision = sum(clipped_counts[n].values()) / sum(candidate_counts[n].values())
            precisions.append(precision)
    
    # 브레버티 페널티 계산
    bp = 1.0
    candidate_len = len(candidate)
    reference_len = len(reference)
    
    if candidate_len > reference_len:
        bp = math.exp(1 - reference_len / candidate_len)
    
    # BLEU 점수 계산
    if sum(precisions) == 0:
        return 0.0
    
    bleu = bp * math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n)
    
    return bleu

# BLEU 계산 예시
reference = "the cat is on the mat".split()
candidate1 = "the cat is on a mat".split()
candidate2 = "the cat sat on the mat".split()

bleu1 = calculate_bleu(reference, candidate1)
bleu2 = calculate_bleu(reference, candidate2)

print(f"참조: {' '.join(reference)}")
print(f"후보 1: {' '.join(candidate1)}, BLEU: {bleu1:.4f}")
print(f"후보 2: {' '.join(candidate2)}, BLEU: {bleu2:.4f}")
```

#### ROUGE 점수 계산
```python
def get_ngrams_list(tokens: List[str], n: int) -> List[List[str]]:
    """n-그램 리스트 추출"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tokens[i:i+n])
    return ngrams

def calculate_rouge_n(reference: List[str], candidate: List[str], n: int) -> float:
    """ROUGE-N 점수 계산"""
    
    # 참조와 후보의 n-그램
    ref_ngrams = set(get_ngrams_list(reference, n))
    cand_ngrams = get_ngrams_list(candidate, n)
    
    # 일치하는 n-그램 수
    overlap = 0
    for ngram in cand_ngrams:
        if ngram in ref_ngrams:
            overlap += 1
    
    # 재현률 계산
    if len(cand_ngrams) == 0:
        return 0.0
    
    recall = overlap / len(ref_ngrams)
    
    return recall

def calculate_rouge_l(reference: List[str], candidate: List[str]) -> float:
    """ROUGE-L 점수 계산"""
    
    # 참조와 후보의 LCS (Longest Common Subsequence)
    m = len(reference)
    n = len(candidate)
    
    # DP 테이블 초기화
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # LCS 계산
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == candidate[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    
    # ROUGE-L 계산
    if len(reference) == 0:
        return 0.0
    
    rouge_l = lcs_len / len(reference)
    
    return rouge_l

def calculate_rouge(reference: str, candidate: str) -> dict:
    """ROUGE 점수 계산"""
    
    # 토큰화
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    # ROUGE-1, ROUGE-2, ROUGE-L 계산
    rouge_1 = calculate_rouge_n(ref_tokens, cand_tokens, 1)
    rouge_2 = calculate_rouge_n(ref_tokens, cand_tokens, 2)
    rouge_l = calculate_rouge_l(ref_tokens, cand_tokens)
    
    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l
    }

# ROUGE 계산 예시
reference = "the cat was sitting on the mat"
candidate1 = "the cat sat on a mat"
candidate2 = "the cat was on the mat"

rouge1 = calculate_rouge(reference, candidate1)
rouge2 = calculate_rouge(reference, candidate2)

print(f"참조: {reference}")
print(f"후보 1: {candidate1}")
print(f"  ROUGE-1: {rouge1['rouge-1']:.4f}")
print(f"  ROUGE-2: {rouge1['rouge-2']:.4f}")
print(f"  ROUGE-L: {rouge1['rouge-l']:.4f}")
print()
print(f"후보 2: {candidate2}")
print(f"  ROUGE-1: {rouge2['rouge-1']:.4f}")
print(f"  ROUGE-2: {rouge2['rouge-2']:.4f}")
print(f"  ROUGE-L: {rouge2['rouge-l']:.4f}")
```

#### Perplexity 계산
```python
import torch
import torch.nn.functional as F

def calculate_perplexity(model, tokenizer, text: str, stride: int = 512) -> float:
    """Perplexity 계산"""
    
    # 텍스트 토큰화
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids']
    
    # 모델 평가 모드
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    # 슬라이딩 윈도우로 평가
    with torch.no_grad():
        for i in range(0, input_ids.size(1), stride):
            # 윈도우 추출
            window = input_ids[:, i:i+stride]
            
            # 타겟 생성 (다음 토큰 예측)
            targets = window[:, 1:].contiguous()
            inputs = window[:, :-1].contiguous()
            
            if inputs.size(1) == 0 or targets.size(1) == 0:
                continue
            
            # 순전파
            outputs = model(inputs)
            
            # 손실 계산
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1),
                reduction='mean'
            )
            
            total_loss += loss.item() * targets.size(1)
            total_tokens += targets.size(1)
    
    # 평균 손실과 Perplexity 계산
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

# Perplexity 계산 예시
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_text = "The quick brown fox jumps over the lazy dog"
ppl = calculate_perplexity(model, tokenizer, test_text)

print(f"텍스트: {test_text}")
print(f"Perplexity: {ppl:.2f}")
```

### 2. 인간 평가 설계와 구현 (30분)

#### 인간 평가 데이터 수집
```python
import json
import pandas as pd
from typing import List, Dict, Any

class HumanEvaluationCollector:
    def __init__(self):
        self.evaluations = []
    
    def add_evaluation(self, model_name: str, prompt: str, 
                     model_output: str, human_rating: float, 
                     criteria: Dict[str, Any], evaluator_id: str):
        """평가 데이터 추가"""
        
        evaluation = {
            'model_name': model_name,
            'prompt': prompt,
            'model_output': model_output,
            'human_rating': human_rating,
            'criteria': criteria,
            'evaluator_id': evaluator_id,
            'timestamp': pd.Timestamp.now()
        }
        
        self.evaluations.append(evaluation)
    
    def save_evaluations(self, filepath: str):
        """평가 데이터 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, ensure_ascii=False, indent=2)
    
    def load_evaluations(self, filepath: str):
        """평가 데이터 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.evaluations = json.load(f)
    
    def get_dataframe(self) -> pd.DataFrame:
        """평가 데이터를 DataFrame으로 변환"""
        return pd.DataFrame(self.evaluations)
    
    def calculate_inter_rater_reliability(self) -> Dict[str, float]:
        """평가자 간 신뢰도 계산"""
        
        # 모델별 평가자 그룹화
        model_evaluators = {}
        for eval in self.evaluations:
            model_name = eval['model_name']
            evaluator_id = eval['evaluator_id']
            
            if model_name not in model_evaluators:
                model_evaluators[model_name] = {}
            
            if evaluator_id not in model_evaluators[model_name]:
                model_evaluators[model_name][evaluator_id] = []
            
            model_evaluators[model_name][evaluator_id].append(eval['human_rating'])
        
        # ICC (Intraclass Correlation Coefficient) 계산
        reliability_scores = {}
        for model_name, evaluators in model_evaluators.items():
            if len(evaluators) < 2:
                reliability_scores[model_name] = 0.0
                continue
            
            # 간단한 ICC 계산 (실제로는 더 복잡한 통계적 방법 사용)
            ratings = list(evaluators.values())
            
            # 평가자 간 상관계계 계산
            import numpy as np
            correlation_matrix = np.corrcoef(np.array(ratings).T)
            
            # 평균 상관계계
            avg_correlation = 0
            count = 0
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    if i != j:
                        avg_correlation += correlation_matrix[i, j]
                        count += 1
            
            if count > 0:
                avg_correlation /= count
            
            reliability_scores[model_name] = avg_correlation
        
        return reliability_scores

# 인간 평가 데이터 수집 예시
collector = HumanEvaluationCollector()

# 평가 기준 정의
criteria = {
    'relevance': {
        'description': '프롬프트와의 관련성',
        'scale': 1-5,
        'weight': 0.3
    },
    'coherence': {
        'description': '응답의 일관성',
        'scale': 1-5,
        'weight': 0.2
    },
    'accuracy': {
        'description': '사실적 정확성',
        'scale': 1-5,
        'weight': 0.3
    },
    'safety': {
        'description': '안전성 (유해한 내용 없음)',
        'scale': 1-5,
        'weight': 0.2
    }
}

# 샘플 평가 데이터 추가
evaluations = [
    {
        'model_name': 'Model A',
        'prompt': '대한민국의 수도는 어디인가?',
        'model_output': '대한민국의 수도는 서울입니다.',
        'human_rating': 4.5,
        'criteria': {
            'relevance': 5,
            'coherence': 4,
            'accuracy': 5,
            'safety': 4
        },
        'evaluator_id': 'eval_001'
    },
    {
        'model_name': 'Model B',
        'prompt': '대한민국의 수도는 어디인가?',
        'model_output': '대한민국의 수도는 부산입니다.',
        'human_rating': 2.0,
        'criteria': {
            'relevance': 2,
            'coherence': 3,
            'accuracy': 1,
            'safety': 2
        },
        'evaluator_id': 'eval_001'
    },
    {
        'model_name': 'Model A',
        'prompt': '기후 변화의 주요 원인은 무엇인가?',
        'model_output': '기후 변화의 주요 원인은 온실가스 배출입니다.',
        'human_rating': 4.0,
        'criteria': {
            'relevance': 4,
            'coherence': 4,
            'accuracy': 4,
            'safety': 4
        },
        'evaluator_id': 'eval_002'
    },
    {
        'model_name': 'Model B',
        'prompt': '기후 변화의 주요 원인은 무엇인가?',
        'model_output': '기후 변화는 자연적인 현상입니다.',
        'human_rating': 2.5,
        'criteria': {
            'relevance': 3,
            'coherence': 3,
            'accuracy': 2,
            'safety': 2
        },
        'evaluator_id': 'eval_002'
    }
]

# 평가 데이터 추가
for eval in evaluations:
    collector.add_evaluation(**eval)

# 평가 데이터 저장
collector.save_evaluations('human_evaluations.json')

# 평가자 간 신뢰도 계산
reliability = collector.calculate_inter_rater_reliability()
print("평가자 간 신뢰도:")
for model, score in reliability.items():
    print(f"  {model}: {score:.4f}")

# 평가 데이터 DataFrame으로 변환
df = collector.get_dataframe()
print("\n평가 데이터:")
print(df.head())
```

#### 인간 평가 분석
```python
def analyze_human_evaluations(evaluation_file: str):
    """인간 평가 데이터 분석"""
    
    # 평가 데이터 로드
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
    
    df = pd.DataFrame(evaluations)
    
    # 모델별 평가 분석
    model_stats = df.groupby('model_name')['human_rating'].agg(['mean', 'std', 'count'])
    
    print("=== 모델별 평가 통계 ===")
    for model, stats in model_stats.iterrows():
        print(f"모델: {model}")
        print(f"  평균 점수: {stats['mean']:.2f}")
        print(f"  표준편차: {stats['std']:.2f}")
        print(f"  평가 수: {stats['count']}")
        print()
    
    # 기준별 평가 분석
    criteria_analysis = {}
    
    for eval in evaluations:
        model = eval['model_name']
        if model not in criteria_analysis:
            criteria_analysis[model] = {}
        
        for criterion, score in eval['criteria'].items():
            if criterion not in criteria_analysis[model]:
                criteria_analysis[model][criterion] = []
            
            criteria_analysis[model][criterion].append(score)
    
    # 기준별 평균 점수 계산
    print("=== 모델별 기준별 평균 점수 ===")
    for model, criteria_scores in criteria_analysis.items():
        print(f"모델: {model}")
        for criterion, scores in criteria_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"  {criterion}: {avg_score:.2f}")
        print()
    
    # 통계적 유의성 검정
    from scipy import stats
    
    print("=== 모델 간 통계적 유의성 검정 ===")
    models = df['model_name'].unique()
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            # 두 모델의 평가 점수
            scores1 = df[df['model_name'] == model1]['human_rating']
            scores2 = df[df['model_name'] == model2]['human_rating']
            
            # t-검정
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            
            print(f"{model1} vs {model2}:")
            print(f"  t-통계량: {t_stat:.4f}")
            print(f"  p-값: {p_value:.4f}")
            print(f"  유의미: {'유의미' if p_value < 0.05 else '무의미'}")
            print()
    
    return model_stats, criteria_analysis

# 인간 평가 분석
model_stats, criteria_analysis = analyze_human_evaluations('human_evaluations.json')
```

### 3. 편향성 측정과 분석 (30분)

#### 성별 편향 측정
```python
import re
from collections import Counter
import numpy as np

def calculate_gender_bias(texts: List[str], gender_terms: Dict[str, List[str]]) -> Dict[str, float]:
    """성별 편향 측정"""
    
    # 성별 관련 용어 빈도 계산
    gender_counts = {gender: 0 for gender in gender_terms}
    total_gender_terms = 0
    
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        
        for gender, terms in gender_terms.items():
            for term in terms:
                if term in words:
                    gender_counts[gender] += words.count(term)
                    total_gender_terms += words.count(term)
    
    # 성별 편향 점수 계산
    bias_scores = {}
    
    if total_gender_terms == 0:
        return {gender: 0.0 for gender in gender_terms}
    
    for gender, count in gender_counts.items():
        # 기대 빈도 (모든 성별이 동일한 빈도를 가질 경우)
        expected_count = total_gender_terms / len(gender_terms)
        
        # 편향 점수 (실제 빈도 - 기대 빈도) / 기대 빈도
        bias_score = (count - expected_count) / expected_count if expected_count > 0 else 0
        
        bias_scores[gender] = bias_score
    
    return bias_scores

def analyze_gender_associations(model, tokenizer, prompts: List[str], 
                           male_terms: List[str], female_terms: List[str]):
    """모델의 성별 연관성 분석"""
    
    model.eval()
    gender_associations = {gender: [] for gender in ['male', 'female']}
    
    with torch.no_grad():
        for prompt in prompts:
            # 프롬프트 토큰화
            inputs = tokenizer(prompt, return_tensors='pt')
            
            # 텍스트 생성
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 생성된 텍스트 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 성별 관련 용어 빈도 계산
            words = re.findall(r'\b\w+\b', generated_text.lower())
            
            male_count = sum(1 for word in words if word in male_terms)
            female_count = sum(1 for word in words if word in female_terms)
            
            gender_associations['male'].append(male_count)
            gender_associations['female'].append(female_count)
    
    # 평균 성별 연관성 계산
    avg_male_association = np.mean(gender_associations['male'])
    avg_female_association = np.mean(gender_associations['female'])
    
    # 성별 편향 점수
    gender_bias = (avg_male_association - avg_female_association) / (avg_male_association + avg_female_association + 1e-8)
    
    return {
        'male_association': avg_male_association,
        'female_association': avg_female_association,
        'gender_bias': gender_bias,
        'associations': gender_associations
    }

# 성별 편향 측정 예시
gender_terms = {
    'male': ['he', 'him', 'his', 'man', 'boy', 'father', 'son', 'brother', 'uncle'],
    'female': ['she', 'her', 'hers', 'woman', 'girl', 'mother', 'daughter', 'sister', 'aunt']
}

# 샘플 텍스트
sample_texts = [
    "The doctor arrived and examined the patient.",
    "The nurse helped the patient to feel comfortable.",
    "The engineer fixed the machine.",
    "The teacher explained the concept to the students."
]

# 성별 편향 계산
gender_bias = calculate_gender_bias(sample_texts, gender_terms)

print("=== 성별 편향 분석 ===")
for gender, bias in gender_bias.items():
    print(f"{gender}: {bias:.4f}")

# 모델의 성별 연관성 분석 (가상 예시)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained("model_name")
# tokenizer = AutoTokenizer.from_pretrained("model_name")

# prompts = [
#     "The CEO announced the new strategy.",
#     "The nurse provided care to the patient.",
#     "The engineer designed the system."
# ]

# gender_associations = analyze_gender_associations(model, tokenizer, prompts, gender_terms['male'], gender_terms['female'])

# print("\n=== 모델의 성별 연관성 분석 ===")
# print(f"남성 연관성: {gender_associations['male_association']:.2f}")
# print(f"여성 연관성: {gender_associations['female_association']:.2f}")
# print(f"성별 편향: {gender_associations['gender_bias']:.4f}")
```

#### 문화 편향 측정
```python
def calculate_cultural_bias(texts: List[str], cultural_terms: Dict[str, List[str]]) -> Dict[str, float]:
    """문화 편향 측정"""
    
    # 문화 관련 용어 빈도 계산
    cultural_counts = {culture: 0 for culture in cultural_terms}
    total_cultural_terms = 0
    
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        
        for culture, terms in cultural_terms.items():
            for term in terms:
                if term in words:
                    cultural_counts[culture] += words.count(term)
                    total_cultural_terms += words.count(term)
    
    # 문화 편향 점수 계산
    bias_scores = {}
    
    if total_cultural_terms == 0:
        return {culture: 0.0 for culture in cultural_terms}
    
    for culture, count in cultural_counts.items():
        # 기대 빈도
        expected_count = total_cultural_terms / len(cultural_terms)
        
        # 편향 점수
        bias_score = (count - expected_count) / expected_count if expected_count > 0 else 0
        
        bias_scores[culture] = bias_score
    
    return bias_scores

def analyze_cultural_stereotypes(model, tokenizer, prompts: List[str], 
                             cultural_contexts: Dict[str, str]):
    """모델의 문화적 고정관념 분석"""
    
    model.eval()
    stereotype_scores = {culture: [] for culture in cultural_contexts.keys()}
    
    with torch.no_grad():
        for prompt in prompts:
            # 프롬프트 토큰화
            inputs = tokenizer(prompt, return_tensors='pt')
            
            # 텍스트 생성
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 생성된 텍스트 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 문화적 고정관념과의 유사성 측정
            for culture, context in cultural_contexts.items():
                # 간단한 유사도 측정 (실제로는 더 정교한 방법 사용)
                similarity = len(set(generated_text.lower().split()) & set(context.lower().split())) / len(set(context.lower().split()))
                
                stereotype_scores[culture].append(similarity)
    
    # 평균 문화적 고정관념 점수 계산
    avg_stereotype_scores = {}
    for culture, scores in stereotype_scores.items():
        avg_stereotype_scores[culture] = np.mean(scores)
    
    return avg_stereotype_scores

# 문화 편향 측정 예시
cultural_terms = {
    'western': ['america', 'europe', 'christianity', 'democracy', 'individualism'],
    'eastern': ['asia', 'china', 'buddhism', 'collectivism', 'harmony'],
    'middle_eastern': ['islam', 'arab', 'desert', 'oil', 'family']
}

# 샘플 텍스트
sample_texts = [
    "The Western approach emphasizes individual rights and freedoms.",
    "Eastern cultures often prioritize community harmony.",
    "The Middle Eastern region has a rich history of trade.",
    "Globalization connects different cultural perspectives."
]

# 문화 편향 계산
cultural_bias = calculate_cultural_bias(sample_texts, cultural_terms)

print("\n=== 문화 편향 분석 ===")
for culture, bias in cultural_bias.items():
    print(f"{culture}: {bias:.4f}")

# 모델의 문화적 고정관념 분석 (가상 예시)
cultural_contexts = {
    'western': 'freedom democracy individual rights liberty',
    'eastern': 'harmony community collectivism family',
    'middle_eastern': 'oil desert family tradition religion'
}

# prompts = [
#     "Describe the ideal political system.",
#     "What makes a society successful?",
#     "Explain the concept of family values."
# ]

# stereotype_scores = analyze_cultural_stereotypes(model, tokenizer, prompts, cultural_contexts)

# print("\n=== 모델의 문화적 고정관념 분석 ===")
# for culture, score in stereotype_scores.items():
#     print(f"{culture}: {score:.4f}")
```

## 과제

### 1. 자동 평가 지표 과제
- BLEU, ROUGE, Perplexity 등 주요 평가 지표 구현
- 다양한 평가 지표의 특성과 한계 분석
- 평가 지표의 조합을 통한 종합 평가 방법 연구

### 2. 인간 평가 과제
- 인간 평가 실험 설계와 수행
- 평가자 간 신뢰도 분석과 개선 방안
- 자동 평가와 인간 평가의 상관관계 연구

### 3. 편향성 평가 과제
- 성별, 인종, 문화 등 다양한 편향성 측정
- 편향성 완화 방법 구현과 실험
- 공정성 평가 지표 개발과 적용

## 추가 학습 자료

### 논문
- "BLEU: a Method for Automatic Evaluation of Machine Translation" (Papineni et al., 2002)
- "ROUGE: A Package for Automatic Evaluation of Summaries" (Lin, 2004)
- "Human Evaluation of Language Models" (Kiela et al., 2021)

### 온라인 자료
- [HELM (Holistic Evaluation of Language Models)](https://stanford-crfm.github.io/helm/latest/)
- [BigBench (Beyond the Imitation Game Benchmark)](https://github.com/suzgunmiracle/BigBench)
- [TruthfulQA Benchmark](https://truthfulqa.ai/)

### 평가 도구
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [Rouge Score](https://pypi.org/project/rouge-score/)