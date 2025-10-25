# 15주차: LLM 윤리와 안전성

## 강의 목표
- LLM의 주요 윤리적 문제와 도전 과제 이해
- 편향성, 공정성, 투명성의 중요성과 측정 방법 습득
- 유해 콘텐츠 생성 방지와 안전성 확보 기법 파악
- LLM 개발과 배포의 윤리적 가이드라인과 법적 고려사항 학습
- 실제 LLM 프로젝트에서의 윤리적 고려사항과 안전성 평가 능력 배양

## 이론 강의 (90분)

### 1. LLM의 윤리적 문제 (25분)

#### 주요 윤리적 문제
**편향성과 차별**
- 성별, 인종, 문화적 편향: 특정 집단에 대한 선호나 차별
- 사회경제적 편향: 특정 사회계층에 대한 선호나 차별
- 지역적 편향: 특정 지역에 대한 선호나 차별
- 연령적 편향: 특정 연령대에 대한 선호나 차별

**프라이버시와 데이터 보호**
- 개인정보 유출: 훈련 데이터에 포함된 개인정보 유출
- 데이터 무단 사용: 저작권이 있는 데이터의 무단 사용
- 데이터 보안: 데이터의 부적절한 보안과 유출
- 데이터 소스 투명성: 데이터 소스의 불투명한 사용

**오해와 허위 정보**
- 환각 생성: 사실이 아닌 정보를 사실처럼 생성
- 부정확한 정보 생성: 신뢰할 수 없는 정보 생성
- 의도적 오해 유도: 특정 목적을 위한 오해 생성
- 신뢰성 저하: 사용자의 신뢰를 저하하는 정보 생성

**책임과 책임 소재**
- AI 결정의 책임: AI 결정에 대한 책임 소재 불분명
- 오류의 영향: AI 오류가 미치는 영향에 대한 책임
- 통제 가능성: AI 시스템의 통제 가능성과 책임
- 법적 책임: AI 시스템의 법적 책임 소재

#### 윤리적 문제의 원인
**데이터 편향**
- 훈련 데이터의 편향: 훈련 데이터에 포함된 편향이 모델에 반영
- 데이터의 대표성 부족: 특정 집단이 과소표현된 데이터
- 역사적 편향: 과거의 편견된 데이터가 현재에 영향
- 문화적 편향: 특정 문화의 관점이 지배적인 데이터

**알고리즘 설계**
- 목적 함수의 편향: 특정 목적을 최적화하는 알고리즘의 편향
- 평가 지표의 편향: 특정 지표를 최적화하는 알고리즘의 편향
- 아키텍처의 편향: 특정 아키텍처가 특정 종류의 편향을 증폭
- 최적화의 부작용: 특정 목적 최적화가 다른 목적에 미치는 영향

**사회적 맥락**
- 기술적 중립성의 환상: 기술이 사회적으로 중립적이라는 환상
- 사회적 영향 무시: 기술이 사회에 미치는 영향을 고려하지 않음
- 윤리적 검토 부족: 기술 개발 시 윤리적 검토의 부족
- 이해관계의 단절: 개발자와 사용자 간의 이해관계 단절

### 2. 편향성과 공정성 (25분)

#### 편향성의 종류와 측정
**통계적 편향**
- 정의: 특정 집단에 대한 통계적 차이
- 측정: 집단별 성능 차이, 표준 편차, 분산 분석
- 예시: 특정 성별에 대한 낮은 인식률
- 해결: 데이터 균형화, 알고리즘 수정

**인지적 편향**
- 정의: 특정 집단에 대한 인지적 선호나 차별
- 측정: 암묵적 연관 테스트, IAT (Implicit Association Test)
- 예시: 특정 인종에 대한 부정적 연관
- 해결: 편향성 감지, 알고리즘 수정

**문화적 편향**
- 정의: 특정 문화의 가치관이나 관점을 우선시키는 편향
- 측정: 문화적 표현의 빈도와 감성 분석
- 예시: 특정 문화의 관점을 우선시키는 표현
- 해결: 다문화 데이터, 문화적 감수성 증가

**사회경제적 편향**
- 정의: 특정 사회계층에 대한 선호나 차별
- 측정: 사회경제적 지위와 표현의 연관성 분석
- 예시: 특정 사회계층에 대한 부정적 묘사
- 해결: 사회경제적 다양성 증가, 편향성 감지

#### 공정성 확보 방법
**데이터 다양성**
- 균형잡힌 데이터: 다양한 집단을 균형있게 포함하는 데이터
- 과소표현 집단 강화: 과소표현된 집단의 데이터 강화
- 편향적 데이터 제거: 편향적 표현을 포함하는 데이터 제거
- 다양성 평가: 데이터의 다양성을 정기적으로 평가

**알고리즘 수정**
- 편향성 감지: 알고리즘의 편향성을 자동으로 감지
- 편향성 보정: 감지된 편향성을 보정하는 메커니즘
- 공정성 제약: 알고리즘에 공정성을 보장하는 제약 조건
- 편향성 평가: 알고리즘의 편향성을 정기적으로 평가

**투명성과 설명 가능성**
- 결정 과정 투명성: AI 결정 과정을 투명하게 공개
- 편향성 보고: 모델의 편향성을 정기적으로 보고
- 설명 가능성: AI 결정을 설명할 수 있는 능력 제공
- 피드백 메커니즘: 사용자 피드백을 통한 편향성 개선

#### 공정성 평가 지표
**집단별 성능 평가**
- 정확성: 각 집단에서의 모델 정확성
- 재현률: 각 집단에서의 모델 재현률
- F1 점수: 각 집단에서의 모델 F1 점수
- 성능 격차: 집단 간 성능 격차 측정

**편향성 지수**
- 성별 편향 지수: 성별에 대한 모델의 편향 정도
- 인종 편향 지수: 인종에 대한 모델의 편향 정도
- 문화 편향 지수: 문화에 대한 모델의 편향 정도
- 종합 편향 지수: 여러 편향을 종합한 지수

**공정성 기준**
- 80-20 규칙: 모든 집단에서 80% 이상의 성능 달성
- 4/5 규칙: 모든 집단에서 4/5 이상의 성능 달성
- 차이 비율: 집단 간 성능 차이가 특정 비율 이하
- 통계적 유의성: 집단 간 성능 차이가 통계적으로 유의미하지 않음

### 3. 유해 콘텐츠 방지 (20분)

#### 유해 콘텐츠의 종류
**폭력과 증오**
- 직접적 폭력: 폭력적인 행위나 언어의 직접적 묘사
- 간접적 폭력: 폭력을 조장하거나 정당화하는 내용
- 증오: 불법적인 활동이나 물질의 제조 방법
- 자해: 자해나 자살을 조장하거나 유도하는 내용
- 테러리즘: 테러 활동을 조장하거나 지원하는 내용

**증오와 불법 활동**
- 마약 제조: 마약이나 약물의 제조 방법
- 무기 제조: 무기나 폭발물의 제조 방법
- 해킹: 컴퓨터 시스템이나 네트워크의 해킹 방법
- 사기: 금융 사기나 신원 사기의 방법
- 불법 정보: 불법적인 정보나 활동에 대한 정보

**혐오와 차별**
- 인종차별: 특정 인종에 대한 차별적 표현
- 성차별: 특정 성별에 대한 차별적 표현
- 종교차별: 특정 종교에 대한 차별적 표현
- 장애인 차별: 특정 장애인에 대한 차별적 표현
- 성소수자 차별: 특정 성적 지향에 대한 차별적 표현

**기타 유해 콘텐츠**
- 아동학대: 아동에 대한 성적 착취나 학대
- 동물학대: 동물에 대한 잔인한 행위
- 환경 파괴: 환경 파괴를 조장하는 내용
- 건강 위협: 건강에 해로운 정보나 행위
- 사생활 파괴: 사생활을 파괴하는 내용

#### 유해 콘텐츠 방지 기법
**사전 필터링**
- 키워드 필터링: 유해한 키워드를 포함하는 입력 필터링
- 정규표현식 필터링: 유해한 패턴을 포함하는 입력 필터링
- 분류기 기반 필터링: 유해성 분류기를 통한 입력 필터링
- 블랙리스트 기반 필터링: 유해한 사이트나 출처를 블랙리스트에 추가

**사후 검토**
- 출력 검토: 생성된 출력의 유해성 검토
- 사용자 신고: 사용자가 유해한 출력을 신고할 수 있는 메커니즘
- 자동 수정: 유해한 출력을 자동으로 수정하거나 차단
- 로그 기록: 유해한 출력의 시도와 결과를 기록

**안전성 강화**
- 안전성 미세조정: 안전성을 높이기 위한 모델 미세조정
- 안전성 평가: 모델의 안전성을 정기적으로 평가
- 안전성 테스트: 다양한 공격 시나리오에 대한 안전성 테스트
- 안전성 모니터링: 모델의 안전성을 실시간으로 모니터링

#### 유해 콘텐츠 대응 전략
**기술적 대응**
- 필터링 강화: 더 정교한 필터링 기법 개발
- 탐지 알고리즘: 유해한 콘텐츠를 더 효과적으로 탐지하는 알고리즘
- 차단 메커니즘: 유해한 콘텐츠를 더 효과적으로 차단하는 메커니즘
- 복구 시스템: 유해한 콘텐츠 발생 시의 신속한 복구 시스템

**정책적 대응**
- 사용 정책: 명확한 사용 정책 수립과 공개
- 위반 처리: 사용 정책 위반 시의 처리 절차 수립
- 책임자 지정: 안전성 관리를 위한 책임자 지정
- 외부 전문가와 협력: 외부 전문가와의 안전성 관리 협력

**교육적 대응**
- 사용자 교육: 사용자에게 안전한 사용법 교육
- 인식 향상: 사용자의 유해한 콘텐츠 인식 능력 향상
- 신고 장려: 사용자의 유해한 콘텐츠 신고 장려
- 커뮤니티 참여: 안전한 커뮤니티 구축과 참여 장려

### 4. LLM 개발과 배포의 윤리적 고려사항 (20분)

#### 개발 단계의 윤리적 고려사항
**데이터 수집과 처리**
- 동의 기반 수집: 데이터 주체의 명확한 동의 기반 수집
- 최소화 원칙: 목적 달성에 필요한 최소한의 데이터만 수집
- 익명화 처리: 개인정보 식별자 제거나 익명화 처리
- 데이터 보안: 수집된 데이터의 안전한 보관과 처리

**모델 설계와 훈련**
- 편향성 고려: 모델 설계 시 편향성을 고려
- 안전성 통합: 모델에 안전성을 통합하는 설계
- 투명성 확보: 모델의 결정 과정을 투명하게 공개
- 윤리적 검토: 모델의 윤리적 영향을 검토

**테스트와 평가**
- 다양한 시나리오 테스트: 다양한 사용 시나리오에서의 모델 테스트
- 편향성 평가: 모델의 편향성을 체계적으로 평가
- 안전성 평가: 모델의 안전성을 체계적으로 평가
- 윤리적 영향 평가: 모델의 사회적 영향을 평가

#### 배포 단계의 윤리적 고려사항
**사용자 보호**
- 명확한 사용 정책: 명확한 사용 정책 수립과 공개
- 연령 제한: 미성년자에게 부적절한 콘텐츠 제한
- 콘텐츠 필터링: 유해한 콘텐츠를 필터링하는 메커니즘
- 사용자 통제: 사용자가 모델의 동작을 통제할 수 있는 기능

**투명성과 설명 가능성**
- AI 사용 표시: AI가 생성했음을 명확하게 표시
- 결정 과정 설명: AI의 결정 과정을 설명할 수 있는 기능 제공
- 불확실성 인정: AI의 불확실성을 인정하고 공개
- 오류 보고: AI의 오류를 투명하게 보고하고 수정

**책임과 책임 소재**
- 명확한 책임 소재: AI 시스템의 책임 소재를 명확히 함
- 오류 보상: AI 오류로 인한 피해에 대한 보상 메커니즘
- 법적 준수: 관련 법규와 규정을 준수
- 보험 가입: AI 시스템의 위험을 보상하는 보험 가입

#### 법적 고려사항
**개인정보 보호법**
- GDPR: 유럽연합의 개인정보 보호법
- CCPA: 캘나다의 개인정보 보호법
- PIPEDA: 브라질의 개인정보 보호법
- 개인정보 보호법: 국가별 개인정보 보호법 준수

**콘텐츠 규제**
- 온라인 콘텐츠 규제: 온라인 콘텐츠에 대한 법적 규제
- 저작권법: 저작권이 있는 콘텐츠의 사용 규제
- 플랫폼 책임법: 플랫폼의 콘텐츠에 대한 책임
- 콘텐츠 등급: 콘텐츠의 등급 분류와 규제

**AI 규제**
- AI 윤리 가이드라인: AI 개발과 사용에 대한 윤리적 가이드라인
- AI 위험 관리: AI 시스템의 위험을 관리하기 위한 규제
- AI 투명성: AI 시스템의 투명성에 대한 규제
- AI 책임: AI 시스템의 책임에 대한 규제

## 실습 세션 (90분)

### 1. 편향성 측정과 분석 (30분)

#### 성별 편향성 측정
```python
import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple

def calculate_gender_bias(texts: List[str], gender_terms: Dict[str, List[str]]) -> Dict[str, float]:
    """성별 편향성 측정"""
    
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

# 성별 편향성 측정 예시
gender_terms = {
    'male': ['he', 'him', 'his', 'man', 'boy', 'father', 'son', 'brother', 'uncle', 'mr', 'sir'],
    'female': ['she', 'her', 'hers', 'woman', 'girl', 'mother', 'daughter', 'sister', 'aunt', 'mrs', 'miss']
}

# 샘플 텍스트
sample_texts = [
    "The doctor arrived and examined the patient.",
    "The nurse helped the patient to feel comfortable.",
    "The engineer fixed the machine.",
    "The teacher explained the concept to the students."
]

# 성별 편향성 계산
gender_bias = calculate_gender_bias(sample_texts, gender_terms)

print("=== 성별 편향성 분석 ===")
for gender, bias in gender_bias.items():
    print(f"{gender}: {bias:.4f}")

# 모델의 성별 연관성 분석 (가상 예시)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained("model_name")
# tokenizer = AutoTokenizer.from_pretrained("model_name")

# prompts = [
#     "The CEO announced a new strategy.",
#     "The nurse provided care to the patient.",
#     "The engineer designed a system."
# ]

# gender_associations = analyze_gender_associations(model, tokenizer, prompts, gender_terms['male'], gender_terms['female'])

# print("\n=== 모델의 성별 연관성 분석 ===")
# print(f"남성 연관성: {gender_associations['male_association']:.2f}")
# print(f"여성 연관성: {gender_associations['female_association']:.2f}")
# print(f"성별 편향: {gender_associations['gender_bias']:.4f}")
```

#### 인종 편향성 측정
```python
def calculate_racial_bias(texts: List[str], racial_terms: Dict[str, List[str]]) -> Dict[str, float]:
    """인종 편향성 측정"""
    
    # 인종 관련 용어 빈도 계산
    racial_counts = {race: 0 for race in racial_terms}
    total_racial_terms = 0
    
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        
        for race, terms in racial_terms.items():
            for term in terms:
                if term in words:
                    racial_counts[race] += words.count(term)
                    total_racial_terms += words.count(term)
    
    # 인종 편향 점수 계산
    bias_scores = {}
    
    if total_racial_terms == 0:
        return {race: 0.0 for race in racial_terms}
    
    for race, count in racial_counts.items():
        # 기대 빈도
        expected_count = total_racial_terms / len(racial_terms)
        
        # 편향 점수
        bias_score = (count - expected_count) / expected_count if expected_count > 0 else 0
        
        bias_scores[race] = bias_score
    
    return bias_scores

# 인종 편향성 측정 예시
racial_terms = {
    'white': ['caucasian', 'european', 'western', 'american', 'british', 'french', 'german'],
    'black': ['african', 'african american', 'black', 'caribbean', 'jamaican', 'nigerian', 'kenyan'],
    'asian': ['chinese', 'japanese', 'korean', 'vietnamese', 'indian', 'pakistani', 'filipino'],
    'hispanic': ['mexican', 'spanish', 'colombian', 'argentinian', 'peruvian', 'chilean', 'cuban'],
    'middle_eastern': ['arab', 'iranian', 'iraqi', 'saudi', 'egyptian', 'turkish', 'israeli', 'palestinian']
}

# 샘플 텍스트
sample_texts = [
    "The European countries have a rich cultural heritage.",
    "African nations are developing rapidly.",
    "Asian economies are growing fast.",
    "Hispanic communities contribute significantly to culture.",
    "Middle Eastern regions have complex histories."
]

# 인종 편향성 계산
racial_bias = calculate_racial_bias(sample_texts, racial_terms)

print("\n=== 인종 편향성 분석 ===")
for race, bias in racial_bias.items():
    print(f"{race}: {bias:.4f}")
```

### 2. 유해 콘텐츠 필터링 구현 (30분)

#### 키워드 기반 필터링
```python
import re
from typing import List, Dict, Set

class HarmfulContentFilter:
    def __init__(self):
        # 유해한 키워드 목록
        self.harmful_keywords = {
            'violence': ['kill', 'murder', 'attack', 'violence', 'fight', 'weapon', 'bomb', 'terror'],
            'hate': ['hate', 'racist', 'sexist', 'homophobic', 'transphobic', 'islamophobic', 'antisemitic'],
            'self_harm': ['suicide', 'self_harm', 'cut', 'hurt myself', 'end my life'],
            'illegal': ['drug', 'cocaine', 'heroin', 'meth', 'weapon', 'bomb', 'hack', 'steal'],
            'adult': ['porn', 'sex', 'nude', 'explicit', 'adult', 'xxx', 'nsfw']
        }
        
        # 유해한 정규표현식 패턴
        self.harmful_patterns = {
            'instructions': [
                r'(how to|step by step|guide|tutorial)\s+(kill|murder|attack|make|create|build)',
                r'(secret|hidden|backdoor)\s+(password|key|access)',
                r'(bypass|override|disable)\s+(filter|block|restriction)'
            ],
            'descriptions': [
                r'(graphic|detailed|explicit)\s+(violence|gore|blood|death)',
                r'(sexual|erotic)\s+(content|material|act|behavior)',
                r'(illegal|criminal)\s+(activity|behavior|method)'
            ]
        }
    
    def is_harmful(self, text: str) -> Tuple[bool, Dict[str, List[str]]]:
        """텍스트의 유해성 판단"""
        
        text_lower = text.lower()
        detected_categories = {}
        is_harmful = False
        
        # 키워드 기반 검사
        for category, keywords in self.harmful_keywords.items():
            detected_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    detected_keywords.append(keyword)
                    is_harmful = True
            
            if detected_keywords:
                detected_categories[category] = detected_keywords
        
        # 정규표현식 패턴 기반 검사
        for pattern_type, patterns in self.harmful_patterns.items():
            detected_patterns = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    detected_patterns.extend(matches)
                    is_harmful = True
            
            if detected_patterns:
                if pattern_type not in detected_categories:
                    detected_categories[pattern_type] = []
                detected_categories[pattern_type].extend(detected_patterns)
        
        return is_harmful, detected_categories
    
    def filter_response(self, response: str) -> Tuple[str, bool]:
        """응답 필터링"""
        
        is_harmful, detected_categories = self.is_harmful(response)
        
        if is_harmful:
            # 유해한 콘텐츠 차단 메시지
            filtered_response = "I'm sorry, I can't provide that information as it may be harmful."
            
            # 로그 기록
            print(f"유해한 콘텐츠 차단: {detected_categories}")
            
            return filtered_response, True
        else:
            return response, False

# 유해 콘텐츠 필터링 예시
filter = HarmfulContentFilter()

# 테스트 케이스
test_cases = [
    "How to make a bomb",
    "I hate people from that country",
    "I want to hurt myself",
    "The weather is nice today",
    "What is the capital of France?"
]

print("=== 유해 콘텐츠 필터링 테스트 ===")
for i, test_case in enumerate(test_cases):
    filtered_response, was_filtered = filter.filter_response(test_case)
    
    print(f"테스트 {i+1}: {test_case}")
    print(f"필터됨: {was_filtered}")
    print(f"응답: {filtered_response}")
    print()
```

#### 분류기 기반 필터링
```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SafetyClassifier:
    def __init__(self, model_name: str = "unitary/toxic-bert"):
        # 안전성 분류 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # 유해성 카테고리
        self.toxicity_categories = [
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 
            'identity_attack', 'sexual_explicit'
        ]
    
    def classify_text(self, text: str) -> Dict[str, float]:
        """텍스트의 유해성 분류"""
        
        # 텍스트 토큰화
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        # 분류 수행
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # 결과 변환
        results = {}
        for i, category in enumerate(self.toxicity_categories):
            results[category] = probabilities[0, i].item()
        
        return results
    
    def is_toxic(self, text: str, threshold: float = 0.5) -> Tuple[bool, Dict[str, float]]:
        """텍스트의 유해성 판단"""
        
        classification = self.classify_text(text)
        
        # 가장 높은 유해성 점수
        max_toxicity = max(classification.values())
        
        # 임계값과 비교
        is_toxic = max_toxicity > threshold
        
        return is_toxic, classification
    
    def filter_response(self, response: str, threshold: float = 0.5) -> Tuple[str, bool]:
        """응답 필터링"""
        
        is_toxic, classification = self.is_toxic(response, threshold)
        
        if is_toxic:
            # 유해한 콘텐츠 차단 메시지
            filtered_response = "I'm sorry, I can't provide that response as it may be inappropriate."
            
            # 로그 기록
            print(f"유해한 콘텐츠 차단: {classification}")
            
            return filtered_response, True
        else:
            return response, False

# 안전성 분류기 기반 필터링 예시
safety_classifier = SafetyClassifier()

# 테스트 케이스
test_cases = [
    "You are an idiot and I hate you",
    "The weather is beautiful today",
    "I want to kill everyone",
    "What is the capital of France?"
]

print("\n=== 안전성 분류기 기반 필터링 테스트 ===")
for i, test_case in enumerate(test_cases):
    filtered_response, was_filtered = safety_classifier.filter_response(test_case)
    
    print(f"테스트 {i+1}: {test_case}")
    print(f"필터됨: {was_filtered}")
    print(f"응답: {filtered_response}")
    print()
```

### 3. 윤리적 LLM 설계 (30분)

#### 윤리적 제약 조건 통합
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class EthicalLLM(nn.Module):
    def __init__(self, base_model_name: str, safety_threshold: float = 0.5):
        super(EthicalLLM, self).__init__()
        
        # 기본 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 안전성 분류기
        self.safety_classifier = SafetyClassifier()
        self.safety_threshold = safety_threshold
        
        # 윤리적 제약 조건 레이어
        self.ethics_layer = nn.Linear(self.base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None):
        # 기본 모델 순전파
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        # 윤리적 점수 계산
        ethics_score = torch.sigmoid(self.ethics_layer(hidden_states.mean(dim=1)))
        
        # 윤리적 제약 조건 적용
        if ethics_score.mean() < self.safety_threshold:
            # 안전한 경우: 원본 로짓
            constrained_logits = logits
        else:
            # 위험한 경우: 로짓 조정
            constrained_logits = logits - 10.0 * (1 - ethics_score)
        
        return constrained_logits
    
    def generate_with_ethics(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        """윤리적 제약 조건을 통한 텍스트 생성"""
        
        # 입력 토큰화
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # 생성
        self.eval()
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 생성된 텍스트의 안전성 검사
        is_toxic, classification = self.safety_classifier.is_toxic(generated_text)
        
        if is_toxic:
            # 유해한 경우: 안전한 메시지 반환
            safe_response = "I'm sorry, I can't provide that response as it may be inappropriate."
            return safe_response
        else:
            return generated_text

# 윤리적 LLM 예시
ethical_llm = EthicalLLM("microsoft/DialoGPT-medium")

# 테스트
prompt = "Tell me a joke"
response = ethical_llm.generate_with_ethics(prompt)

print(f"프롬프트: {prompt}")
print(f"응답: {response}")
```

#### 투명성과 설명 가능성 증가
```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class ExplainableLLM(nn.Module):
    def __init__(self, base_model_name: str):
        super(ExplainableLLM, self).__init__()
        
        # 기본 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 어텐션 가중치 저장 (설명 가능성을 위해)
        self.attention_weights = []
        
        # 설명 가능성 레이어
        self.explanation_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask=None):
        # 기본 모델 순전파
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions
        
        # 어텐션 가중치 저장
        self.attention_weights = attentions
        
        # 설명 가능성 점수 계산
        hidden_states = outputs.hidden_states[-1]
        explainability_score = self.explanation_head(hidden_states.mean(dim=1))
        
        return logits, explainability_score
    
    def generate_with_explanation(self, prompt: str, max_length: int = 100):
        """설명 가능성을 증가한 텍스트 생성"""
        
        # 입력 토큰화
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # 생성
        self.eval()
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_attentions=True
            )
        
        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 설명 가능성 정보 추출
        _, explainability_score = self.forward(inputs['input_ids'])
        
        # 어텐션 가중치 분석
        attention_patterns = []
        for layer_attention in self.attention_weights:
            # 각 레이어의 어텐션 패턴 분석
            layer_attention = layer_attention[0]  # [batch_size, num_heads, seq_len, seq_len]
            
            # 가장 높은 어텐션 헤드 찾기
            max_attention_heads = []
            for head in range(layer_attention.size(1)):
                head_attention = layer_attention[head]
                max_attention = torch.max(head_attention).item()
                max_pos = torch.argmax(head_attention.view(-1)).item()
                
                max_attention_heads.append({
                    'head': head,
                    'max_attention': max_attention,
                    'max_position': max_pos
                })
            
            attention_patterns.append(max_attention_heads)
        
        return {
            'generated_text': generated_text,
            'explainability_score': explainability_score.mean().item(),
            'attention_patterns': attention_patterns
        }

# 설명 가능성 증가 LLM 예시
explainable_llm = ExplainableLLM("microsoft/DialoGPT-medium")

# 테스트
prompt = "Why is the sky blue?"
result = explainable_llm.generate_with_explanation(prompt)

print(f"프롬프트: {prompt}")
print(f"생성된 텍스트: {result['generated_text']}")
print(f"설명 가능성 점수: {result['explainability_score']:.4f}")
print("어텐션 패턴:")
for layer_idx, patterns in enumerate(result['attention_patterns']):
    print(f"  레이어 {layer_idx}:")
    for pattern in patterns:
        print(f"    헤드 {pattern['head']}: 최대 어텐션 {pattern['max_attention']:.4f}, 위치 {pattern['max_position']}")
```

## 과제

### 1. 편향성 측정 과제
- 다양한 편향성(성별, 인종, 문화 등) 측정 도구 구현
- 실제 LLM의 편향성 분석과 평가
- 편향성 완화 방안 연구와 구현

### 2. 유해 콘텐츠 방지 과제
- 다양한 유해 콘텐츠 필터링 기법 구현과 비교
- 실시간 유해 콘텐츠 탐지 시스템 구현
- 유해 콘텐츠 대응 전략 설계와 평가

### 3. 윤리적 LLM 설계 과제
- 윤리적 제약 조건을 통합한 LLM 아키텍처 설계
- 투명성과 설명 가능성을 증가하는 기법 구현
- 윤리적 LLM의 성능과 안전성 평가

## 추가 학습 자료

### 논문
- "The Moral Machine" (Moor, 2006)
- "Ethical and Social Issues in the Information Age" (Moor, 2018)
- "Gender Shades: Intersectionality in AI" (Noble, 2018)
- "Algorithms of Oppression" (Noble, 2012)

### 온라인 자료
- [Partnership on AI](https://www.partnershiponai.org/)
- [AI Ethics Guidelines](https://www.ethicsguidelines.ai/)
- [Fairlearn](https://fairlearn.org/)

### 규제와 가이드라인
- [EU AI Act](https://artificialintelligence.ec.europa.eu/)
- [NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/)
- [OECD AI Principles](https://oecd.ai/en/ai-principles/)