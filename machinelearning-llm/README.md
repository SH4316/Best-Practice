# 대규모 언어 모델(LLM) 개발 과정

## 개요

본 저장소는 대학원 석사 과정의 **대규모 언어 모델(LLM) 개발** 강의 자료를 포함하고 있습니다. 이 과정은 LLM 개발에 필요한 이론적 기초와 실용적 구현 기술을 체계적으로 다루며, 수학적 기초부터 최신 연구 동향까지 전체 파이프라인을 학습합니다.

## 강의 목표

1. LLM 개발에 필요한 수학적, 프로그래밍적 기초 지식 습득
2. 트랜스포머 아키텍처와 LLM의 핵심 원리 이해
3. LLM 훈련, 미세조정, 평가 방법에 대한 실용적 기술 습득
4. LLM 연구 방법론과 석사 논문 준비를 위한 기반 마련
5. 실제 프로젝트를 통한 LLM 개발 경험 축적

## 강의 구조

### 1부: 기초 다지기 (1-4주)
- **1주차**: 과정 소개 및 LLM 개요
- **2주차**: 수학적 기초
- **3주차**: 머신러닝 기초
- **4주차**: 딥러닝 기초

### 2부: 딥러닝 심화와 트랜스포머 (5-8주)
- **5주차**: 심화 딥러닝 아키텍처
- **6주차**: 트랜스포머 아키텍처 (1)
- **7주차**: 트랜스포머 아키텍처 (2)
- **8주차**: 중간 프로젝트

### 3부: LLM 훈련과 미세조정 (9-12주)
- **9주차**: LLM 훈련 기초
- **10주차**: 사전 훈련 전략
- **11주차**: 미세조정 방법론
- **12주차**: 고급 미세조정 기법

### 4부: LLM 평가와 응용 (13-15주)
- **13주차**: LLM 평가 방법론
- **14주차**: LLM 응용 및 시스템 통합
- **15주차**: 최신 연구 동향과 특수 주제

### 5부: 연구 방법론과 프로젝트 (16주)
- **16주차**: 최종 프로젝트 발표 및 연구 방법론

## 저장소 구조

```
├── README.md                           # 이 파일
├── course_syllabus.md                  # 강의 계획서
├── assessment_grading_structure.md     # 평가 및 등급 구조
├── project_guidelines_evaluation.md   # 프로젝트 가이드라인과 평가 기준
├── reading_lists_resources.md         # 학습 자료와 참고문헌 목록
├── phase1_practical_assignments.md     # 1부 실습 과제
├── phase1_week1_introduction.md        # 1주차: 과정 소개 및 LLM 개요
├── phase1_week2_mathematical_foundations.md  # 2주차: 수학적 기초
├── phase1_week3_machine_learning_basics.md   # 3주차: 머신러닝 기초
├── phase1_week4_deep_learning_basics.md     # 4주차: 딥러닝 기초
├── phase2_practical_assignments.md     # 2부 실습 과제
├── phase2_week5_advanced_dl_architectures.md # 5주차: 심화 딥러닝 아키텍처
├── phase2_week6_transformer_architecture1.md # 6주차: 트랜스포머 아키텍처 (1)
├── phase2_week7_transformer_architecture2.md # 7주차: 트랜스포머 아키텍처 (2)
├── phase2_week8_midterm_project.md     # 8주차: 중간 프로젝트
├── phase3_practical_assignments.md     # 3부 실습 과제
├── phase3_week9_llm_training_fundamentals.md  # 9주차: LLM 훈련 기초
├── phase3_week10_pretraining_strategies.md   # 10주차: 사전 훈련 전략
├── phase3_week11_fine_tuning_methods.md      # 11주차: 미세조정 방법론
├── phase4_practical_assignments.md     # 4부 실습 과제
├── phase4_week12_advanced_fine_tuning.md     # 12주차: 고급 미세조정 기법
├── phase5_week13_llm_evaluation.md      # 13주차: LLM 평가 방법론
├── phase5_week14_llm_applications.md    # 14주차: LLM 응용 및 시스템 통합
├── phase5_week15_llm_ethics_safety.md   # 15주차: 최신 연구 동향과 특수 주제
├── phase6_week16_research_methodology.md # 16주차: 연구 방법론
├── phase6_week17_thesis_development.md  # 석사 논문 개발
└── phase6_week18_final_project.md      # 최종 프로젝트
```

## 평가 방법

- **중간 프로젝트 (25%)**: 간단한 트랜스포머 모델 구현
- **실습 과제 (30%)**: 주별 실습 과제 수행
- **최종 프로젝트 (35%)**: LLM 관련 자유 주제 프로젝트
- **참여도 (10%)**: 수업 참여, 논문 토론, 질의응답

## 개발 환경 요구사항

- Python 3.8+
- PyTorch 1.12+
- Jupyter Notebook
- Hugging Face Transformers
- GPU 권장 (NVIDIA RTX 3060 이상)

## 핵심 참고 자료

### 교재
- "Speech and Language Processing" by Dan Jurafsky and James H. Martin
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Natural Language Processing" by Jacob Eisenstein

### 필수 논문
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)
- "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT, Ouyang et al., 2022)

## 학습 성공 전략

1. **이론과 실습의 균형**: 이론적 이해와 코드 구현을 병행
2. **점진적 학습**: 기초부터 심화까지 체계적으로 접근
3. **능동적 참여**: 논문 읽기, 코드 분석, 질문하기
4. **프로젝트 중심 학습**: 실제 문제 해결에 집중
5. **동료 학습**: 팀 프로젝트, 코드 리뷰, 아이디어 공유

## 시작하기

1. 이 저장소를 클론합니다:
   ```bash
   git clone https://github.com/your-username/machinelearning-llm.git
   cd machinelearning-llm
   ```

2. 개발 환경을 설정합니다:
   ```bash
   pip install -r requirements.txt
   ```

3. [`course_syllabus.md`](course_syllabus.md)를 참고하여 학습 계획을 세웁니다.

4. 주차별 자료를 따라 학습을 진행합니다.

## 기여 방법

이 저장소는 교육용으로 만들어졌으며, 개선 제안이나 오류 신고는 언제나 환영합니다. 이슈를 열거나 풀 리퀘스트를 제출하여 기여해주세요.

## 라이선스

이 저장소의 내용은 교육 목적으로 사용할 수 있으며, 상업적 이용 시 별도의 허가가 필요합니다.

## 연락처

궁금한 점이 있으시면 이슈를 열어주시거나 담당 교수에게 연락해주세요.

---

**참고**: 이 저장소는 교육 목적으로 만들어졌으며, 학생들의 학습을 돕기 위해 제공됩니다. 자유롭게 활용하되, 출처를 명확히 밝혀주시기 바랍니다.