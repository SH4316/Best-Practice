# Scikit-learn 베스트 프랙티스 강의 구조

## 강의 흐름 다이어그램

```mermaid
graph TD
    A[모듈 1: scikit-learn 소개] --> B[모듈 2: 데이터 전처리]
    A --> C[모듈 3: 회귀 기법]
    A --> D[모듈 4: 분류 기법]
    B --> C
    B --> D
    C --> E[모듈 5: 비지도 학습]
    D --> E
    C --> F[모듈 6: 모델 선택]
    D --> F
    E --> F
    F --> G[모듈 7: 모델 평가]
    G --> H[모듈 8: 파이프라인 구축]
    H --> I[모듈 9: 고급 기법]
    I --> J[모듈 10: 모델 배포]
    
    style A fill:#e1f5fe
    style B fill:#e8f5e9
    style C fill:#e8f5e9
    style D fill:#e8f5e9
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#fce4ec
    style I fill:#fce4ec
    style J fill:#fce4ec
```

## 모듈 난이도 진행

```mermaid
graph LR
    A[초급] --> B[중급] --> C[고급]
    
    A1[모듈 1] --> A
    A2[모듈 2] --> A
    A3[모듈 3] --> A
    A4[모듈 4] --> A
    
    B1[모듈 5] --> B
    B2[모듈 6] --> B
    B3[모듈 7] --> B
    B4[모듈 8] --> B
    
    C1[모듈 9] --> C
    C2[모듈 10] --> C
```

## 다양한 대상을 위한 학습 경로

### 초급자용
```mermaid
flowchart TD
    Start[학습 시작] --> M1[모듈 1: 기초]
    M1 --> M2[모듈 2: 전처리]
    M2 --> M3[모듈 3: 회귀]
    M3 --> M4[모듈 4: 분류]
    M4 --> M7[모듈 7: 평가]
    M7 --> Practice[연습문제]
    Practice --> End[초급 트랙 완료]
```

### 중급 사용자용
```mermaid
flowchart TD
    Start[학습 시작] --> Review[모듈 1-4 검토]
    Review --> M5[모듈 5: 비지도]
    M5 --> M6[모듈 6: 모델 선택]
    M6 --> M7[모듈 7: 평가]
    M7 --> M8[모듈 8: 파이프라인]
    M8 --> M9[모듈 9: 고급]
    M9 --> Practice[고급 연습]
    Practice --> End[중급 트랙 완료]
```

### 고급 사용자용
```mermaid
flowchart TD
    Start[학습 시작] --> M6[모듈 6: 모델 선택]
    M6 --> M8[모듈 8: 파이프라인]
    M8 --> M9[모듈 9: 고급]
    M9 --> M10[모듈 10: 배포]
    M10 --> Project[캡스톤 프로젝트]
    Project --> End[고급 트랙 완료]
```

## 핵심 개념 커버리지

```mermaid
mindmap
  root((Scikit-learn 베스트 프랙티스))
    기초
      ML 개념
      API 설계
      설치
    데이터 준비
      전처리
      특성 공학
      문제 처리
    지도 학습
      회귀
      분류
      베스트 프랙티스
    비지도 학습
      군집화
      차원 축소
    모델 최적화
      선택
      튜닝
      평가
    고급 주제
      앙상블
      해석 가능성
      파이프라인
    프로덕션
      배포
      모니터링
      확장