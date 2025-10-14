# 모듈 1: 이론적 개념

## 머신러닝 기초

### 머신러닝이란?

머신러닝은 인공지능의 하위 분야로, 명시적으로 프로그래밍되지 않고도 경험을 통해 특정 작업에서 성능을 향상시킬 수 있는 알고리즘과 통계 모델을 개발하는 데 중점을 둡니다. 머신러닝 시스템은 엄격한 지침을 따르는 대신 데이터에서 패턴을 학습하여 예측이나 결정을 내립니다.

### 머신러닝의 유형

#### 1. 지도 학습 (Supervised Learning)
지도 학습은 입력 특성을 알려진 출력 레이블에 매핑하는 것을 학습하는 레이블된 데이터에서 학습하는 것을 포함합니다.

**주요 특징:**
- 훈련 데이터에는 입력 특성과 타겟 레이블이 모두 포함됨
- 목표: 새로운 입력에 대한 출력을 예측하는 매핑 함수 학습
- 일반적인 작업: 분류(Classification)와 회귀(Regression)

**예시:**
- 이메일 스팸 감지 (분류)
- 주택 가격 예측 (회귀)
- 이미지 인식 (분류)

#### 2. 비지도 학습 (Unsupervised Learning)
비지도 학습은 미리 정의된 타겟 변수 없이 레이블되지 않은 데이터에서 패턴을 찾는 것을 포함합니다.

**주요 특징:**
- 훈련 데이터에는 입력 특성만 포함됨
- 목표: 데이터에서 숨겨진 패턴이나 구조 발견
- 일반적인 작업: 군집화(Clustering)와 차원 축소(Dimensionality Reduction)

**예시:**
- 고객 세분화 (군집화)
- 토픽 모델링 (군집화)
- 데이터 압축 (차원 축소)

#### 3. 강화 학습 (Reinforcement Learning)
강화 학습은 환경과 상호작용하고 보상이나 페널티를 받으면서 결정을 내리는 것을 학습하는 에이전트를 포함합니다.

**주요 특징:**
- 에이전트는 시행착오를 통해 학습
- 목표: 시간에 따른 누적 보상 최대화
- 일반적인 작업: 게임 플레이, 로보틱스, 제어 시스템

**예시:**
- 게임 플레이 (체스, 바둑)
- 로봇 내비게이션
- 자원 관리

### 머신러닝 워크플로우

일반적인 머신러닝 프로젝트는 다음 단계를 따릅니다:

1. **문제 정의**: 비즈니스 문제와 ML 목표 정의
2. **데이터 수집**: 다양한 소스에서 관련 데이터 수집
3. **데이터 전처리**: 모델링을 위해 데이터 정리, 변환, 준비
4. **특성 공학**: 관련 특성 선택 및 생성
5. **모델 선택**: 문제에 적합한 알고리즘 선택
6. **모델 훈련**: 준비된 데이터로 모델 훈련
7. **모델 평가**: 적절한 지표를 사용한 모델 성능 평가
8. **모델 튜닝**: 더 나은 성능을 위한 하이퍼파라미터 최적화
9. **모델 배포**: 프로덕션 환경에 모델 배포
10. **모니터링 및 유지보수**: 성능 모니터링 및 필요시 업데이트

### 핵심 용어

- **특성(Features)**: 예측을 위해 사용되는 입력 변수나 속성
- **타겟/레이블(Target/Label)**: 예측하려는 출력 변수
- **훈련 세트(Training Set)**: 모델 훈련에 사용되는 데이터
- **검증 세트(Validation Set)**: 하이퍼파라미터 튜닝에 사용되는 데이터
- **테스트 세트(Test Set)**: 최종 모델 성능 평가에 사용되는 데이터
- **과적합(Overfitting)**: 모델이 훈련 데이터를 너무 잘 학습하여 새로운 데이터에 일반화하지 못하는 현상
- **과소적합(Underfitting)**: 모델이 너무 단순하여 기본 패턴을 포착하지 못하는 현상
- **편향-분산 트레이드오프(Bias-Variance Tradeoff)**: 모델 단순성과 복잡성 사이의 균형

## scikit-learn 소개

### 역사와 철학

scikit-learn은 Python용 오픈소스 머신러닝 라이브러리로, 데이터 마이닝과 데이터 분석을 위한 간단하고 효율적인 도구를 제공합니다. 2007년 David Cournapeau가 Google Summer of Code 프로젝트의 일부로 처음 개발했으며, 그 이후로 가장 인기 있는 머신러닝 라이브러리 중 하나로 성장했습니다.

**핵심 철학:**
- **일관성**: 모든 객체가 일관된 인터페이스 공유
- **검사**: 모든 파라미터가 공개 속성으로 노출
- **제한된 객체 계층**: 알고리즘만 클래스로 표현
- **구성**: 많은 머신러닝 작업이 더 기본적인 알고리즘의 시퀀스로 표현 가능
- **합리적인 기본값**: 라이브러리가 합리적인 기본 파라미터 값 제공

### 주요 특징과 장점

1. **간단하고 일관된 API**: 모든 알고리즘에 걸쳐 통합된 인터페이스
2. **포괄적인 알고리즘 coverage**: 광범위한 지도 및 비지도 학습 알고리즘
3. **훌륭한 문서**: 예제와 함께 상세한 문서
4. **활발한 커뮤니티**: 사용자와 기여자의 대규모 커뮤니티
5. **Python 생태계와의 통합**: NumPy, pandas, matplotlib과 원활하게 작동
6. **성능 최적화**: Cython으로 효율적인 구현
7. **모델 지속성**: 모델 저장 및 로드에 대한 내장 지원

### scikit-learn 생태계

scikit-learn은 더 큰 Python 데이터 과학 생태계의 일부입니다:

- **NumPy**: 과학 컴퓨팅을 위한 기본 패키지
- **pandas**: 데이터 조작 및 분석 라이브러리
- **matplotlib**: 데이터 시각화 라이브러리
- **SciPy**: 과학 컴퓨팅 라이브러리
- **Jupyter**: 대화형 컴퓨팅 환경

## 설치 및 설정

### scikit-learn 설치

#### pip 사용 (권장)
```bash
pip install -U scikit-learn
```

#### conda 사용
```bash
conda install scikit-learn
```

### 설치 확인

```python
import sklearn
print(f"scikit-learn 버전: {sklearn.__version__}")
```

### 의존성

scikit-learn은 다음 패키지를 필요로 합니다:
- NumPy (>= 1.19.1)
- SciPy (>= 1.6.0)
- joblib (>= 1.0.0)
- threadpoolctl (>= 2.0.0)

향상된 기능을 위한 선택적 의존성:
- matplotlib (플로팅용)
- pandas (데이터 조작용)
- seaborn (통계 시각화용)

## scikit-learn API 디자인

### 추정기 API

scikit-learn API 디자인의 핵심은 추정기 인터페이스로, 모든 머신러닝 알고리즘으로 작업하는 일관된 방법을 제공합니다.

#### 주요 구성 요소

1. **추정기 객체**: 모든 알고리즘이 객체로 구현됨
2. **fit() 메서드**: 훈련 데이터에서 추정기 훈련
3. **predict() 메서드**: 새로운 데이터에 대한 예측
4. **transform() 메서드**: 데이터 변환 (전처리용)

#### API 일관성

scikit-learn의 모든 추정기는 다음 규칙을 따릅니다:

```python
# 모든 추정기의 공통 패턴
estimator = Estimator(param1=value1, param2=value2)
estimator.fit(X_train, y_train)  # 데이터로부터 학습
predictions = estimator.predict(X_test)  # 예측
```

### 추정기 유형

#### 1. 예측기 (지도 학습)
예측기는 `fit()`과 `predict()` 메서드를 모두 가집니다.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 2. 변환기 (전처리)
변환기는 `fit()`과 `transform()` 메서드를 가지며, 종종 결합된 `fit_transform()` 메서드를 가집니다.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # 스케일링 파라미터 학습
X_train_scaled = scaler.transform(X_train)  # 변환 적용
X_test_scaled = scaler.transform(X_test)
```

#### 3. 군집화기 (비지도 학습)
군집화기는 `fit()`과 `predict()` 메서드를 가지지만, `predict()`는 새로운 점을 기존 군집에 할당합니다.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)  # 군집 학습
cluster_labels = kmeans.predict(X)  # 점을 군집에 할당
```

### 파라미터 접근 및 모델 속성

모든 추정기 파라미터는 공개 속성으로 접근 가능:

```python
model = LinearRegression(fit_intercept=True)
print(model.fit_intercept)  # 파라미터 접근
```

학습된 파라미터 (밑줄로 끝나는 속성)은 피팅 후 사용 가능:

```python
model.fit(X_train, y_train)
print(model.coef_)  # 학습된 계수
print(model.intercept_)  # 학습된 절편
```

## 재현 가능한 머신러닝을 위한 모범 사례

### 랜덤 시드 설정

재현 가능한 결과를 위해 항상 랜덤 시드를 설정하세요:

```python
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
```

### 데이터 분할

데이터 유출을 피하기 위해 항상 전처리 전에 데이터를 분할하세요:

```python
# 올바른 접근 방식
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 테스트 데이터에 동일한 스케일러 사용
```

### 문서화 및 코드 구성

- 의미 있는 변수 이름 사용
- 주요 결정을 설명하는 주석 추가
- 코드를 논리적 섹션으로 구성
- 모델 파라미터와 버전 추적

## 일반적인 함정 및 피하는 방법

### 1. 데이터 유출
**문제**: 훈련 중에 테스트 세트의 정보 사용
**해결책**: 항상 전처리 전에 데이터를 분하고 훈련 데이터만 전처리기 피팅에 사용

### 2. 랜덤 상태 무시
**문제**: 실행 간에 결과가 다름
**해결책**: 재현성을 위해 항상 random_state 파라미터 설정

### 3. 모델 가정 이해 부족
**문제**: 데이터에 부적절한 모델 사용
**해결책**: 각 알고리즘의 가정과 제한 사항 학습

### 4. 훈련 데이터에 과적합
**문제**: 모델이 훈련 데이터에서는 잘 동작하지만 새로운 데이터에서는 성능이 저하됨
**해결책**: 적절한 검증 기법과 정규화 사용

이 이론적 기초는 scikit-learn의 디자인 철학과 모범 사례를 이해하는 기반을 제공합니다. 다음 섹션에서는 이 개념들을 실용적인 코드 예제를 통해 적용해 보겠습니다.