# 머신러닝 기초 수식 기호 설명

## 1. 선형 회귀 (Linear Regression) {#linear-regression}

### 모델: $y = \mathbf{w}^T\mathbf{x} + b$

- **$y$**: 예측값 (스칼라)
  - 종속 변수 또는 목표값
  - 회귀 문제에서 예측하고자 하는 연속적인 값

- **$\mathbf{w}$**: 가중치 벡터 (weight vector)
  - $\mathbf{w} = [w_1, w_2, ..., w_n]^T$ 형태의 벡터
  - 각 입력 특성의 중요도나 영향력을 나타내는 파라미터
  - 학습을 통해 최적값이 결정됨

- **$\mathbf{w}^T$**: 가중치 벡터의 전치 (transpose)
  - 벡터를 행 벡터로 변환
  - 내적 계산을 위해 사용

- **$\mathbf{x}$**: 입력 특성 벡터 (input feature vector)
  - $\mathbf{x} = [x_1, x_2, ..., x_n]^T$ 형태의 벡터
  - $x_i$는 i번째 특성값

- **$\mathbf{w}^T\mathbf{x}$**: 내적 (dot product)
  - $\sum_{i=1}^{n} w_i x_i$와 동일
  - 가중치와 입력 특성의 가중합

- **$b$**: 편향 (bias)
  - 절편(intercept)으로도 불림
  - 모든 입력이 0일 때의 예측값
  - 모델의 유연성을 높임

### 손실 함수: $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

- **$L$**: 손실 함수 (loss function)
  - 모델의 예측 오차를 측정하는 함수
  - 최소화해야 할 목표 함수

- **$n$**: 데이터 샘플의 개수
  - 훈련 데이터셋의 전체 샘플 수

- **$\sum_{i=1}^{n}$**: 시그마 (summation)
  - i=1부터 n까지의 합을 의미
  - 모든 데이터 샘플에 대한 오차의 합

- **$y_i$**: i번째 샘플의 실제값 (ground truth)
  - 정답 레이블

- **$\hat{y}_i$**: i번째 샘플의 예측값 (predicted value)
  - 모델이 예측한 값
  - hat(^) 기호는 추정치를 의미

- **$(y_i - \hat{y}_i)^2$**: 제곱 오차 (squared error)
  - 실제값과 예측값의 차이를 제곱
  - 오차의 크기를 측정하고 음수를 양수로 변환

- **$\frac{1}{n}$**: 평균을 구하기 위한 정규화 상수
  - 모든 샘플의 평균 제곱 오차(MSE)를 계산

## 2. 다항 회귀 (Polynomial Regression) {#polynomial-regression}

### 모델: $y = \sum_{i=0}^{d} w_i x^i$

- **$\sum_{i=0}^{d}$**: i=0부터 d까지의 합
  - d는 다항식의 차수(degree)

- **$w_i$**: i차항의 가중치
  - $w_0$는 상수항(절편)
  - $w_1$는 1차항의 계수
  - $w_d$는 d차항의 계수

- **$x^i$**: 입력 변수 x의 i제곱
  - $x^0 = 1$ (상수항)
  - $x^1 = x$ (1차항)
  - $x^2$ (2차항), $x^3$ (3차항) 등

## 3. 로지스틱 회귀 (Logistic Regression) {#logistic-regression}

### 모델: $P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$

- **$P(y=1|\mathbf{x})$**: 조건부 확률
  - 입력 $\mathbf{x}$가 주어졌을 때 $y=1$일 확률
  - 0과 1 사이의 값을 가짐

- **$\sigma(z)$**: 시그모이드 함수 (sigmoid function)
  - 임의의 실수를 0과 1 사이의 값으로 변환
  - 활성화 함수의 일종

### 시그모이드 함수: $\sigma(z) = \frac{1}{1 + e^{-z}}$

- **$e$**: 자연상수 (약 2.71828)
  - 자연 로그의 밑

- **$e^{-z}$**: $e$의 $-z$제곱
  - 지수 함수

- **$\frac{1}{1 + e^{-z}}$**: 시그모이드 함수의 정의
  - 모든 실수 입력에 대해 0과 1 사이의 출력을 반환
  - $z \to \infty$일 때 1에 수렴
  - $z \to -\infty$일 때 0에 수렴
  - $z = 0$일 때 0.5

### 손실 함수: $L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$

- **$-\frac{1}{n}$**: 음수 부호와 평균화
  - 로그 함수의 특성상 손실을 최소화하기 위해 음수 부호 사용
  - 최대화 문제를 최소화 문제로 변환

- **$[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$**: 교차 엔트로피
  - $y_i = 1$일 때: $-\log\hat{y}_i$ (예측이 1에 가까워야 손실이 작아짐)
  - $y_i = 0$일 때: $-\log(1-\hat{y}_i)$ (예측이 0에 가까워야 손실이 작아짐)

- **$\log$**: 자연로그 (natural logarithm)
  - 밑이 $e$인 로그 함수

## 4. 결정 트리 (Decision Trees) {#decision-trees}

### 정보 이득: $IG = H_{\text{parent}} - \sum_{\text{children}} \frac{N_{\text{child}}}{N_{\text{parent}}} H_{\text{child}}$

- **$IG$**: 정보 이득 (Information Gain)
  - 분할 전후의 엔트로피 차이
  - 클수록 좋은 분할

- **$H_{\text{parent}}$**: 부모 노드의 엔트로피
  - 분할 전의 불확실성

- **$\sum_{\text{children}}$**: 모든 자식 노드에 대한 합

- **$\frac{N_{\text{child}}}{N_{\text{parent}}}$**: 가중치
  - 자식 노드의 샘플 수 / 부모 노드의 샘플 수
  - 자식 노드의 상대적 크기를 반영

- **$H_{\text{child}}$**: 자식 노드의 엔트로피
  - 분할 후의 불확실성

### 지니 불순도: $G = 1 - \sum_{i=1}^{C} p_i^2$

- **$G$**: 지니 불순도 (Gini Impurity)
  - 0에 가까울수록 순수 (한 클래스만 존재)
  - 0.5에 가까울수록 불순 (클래스가 균등하게 분포)

- **$C$**: 클래스의 총 개수

- **$p_i$**: i번째 클래스의 비율
  - $p_i = \frac{\text{클래스 } i \text{의 샘플 수}}{\text{전체 샘플 수}}$

- **$p_i^2$**: i번째 클래스 비율의 제곱
  - 클래스의 순도를 측정

- **$\sum_{i=1}^{C} p_i^2$**: 모든 클래스 비율 제곱의 합
  - 순도를 나타내는 지표

## 5. K-평균 클러스터링 (K-Means Clustering) {#k-means-clustering}

### 목적 함수: $J = \sum_{i=1}^{k}\sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$

- **$J$**: 목적 함수 (Objective Function)
  - 클러스터 내 분산의 합
  - 최소화해야 할 목표

- **$k$**: 클러스터의 개수

- **$\sum_{i=1}^{k}$**: 모든 클러스터에 대한 합

- **$C_i$**: i번째 클러스터에 속한 데이터 포인트들의 집합

- **$\mathbf{x} \in C_i$**: 클러스터 $C_i$에 속하는 데이터 포인트 $\mathbf{x}$

- **$\boldsymbol{\mu}_i$**: i번째 클러스터의 중심점 (centroid)
  - 해당 클러스터에 속한 모든 데이터 포인트의 평균

- **$\|\mathbf{x} - \boldsymbol{\mu}_i\|^2$**: 유클리드 거리의 제곱
  - 데이터 포인트와 클러스터 중심점 간의 거리
  - $\|\mathbf{v}\|$는 벡터 $\mathbf{v}$의 노름(norm)을 의미

## 6. 정규화 (Regularization) {#regularization}

### L1 정규화: $L = \text{Original Loss} + \lambda\sum_{i}|w_i|$

- **$\lambda$**: 정규화 강도 (regularization strength)
  - 하이퍼파라미터
  - 클수록 정규화 효과가 강해짐

- **$\sum_{i}|w_i|$**: L1 노름 (L1 norm)
  - 모든 가중치의 절대값 합
  - 희소성(sparse)을 유도하는 효과

### L2 정규화: $L = \text{Original Loss} + \lambda\sum_{i}w_i^2$

- **$\sum_{i}w_i^2$**: L2 노름 (L2 norm)
  - 모든 가중치의 제곱 합
  - 가중치 감쇠(weight decay) 효과

## 7. 일반적인 수학 기호

- **벡터 표기법**:
  - **굵은 소문자**: $\mathbf{x}, \mathbf{w}, \boldsymbol{\mu}$ (벡터)
  - **보통 소문자**: $x, w, \mu$ (스칼라)
  - **보통 대문자**: $X, W, M$ (행렬)

- **첨자 표기법**:
  - **$x_i$**: i번째 요소 또는 i번째 샘플
  - **$w_j$**: j번째 가중치
  - **$\hat{y}_i$**: i번째 샘플의 예측값

- **집합 표기법**:
  - **$\in$**: 속한다 (belongs to)
  - **$\sum$**: 합 (summation)
  - **$\prod$**: 곱 (product)

- **함수 표기법**:
  - **$f(x)$**: 함수 f의 입력 x에 대한 출력
  - **$f: X \to Y$**: 함수 f는 집합 X에서 집합 Y로의 매핑

- **확률 표기법**:
  - **$P(A)$**: 사건 A가 발생할 확률
  - **$P(A|B)$**: B가 주어졌을 때 A가 발생할 조건부 확률
  - **$E[X]$**: 확률 변수 X의 기댓값

- **미적분 표기법**:
  - **$\frac{\partial f}{\partial x}$**: 함수 f를 x에 대한 편미분
  - **$\nabla f$**: 함수 f의 기울기 (gradient)
  - **$\int f(x)dx$**: 함수 f의 적분