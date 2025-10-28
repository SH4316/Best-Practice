# 3주차: 머신러닝 기초

## 강의 목표
- 머신러닝의 기본 개념과 종류 이해
- 지도 학습과 비지도 학습의 원리와 알고리즘 습득
- 회귀와 분류 문제의 해결 방법 학습
- 과적합과 정규화의 개념과 해결책 이해
- 머신러닝 모델 평가 방법 습득

## 이론 강의 (90분)

### 1. 머신러닝 개요 (20분)

#### 머신러닝의 정의와 역사
**정의**
- 톰 미첼의 정의: "경험(E)을 통해 특정 작업(T)의 성능(P)을 향상시키는 컴퓨터 프로그램"
- 전통적 프로그래밍과의 차이: 명시적 규칙 대신 데이터에서 패턴 학습
- LLM과의 관계: 대규모 언어 모델은 머신러닝의 특정 분야

**머신러닝의 역사**
- 1950년대: 퍼셉트론, 초기 신경망
- 1980-90년대: 결정 트리, 서포트 벡터 머신, 앙상블 방법
- 2000년대: 딥러닝의 부상, 컨볼루션 신경망
- 2010년대-현재: 대규모 모델, 트랜스포머, LLM

#### 머신러닝의 종류

**지도 학습(Supervised Learning)**
- 정의: 레이블된 데이터로부터 입력-출력 관계 학습
- 특징: 정답이 있는 데이터, 명시적 피드백
- 예시: 스팸 메일 분류, 이미지 인식, 번역
- LLM 적용: 지도 미세조정(SFT), 명령어 따르기 학습

**비지도 학습(Unsupervised Learning)**
- 정의: 레이블 없는 데이터에서 숨겨된 구조 발견
- 특징: 정답이 없는 데이터, 암시적 패턴 발견
- 예시: 클러스터링, 차원 축소, 이상 탐지
- LLM 적용: 사전 훈련, 표현 학습

**강화 학습(Reinforcement Learning)**
- 정의: 환경과의 상호작용을 통해 최적 행동 정책 학습
- 특징: 보상과 페널티, 시행착오 학습
- 예시: 게임 AI, 로봇 제어
- LLM 적용: 인간 피드백을 통한 강화 학습(RLHF)

**준지도 학습(Semi-supervised Learning)**
- 정의: 소량의 레이블된 데이터와 대량의 레이블 없는 데이터 활용
- LLM 적용: 자기 지도 학습, 마스크드 언어 모델링

### 2. 지도 학습 (35분)

#### 회귀(Regression)
**개념**
- 정의: 연속적인 목표값 예측
- 목표: 입력 변수와 출력 변수 간의 함수 관계 모델링
- 평가 지표: 평균 제곱 오차(MSE), 평균 절대 오차(MAE), 결정 계수(R²)

**선형 회귀(Linear Regression)**
- 모델: [$y = \mathbf{w}^T\mathbf{x} + b$](mathematical_symbols_explanation.md#linear-regression)
- 손실 함수: [$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$](mathematical_symbols_explanation.md#linear-regression)
- 최적화: 정규 방정식 또는 경사 하강법
- LLM 적용: 언어 모델의 출력 레이어, 회귀 헤드

**비선형 회귀**
- 다항 회귀: [$y = \sum_{i=0}^{d} w_i x^i$](mathematical_symbols_explanation.md#polynomial-regression)
- 결정 트리 회귀: 트리 구조를 이용한 비선형 모델링
- 신경망 회귀: 다층 퍼셉트론을 이용한 복잡한 함수 근사

#### 분류(Classification)
**개념**
- 정의: 이산적인 클래스 레이블 예측
- 이진 분류: 두 개의 클래스 (예: 스팸/정상)
- 다중 분류: 세 개 이상의 클래스 (예: 감성 분류)
- 다중 레이블 분류: 하나의 샘플이 여러 클래스에 속할 수 있음

**로지스틱 회귀(Logistic Regression)**
- 모델: [$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$](mathematical_symbols_explanation.md#logistic-regression)
- 시그모이드 함수: [$\sigma(z) = \frac{1}{1 + e^{-z}}$](mathematical_symbols_explanation.md#logistic-regression)
- 손실 함수: 교차 엔트로피 [$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$](mathematical_symbols_explanation.md#logistic-regression)
- LLM 적용: 다음 단어 예측, 텍스트 분류

**결정 트리(Decision Trees)**
- 원리: 특성 공간을 재귀적으로 분할
- 정보 이득: [$IG = H_{\text{parent}} - \sum_{\text{children}} \frac{N_{\text{child}}}{N_{\text{parent}}} H_{\text{child}}$](mathematical_symbols_explanation.md#decision-trees)
- 지니 불순도: [$G = 1 - \sum_{i=1}^{C} p_i^2$](mathematical_symbols_explanation.md#decision-trees)
- 장점: 해석 가능성, 비선형성 처리
- 단점: 과적합 경향

**서포트 벡터 머신(Support Vector Machines)**
- 원리: 마진 최대화를 통한 분리 초평면 찾기
- 하드 마진: 완벽한 분리
- 소프트 마진: 오류 허용
- 커널 트릭: 비선형 매핑을 통한 고차원 공간에서의 분리
- LLM 적용: 텍스트 분류, 의미 분석

### 3. 비지도 학습 (20분)

#### 클러스터링(Clustering)
**K-평균(K-Means)**
- 원리: 거리 기반 클러스터링
- 알고리즘:
  1. 초기 중심점 선택
  2. 각 데이터를 가장 가까운 중심점에 할당
  3. 중심점 재계산
  4. 수렴할 때까지 반복
- 목적 함수: [$J = \sum_{i=1}^{k}\sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$](mathematical_symbols_explanation.md#k-means-clustering)
- LLM 적용: 문서 클러스터링, 토픽 모델링

**계층적 클러스터링(Hierarchical Clustering)**
- 응집적: 바닥에서 위로 병합
- 분할적: 위에서 아래로 분할
- 덴드로그램: 클러스터링 과정 시각화
- LLM 적용: 의미 유사성에 기반한 단어 계층 구조

#### 차원 축소(Dimensionality Reduction)
**주성분 분석(Principal Component Analysis, PCA)**
- 원리: 데이터의 분산을 최대로 보존하는 저차원 투영
- 고유값 분해를 통한 주성분 찾기
- LLM 적용: 단어 임베딩 차원 축소, 시각화

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- 원리: 고차원 데이터의 국소 구조 보존
- LLM 적용: 단어 임베딩 시각화, 클러스터 구조 확인

### 4. 과적합과 정규화 (15분)

#### 과적합(Overfitting)
**정의와 원인**
- 정의: 훈련 데이터에 과도하게 적응하여 일반화 성능 저하
- 원인: 모델 복잡도 과대, 데이터 부족, 노이즈 과적합
- 증상: 훈련 성능은 높지만 검증/테스트 성능은 낮음

**과적합 진단**
- 학습 곡선: 훈련 손실과 검증 손실의 간격
- 교차 검증: 데이터 분할에 따른 성능 변화
- LLM 적용: 훈련 손실과 검증 손실 모니터링

#### 정규화(Regularization)
**L1 정규화 (Lasso)**
- 손실 함수: [$L = \text{Original Loss} + \lambda\sum_{i}|w_i|$](mathematical_symbols_explanation.md#regularization)
- 특징: 희소성 유도, 특성 선택
- LLM 적용: 모델 압축, 중요한 파라미터 식별

**L2 정규화 (Ridge)**
- 손실 함수: [$L = \text{Original Loss} + \lambda\sum_{i}w_i^2$](mathematical_symbols_explanation.md#regularization)
- 특징: 가중치 감쇠, 작은 가중치 선호
- LLM 적용: 가중치 감쇠, 과적합 방지

**드롭아웃(Dropout)**
- 원리: 훈련 중 무작위로 뉴런 비활성화
- 효과: 앙상블 효과, 과적합 방지
- LLM 적용: 트랜스포머의 드롭아웃 레이어

**조기 종료(Early Stopping)**
- 원리: 검증 성능이 더 이상 향상되지 않을 때 훈련 중단
- LLM 적용: 최적 훈련 시점 결정

## 실습 세션 (90분)

### 1. scikit-learn을 이용한 기본 머신러닝 알고리즘 (30분)

#### 선형 회귀 구현
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"훈련 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"테스트 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
print(f"기울기: {model.coef_[0][0]:.4f}, 절편: {model.intercept_[0]:.4f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.7, label='Training data')
plt.scatter(X_test, y_test, alpha=0.7, label='Test data')
plt.plot(X, model.predict(X), 'r-', label='Linear regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
```

#### 로지스틱 회귀 구현
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 평가
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"훈련 정확도: {train_acc:.4f}")
print(f"테스트 정확도: {test_acc:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_test_pred))

# 결정 경계 시각화
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

plot_decision_boundary(model, X, y)
```

### 2. 과적합과 정규화 실험 (30분)

#### 다항 회귀에서의 과적합
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 비선형 데이터 생성
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 다양한 차수의 다항 회귀 모델
degrees = [1, 2, 10, 15]
plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees):
    # 파이프라인 생성
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression)
    ])
    
    # 모델 학습
    pipeline.fit(X_train, y_train)
    
    # 예측
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # 평가
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # 시각화
    plt.subplot(2, 2, i + 1)
    X_smooth = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_smooth = pipeline.predict(X_smooth)
    
    plt.scatter(X_train, y_train, alpha=0.7, label='Training data')
    plt.scatter(X_test, y_test, alpha=0.7, label='Test data')
    plt.plot(X_smooth, y_smooth, 'r-', label=f'Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Degree {degree}: Train MSE={train_mse:.2f}, Test MSE={test_mse:.2f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
```

#### 정규화 효과 비교
```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# 데이터 생성
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
# 실제 중요한 특성은 3개만
true_coef = np.array([3, -2, 1.5] + [0] * (n_features - 3))
y = X @ true_coef + np.random.randn(n_samples) * 0.5

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 다양한 정규화 강도로 모델 학습
alphas = [0, 0.01, 0.1, 1, 10]

plt.figure(figsize=(15, 5))

# 선형 회귀 (정규화 없음)
plt.subplot(1, 3, 1)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
plt.bar(range(n_features), lr.coef_)
plt.title('Linear Regression (No Regularization)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.xticks(range(n_features))

# Ridge 정규화
plt.subplot(1, 3, 2)
ridge_coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)

ridge_coefs = np.array(ridge_coefs)
for i in range(n_features):
    plt.plot(alphas, ridge_coefs[:, i], label=f'Feature {i}')
plt.xscale('log')
plt.title('Ridge Regularization')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Lasso 정규화
plt.subplot(1, 3, 3)
lasso_coefs = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)
for i in range(n_features):
    plt.plot(alphas, lasso_coefs[:, i], label=f'Feature {i}')
plt.xscale('log')
plt.title('Lasso Regularization')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```

### 3. 교차 검증과 모델 선택 (30분)

#### K-폴드 교차 검증
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# K-폴드 교차 검증
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# 최적의 k 값 찾기
optimal_k = k_values[np.argmax(cv_scores)]
max_score = max(cv_scores)

print(f"최적의 k: {optimal_k}, 최고 교차 검증 정확도: {max_score:.4f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Optimal k Selection')
plt.grid(True)
plt.show()
```

#### 학습 곡선을 통한 과적합 진단
```python
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

# 학습 곡선 생성
def plot_learning_curve(estimator, title, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# SVM 모델의 학습 곡선
plot_learning_curve(SVC(gamma=0.01), "Learning Curve (SVM, gamma=0.01)", X, y, cv=5)
plot_learning_curve(SVC(gamma=0.001), "Learning Curve (SVM, gamma=0.001)", X, y, cv=5)
```

## 과제

### 1. 회귀 과제
- 보스턴 주택 가격 데이터셋에 대한 다양한 회귀 모델 비교
- 다항 회귀에서의 최적 차수 선택
- 정규화 강도에 따른 성능 변화 분석

### 2. 분류 과제
- 와인 데이터셋에 대한 다양한 분류 알고리즘 비교
- 특성 스케일링이 모델 성능에 미치는 영향 분석
- 혼동 행렬을 통한 분류 결과 상세 분석

### 3. 비지도 학습 과제
- K-평균 클러스터링에서 최적 클러스터 수 찾기
- PCA를 이용한 차원 축소 및 시각화
- t-SNE와 PCA의 차이점 비교 분석

## 추가 학습 자료

### 온라인 강의
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [StatQuest with Josh Starmer](https://www.youtube.com/c/statquest)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

### 교재
- "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher M. Bishop
- "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, Jerome Friedman

### 온라인 자료
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Papers with Code](https://paperswithcode.com/)
- [Kaggle Learn](https://www.kaggle.com/learn)