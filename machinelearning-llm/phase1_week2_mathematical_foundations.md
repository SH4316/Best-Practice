# 2주차: 수학적 기초

## 강의 목표
- LLM 개발에 필요한 핵심 수학적 개념 이해
- 선형대수, 미적분학, 확률과 통계의 딥러닝 적용 방법 습득
- 수학적 개념을 코드로 구현하는 능력 배양
- LLM의 수학적 원리에 대한 직관 형성

## 이론 강의 (90분)

### 1. 선형대수 (30분)

#### 벡터와 행렬 기초
**벡터(Vector)**
- 정의: 크기와 방향을 가진 수학적 객체
- 표기법: $\mathbf{v} = [v_1, v_2, ..., v_n]^T$
- LLM에서의 의미: 단어 임베딩, 특성 벡터
- 연산:
  - 덧셈: $\mathbf{a} + \mathbf{b} = [a_1 + b_1, a_2 + b_2, ..., a_n + b_n]^T$
  - 스칼라 곱: $c\mathbf{v} = [cv_1, cv_2, ..., cv_n]^T$
  - 내적: $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$

**행렬(Matrix)**
- 정의: 수치를 직사각형 배열로 나열한 것
- 표기법: $\mathbf{A} \in \mathbb{R}^{m \times n}$
- LLM에서의 의미: 가중치 행렬, 어텐션 가중치
- 연산:
  - 곱셈: $(\mathbf{AB})_{ij} = \sum_{k} A_{ik} B_{kj}$
  - 전치: $(\mathbf{A}^T)_{ij} = A_{ji}$

#### 고유값 분해와 특이값 분해
**고유값 분해(Eigenvalue Decomposition)**
- 정의: $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$
- 의미: 행렬의 핵심 방향과 스케일링
- LLM 적용: 주성분 분석, 차원 축소

**특이값 분해(Singular Value Decomposition, SVD)**
- 정의: $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$
- 의미: 모든 행렬에 적용 가능한 분해
- LLM 적용: 행렬 근사화, 차원 축소, 잠재 의미 분석

#### 선형대수의 LLM 적용
- **단어 임베딩**: 단어를 고차원 벡터 공간에 표현
- **어텐션 메커니즘**: 쿼리, 키, 값 벡터의 내적 계산
- **가중치 행렬**: 신경망의 파라미터 표현
- **행렬 곱**: 신경망의 순전파 연산

### 2. 미적분학 (30분)

#### 도함수와 편도함수
**도함수(Derivative)**
- 정의: 함수의 순간 변화율
- 표기법: $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$
- LLM에서의 의미: 손실 함수의 기울기

**편도함수(Partial Derivative)**
- 정의: 다변수 함수에서 한 변수에 대한 도함수
- 표기법: $\frac{\partial f}{\partial x_i}$
- LLM에서의 의미: 각 파라미터에 대한 손실의 변화율

#### 연쇄 법칙(Chain Rule)
- 정의: 합성 함수의 도함수 계산 법칙
- 표기법: $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$
- 다변수 연쇄 법칙: $\frac{\partial f}{\partial x} = \sum_{i} \frac{\partial f}{\partial u_i} \cdot \frac{\partial u_i}{\partial x}$
- LLM에서의 의미: 역전파 알고리즘의 수학적 기반

#### 최적화 기초
**경사 하강법(Gradient Descent)**
- 원리: 기울기의 반대 방향으로 이동하여 최소값 찾기
- 수식: $\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$
- LLM 적용: 모델 파라미터 최적화

**고급 최적화 알고리즘**
- 모멘텀: 이전 기울기 정보 활용
- AdaGrad: 파라미터별 학습률 조정
- RMSprop: 이전 기울기의 지수 이동 평균 활용
- Adam: 모멘텀과 RMSprop의 결합

### 3. 확률과 통계 (30분)

#### 확률 분포
**이산 확률 분포**
- 정의: 유한한 표본 공간에서의 확률 분포
- 예시: 베르누이, 이항, 포아송 분포
- LLM 적용: 다음 단어 예측의 확률 분포

**연속 확률 분포**
- 정의: 연속적인 표본 공간에서의 확률 분포
- 예시: 정규 분포, 균등 분포
- LLM 적용: 가중치 초기화, 드롭아웃

#### 베이즈 정리
- 정의: 조건부 확률 관계
- 수식: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- LLM 적용: 베이즈 언어 모델, 불확실성 추정

#### 기댓값과 분산
**기댓값(Expected Value)**
- 정의: 확률 변수의 평균값
- 이산: $E[X] = \sum_{x} x \cdot P(X=x)$
- 연속: $E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$
- LLM 적용: 손실 함수의 기댓값 최소화

**분산(Variance)**
- 정의: 확률 변수의 흩어진 정도
- 수식: $\text{Var}(X) = E[(X - E[X])^2]$
- LLM 적용: 모델 예측의 불확실성 측정

#### 통계적 추론
- 최대 가능도 추정(MLE): 관측된 데이터를 가장 잘 설명하는 파라미터 찾기
- 최대 사후 확률 추정(MAP): 사전 지식을 고려한 파라미터 추정
- LLM 적용: 언어 모델의 파라미터 학습

## 실습 세션 (90분)

### 1. NumPy를 이용한 선형대수 연산 (30분)

#### 벡터와 행렬 기본 연산
```python
import numpy as np

# 벡터 생성
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 벡터 연산
print("벡터 덧셈:", a + b)
print("벡터 내적:", np.dot(a, b))
print("벡터 노름:", np.linalg.norm(a))

# 행렬 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 행렬 연산
print("행렬 곱셈:\n", np.dot(A, B))
print("행렬 전치:\n", A.T)
print("행렬식:", np.linalg.det(A))
```

#### 고유값 분해와 특이값 분해
```python
# 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(A)
print("고유값:", eigenvalues)
print("고유벡터:\n", eigenvectors)

# 특이값 분해
U, S, Vt = np.linalg.svd(A)
print("U:\n", U)
print("특이값:", S)
print("Vt:\n", Vt)

# SVD를 이용한 행렬 재구성
A_reconstructed = U @ np.diag(S) @ Vt
print("재구성된 행렬:\n", A_reconstructed)
```

### 2. 미적분학적 개념의 코드 구현 (30분)

#### 수치적 도함수 계산
```python
def numerical_derivative(f, x, h=1e-5):
    """수치적 도함수 계산"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 예시 함수
def f(x):
    return x**2 + 2*x + 1

# 도함수 계산
x = 2.0
derivative = numerical_derivative(f, x)
print(f"f({x})의 도함수: {derivative}")

# 편도함수 계산
def numerical_partial_derivative(f, x, var_idx, h=1e-5):
    """다변수 함수의 편도함수 계산"""
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[var_idx] += h
    x_minus[var_idx] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)

# 다변수 함수 예시
def g(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]

x = np.array([1.0, 2.0])
partial_x0 = numerical_partial_derivative(g, x, 0)
partial_x1 = numerical_partial_derivative(g, x, 1)
print(f"g({x})의 x0에 대한 편도함수: {partial_x0}")
print(f"g({x})의 x1에 대한 편도함수: {partial_x1}")
```

#### 경사 하강법 구현
```python
def gradient_descent(f, grad_f, initial_x, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """경사 하강법 구현"""
    x = initial_x.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - learning_rate * gradient
        
        # 수렴 확인
        if np.linalg.norm(x_new - x) < tolerance:
            break
            
        x = x_new
        history.append(x.copy())
        
    return x, history

# 2차 함수 예시: f(x) = x^2 + y^2
def quadratic_function(x):
    return x[0]**2 + x[1]**2

def quadratic_gradient(x):
    return np.array([2*x[0], 2*x[1]])

# 경사 하강법 실행
initial_x = np.array([3.0, 4.0])
optimal_x, history = gradient_descent(quadratic_function, quadratic_gradient, initial_x)

print(f"초기값: {initial_x}")
print(f"최적값: {optimal_x}")
print(f"최소 함수값: {quadratic_function(optimal_x)}")
```

### 3. 확률과 통계 구현 (30분)

#### 확률 분포 시각화
```python
import matplotlib.pyplot as plt

# 정규 분포
def normal_distribution(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# 정규 분포 시각화
x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, normal_distribution(x), label='Standard Normal')
plt.plot(x, normal_distribution(x, mu=1, sigma=0.5), label='N(1, 0.5)')
plt.plot(x, normal_distribution(x, mu=-1, sigma=2), label='N(-1, 2)')
plt.title('Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

#### 베이즈 정리 구현
```python
def bayes_theorem(prior_A, likelihood_B_given_A, marginal_B):
    """베이즈 정리 계산"""
    return (likelihood_B_given_A * prior_A) / marginal_B

# 예시: 질병 진단
# 사전 확률: 특정 질병에 걸릴 확률
prior_disease = 0.01

# 우도: 질병이 있을 때 검사가 양성일 확률
likelihood_positive_given_disease = 0.99

# 주변 확률: 검사가 양성일 전체 확률
# P(Positive) = P(Positive|Disease)P(Disease) + P(Positive|No Disease)P(No Disease)
likelihood_positive_given_no_disease = 0.05
marginal_positive = (likelihood_positive_given_disease * prior_disease + 
                    likelihood_positive_given_no_disease * (1 - prior_disease))

# 사후 확률: 검사가 양성일 때 실제로 질병이 있을 확률
posterior_disease_given_positive = bayes_theorem(prior_disease, 
                                               likelihood_positive_given_disease, 
                                               marginal_positive)

print(f"검사가 양성일 때 실제로 질병이 있을 확률: {posterior_disease_given_positive:.4f}")
```

#### 기댓값과 분산 계산
```python
def calculate_expectation(values, probabilities):
    """이산 확률 변수의 기댓값 계산"""
    return np.sum(values * probabilities)

def calculate_variance(values, probabilities):
    """이산 확률 변수의 분산 계산"""
    expectation = calculate_expectation(values, probabilities)
    return np.sum((values - expectation)**2 * probabilities)

# 주사위 예시
values = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

expectation = calculate_expectation(values, probabilities)
variance = calculate_variance(values, probabilities)

print(f"주사위의 기댓값: {expectation}")
print(f"주사위의 분산: {variance}")
print(f"주사위의 표준편차: {np.sqrt(variance)}")
```

## 과제

### 1. 선형대수 과제
- 3x3 행렬의 고유값 분해와 특이값 분해 구현
- SVD를 이용한 이미지 압축 시뮬레이션
- 단어 임베딩 벡터 간의 코사인 유사도 계산

### 2. 미적분학 과제
- 다양한 활성화 함수(sigmoid, tanh, ReLU)의 도함수 구현
- 2차원 함수에서의 경사 하강법 시각화
- 연쇄 법칙을 이용한 간단한 신경망의 역전파 구현

### 3. 확률과 통계 과제
- 다양한 확률 분포의 시각화와 특성 분석
- 베이즈 정리를 이용한 스팸 필터 간단 구현
- 최대 가능도 추정을 이용한 정규 분포 파라미터 추정

## 추가 학습 자료

### 온라인 강의
- [3Blue1Brown - Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [3Blue1Brown - Calculus](https://www.3blue1brown.com/topics/calculus)
- [Khan Academy - Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)

### 교재
- "Linear Algebra and Its Applications" by Gilbert Strang
- "Calculus" by James Stewart
- "Probability and Statistics" by Morris H. DeGroot

### 온라인 자료
- [Mathematics for Machine Learning](https://mml-book.com/)
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
- [Probability Course](https://www.probabilitycourse.com/)