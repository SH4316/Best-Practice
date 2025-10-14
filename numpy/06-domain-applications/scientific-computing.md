# NumPy 과학 계산 응용

## 과학 계산에서의 NumPy

NumPy는 과학 계산의 핵심 라이브러리로, 수치 해석, 미분 방정식, 선형대수, 통계물리 등 다양한 과학 분야에서 사용됩니다. SciPy, Matplotlib 등 다른 과학 계산 라이브러리의 기반이 됩니다.

## 수치 해석

### 수치 미분

```python
import numpy as np

# 전진 차분법 (Forward Difference)
def forward_diff(f, x, h=1e-5):
    """전진 차분법을 이용한 수치 미분"""
    return (f(x + h) - f(x)) / h

# 후진 차분법 (Backward Difference)
def backward_diff(f, x, h=1e-5):
    """후진 차분법을 이용한 수치 미분"""
    return (f(x) - f(x - h)) / h

# 중앙 차분법 (Central Difference)
def central_diff(f, x, h=1e-5):
    """중앙 차분법을 이용한 수치 미분"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 테스트 함수
def f(x):
    return x**3 + 2*x**2 + x + 1

def df(x):
    """진 미분값"""
    return 3*x**2 + 4*x + 1

# 테스트 지점
x_values = np.array([0, 1, 2, 3])
h = 1e-5

print("수치 미분 비교:")
print("x\t진 미분\t전진 차분\t후진 차분\t중앙 차분")
for x in x_values:
    true_val = df(x)
    forward_val = forward_diff(f, x, h)
    backward_val = backward_diff(f, x, h)
    central_val = central_diff(f, x, h)
    
    print(f"{x}\t{true_val:.6f}\t{forward_val:.6f}\t{backward_val:.6f}\t{central_val:.6f}")

# 오차 분석
h_values = np.logspace(-1, -10, 10)
x = 1.0
true_val = df(x)

errors_forward = np.abs(forward_diff(f, x, h_values) - true_val)
errors_backward = np.abs(backward_diff(f, x, h_values) - true_val)
errors_central = np.abs(central_diff(f, x, h_values) - true_val)

print("\n오차 분석 (h=1e-5):")
print(f"전진 차분 오차: {errors_forward[4]:.10f}")
print(f"후진 차분 오차: {errors_backward[4]:.10f}")
print(f"중앙 차분 오차: {errors_central[4]:.10f}")
```

### 수치 적분

```python
# 사각형 법칙 (Rectangle Rule)
def rectangle_rule(f, a, b, n=1000):
    """사각형 법칙을 이용한 수치 적분"""
    h = (b - a) / n
    x = np.linspace(a, b - h, n)  # 마지막 점 제외
    return h * np.sum(f(x))

# 사다리꼴 법칙 (Trapezoidal Rule)
def trapezoidal_rule(f, a, b, n=1000):
    """사다리꼴 법칙을 이용한 수치 적분"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# 심슨 법칙 (Simpson's Rule)
def simpson_rule(f, a, b, n=1000):
    """심슨 법칙을 이용한 수치 적분 (n은 짝수)"""
    if n % 2 != 0:
        n += 1  # 짝수로 만들기
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

# 테스트 함수
def f(x):
    return x**2

def F(x):
    """진 적분값"""
    return x**3 / 3

# 적분 구간
a, b = 0, 1
true_val = F(b) - F(a)

# 수치 적분
n = 100
rect_val = rectangle_rule(f, a, b, n)
trap_val = trapezoidal_rule(f, a, b, n)
simpson_val = simpson_rule(f, a, b, n)

print("수치 적분 비교:")
print(f"진 적분값: {true_val:.10f}")
print(f"사각형 법칙: {rect_val:.10f} (오차: {abs(rect_val - true_val):.10f})")
print(f"사다리꼴 법칙: {trap_val:.10f} (오차: {abs(trap_val - true_val):.10f})")
print(f"심슨 법칙: {simpson_val:.10f} (오차: {abs(simpson_val - true_val):.10f})")
```

### 방정식의 근 찾기

```python
# 이분법 (Bisection Method)
def bisection(f, a, b, tol=1e-6, max_iter=100):
    """이분법을 이용한 근 찾기"""
    if f(a) * f(b) >= 0:
        raise ValueError("f(a)와 f(b)의 부호가 달라야 합니다.")
    
    for i in range(max_iter):
        c = (a + b) / 2
        if f(c) == 0 or (b - a) / 2 < tol:
            return c, i + 1
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2, max_iter

# 뉴턴-랩슨법 (Newton-Raphson Method)
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """뉴턴-랩슨법을 이용한 근 찾기"""
    x = x0
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if dfx == 0:
            raise ValueError("도함수가 0이 되면 안 됩니다.")
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new, i + 1
        
        x = x_new
    
    return x, max_iter

# 테스트 함수
def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

# 근 찾기
print("방정식 근 찾기 (x³ - x - 2 = 0):")

# 이분법
root_bisection, iter_bisection = bisection(f, 1, 2)
print(f"이분법: 근 = {root_bisection:.6f}, 반복 = {iter_bisection}")

# 뉴턴-랩슨법
root_newton, iter_newton = newton_raphson(f, df, 1.5)
print(f"뉴턴-랩슨법: 근 = {root_newton:.6f}, 반복 = {iter_newton}")

# 검증
print(f"근 검증: f({root_bisection:.6f}) = {f(root_bisection):.10f}")
```

## 미분 방정식

### 오일러 방법 (Euler's Method)

```python
# 오일러 방법을 이용한 1차 미분 방정식 해석
def euler_method(f, t0, y0, t_end, h):
    """오일러 방법을 이용한 미분 방정식 수치 해석"""
    n_steps = int((t_end - t0) / h) + 1
    t = np.linspace(t0, t_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    
    return t, y

# 테스트 미분 방정식: dy/dt = -2t * y², y(0) = 1
def f(t, y):
    return -2 * t * y**2

# 해석해: y(t) = 1 / (1 + t²)
def analytical_solution(t):
    return 1 / (1 + t**2)

# 수치 해석
t0, y0 = 0, 1
t_end = 2
h = 0.1

t_num, y_num = euler_method(f, t0, y0, t_end, h)
y_analytical = analytical_solution(t_num)

# 오차 계산
error = np.abs(y_num - y_analytical)
max_error = np.max(error)

print("오일러 방법 미분 방정식 해석:")
print(f"스텝 크기: {h}")
print(f"최대 오차: {max_error:.6f}")

# 다른 스텝 크기와 비교
h_values = [0.5, 0.2, 0.1, 0.05, 0.01]
print("\n스텝 크기에 따른 최대 오차:")
for h_val in h_values:
    t_val, y_val = euler_method(f, t0, y0, t_end, h_val)
    y_analytical_val = analytical_solution(t_val)
    error_val = np.max(np.abs(y_val - y_analytical_val))
    print(f"h = {h_val}: 최대 오차 = {error_val:.6f}")
```

### 룽게-쿠타 방법 (Runge-Kutta Method)

```python
# 4차 룽게-쿠타 방법 (RK4)
def rk4_method(f, t0, y0, t_end, h):
    """4차 룽게-쿠타 방법을 이용한 미분 방정식 수치 해석"""
    n_steps = int((t_end - t0) / h) + 1
    t = np.linspace(t0, t_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    
    for i in range(n_steps - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h/2, y[i] + k2/2)
        k4 = h * f(t[i] + h, y[i] + k3)
        
        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y

# 테스트 미분 방정식: dy/dt = -2t * y², y(0) = 1
def f(t, y):
    return -2 * t * y**2

# 해석해: y(t) = 1 / (1 + t²)
def analytical_solution(t):
    return 1 / (1 + t**2)

# 수치 해석 비교
t0, y0 = 0, 1
t_end = 2
h = 0.2

# 오일러 방법
t_euler, y_euler = euler_method(f, t0, y0, t_end, h)
error_euler = np.max(np.abs(y_euler - analytical_solution(t_euler)))

# RK4 방법
t_rk4, y_rk4 = rk4_method(f, t0, y0, t_end, h)
error_rk4 = np.max(np.abs(y_rk4 - analytical_solution(t_rk4)))

print("미분 방정식 해석 방법 비교:")
print(f"스텝 크기: {h}")
print(f"오일러 방법 최대 오차: {error_euler:.6f}")
print(f"RK4 방법 최대 오차: {error_rk4:.6f}")
print(f"오차 비율: {error_euler/error_rk4:.1f}배")
```

## 선형대수 응용

### 연립 방정식 해결

```python
# 가우스 소거법 (Gaussian Elimination)
def gaussian_elimination(A, b):
    """가우스 소거법을 이용한 연립 방정식 해결"""
    n = len(b)
    # 증강 행렬 생성
    augmented = np.column_stack([A, b])
    
    # 전진 소거
    for i in range(n):
        # 부분 피벗팅
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # 대각 원소로 나누기
        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot
        
        # 아래 행들에서 0 만들기
        for j in range(i + 1, n):
            factor = augmented[j, i]
            augmented[j] = augmented[j] - factor * augmented[i]
    
    # 후진 대입
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i, -1] - np.sum(augmented[i, i+1:-1] * x[i+1:])
    
    return x

# 테스트
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

# 가우스 소거법
x_gauss = gaussian_elimination(A, b)

# NumPy 내장 함수
x_numpy = np.linalg.solve(A, b)

print("연립 방정식 해결:")
print("계수 행렬 A:")
print(A)
print("상수 벡터 b:", b)
print(f"가우스 소거법 해: {x_gauss}")
print(f"NumPy 해: {x_numpy}")
print(f"해 차이: {np.max(np.abs(x_gauss - x_numpy)):.10f}")

# 검증
print(f"검증 Ax = b: {A @ x_gauss}")
```

### 고유값 문제

```python
# 거듭제곱법 (Power Method) - 최대 고유값 계산
def power_method(A, max_iter=1000, tol=1e-6):
    """거듭제곱법을 이용한 최대 고유값과 고유벡터 계산"""
    n = A.shape[0]
    
    # 초기 벡터
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for i in range(max_iter):
        x_new = A @ x
        eigenvalue = np.dot(x, x_new)
        x_new = x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            return eigenvalue, x_new, i + 1
        
        x = x_new
    
    return eigenvalue, x, max_iter

# 역거듭제곱법 (Inverse Power Method) - 최소 고유값 계산
def inverse_power_method(A, max_iter=1000, tol=1e-6):
    """역거듭제곱법을 이용한 최소 고유값과 고유벡터 계산"""
    n = A.shape[0]
    
    # 초기 벡터
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    # LU 분해 (실제로는 더 효율적인 방법 사용)
    for i in range(max_iter):
        # Ax = x_new 풀기 (실제로는 LU 분해 사용)
        x_new = np.linalg.solve(A, x)
        eigenvalue = np.dot(x, x_new)
        x_new = x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            return 1/eigenvalue, x_new, i + 1
        
        x = x_new
    
    return 1/eigenvalue, x, max_iter

# 테스트 행렬
A = np.array([[4, -2, 1], 
              [-2, 4, -2], 
              [1, -2, 3]], dtype=float)

# NumPy 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(A)
max_eigenval_idx = np.argmax(np.abs(eigenvalues))
min_eigenval_idx = np.argmin(np.abs(eigenvalues))

numpy_max_eigenval = eigenvalues[max_eigenval_idx]
numpy_min_eigenval = eigenvalues[min_eigenval_idx]

# 거듭제곱법
power_max_eigenval, power_eigvec, power_iter = power_method(A)
inverse_min_eigenval, inverse_eigvec, inverse_iter = inverse_power_method(A)

print("고유값 계산:")
print(f"NumPy 최대 고유값: {numpy_max_eigenval:.6f}")
print(f"거듭제곱법 최대 고유값: {power_max_eigenval:.6f} (반복: {power_iter})")
print(f"NumPy 최소 고유값: {numpy_min_eigenval:.6f}")
print(f"역거듭제곱법 최소 고유값: {inverse_min_eigenval:.6f} (반복: {inverse_iter})")
```

## 푸리에 변환

### 이산 푸리에 변환 (DFT)

```python
# 이산 푸리에 변환 (DFT)
def dft(x):
    """이산 푸리에 변환"""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

# 역 이산 푸리에 변환 (IDFT)
def idft(X):
    """역 이산 푸리에 변환"""
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, X) / N

# 테스트 신호
N = 100
t = np.linspace(0, 1, N)
freq1, freq2 = 5, 15  # 5Hz와 15Hz 성분
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# DFT 계산
dft_result = dft(signal)

# 주파수 축
freqs = np.fft.fftfreq(N, d=t[1] - t[0])

# 진폭 스펙트럼
amplitude = np.abs(dft_result)

# NumPy FFT와 비교
fft_result = np.fft.fft(signal)

print("이산 푸리에 변환 비교:")
print(f"신호 길이: {N}")
print(f"최대 진폭 (DFT): {np.max(amplitude):.6f}")
print(f"최대 진폭 (FFT): {np.max(np.abs(fft_result)):.6f}")
print(f"최대 차이: {np.max(np.abs(dft_result - fft_result)):.10f}")

# 주파수 성분 확인
positive_freqs = freqs[:N//2]
positive_amplitude = amplitude[:N//2]

# 가장 큰 진폭을 가진 주파수
top_indices = np.argsort(positive_amplitude)[-3:][::-1]  # 상위 3개
print("\n주요 주파수 성분:")
for idx in top_indices:
    print(f"주파수: {positive_freqs[idx]:.1f}Hz, 진폭: {positive_amplitude[idx]:.6f}")
```

## 다음 학습 내용

다음으로는 이미지 처리 응용에 대해 알아보겠습니다. [`image-processing.md`](image-processing.md)를 참조하세요.