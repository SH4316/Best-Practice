# NumPy 벡터화(Vectorization)

## 벡터화란?

벡터화는 반복문 대신 배열 연산을 사용하여 데이터를 한 번에 처리하는 기법입니다. NumPy의 벡터화 연산은 C나 Fortran으로 구현되어 있어 Python 반복문보다 훨씬 빠릅니다.

## 벡터화의 장점

1. **성능 향상**: C 수준의 최적화된 코드로 실행
2. **코드 간결성**: 반복문 없이 간결한 표현
3. **가독성**: 의도가 명확하게 드러남
4. **메모리 효율성**: 임시 변수 생성 최소화

## 기본 벡터화 연산

### 산술 연산 벡터화

```python
import numpy as np
import time

# 큰 데이터 생성
size = 1000000
a = np.random.rand(size)
b = np.random.rand(size)

# Python 반복문 (느림)
start = time.time()
result_loop = np.empty(size)
for i in range(size):
    result_loop[i] = a[i] + b[i]
loop_time = time.time() - start

# NumPy 벡터화 (빠름)
start = time.time()
result_vectorized = a + b
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

### 수학 함수 벡터화

```python
# 데이터 생성
x = np.linspace(0, 2*np.pi, 1000000)

# Python 반복문
start = time.time()
result_loop = np.empty_like(x)
for i in range(len(x)):
    result_loop[i] = np.sin(x[i]) * np.cos(x[i])
loop_time = time.time() - start

# NumPy 벡터화
start = time.time()
result_vectorized = np.sin(x) * np.cos(x)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

## 조건부 연산 벡터화

### where 함수 사용

```python
# 데이터 생성
data = np.random.randn(1000000)

# Python 반복문
start = time.time()
result_loop = np.empty_like(data)
for i in range(len(data)):
    if data[i] > 0:
        result_loop[i] = 1
    else:
        result_loop[i] = 0
loop_time = time.time() - start

# NumPy where 함수
start = time.time()
result_vectorized = np.where(data > 0, 1, 0)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

### 불리언 인덱싱

```python
# 데이터 생성
data = np.random.randn(1000000)
threshold = 0.5

# Python 반복문
start = time.time()
filtered_loop = []
for i in range(len(data)):
    if data[i] > threshold:
        filtered_loop.append(data[i])
filtered_loop = np.array(filtered_loop)
loop_time = time.time() - start

# NumPy 불리언 인덱싱
start = time.time()
filtered_vectorized = data[data > threshold]
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

## 집계 연산 벡터화

### 축별 연산

```python
# 2차원 데이터 생성
data = np.random.rand(1000, 1000)

# Python 반복문 (행별 합계)
start = time.time()
row_sums_loop = np.empty(1000)
for i in range(1000):
    row_sums_loop[i] = np.sum(data[i, :])
loop_time = time.time() - start

# NumPy 벡터화 (행별 합계)
start = time.time()
row_sums_vectorized = np.sum(data, axis=1)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

### 누적 연산

```python
# 데이터 생성
data = np.random.rand(100000)

# Python 반복문 (누적 합계)
start = time.time()
cumsum_loop = np.empty_like(data)
cumsum_loop[0] = data[0]
for i in range(1, len(data)):
    cumsum_loop[i] = cumsum_loop[i-1] + data[i]
loop_time = time.time() - start

# NumPy 벡터화 (누적 합계)
start = time.time()
cumsum_vectorized = np.cumsum(data)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

## 사용자 정의 함수 벡터화

### vectorize 함수

```python
# 사용자 정의 함수
def my_function(x, y):
    if x > y:
        return x ** 2
    else:
        return y ** 2

# 데이터 생성
x = np.random.rand(100000)
y = np.random.rand(100000)

# Python 반복문
start = time.time()
result_loop = np.empty_like(x)
for i in range(len(x)):
    result_loop[i] = my_function(x[i], y[i])
loop_time = time.time() - start

# NumPy vectorize
my_vectorized = np.vectorize(my_function)
start = time.time()
result_vectorized = my_vectorized(x, y)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

### frompyfunc 함수

```python
# 사용자 정의 함수
def my_function2(x, y):
    return x + y * 2

# 데이터 생성
x = np.random.rand(100000)
y = np.random.rand(100000)

# NumPy frompyfunc
my_ufunc = np.frompyfunc(my_function2, 2, 1)  # (입력 인자 수, 출력 인자 수)
start = time.time()
result_ufunc = my_ufunc(x, y)
ufunc_time = time.time() - start

# NumPy 벡터화 (더 효율적)
start = time.time()
result_vectorized = x + y * 2
vectorized_time = time.time() - start

print(f"frompyfunc 시간: {ufunc_time:.6f}초")
print(f"기본 벡터화 시간: {vectorized_time:.6f}초")
print(f"기본 벡터화가 {ufunc_time/vectorized_time:.1f}배 더 빠름")
```

## 고급 벡터화 기법

### 브로드캐스팅 활용

```python
# 데이터 생성
data = np.random.rand(1000, 100)
weights = np.random.rand(100)

# Python 반복문
start = time.time()
weighted_loop = np.empty((1000, 100))
for i in range(1000):
    for j in range(100):
        weighted_loop[i, j] = data[i, j] * weights[j]
loop_time = time.time() - start

# NumPy 브로드캐스팅
start = time.time()
weighted_vectorized = data * weights  # 자동 브로드캐스팅
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"브로드캐스팅 시간: {vectorized_time:.6f}초")
print(f"브로드캐스팅이 {loop_time/vectorized_time:.1f}배 더 빠름")
```

### 스트라이드 트릭

```python
# 이동 평균 계산
def moving_average_strides(x, window_size):
    """스트라이드 트릭을 이용한 이동 평균"""
    shape = (x.shape[0] - window_size + 1, window_size)
    strides = (x.strides[0], x.strides[0])
    return np.mean(np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides), axis=1)

# 데이터 생성
data = np.random.rand(100000)
window_size = 100

# Python 반복문
start = time.time()
ma_loop = np.empty(len(data) - window_size + 1)
for i in range(len(data) - window_size + 1):
    ma_loop[i] = np.mean(data[i:i+window_size])
loop_time = time.time() - start

# 스트라이드 트릭
start = time.time()
ma_strides = moving_average_strides(data, window_size)
strides_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"스트라이드 트릭 시간: {strides_time:.6f}초")
print(f"스트라이드 트릭이 {loop_time/strides_time:.1f}배 더 빠름")
```

### 누적 연산의 벡터화

```python
# 누적 최대값 계산
def cumulative_max_strides(x):
    """벡터화된 누적 최대값"""
    return np.maximum.accumulate(x)

# 데이터 생성
data = np.random.rand(100000)

# Python 반복문
start = time.time()
cummax_loop = np.empty_like(data)
cummax_loop[0] = data[0]
for i in range(1, len(data)):
    cummax_loop[i] = max(cummax_loop[i-1], data[i])
loop_time = time.time() - start

# NumPy 누적 연산
start = time.time()
cummax_vectorized = np.maximum.accumulate(data)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

## 실용적인 벡터화 예제

### 이미지 처리

```python
# 이미지 데이터 생성 (100x100 RGB)
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# 그레이스케일 변환
def rgb_to_gray(image):
    """RGB를 그레이스케일로 변환"""
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

# Python 반복문
start = time.time()
gray_loop = np.empty((100, 100))
for i in range(100):
    for j in range(100):
        gray_loop[i, j] = 0.299 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 2]
loop_time = time.time() - start

# NumPy 벡터화
start = time.time()
gray_vectorized = rgb_to_gray(image)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

### 금융 데이터 분석

```python
# 주가 데이터 생성
prices = np.random.randn(1000) * 10 + 100
prices = np.cumsum(prices) + 100

# 일일 수익률 계산
def daily_returns(prices):
    """일일 수익률 계산"""
    return (prices[1:] - prices[:-1]) / prices[:-1]

# Python 반복문
start = time.time()
returns_loop = np.empty(len(prices) - 1)
for i in range(1, len(prices)):
    returns_loop[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
loop_time = time.time() - start

# NumPy 벡터화
start = time.time()
returns_vectorized = daily_returns(prices)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

### 신호 처리

```python
# 신호 데이터 생성
t = np.linspace(0, 1, 100000)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))

# 이동 평균 필터
def moving_average_filter(signal, window_size):
    """이동 평균 필터"""
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='valid')

# Python 반복문
window_size = 100
start = time.time()
filtered_loop = np.empty(len(signal) - window_size + 1)
for i in range(len(signal) - window_size + 1):
    filtered_loop[i] = np.mean(signal[i:i+window_size])
loop_time = time.time() - start

# NumPy 컨볼루션
start = time.time()
filtered_vectorized = moving_average_filter(signal, window_size)
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"벡터화가 {loop_time/vectorized_time:.1f}배 더 빠름")
```

## 벡터화 한계와 대안

### 벡터화가 적합하지 않은 경우

```python
# 순차 의존성이 있는 경우 (피보나치 수열)
def fibonacci_sequence(n):
    """피보나치 수열 생성"""
    result = np.empty(n)
    result[0] = 0
    result[1] = 1
    for i in range(2, n):
        result[i] = result[i-1] + result[i-2]
    return result

# 이 경우 벡터화가 어려움
fib = fibonacci_sequence(1000)
print(fib[:10])  # [ 0.  1.  1.  2.  3.  5.  8. 13. 21. 34.]
```

### Numba와의 비교

```python
# Numba를 사용한 JIT 컴파일 (설치 필요: pip install numba)
try:
    from numba import jit
    
    @jit(nopython=True)
    def numba_function(x, y):
        result = np.empty_like(x)
        for i in range(len(x)):
            if x[i] > y[i]:
                result[i] = x[i] ** 2
            else:
                result[i] = y[i] ** 2
        return result
    
    # 데이터 생성
    x = np.random.rand(1000000)
    y = np.random.rand(1000000)
    
    # Numba 실행 (첫 실행은 컴파일 시간 포함)
    start = time.time()
    result_numba = numba_function(x, y)
    numba_time = time.time() - start
    
    # 두 번째 실행 (컴파일된 코드)
    start = time.time()
    result_numba = numba_function(x, y)
    numba_time2 = time.time() - start
    
    print(f"Numba 첫 실행 시간: {numba_time:.6f}초")
    print(f"Numba 두 번째 실행 시간: {numba_time2:.6f}초")
    
except ImportError:
    print("Numba가 설치되지 않았습니다. 'pip install numba'로 설치하세요.")
```

## 벡터화 모범 사례

1. **가능한 한 NumPy 내장 함수 사용**: 내장 함수는 이미 최적화되어 있음
2. **불리언 인덱싱 활용**: 조건부 선택은 불리언 인덱싱으로
3. **축(axis) 매개변수 활용**: 다차원 배열 연산은 축 지정으로
4. **브로드캐스팅 이해**: 자동 브로드캐스팅 규칙을 이해하고 활용
5. **메모리 레이아웃 고려**: C 순서와 Fortran 순서의 차이 이해

## 다음 학습 내용

다음으로는 메모리 관리에 대해 알아보겠습니다. [`memory-management.md`](memory-management.md)를 참조하세요.