# NumPy 커스텀 함수

## 커스텀 함수란?

NumPy는 내장된 유니버설 함수(UFunc) 외에도 사용자가 직접 커스텀 함수를 만들어 벡터화 연산을 수행할 수 있습니다. 이를 통해 NumPy의 성능을 유지하면서 복잡한 연산을 효율적으로 처리할 수 있습니다.

## vectorize 함수

### 기본 vectorize 사용법

```python
import numpy as np

# 일반 Python 함수
def my_function(x, y):
    """두 값 중 더 큰 값의 제곱을 반환"""
    if x > y:
        return x ** 2
    else:
        return y ** 2

# vectorize로 래핑
vectorized_func = np.vectorize(my_function)

# 배열에 적용
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

result = vectorized_func(a, b)
print(f"원본 a: {a}")
print(f"원본 b: {b}")
print(f"결과: {result}")
```

### vectorize 옵션

```python
# 데이터 타입 지정
def add_strings(x, y):
    return f"{x}_{y}"

vectorized_add = np.vectorize(add_strings, otypes=[object])  # 출력 타입 지정

str_arr1 = np.array(['a', 'b', 'c'])
str_arr2 = np.array(['x', 'y', 'z'])

result = vectorized_add(str_arr1, str_arr2)
print(f"문자열 결합 결과: {result}")

# 여러 출력
def calc_stats(x):
    return np.mean(x), np.std(x)

vectorized_stats = np.vectorize(calc_stats, otypes=[float, float])

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
means, stds = vectorized_stats(data)

print(f"평균: {means}")
print(f"표준편차: {stds}")
```

## frompyfunc 함수

### 기본 frompyfunc 사용법

```python
# frompyfunc로 Python 함수를 UFunc로 변환
def my_operation(x, y):
    """복잡한 연산"""
    return (x + y) * (x - y) / 2

# frompyfunc로 변환 (입력 인자 수, 출력 인자 수)
ufunc_operation = np.frompyfunc(my_operation, 2, 1)

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

result = ufunc_operation(a, b)
print(f"frompyfunc 결과: {result}")
print(f"결과 타입: {result.dtype}")  # 객체 타입
```

### frompyfunc와 vectorize 비교

```python
import time

# 성능 비교
def complex_operation(x):
    """복잡한 연산"""
    return np.sin(x) * np.cos(x) + np.sqrt(abs(x))

# vectorize
vec_func = np.vectorize(complex_operation)

# frompyfunc
pyfunc = np.frompyfunc(complex_operation, 1, 1)

# 테스트 데이터
data = np.random.rand(100000)

# vectorize 성능
start = time.time()
result_vec = vec_func(data)
vec_time = time.time() - start

# frompyfunc 성능
start = time.time()
result_pyfunc = pyfunc(data)
pyfunc_time = time.time() - start

# NumPy 내장 함수 (벡터화)
start = time.time()
result_numpy = np.sin(data) * np.cos(data) + np.sqrt(np.abs(data))
numpy_time = time.time() - start

print(f"vectorize 시간: {vec_time:.6f}초")
print(f"frompyfunc 시간: {pyfunc_time:.6f}초")
print(f"NumPy 내장 시간: {numpy_time:.6f}초")
print(f"결과 동일성: {np.allclose(result_vec.astype(float), result_numpy)}")
```

## C/C++ 확장

### Cython을 이용한 성능 최적화

```python
# Cython 예제 (실제로는 .pyx 파일로 작성)
# 이 코드는 설명을 위한 예제이며 실제로는 별도의 Cython 설정 필요

"""
# my_functions.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_sum(np.ndarray[np.float64_t, ndim=1] arr):
    cdef int i
    cdef double total = 0.0
    cdef int n = arr.shape[0]
    
    for i in range(n):
        total += arr[i]
    
    return total
"""

# Cython 함수 사용 예제 (가상 코드)
try:
    # 실제로는 Cython 컴파일 필요
    import my_functions
    
    data = np.random.rand(1000000)
    
    # Python sum
    start = time.time()
    python_sum = sum(data)
    python_time = time.time() - start
    
    # NumPy sum
    start = time.time()
    numpy_sum = np.sum(data)
    numpy_time = time.time() - start
    
    # Cython sum
    start = time.time()
    cython_sum = my_functions.cython_sum(data)
    cython_time = time.time() - start
    
    print(f"Python sum 시간: {python_time:.6f}초")
    print(f"NumPy sum 시간: {numpy_time:.6f}초")
    print(f"Cython sum 시간: {cython_time:.6f}초")
    
except ImportError:
    print("Cython 모듈이 없습니다. 이 예제는 Cython 컴파일이 필요합니다.")
```

### Numba를 이용한 JIT 컴파일

```python
# Numba를 이용한 JIT 컴파일
try:
    from numba import jit
    import numba
    
    # Numba 데코레이터로 함수 최적화
    @jit(nopython=True)
    def numba_complex_operation(x, y):
        """Numba 최적화 함수"""
        result = np.empty_like(x)
        for i in range(len(x)):
            if x[i] > y[i]:
                result[i] = x[i] ** 2
            else:
                result[i] = y[i] ** 2
        return result
    
    # 테스트 데이터
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)
    
    # 첫 실행 (컴파일 포함)
    start = time.time()
    result_numba = numba_complex_operation(a, b)
    first_run_time = time.time() - start
    
    # 두 번째 실행 (컴파일된 코드)
    start = time.time()
    result_numba = numba_complex_operation(a, b)
    second_run_time = time.time() - start
    
    # NumPy 버전
    start = time.time()
    result_numpy = np.where(a > b, a**2, b**2)
    numpy_time = time.time() - start
    
    print(f"Numba 첫 실행 시간: {first_run_time:.6f}초")
    print(f"Numba 두 번째 실행 시간: {second_run_time:.6f}초")
    print(f"NumPy 시간: {numpy_time:.6f}초")
    print(f"결과 동일성: {np.allclose(result_numba, result_numpy)}")
    
except ImportError:
    print("Numba가 설치되지 않았습니다. 'pip install numba'로 설치하세요.")
```

## 고급 커스텀 함수 기법

### generalize 함수

```python
# np.vectorize의 generalize 메서드
def my_func(x, y):
    return x + y

vectorized = np.vectorize(my_func)

# 일반화된 함수 생성
generalized = vectorized.generalize((2, 0))  # (입력 차원, 출력 차원)

# 다차원 배열에 적용
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])

result = generalized(a, b)
print(f"일반화된 함수 결과:\n{result}")
```

### 스트라이드 트릭과 커스텀 함수

```python
# 스트라이드 트릭을 이용한 이동 평균 함수
def moving_average_strides(arr, window_size):
    """스트라이드 트릭을 이용한 효율적인 이동 평균"""
    shape = (arr.shape[0] - window_size + 1, window_size)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return np.mean(windows, axis=1)

# 일반 이동 평균 함수
def moving_average_loop(arr, window_size):
    """반복문을 이용한 이동 평균"""
    result = np.empty(len(arr) - window_size + 1)
    for i in range(len(result)):
        result[i] = np.mean(arr[i:i+window_size])
    return result

# 성능 비교
data = np.random.rand(100000)
window_size = 100

# 스트라이드 트릭
start = time.time()
result_strides = moving_average_strides(data, window_size)
strides_time = time.time() - start

# 반복문
start = time.time()
result_loop = moving_average_loop(data, window_size)
loop_time = time.time() - start

print(f"스트라이드 트릭 시간: {strides_time:.6f}초")
print(f"반복문 시간: {loop_time:.6f}초")
print(f"성능 향상: {loop_time/strides_time:.1f}배")
print(f"결과 동일성: {np.allclose(result_strides, result_loop)}")
```

## 실용적인 커스텀 함수 예제

### 이미지 필터

```python
# 커스텀 이미지 필터
def custom_filter(image, kernel):
    """커스텀 컨볼루션 필터"""
    # 경계 처리를 위한 패딩
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # 결과 배열 초기화
    result = np.empty_like(image)
    
    # 컨볼루션 연산
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            result[i, j] = np.sum(region * kernel)
    
    return result

# 벡터화된 필터
def vectorized_filter(image, kernel):
    """벡터화된 컨볼루션 필터"""
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # 스트라이드 트릭으로 윈도우 생성
    shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1])
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    
    # 벡터화된 컨볼루션
    return np.sum(windows * kernel, axis=(2, 3))

# 테스트
image = np.random.rand(100, 100)
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9  # 평균 필터

# 일반 필터
start = time.time()
result_normal = custom_filter(image, kernel)
normal_time = time.time() - start

# 벡터화된 필터
start = time.time()
result_vectorized = vectorized_filter(image, kernel)
vectorized_time = time.time() - start

print(f"일반 필터 시간: {normal_time:.6f}초")
print(f"벡터화된 필터 시간: {vectorized_time:.6f}초")
print(f"성능 향상: {normal_time/vectorized_time:.1f}배")
print(f"결과 동일성: {np.allclose(result_normal, result_vectorized)}")
```

### 금융 계산

```python
# 금융 옵션 가격 계산 (블랙-숄즈 모델)
def black_scholes(S, K, T, r, sigma):
    """블랙-숄즈 옵션 가격 계산"""
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

# 벡터화된 블랙-숄즈
vectorized_bs = np.vectorize(black_scholes, otypes=[float, float])

# 여러 옵션 가격 계산
S = np.array([100, 105, 110, 115, 120])  # 기초자산 가격
K = 100  # 행사가
T = 0.25  # 만기 (1/4년)
r = 0.05  # 무위험 이자율
sigma = 0.2  # 변동성

call_prices, put_prices = vectorized_bs(S, K, T, r, sigma)

print("기초자산 가격:", S)
print("콜 옵션 가격:", call_prices)
print("풋 옵션 가격:", put_prices)
```

### 신호 처리

```python
# 커스텀 신호 처리 함수
def apply_envelope(signal, window_size=100):
    """신호의 엔벨로프 계산"""
    # 힐버트 변환을 간단히 구현 (실제로는 scipy.signal.hilbert 사용)
    analytic_signal = signal + 1j * np.zeros_like(signal)
    
    # 이동 평균으로 엔벨로프 근사
    envelope = np.zeros_like(signal)
    half_window = window_size // 2
    
    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        envelope[i] = np.sqrt(np.mean(np.abs(analytic_signal[start:end]) ** 2))
    
    return envelope

# 벡터화된 엔벨로프
def vectorized_envelope(signal, window_size=100):
    """벡터화된 엔벨로프 계산"""
    half_window = window_size // 2
    result = np.empty_like(signal)
    
    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        result[i] = np.sqrt(np.mean(signal[start:end] ** 2))
    
    return result

# 테스트 신호
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 10 * t) * np.exp(-t * 2)  # 감쇠 사인파

# 엔벨로프 계산
envelope = apply_envelope(signal)
envelope_vec = vectorized_envelope(signal)

print(f"신호 길이: {len(signal)}")
print(f"엔벨로프 동일성: {np.allclose(envelope, envelope_vec)}")
```

## 커스텀 함수 최적화 팁

### 성능 최적화 전략

```python
# 최적화 전략 비교
def strategy1(data):
    """전략 1: Python 반복문"""
    result = np.empty_like(data)
    for i in range(len(data)):
        result[i] = np.sin(data[i]) * np.cos(data[i])
    return result

def strategy2(data):
    """전략 2: NumPy 벡터화"""
    return np.sin(data) * np.cos(data)

def strategy3(data):
    """전략 3: vectorize 사용"""
    def func(x):
        return np.sin(x) * np.cos(x)
    return np.vectorize(func)(data)

def strategy4(data):
    """전략 4: frompyfunc 사용"""
    def func(x):
        return np.sin(x) * np.cos(x)
    return np.frompyfunc(func, 1, 1)(data).astype(float)

# 성능 비교
data = np.random.rand(100000)

strategies = [strategy1, strategy2, strategy3, strategy4]
strategy_names = ["Python 반복문", "NumPy 벡터화", "vectorize", "frompyfunc"]

for name, strategy in zip(strategy_names, strategies):
    start = time.time()
    result = strategy(data)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.6f}초")
```

## 커스텀 함수 모범 사례

1. **내장 함수 우선**: NumPy 내장 함수가 있는 경우 항상 우선 사용
2. **벡터화 최우선**: 반복문 대신 벡터화 연산으로 구현
3. **성능 테스트**: 다양한 구현 방법의 성능을 비교하여 최적의 방법 선택
4. **메모리 고려**: 대용량 데이터는 메모리 사용량을 고려한 구현
5. **JIT 컴파일 고려**: 복잡한 연산은 Numba와 같은 JIT 컴파일러 고려

## 다음 학습 내용

다음으로는 도메인 응용에 대해 알아보겠습니다. [`../06-domain-applications/data-science.md`](../06-domain-applications/data-science.md)를 참조하세요.