# NumPy 최적화 팁

## 소개

NumPy 코드의 성능을 최적화하는 것은 대용량 데이터를 처리할 때 매우 중요합니다. 이 문서에서는 NumPy 코드를 더 빠르고 효율적으로 만드는 다양한 기법과 팁을 소개합니다.

## 벡터화 최적화

### 브로드캐스팅 활용

```python
import numpy as np
import time

# 나쁜 예제: 반복문 사용
def inefficient_normalization(data):
    """반복문을 이용한 데이터 정규화"""
    normalized = np.empty_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            normalized[i, j] = (data[i, j] - np.mean(data[i, :])) / np.std(data[i, :])
    return normalized

# 좋은 예제: 브로드캐스팅 활용
def efficient_normalization(data):
    """브로드캐스팅을 이용한 데이터 정규화"""
    means = np.mean(data, axis=1, keepdims=True)  # (n, 1)
    stds = np.std(data, axis=1, keepdims=True)    # (n, 1)
    return (data - means) / stds

# 성능 비교
data = np.random.rand(1000, 1000)

start = time.time()
result_inefficient = inefficient_normalization(data)
inefficient_time = time.time() - start

start = time.time()
result_efficient = efficient_normalization(data)
efficient_time = time.time() - start

print(f"비효율적 방법: {inefficient_time:.6f}초")
print(f"효율적 방법: {efficient_time:.6f}초")
print(f"성능 향상: {inefficient_time/efficient_time:.1f}배")
print(f"결과 동일성: {np.allclose(result_inefficient, result_efficient)}")
```

### 유니버설 함수 활용

```python
# 나쁜 예제: Python 함수와 반복문
def inefficient_exp(data):
    """Python 함수와 반복문을 이용한 지수 함수"""
    result = np.empty_like(data)
    for i in range(data.size):
        result.flat[i] = np.exp(data.flat[i])
    return result

# 좋은 예제: NumPy 유니버설 함수
def efficient_exp(data):
    """NumPy 유니버설 함수를 이용한 지수 함수"""
    return np.exp(data)

# 성능 비교
data = np.random.rand(1000000)

start = time.time()
result_inefficient = inefficient_exp(data)
inefficient_time = time.time() - start

start = time.time()
result_efficient = efficient_exp(data)
efficient_time = time.time() - start

print(f"비효율적 방법: {inefficient_time:.6f}초")
print(f"효율적 방법: {efficient_time:.6f}초")
print(f"성능 향상: {inefficient_time/efficient_time:.1f}배")
```

## 메모리 최적화

### in-place 연산

```python
# in-place 연산 vs 일반 연산
data = np.random.rand(1000000)

# 일반 연산 (새 배열 생성)
start = time.time()
result = data * 2 + 1
normal_time = time.time() - start

# in-place 연산 (기존 배열 수정)
data_copy = data.copy()
start = time.time()
data_copy *= 2
data_copy += 1
inplace_time = time.time() - start

print(f"일반 연산 시간: {normal_time:.6f}초")
print(f"in-place 연산 시간: {inplace_time:.6f}초")
print(f"성능 향상: {normal_time/inplace_time:.2f}배")

# out 매개변수를 이용한 in-place 연산
data_copy = data.copy()
start = time.time()
np.add(data_copy, 1, out=data_copy)  # in-place 덧셈
np.multiply(data_copy, 2, out=data_copy)  # in-place 곱셈
out_time = time.time() - start

print(f"out 매개변수 시간: {out_time:.6f}초")
```

### 메모리 레이아웃 최적화

```python
# 메모리 레이아웃에 따른 성능 차이
size = 10000

# C 순서 (행 우선)
c_array = np.random.rand(size, size)
print(f"C 순서 배열: {c_array.flags['C_CONTIGUOUS']}, {c_array.flags['F_CONTIGUOUS']}")

# Fortran 순서 (열 우선)
f_array = np.array(c_array, order='F')
print(f"F 순서 배열: {f_array.flags['C_CONTIGUOUS']}, {f_array.flags['F_CONTIGUOUS']}")

# 행별 합계
start = time.time()
row_sum_c = np.sum(c_array, axis=1)
row_sum_time_c = time.time() - start

start = time.time()
row_sum_f = np.sum(f_array, axis=1)
row_sum_time_f = time.time() - start

# 열별 합계
start = time.time()
col_sum_c = np.sum(c_array, axis=0)
col_sum_time_c = time.time() - start

start = time.time()
col_sum_f = np.sum(f_array, axis=0)
col_sum_time_f = time.time() - start

print(f"C 순서 행별 합계: {row_sum_time_c:.6f}초")
print(f"F 순서 행별 합계: {row_sum_time_f:.6f}초")
print(f"C 순서 열별 합계: {col_sum_time_c:.6f}초")
print(f"F 순서 열별 합계: {col_sum_time_f:.6f}초")

print(f"행별 합계 성능 차이: {row_sum_time_f/row_sum_time_c:.2f}배")
print(f"열별 합계 성능 차이: {col_sum_time_c/col_sum_time_f:.2f}배")
```

### 데이터 타입 최적화

```python
# 데이터 타입에 따른 메모리 사용량과 성능
size = 1000000

# 다양한 데이터 타입
dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]

for dtype in dtypes:
    data = np.random.randint(0, 100, size).astype(dtype)
    
    # 메모리 사용량
    memory_usage = data.nbytes
    
    # 연산 성능
    start = time.time()
    result = data + 1
    operation_time = time.time() - start
    
    print(f"{dtype.__name__}: 메모리 {memory_usage//1024}KB, 연산 {operation_time:.6f}초")
```

## 알고리즘 최적화

### 효율적인 알고리즘 선택

```python
# 소수 찾기 알고리즘 비교
def sieve_of_eratosthenes(n):
    """에라토스테네스의 체를 이용한 소수 찾기"""
    sieve = np.ones(n+1, dtype=bool)
    sieve[0:2] = False
    
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    
    return np.nonzero(sieve)[0]

def naive_prime_finder(n):
    """단순한 소수 찾기"""
    primes = []
    for num in range(2, n+1):
        is_prime = True
        for i in range(2, int(np.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return np.array(primes)

# 성능 비교
n = 100000

start = time.time()
primes_sieve = sieve_of_eratosthenes(n)
sieve_time = time.time() - start

start = time.time()
primes_naive = naive_prime_finder(n)
naive_time = time.time() - start

print(f"에라토스테네스의 체: {sieve_time:.6f}초")
print(f"단순한 방법: {naive_time:.6f}초")
print(f"성능 향상: {naive_time/sieve_time:.1f}배")
print(f"소수 개수: {len(primes_sieve)}")
```

### 스트라이드 트릭 활용

```python
# 스트라이드 트릭을 이용한 이동 평균
def moving_average_strides(arr, window_size):
    """스트라이드 트릭을 이용한 효율적인 이동 평균"""
    shape = (arr.shape[0] - window_size + 1, window_size)
    strides = (arr.strides[0], arr.strides[0])
    return np.mean(np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides), axis=1)

def moving_average_loop(arr, window_size):
    """반복문을 이용한 이동 평균"""
    result = np.empty(len(arr) - window_size + 1)
    for i in range(len(result)):
        result[i] = np.mean(arr[i:i+window_size])
    return result

# 성능 비교
data = np.random.rand(100000)
window_size = 100

start = time.time()
result_strides = moving_average_strides(data, window_size)
strides_time = time.time() - start

start = time.time()
result_loop = moving_average_loop(data, window_size)
loop_time = time.time() - start

print(f"스트라이드 트릭: {strides_time:.6f}초")
print(f"반복문: {loop_time:.6f}초")
print(f"성능 향상: {loop_time/strides_time:.1f}배")
print(f"결과 동일성: {np.allclose(result_strides, result_loop)}")
```

## 병렬 처리 최적화

### NumPy의 내장 병렬 처리

```python
# NumPy의 내장 병렬 처리 활용
# 여러 연산을 한 번에 수행
data = np.random.rand(1000000)

# 개별 연산
start = time.time()
mean_val = np.mean(data)
std_val = np.std(data)
min_val = np.min(data)
max_val = np.max(data)
individual_time = time.time() - start

# 벡터화된 연산 (NumPy 내부에서 병렬화)
start = time.time()
stats = np.array([np.mean(data), np.std(data), np.min(data), np.max(data)])
vectorized_time = time.time() - start

print(f"개별 연산: {individual_time:.6f}초")
print(f"벡터화된 연산: {vectorized_time:.6f}초")
```

### 외부 라이브러리 활용

```python
# Numba를 이용한 JIT 컴파일 (설치 필요)
try:
    from numba import jit
    
    # Numba 최적화 함수
    @jit(nopython=True)
    def numba_mandelbrot(width, height, max_iter):
        """Numba를 이용한 맨델브로 집합 계산"""
        result = np.zeros((height, width), dtype=np.int32)
        
        for y in range(height):
            for x in range(width):
                # 복소수 평면으로 변환
                c0 = complex(x * 3.0 / width - 2, y * 2.0 / height - 1)
                c = c0
                z = 0
                
                for i in range(max_iter):
                    if abs(z) > 2:
                        result[y, x] = i
                        break
                    z = z*z + c
                else:
                    result[y, x] = max_iter
        
        return result
    
    # 성능 비교
    width, height = 1000, 1000
    max_iter = 100
    
    # 첫 실행 (컴파일 포함)
    start = time.time()
    result_numba = numba_mandelbrot(width, height, max_iter)
    first_run_time = time.time() - start
    
    # 두 번째 실행 (컴파일된 코드)
    start = time.time()
    result_numba = numba_mandelbrot(width, height, max_iter)
    second_run_time = time.time() - start
    
    print(f"Numba 첫 실행: {first_run_time:.6f}초")
    print(f"Numba 두 번째 실행: {second_run_time:.6f}초")
    
except ImportError:
    print("Numba가 설치되지 않았습니다. 'pip install numba'로 설치하세요.")
```

## 입출력 최적화

### 효율적인 파일 입출력

```python
# 효율적인 배열 저장/로드
data = np.random.rand(10000, 10000)

# 일반 텍스트 파일 저장/로드
start = time.time()
np.savetxt('data.txt', data)
txt_save_time = time.time() - start

start = time.time()
loaded_txt = np.loadtxt('data.txt')
txt_load_time = time.time() - start

# 바이너리 파일 저장/로드
start = time.time()
data.tofile('data.bin')
bin_save_time = time.time() - start

start = time.time()
loaded_bin = np.fromfile('data.bin', dtype=data.dtype).reshape(data.shape)
bin_load_time = time.time() - start

# NumPy 포맷 파일 저장/로드
start = time.time()
np.save('data.npy', data)
npy_save_time = time.time() - start

start = time.time()
loaded_npy = np.load('data.npy')
npy_load_time = time.time() - start

print(f"텍스트 저장: {txt_save_time:.6f}초")
print(f"텍스트 로드: {txt_load_time:.6f}초")
print(f"바이너리 저장: {bin_save_time:.6f}초")
print(f"바이너리 로드: {bin_load_time:.6f}초")
print(f"NumPy 저장: {npy_save_time:.6f}초")
print(f"NumPy 로드: {npy_load_time:.6f}초")

# 파일 크기 비교
import os
txt_size = os.path.getsize('data.txt')
bin_size = os.path.getsize('data.bin')
npy_size = os.path.getsize('data.npy')

print(f"텍스트 파일 크기: {txt_size//1024}KB")
print(f"바이너리 파일 크기: {bin_size//1024}KB")
print(f"NumPy 파일 크기: {npy_size//1024}KB")
```

### 메모리 매핑 활용

```python
# 메모리 매핑을 이용한 대용량 파일 처리
# 큰 배열 생성 및 저장
large_data = np.random.rand(5000, 5000)
large_data.tofile('large_data.bin')

# 일반적인 방법 (전체를 메모리에 로드)
start = time.time()
normal_loaded = np.fromfile('large_data.bin', dtype=large_data.dtype).reshape(large_data.shape)
normal_time = time.time() - start

# 메모리 매핑 (필요한 부분만 로드)
start = time.time()
mmap_array = np.memmap('large_data.bin', dtype=large_data.dtype, mode='r', shape=large_data.shape)
mmap_time = time.time() - start

# 메모리 매핑된 배열의 일부만 접근
start = time.time()
partial_data = mmap_array[1000:2000, 1000:2000]
partial_time = time.time() - start

print(f"일반 로드 시간: {normal_time:.6f}초")
print(f"메모리 매핑 시간: {mmap_time:.6f}초")
print(f"부분 접근 시간: {partial_time:.6f}초")
```

## 최적화 도구

### 프로파일링

```python
# 간단한 프로파일링 도구
def profile_function(func, *args, **kwargs):
    """함수 실행 시간 측정"""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

# 테스트 함수들
def function1(data):
    return np.sum(data ** 2)

def function2(data):
    return np.sum(np.square(data))

def function3(data):
    return np.dot(data, data)

# 프로파일링
data = np.random.rand(1000000)

for func in [function1, function2, function3]:
    result, elapsed = profile_function(func, data)
    print(f"{func.__name__}: {elapsed:.6f}초, 결과: {result:.6f}")
```

### 라인 프로파일러

```python
# 라인별 프로파일링 (line_profiler 설치 필요)
try:
    from line_profiler import LineProfiler
    
    def complex_function(data):
        """복잡한 함수"""
        result = np.zeros_like(data)
        
        # 전처리
        processed = np.sqrt(np.abs(data))
        
        # 변환
        transformed = np.sin(processed) + np.cos(processed)
        
        # 후처리
        result = transformed / np.max(transformed)
        
        return result
    
    # 라인 프로파일러 설정
    lp = LineProfiler()
    lp_wrapper = lp(complex_function)
    
    # 프로파일링 실행
    data = np.random.rand(100000)
    lp_wrapper(data)
    
    # 결과 출력 (실제로는 print_stats()로 출력)
    print("라인 프로파일러 결과:")
    lp.print_stats()
    
except ImportError:
    print("line_profiler가 설치되지 않았습니다. 'pip install line_profiler'로 설치하세요.")
```

## 최적화 체크리스트

### 성능 최적화 확인 사항

1. **벡터화**: 반복문 대신 NumPy 벡터화 연산 사용
2. **브로드캐스팅**: 명시적 차원 추가로 브로드캐스팅 활용
3. **in-place 연산**: 가능한 경우 in-place 연산으로 메모리 절약
4. **데이터 타입**: 적절한 데이터 타입 선택으로 메모리 최적화
5. **메모리 레이아웃**: C 순서와 Fortran 순서의 차이 이해
6. **알고리즘**: 효율적인 알고리즘 선택
7. **스트라이드 트릭**: 적절한 경우 스트라이드 트릭 활용
8. **입출력**: 바이너리 포맷과 메모리 매핑 활용
9. **프로파일링**: 병목 지점 식별 및 최적화
10. **외부 라이브러리**: Numba 등 외부 라이브러리 활용

### 코드 리뷰 질문

1. 반복문을 벡터화 연산으로 바꿀 수 있는가?
2. 불필요한 배열 복사를 제거할 수 있는가?
3. 더 적절한 데이터 타입이 있는가?
4. 브로드캐스팅을 더 효율적으로 사용할 수 있는가?
5. 메모리 레이아웃을 최적화할 수 있는가?
6. 알고리즘을 더 효율적으로 바꿀 수 있는가?

## 다음 학습 내용

다음으로는 디버깅 기법에 대해 알아보겠습니다. [`debugging.md`](debugging.md)를 참조하세요.