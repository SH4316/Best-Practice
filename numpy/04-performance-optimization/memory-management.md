# NumPy 메모리 관리

## NumPy 메모리 구조 이해

NumPy 배열은 메모리에 연속적인 블록으로 저장되며, 이는 효율적인 메모리 접근과 연산을 가능하게 합니다.

### 메모리 레이아웃

```python
import numpy as np

# C 순서 (행 우선, 기본값)
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(arr_c.flags['C_CONTIGUOUS'])  # True
print(arr_c.flags['F_CONTIGUOUS'])  # False

# Fortran 순서 (열 우선)
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(arr_f.flags['C_CONTIGUOUS'])  # False
print(arr_f.flags['F_CONTIGUOUS'])  # True

# 메모리 사용량 확인
print(f"C 순서 배열 크기: {arr_c.nbytes} 바이트")
print(f"F 순서 배열 크기: {arr_f.nbytes} 바이트")
```

### 데이터 타입과 메모리

```python
# 다양한 데이터 타입의 메모리 사용량
dtypes = [np.int8, np.int16, np.int32, np.int64, 
          np.float32, np.float64, np.complex128]

for dtype in dtypes:
    arr = np.zeros(1000, dtype=dtype)
    print(f"{dtype.__name__}: {arr.nbytes} 바이트, 요소당 {arr.itemsize} 바이트")
```

## 뷰(View)와 복사(Copy)

### 뷰와 복사의 차이

```python
# 원본 배열
original = np.array([1, 2, 3, 4, 5])

# 뷰 생성
view = original[1:4]
print(f"원본: {original}")
print(f"뷰: {view}")

# 뷰 수정 시 원본도 변경
view[0] = 100
print(f"뷰 수정 후 원본: {original}")
print(f"뷰 수정 후 뷰: {view}")

# 복사 생성
copy = original[1:4].copy()
print(f"복사: {copy}")

# 복사 수정 시 원본은 변경되지 않음
copy[0] = 200
print(f"복사 수정 후 원본: {original}")
print(f"복사 수정 후 복사: {copy}")

# 메모리 공유 확인
print(f"뷰와 원본 메모리 공유: {np.shares_memory(view, original)}")
print(f"복사와 원본 메모리 공유: {np.shares_memory(copy, original)}")
```

### 명시적 복사와 암시적 복사

```python
# 명시적 복사
arr = np.array([1, 2, 3, 4, 5])
explicit_copy = arr.copy()
print(f"명시적 복사 메모리 공유: {np.shares_memory(arr, explicit_copy)}")

# 암시적 복사가 발생하는 경우
arr = np.array([1, 2, 3, 4, 5])
slice_view = arr[1:4]  # 뷰
reshape_view = arr.reshape(5, 1)  # 뷰 (가능한 경우)

# 암시적 복사가 발생하는 경우
arr = np.array([1, 2, 3, 4, 5])
fancy_indexed = arr[[0, 2, 4]]  # 복사
boolean_indexed = arr[arr > 2]  # 복사

print(f"팬시 인덱싱 메모리 공유: {np.shares_memory(arr, fancy_indexed)}")
print(f"불리언 인덱싱 메모리 공유: {np.shares_memory(arr, boolean_indexed)}")
```

## 메모리 효율적인 연산

### in-place 연산

```python
import time

# 큰 배열 생성
size = 10000000
a = np.random.rand(size)
b = np.random.rand(size)

# 일반 연산 (새 배열 생성)
start = time.time()
c = a + b
normal_time = time.time() - start

# in-place 연산 (기존 배열 수정)
a_copy = a.copy()
start = time.time()
a_copy += b
inplace_time = time.time() - start

print(f"일반 연산 시간: {normal_time:.6f}초")
print(f"in-place 연산 시간: {inplace_time:.6f}초")

# in-place 연산 함수들
arr = np.array([1, 2, 3, 4, 5])
np.add(arr, 2, out=arr)  # in-place 덧셈
print(f"in-place 덧셈 후: {arr}")

np.multiply(arr, 2, out=arr)  # in-place 곱셈
print(f"in-place 곱셈 후: {arr}")
```

### 메모리 절약형 연산

```python
# 큰 배열 연산에서 메모리 사용량 최소화
size = 1000000
a = np.random.rand(size)
b = np.random.rand(size)

# 메모리를 많이 사용하는 방식
c1 = a * b
c2 = c1 + 1
c3 = c2 * 2
result1 = np.sin(c3)

# 메모리 효율적인 방식
result2 = 2 * np.sin(a * b + 1)

print(f"두 결과가 동일한가: {np.allclose(result1, result2)}")
```

## 메모리 매핑

### memmap을 사용한 대용량 파일 처리

```python
# 대용량 배열 생성 및 파일로 저장
large_array = np.random.rand(10000, 10000)
filename = 'large_array.dat'

# 파일로 저장
large_array.tofile(filename)

# 메모리 매핑으로 파일 접근
mmap_array = np.memmap(filename, dtype='float64', mode='r', shape=(10000, 10000))

# 일부 데이터만 접근
subset = mmap_array[1000:2000, 1000:2000]
print(f"부분 배열 형태: {subset.shape}")

# 메모리 매핑 수정
mmap_writable = np.memmap(filename, dtype='float64', mode='r+', shape=(10000, 10000))
mmap_writable[0, 0] = 999.0  # 파일 직접 수정

# 변경 확인
print(f"수정된 값: {mmap_array[0, 0]}")
```

## 메모리 프로파일링

### 메모리 사용량 측정

```python
import sys

# 배열 크기에 따른 메모리 사용량
sizes = [100, 1000, 10000, 100000, 1000000]

for size in sizes:
    arr = np.random.rand(size)
    array_memory = arr.nbytes
    total_memory = sys.getsizeof(arr) + array_memory
    print(f"크기 {size}: 배열 메모리 {array_memory} 바이트, 총 {total_memory} 바이트")
```

### 메모리 누수 탐지

```python
# 메모리 누수가 발생할 수 있는 코드 패턴
def memory_leak_example():
    """메모리 누수 예제"""
    arrays = []
    for i in range(1000):
        # 매번 새 배열 생성하고 리스트에 추가
        arrays.append(np.random.rand(10000))
    return arrays

# 메모리 효율적인 대안
def memory_efficient_example():
    """메모리 효율적인 예제"""
    # 하나의 배열을 재사용
    arr = np.empty((1000, 10000))
    for i in range(1000):
        arr[i] = np.random.rand(10000)
    return arr
```

## 데이터 타입 최적화

### 적절한 데이터 타입 선택

```python
# 정수 데이터 타입 최적화
data_int = np.random.randint(0, 100, 1000000)

# 기본 int64 사용
int64_array = data_int.astype(np.int64)
print(f"int64 메모리: {int64_array.nbytes} 바이트")

# 실제 데이터 범위에 맞는 타입 사용
int8_array = data_int.astype(np.int8)
print(f"int8 메모리: {int8_array.nbytes} 바이트")
print(f"메모리 절약: {int64_array.nbytes / int8_array.nbytes}배")

# 실수 데이터 타입 최적화
data_float = np.random.rand(1000000)

# 기본 float64 사용
float64_array = data_float.astype(np.float64)
print(f"float64 메모리: {float64_array.nbytes} 바이트")

# 정밀도가 낮아도 되는 경우 float32 사용
float32_array = data_float.astype(np.float32)
print(f"float32 메모리: {float32_array.nbytes} 바이트")
print(f"메모리 절약: {float64_array.nbytes / float32_array.nbytes}배")
```

### 구조화된 배열

```python
# 구조화된 배열로 메모리 효율성 향상
# 개별 배열 사용
names = np.array(['Alice', 'Bob', 'Charlie'], dtype='U10')
ages = np.array([25, 30, 35], dtype=np.int32)
scores = np.array([85.5, 90.0, 78.5], dtype=np.float32)

print(f"개별 배열 메모리: {names.nbytes + ages.nbytes + scores.nbytes} 바이트")

# 구조화된 배열 사용
dtype = [('name', 'U10'), ('age', 'i4'), ('score', 'f4')]
structured_array = np.zeros(3, dtype=dtype)
structured_array['name'] = names
structured_array['age'] = ages
structured_array['score'] = scores

print(f"구조화된 배열 메모리: {structured_array.nbytes} 바이트")
print(f"메모리 절약: {(names.nbytes + ages.nbytes + scores.nbytes) / structured_array.nbytes}배")
```

## 가비지 컬렉션과 메모리 해제

### 명시적 메모리 해제

```python
# 큰 배열 생성
large_array = np.random.rand(10000, 10000)
print(f"생성된 배열 크기: {large_array.nbytes} 바이트")

# 배열 참조 삭제
del large_array

# 가비지 컬렉션 강제 실행
import gc
gc.collect()

# 메모리 뷰와 참조
arr = np.random.rand(1000, 1000)
view = arr[500:, 500:]  # 뷰 생성

# 원본 배열을 삭제해도 뷰가 존재하면 메모리는 해제되지 않음
del arr
print(f"뷰가 여전히 존재: {view.nbytes} 바이트")

# 뷰도 삭제해야 메모리 완전 해제
del view
gc.collect()
```

## 메모리 최적화 기법

### 스트리밍 처리

```python
# 대용량 데이터를 한 번에 처리하지 않고 조각내어 처리
def process_large_data(filename, chunk_size=10000):
    """대용량 파일을 조각내어 처리"""
    # 파일 크기假设 (실제로는 파일에서 읽어옴)
    total_size = 1000000
    processed_chunks = []
    
    for i in range(0, total_size, chunk_size):
        # 실제로는 파일에서 청크를 읽어옴
        chunk = np.random.rand(min(chunk_size, total_size - i))
        
        # 청크 처리
        processed_chunk = np.sin(chunk) * 2 + 1
        processed_chunks.append(processed_chunk)
    
    # 결과 결합
    return np.concatenate(processed_chunks)

result = process_large_data('large_data.dat')
print(f"처리된 데이터 크기: {result.shape}")
```

### 지연 계산

```python
# 필요할 때만 계산을 수행하여 메모리 사용 최소화
class LazyArray:
    """지연 계산 배열 클래스"""
    def __init__(self, shape, compute_func):
        self.shape = shape
        self.compute_func = compute_func
        self._computed = None
    
    @property
    def array(self):
        if self._computed is None:
            self._computed = self.compute_func()
        return self._computed
    
    def __getitem__(self, index):
        return self.array[index]

# 지연 계산 배열 생성
def expensive_computation():
    """메모리를 많이 사용하는 계산"""
    return np.random.rand(10000, 10000) * 100

lazy_array = LazyArray((10000, 10000), expensive_computation)

# 실제로 접근할 때까지 계산되지 않음
print("배열 생성됨 (아직 계산 안 됨)")
subset = lazy_array[0:100, 0:100]  # 이때 계산 수행
print(f"부분 배열 형태: {subset.shape}")
```

## 실용적인 메모리 최적화 예제

### 이미지 배치 처리

```python
# 여러 이미지를 메모리 효율적으로 처리
def process_images_batch(image_files, batch_size=10):
    """이미지 배치를 메모리 효율적으로 처리"""
    results = []
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        
        # 배치 크기에 맞는 배열 미리 할당
        batch = np.empty((len(batch_files), 256, 256, 3), dtype=np.uint8)
        
        # 이미지 로드 및 처리
        for j, file in enumerate(batch_files):
            # 실제로는 파일에서 이미지 로드
            image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            
            # 이미지 처리
            processed_image = np.flip(image, axis=0)  # 상하 반전
            
            batch[j] = processed_image
        
        # 배치 결과 추가
        results.append(batch)
    
    return results

# 예제 실행
image_files = [f'image_{i}.jpg' for i in range(100)]
batch_results = process_images_batch(image_files)
print(f"처리된 배치 수: {len(batch_results)}")
print(f"각 배치 크기: {batch_results[0].shape}")
```

### 시계열 데이터 처리

```python
# 시계열 데이터를 메모리 효율적으로 처리
def process_time_series(data, window_size=1000, step_size=500):
    """시계열 데이터를 윈도우 단위로 처리"""
    n_samples = len(data)
    results = []
    
    # 미리 결과 크기 계산
    n_windows = (n_samples - window_size) // step_size + 1
    
    for i in range(0, n_samples - window_size + 1, step_size):
        window = data[i:i+window_size]
        
        # 윈도우 통계 계산
        window_stats = {
            'mean': np.mean(window),
            'std': np.std(window),
            'min': np.min(window),
            'max': np.max(window)
        }
        
        results.append(window_stats)
    
    return results

# 예제 실행
time_series_data = np.random.randn(100000)
stats = process_time_series(time_series_data)
print(f"처리된 윈도우 수: {len(stats)}")
```

## 메모리 최적화 모범 사례

1. **적절한 데이터 타입 선택**: 실제 데이터 범위에 맞는 최소 크기의 타입 사용
2. **in-place 연산 활용**: 가능한 경우 in-place 연산으로 임시 배열 생성 최소화
3. **뷰 적극 활용**: 데이터 복사 필요 없는 경우 뷰 사용
4. **메모리 매핑**: 대용량 파일은 memmap으로 접근
5. **청크 처리**: 대용량 데이터는 조각내어 처리
6. **명시적 메모리 해제**: 사용이 끝난 큰 배열은 명시적으로 삭제

## 다음 학습 내용

다음으로는 브로드캐스팅에 대해 알아보겠습니다. [`broadcasting.md`](broadcasting.md)를 참조하세요.