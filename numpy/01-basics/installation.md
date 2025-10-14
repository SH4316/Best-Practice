# NumPy 설치 및 기본 개념

## NumPy란 무엇인가?

NumPy(Numerical Python)는 파이썬에서 과학 계산을 위한 핵심 라이브러리입니다. 다차원 배열 객체와 이 배열들을 다루는 다양한 함수들을 제공합니다.

### NumPy의 주요 특징

- **강력한 다차원 배열 객체**: ndarray라는 효율적인 다차원 배열 제공
- **수학 함수**: 다양한 수학적 함수와 통계 함수 제공
- **통합된 C/C++/Fortran 코드**: 고성능 계산 가능
- **선형대수, 푸리에 변환, 난수 생성**: 과학 계산에 필요한 기능 제공

## NumPy 설치

### pip를 이용한 설치

```bash
pip install numpy
```

### conda를 이용한 설치

```bash
conda install numpy
```

### 특정 버전 설치

```bash
pip install numpy==1.21.0
```

### 최신 개발 버전 설치

```bash
pip install --pre numpy
```

## 설치 확인

NumPy가 올바르게 설치되었는지 확인하려면:

```python
import numpy as np
print(np.__version__)
```

## NumPy 기본 개념

### ndarray 객체

NumPy의 핵심은 ndarray(N-dimensional array) 객체입니다. 이는 동일한 타입의 데이터를 담는 격자(grid)입니다.

```python
import numpy as np

# 1차원 배열 생성
a = np.array([1, 2, 3, 4, 5])
print(a)  # [1 2 3 4 5]

# 2차원 배열 생성
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]
```

### 배열의 속성

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 배열의 차원
print(arr.ndim)  # 2

# 배열의 형태
print(arr.shape)  # (2, 3)

# 배열의 전체 요소 수
print(arr.size)  # 6

# 배열의 데이터 타입
print(arr.dtype)  # int64

# 배열 요소 하나의 바이트 크기
print(arr.itemsize)  # 8 (int64의 경우)
```

## NumPy가 왜 중요한가?

### 성능 비교

```python
import numpy as np
import time

# 큰 리스트 생성
size = 1000000
python_list = list(range(size))
numpy_array = np.arange(size)

# 합계 계산 성능 비교
start_time = time.time()
sum_python = sum(python_list)
python_time = time.time() - start_time

start_time = time.time()
sum_numpy = np.sum(numpy_array)
numpy_time = time.time() - start_time

print(f"Python 리스트: {python_time:.6f}초")
print(f"NumPy 배열: {numpy_time:.6f}초")
print(f"NumPy가 {python_time/numpy_time:.1f}배 더 빠름")
```

### 메모리 효율성

NumPy 배열은 Python 리스트보다 훨씬 적은 메모리를 사용합니다.

```python
import sys

python_list = list(range(1000))
numpy_array = np.arange(1000, dtype=np.int32)

print(f"Python 리스트 메모리: {sys.getsizeof(python_list) + sum(sys.getsizeof(i) for i in python_list)} 바이트")
print(f"NumPy 배열 메모리: {numpy_array.nbytes} 바이트")
```

## NumPy vs Python 리스트

| 특징 | Python 리스트 | NumPy 배열 |
|------|---------------|------------|
| 데이터 타입 | 다양한 타입 혼합 가능 | 단일 타입만 가능 |
| 메모리 사용 | 비효율적 | 효율적 |
| 계산 속도 | 느림 | 빠름 |
| 수학 연산 | 제한적 | 풍부한 함수 제공 |
| 벡터화 연산 | 불가능 | 가능 |

## 다음 학습 내용

다음으로는 배열 생성 방법과 기본 속성에 대해 자세히 알아보겠습니다. [`arrays.md`](arrays.md)를 참조하세요.