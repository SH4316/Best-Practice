# NumPy 배열 생성과 기본 속성

## 배열 생성 방법

### 1. 리스트로부터 배열 생성하기

가장 기본적인 배열 생성 방법은 Python 리스트를 사용하는 것입니다.

```python
import numpy as np

# 1차원 배열
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]

# 2차원 배열
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# 3차원 배열
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr3)
# [[[1 2]
#   [3 4]]
# 
#  [[5 6]
#   [7 8]]]
```

### 2. NumPy 내장 함수로 배열 생성하기

#### zeros(): 모든 요소가 0인 배열

```python
# 1차원 배열
zeros1 = np.zeros(5)
print(zeros1)  # [0. 0. 0. 0. 0.]

# 2차원 배열
zeros2 = np.zeros((3, 4))
print(zeros2)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 데이터 타입 지정
zeros_int = np.zeros((2, 3), dtype=int)
print(zeros_int)
# [[0 0 0]
#  [0 0 0]]
```

#### ones(): 모든 요소가 1인 배열

```python
ones1 = np.ones(4)
print(ones1)  # [1. 1. 1. 1.]

ones2 = np.ones((2, 3), dtype=int)
print(ones2)
# [[1 1 1]
#  [1 1 1]]
```

#### full(): 모든 요소가 특정 값인 배열

```python
full1 = np.full(5, 7)
print(full1)  # [7 7 7 7 7]

full2 = np.full((2, 3), 9)
print(full2)
# [[9 9 9]
#  [9 9 9]]
```

#### eye(): 단위 행렬

```python
eye1 = np.eye(3)
print(eye1)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# k 매개변수로 대각선 위치 조정
eye2 = np.eye(3, k=1)
print(eye2)
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 0.]]
```

#### empty(): 초기화되지 않은 배열

```python
empty1 = np.empty((2, 3))
print(empty1)  # 예측 불가능한 값
```

### 3. 순차적인 값으로 배열 생성하기

#### arange(): Python range와 유사

```python
# 기본 사용법
arr1 = np.arange(10)
print(arr1)  # [0 1 2 3 4 5 6 7 8 9]

# 시작, 끝, 간격
arr2 = np.arange(1, 10, 2)
print(arr2)  # [1 3 5 7 9]

# 실수 간격
arr3 = np.arange(0, 1, 0.1)
print(arr3)  # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
```

#### linspace(): 선형 간격의 값들

```python
# 0부터 1까지 5개의 점
lin1 = np.linspace(0, 1, 5)
print(lin1)  # [0.   0.25 0.5  0.75 1.  ]

# endpoint=False로 끝점 제외
lin2 = np.linspace(0, 1, 5, endpoint=False)
print(lin2)  # [0.  0.2 0.4 0.6 0.8]
```

#### logspace(): 로그 스케일 간격의 값들

```python
log1 = np.logspace(0, 2, 5)  # 10^0부터 10^2까지 5개의 점
print(log1)  # [  1.           3.16227766  10.          31.6227766  100.        ]
```

### 4. 난수로 배열 생성하기

#### random.rand(): 0과 1 사이의 균일 분포

```python
rand1 = np.random.rand(3)  # 1차원
print(rand1)

rand2 = np.random.rand(2, 3)  # 2차원
print(rand2)
```

#### random.randn(): 표준 정규 분포

```python
randn1 = np.random.randn(3)
print(randn1)

randn2 = np.random.randn(2, 3)
print(randn2)
```

#### random.randint(): 정수 난수

```python
randint1 = np.random.randint(0, 10, 5)  # 0부터 9까지의 정수 5개
print(randint1)

randint2 = np.random.randint(0, 10, (2, 3))  # 2x3 배열
print(randint2)
```

## 배열의 데이터 타입

### 기본 데이터 타입

```python
# 정수 타입
int_arr = np.array([1, 2, 3], dtype=np.int32)
print(int_arr.dtype)  # int32

# 실수 타입
float_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
print(float_arr.dtype)  # float64

# 복소수 타입
complex_arr = np.array([1+2j, 3+4j], dtype=np.complex128)
print(complex_arr.dtype)  # complex128

# 불리언 타입
bool_arr = np.array([True, False, True], dtype=np.bool_)
print(bool_arr.dtype)  # bool
```

### 데이터 타입 변환

```python
arr = np.array([1, 2, 3])
print(arr.dtype)  # int64

# float으로 변환
float_arr = arr.astype(np.float64)
print(float_arr.dtype)  # float64

# 다시 int로 변환
int_arr = float_arr.astype(np.int32)
print(int_arr.dtype)  # int32
```

## 배열의 속성

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 배열의 차원 수
print(arr.ndim)  # 2

# 배열의 형태 (각 차원의 크기)
print(arr.shape)  # (3, 4)

# 배열의 전체 요소 수
print(arr.size)  # 12

# 배열의 데이터 타입
print(arr.dtype)  # int64

# 배열 요소 하나의 바이트 크기
print(arr.itemsize)  # 8 (int64의 경우)

# 배열의 전체 바이트 크기
print(arr.nbytes)  # 96 (12 * 8)
```

## 배열의 형태 변경

```python
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# reshape로 형태 변경
reshaped = arr.reshape(3, 4)
print(reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# -1을 사용하면 자동으로 크기 계산
auto_reshaped = arr.reshape(3, -1)
print(auto_reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 1차원으로 변환 (flatten)
flattened = reshaped.flatten()
print(flattened)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]
```

## 배열 복사

```python
a = np.array([1, 2, 3])

# 뷰(view): 같은 데이터를 참조
b = a.view()
b[0] = 100
print(a)  # [100   2   3]  원본도 변경됨

# 복사(copy): 독립적인 배열
c = a.copy()
c[0] = 200
print(a)  # [100   2   3]  원본은 변경되지 않음
```

## 다음 학습 내용

다음으로는 NumPy 배열의 기본 연산에 대해 알아보겠습니다. [`basic-operations.md`](basic-operations.md)를 참조하세요.