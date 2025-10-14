# NumPy 유니버설 함수(UFunc)

## 유니버설 함수란?

유니버설 함수(UFunc, Universal Function)는 NumPy에서 ndarray 배열의 각 요소에 대해 개별적으로 연산을 수행하는 함수입니다. 이 함수들은 벡터화 연산을 지원하여 반복문 없이 빠른 연산을 가능하게 합니다.

## 기본 산술 연산 UFunc

### 배열과 스칼라 연산

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# 덧셈
print(np.add(arr, 2))  # [3 4 5 6 7]
print(arr + 2)         # [3 4 5 6 7] (동일)

# 뺄셈
print(np.subtract(arr, 1))  # [0 1 2 3 4]
print(arr - 1)              # [0 1 2 3 4] (동일)

# 곱셈
print(np.multiply(arr, 3))  # [ 3  6  9 12 15]
print(arr * 3)              # [ 3  6  9 12 15] (동일)

# 나눗셈
print(np.divide(arr, 2))    # [0.5 1.  1.5 2.  2.5]
print(arr / 2)              # [0.5 1.  1.5 2.  2.5] (동일)

# 제곱
print(np.power(arr, 2))     # [ 1  4  9 16 25]
print(arr ** 2)             # [ 1  4  9 16 25] (동일)

# 나머지
print(np.mod(arr, 2))       # [1 0 1 0 1]
print(arr % 2)              # [1 0 1 0 1] (동일)
```

### 배열 간 연산

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 덧셈
print(np.add(a, b))  # [ 6  8 10 12]
print(a + b)         # [ 6  8 10 12] (동일)

# 뺄셈
print(np.subtract(a, b))  # [-4 -4 -4 -4]
print(a - b)              # [-4 -4 -4 -4] (동일)

# 곱셈
print(np.multiply(a, b))  # [ 5 12 21 32]
print(a * b)              # [ 5 12 21 32] (동일)

# 나눗셈
print(np.divide(a, b))    # [0.2        0.33333333 0.42857143 0.5       ]
print(a / b)              # [0.2        0.33333333 0.42857143 0.5       ] (동일)

# 제곱
print(np.power(a, b))     # [   1   64 2187 65536]
print(a ** b)             # [   1   64 2187 65536] (동일)
```

## 수학 함수 UFunc

### 기본 수학 함수

```python
arr = np.array([1, 4, 9, 16, 25])

# 제곱근
print(np.sqrt(arr))  # [1. 2. 3. 4. 5.]

# 제곱
print(np.square(arr))  # [  1  16  81 256 625]

# 절대값
arr_neg = np.array([-1, -2, -3, 4, 5])
print(np.abs(arr_neg))  # [1 2 3 4 5]

# 부호 함수
print(np.sign(arr_neg))  # [-1 -1 -1  1  1]

# 올림
arr_float = np.array([1.2, 2.5, 3.7, 4.1])
print(np.ceil(arr_float))  # [2. 3. 4. 5.]

# 내림
print(np.floor(arr_float))  # [1. 2. 3. 4.]

# 반올림
print(np.round(arr_float))  # [1. 2. 4. 4.]

# 소수점 이하 버림
print(np.trunc(arr_float))  # [1. 2. 3. 4.]
```

### 지수와 로그 함수

```python
arr = np.array([1, 2, 3, 4, 5])

# 지수 함수 (e^x)
print(np.exp(arr))  # [  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591 ]

# 2^x
print(np.exp2(arr))  # [ 2.  4.  8. 16. 32.]

# 자연 로그 (ln)
print(np.log(arr))  # [0.         0.69314718 1.09861229 1.38629436 1.60943791]

# 상용 로그 (log10)
print(np.log10(arr))  # [0.         0.30103    0.47712125 0.60205999 0.69897   ]

# 밑이 2인 로그 (log2)
print(np.log2(arr))  # [0.        1.        1.585    2.        2.32192809]

# 로그(1+x)
print(np.log1p(arr))  # [0.69314718 1.09861229 1.38629436 1.60943791 1.79175947]
```

### 삼각함수

```python
# 라디안 값
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

# 사인
print(np.sin(angles))  # [0.         0.5        0.70710678 0.8660254  1.        ]

# 코사인
print(np.cos(angles))  # [1.00000000e+00 8.66025404e-01 7.07106781e-01 5.00000000e-01 6.12323400e-17]

# 탄젠트
print(np.tan(angles))  # [0.00000000e+00 5.77350269e-01 1.00000000e+00 1.73205081e+00 1.63312394e+16]

# 역삼각함수
values = np.array([0, 0.5, 0.70710678, 0.8660254, 1.0])
print(np.arcsin(values))  # [0.         0.52359878 0.78539816 1.04719755 1.57079633]
print(np.arccos(values))  # [1.57079633 1.04719755 0.78539816 0.52359878 0.        ]
print(np.arctan(values))  # [0.         0.46364761 0.61547971 0.71372438 0.78539816]

# 도-라디안 변환
degrees = np.array([0, 30, 45, 60, 90])
print(np.radians(degrees))  # [0.         0.52359878 0.78539816 1.04719755 1.57079633]
print(np.degrees(angles))   # [ 0. 30. 45. 60. 90.]
```

### 쌍곡선 함수

```python
x = np.array([0, 1, 2, 3])

# 쌍곡선 사인
print(np.sinh(x))  # [ 0.          1.17520119  3.62686041 10.01787493]

# 쌍곡선 코사인
print(np.cosh(x))  # [ 1.          1.54308063  3.76219569 10.067662  ]

# 쌍곡선 탄젠트
print(np.tanh(x))  # [0.         0.76159416 0.96402758 0.99505475]

# 역쌍곡선 함수
print(np.arcsinh(x))  # [0.         0.88137359 1.44363548 1.81844646]
print(np.arccosh(x + 1))  # [0.         1.3169579  2.06343707 2.63391579]
print(np.arctanh(x / 4))  # [0.         0.25541281 0.54930614 0.97161381]
```

## 비교 연산 UFunc

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

# 요소별 비교
print(np.greater(a, b))     # [False False False  True  True]
print(np.greater_equal(a, b))  # [False False  True  True  True]
print(np.less(a, b))        # [ True  True False False False]
print(np.less_equal(a, b))     # [ True  True  True False False]
print(np.equal(a, b))       # [False False  True False False]
print(np.not_equal(a, b))   # [ True  True False  True  True]

# 스칼라와 비교
print(np.greater(a, 3))     # [False False False  True  True]
print(np.less_equal(a, 2))  # [ True  True False False False]
```

## 논리 연산 UFunc

```python
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

# 논리 연산
print(np.logical_and(a, b))  # [ True False False False]
print(np.logical_or(a, b))   # [ True  True  True False]
print(np.logical_not(a))     # [False False  True  True]
print(np.logical_xor(a, b))  # [False  True  True False]
```

## 비트 연산 UFunc

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

# 비트 연산
print(np.bitwise_and(a, b))  # [1 0 3 0 1]
print(np.bitwise_or(a, b))   # [5 6 3 6 5]
print(np.bitwise_xor(a, b))  # [4 6 0 6 4]
print(np.invert(a))          # [-2 -3 -4 -5 -6]

# 비트 이동
print(np.left_shift(a, 1))   # [ 2  4  6  8 10]
print(np.right_shift(a, 1))  # [0 1 1 2 2]
```

## 특수 UFunc

### 조건부 연산

```python
arr = np.array([1, 2, 3, 4, 5])

# where: 조건이 참이면 x, 거짓이면 y
result = np.where(arr > 3, arr, 0)
print(result)  # [0 0 0 4 5]

# select: 여러 조건
conditions = [arr < 2, (arr >= 2) & (arr < 4), arr >= 4]
choices = [arr * 10, arr * 100, arr * 1000]
result = np.select(conditions, choices, default=0)
print(result)  # [  10  200  300 4000 5000]
```

### 누적 연산

```python
arr = np.array([1, 2, 3, 4, 5])

# 누적 합계
print(np.cumsum(arr))  # [ 1  3  6 10 15]

# 누적 곱
print(np.cumprod(arr))  # [  1   2   6  24 120]

# 누적 최소값
print(np.minimum.accumulate(arr))  # [1 1 1 1 1]

# 누적 최대값
print(np.maximum.accumulate(arr))  # [1 2 3 4 5]
```

## UFunc 메서드

UFunc는 배열 연산에 유용한 메서드들을 제공합니다.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# reduce: 배열을 하나의 값으로 축소
print(np.add.reduce(arr))        # [12 15 18] (축 기본값=0)
print(np.add.reduce(arr, axis=1))  # [ 6 15 24]

# accumulate: 누적 연산
print(np.add.accumulate(arr))    # [[ 1  2  3] [ 5  7  9] [12 15 18]]
print(np.add.accumulate(arr, axis=1))  # [[ 1  3  6] [ 4  9 15] [ 7 15 24]]

# reduceat: 지정된 인덱스에서 reduce 수행
print(np.add.reduceat(arr, [0, 2], axis=0))  # [[1 2 3] [11 13 15]]

# outer: 외적 연산
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.add.outer(a, b))
# [[5 6 7]
#  [6 7 8]
#  [7 8 9]]
```

## 사용자 정의 UFunc

```python
# frompyfunc: Python 함수를 UFunc로 변환
def my_function(x, y):
    return x + y * 2

my_ufunc = np.frompyfunc(my_function, 2, 1)  # (입력 인자 수, 출력 인자 수)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = my_ufunc(a, b)
print(result)  # [9 12 15]

# vectorize: 자동 벡터화
def my_function2(x, y):
    if x > y:
        return x
    else:
        return y

my_vectorized = np.vectorize(my_function2)
result2 = my_vectorized(a, b)
print(result2)  # [4 5 6]
```

## 성능 비교: UFunc vs Python 반복문

```python
import time

# 큰 배열 생성
size = 1000000
a = np.random.rand(size)
b = np.random.rand(size)

# UFunc 사용
start = time.time()
c_ufunc = a + b
ufunc_time = time.time() - start

# Python 반복문 사용
start = time.time()
c_loop = np.empty(size)
for i in range(size):
    c_loop[i] = a[i] + b[i]
loop_time = time.time() - start

print(f"UFunc 시간: {ufunc_time:.6f}초")
print(f"반복문 시간: {loop_time:.6f}초")
print(f"UFunc가 {loop_time/ufunc_time:.1f}배 더 빠름")
```

## 다음 학습 내용

다음으로는 통계 연산에 대해 알아보겠습니다. [`statistics.md`](statistics.md)를 참조하세요.