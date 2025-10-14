# NumPy 기본 연산

## 배열과 스칼라 연산

NumPy에서는 배열과 스칼라(단일 값) 간의 연산을 간단하게 수행할 수 있습니다. 이 연산은 배열의 각 요소에 individually 적용됩니다.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# 덧셈
print(arr + 2)  # [3 4 5 6 7]

# 뺄셈
print(arr - 1)  # [0 1 2 3 4]

# 곱셈
print(arr * 3)  # [ 3  6  9 12 15]

# 나눗셈
print(arr / 2)  # [0.5 1.  1.5 2.  2.5]

# 제곱
print(arr ** 2)  # [ 1  4  9 16 25]

# 나머지
print(arr % 2)  # [1 0 1 0 1]
```

## 배열 간 연산

같은 형태의 배열 간 연산은 요소별(element-wise)로 수행됩니다.

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 덧셈
print(a + b)  # [ 6  8 10 12]

# 뺄셈
print(a - b)  # [-4 -4 -4 -4]

# 곱셈
print(a * b)  # [ 5 12 21 32]

# 나눗셈
print(a / b)  # [0.2        0.33333333 0.42857143 0.5       ]

# 제곱
print(a ** b)  # [   1   64 2187 65536]
```

## 유니버설 함수(UFunc)

NumPy는 모든 배열 요소에 적용되는 내장 함수들을 제공합니다. 이를 유니버설 함수(UFunc)라고 부릅니다.

### 기본 수학 함수

```python
arr = np.array([1, 2, 3, 4, 5])

# 제곱근
print(np.sqrt(arr))  # [1.         1.41421356 1.73205081 2.         2.23606798]

# 지수 함수
print(np.exp(arr))  # [  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591 ]

# 자연 로그
print(np.log(arr))  # [0.         0.69314718 1.09861229 1.38629436 1.60943791]

# 상용 로그
print(np.log10(arr))  # [0.         0.30103    0.47712125 0.60205999 0.69897   ]

# 절대값
arr2 = np.array([-1, -2, 3, -4, 5])
print(np.abs(arr2))  # [1 2 3 4 5]

# 반올림
arr3 = np.array([1.2, 2.5, 3.7, 4.1])
print(np.round(arr3))  # [1. 2. 4. 4.]

# 올림
print(np.ceil(arr3))  # [2. 3. 4. 5.]

# 내림
print(np.floor(arr3))  # [1. 2. 3. 4.]
```

### 삼각함수

```python
# 라디안 값
angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

# 사인
print(np.sin(angles))  # [ 0.0000000e+00  1.0000000e+00  1.2246468e-16 -1.0000000e+00 -1.2246468e-16]

# 코사인
print(np.cos(angles))  # [ 1.000000e+00  6.123234e-17 -1.000000e+00 -1.836970e-16  1.000000e+00]

# 탄젠트
print(np.tan(angles))  # [ 0.00000000e+00  1.63312394e+16 -1.22464680e-16  5.44374645e-15 -1.22464680e-16]
```

## 비교 연산

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

# 요소별 비교
print(a > b)  # [False False False  True  True]
print(a < b)  # [ True  True False False False]
print(a == b)  # [False False  True False False]
print(a != b)  # [ True  True False  True  True]
print(a >= b)  # [False False  True  True  True]
print(a <= b)  # [ True  True  True False False]

# 스칼라와 비교
print(a > 3)  # [False False False  True  True]
print(a <= 2)  # [ True  True False False False]
```

## 논리 연산

```python
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

# 논리 AND
print(np.logical_and(a, b))  # [ True False False False]

# 논리 OR
print(np.logical_or(a, b))  # [ True  True  True False]

# 논리 NOT
print(np.logical_not(a))  # [False False  True  True]

# 논리 XOR
print(np.logical_xor(a, b))  # [False  True  True False]
```

## 집계 함수

### 1차원 배열

```python
arr = np.array([1, 2, 3, 4, 5])

# 합계
print(np.sum(arr))  # 15

# 평균
print(np.mean(arr))  # 3.0

# 표준편차
print(np.std(arr))  # 1.4142135623730951

# 분산
print(np.var(arr))  # 2.0

# 최소값
print(np.min(arr))  # 1

# 최대값
print(np.max(arr))  # 5

# 최소값의 인덱스
print(np.argmin(arr))  # 0

# 최대값의 인덱스
print(np.argmax(arr))  # 4

# 누적 합계
print(np.cumsum(arr))  # [ 1  3  6 10 15]

# 누적 곱
print(np.cumprod(arr))  # [  1   2   6  24 120]
```

### 다차원 배열

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 전체 합계
print(np.sum(arr2d))  # 45

# 축별 합계 (axis=0: 행 방향, axis=1: 열 방향)
print(np.sum(arr2d, axis=0))  # [12 15 18]  # 각 열의 합계
print(np.sum(arr2d, axis=1))  # [ 6 15 24]  # 각 행의 합계

# 축별 평균
print(np.mean(arr2d, axis=0))  # [4. 5. 6.]
print(np.mean(arr2d, axis=1))  # [2. 5. 8.]

# 축별 최소값
print(np.min(arr2d, axis=0))  # [1 2 3]
print(np.min(arr2d, axis=1))  # [1 4 7]

# 축별 최대값
print(np.max(arr2d, axis=0))  # [7 8 9]
print(np.max(arr2d, axis=1))  # [3 6 9]
```

## 배열 메서드

NumPy 배열은 객체 메서드 형태로도 많은 함수를 제공합니다.

```python
arr = np.array([1, 2, 3, 4, 5])

# 배열 메서드
print(arr.sum())  # 15
print(arr.mean())  # 3.0
print(arr.std())  # 1.4142135623730951
print(arr.var())  # 2.0
print(arr.min())  # 1
print(arr.max())  # 5
print(arr.argmin())  # 0
print(arr.argmax())  # 4
print(arr.cumsum())  # [ 1  3  6 10 15]
print(arr.cumprod())  # [  1   2   6  24 120]
```

## 조건부 연산

```python
arr = np.array([1, 2, 3, 4, 5])

# 조건에 맞는 요소 선택
print(arr[arr > 3])  # [4 5]

# where 함수
print(np.where(arr > 3, arr, 0))  # [0 0 0 4 5]
# 조건이 참이면 arr 값, 거짓이면 0

# select 함수
conditions = [arr < 2, (arr >= 2) & (arr < 4), arr >= 4]
choices = [arr * 10, arr * 100, arr * 1000]
print(np.select(conditions, choices, default=0))  # [  10  200  300 4000 5000]
```

## 정렬

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 오름차순 정렬
print(np.sort(arr))  # [1 1 2 3 4 5 6 9]

# 내림차순 정렬
print(np.sort(arr)[::-1])  # [9 6 5 4 3 2 1 1]

# 정렬된 인덱스
print(np.argsort(arr))  # [1 3 6 0 2 4 7 5]

# 2차원 배열 정렬
arr2d = np.array([[3, 1, 2], [1, 3, 2], [2, 1, 3]])

# 행별 정렬
print(np.sort(arr2d, axis=1))
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]

# 열별 정렬
print(np.sort(arr2d, axis=0))
# [[1 1 2]
#  [2 3 2]
#  [3 3 3]]
```

## 다음 학습 내용

다음으로는 배열 조작(인덱싱, 슬라이싱, 형태 변환)에 대해 알아보겠습니다. [`../02-array-manipulation/indexing-slicing.md`](../02-array-manipulation/indexing-slicing.md)를 참조하세요.