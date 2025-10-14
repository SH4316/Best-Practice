# NumPy 배열 인덱싱과 슬라이싱

## 기본 인덱싱

NumPy 배열은 Python 리스트와 유사한 인덱싱을 지원하지만, 더 강력한 기능을 제공합니다.

### 1차원 배열 인덱싱

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# 단일 요소 접근
print(arr[0])  # 10
print(arr[2])  # 30
print(arr[-1])  # 50 (음수 인덱스)

# 값 수정
arr[1] = 25
print(arr)  # [10 25 30 40 50]
```

### 2차원 배열 인덱싱

```python
arr2d = np.array([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])

# 단일 요소 접근
print(arr2d[0, 0])  # 1
print(arr2d[1, 2])  # 7
print(arr2d[2, -1])  # 12

# 전체 행 접근
print(arr2d[1])  # [5 6 7 8]

# 값 수정
arr2d[0, 1] = 20
print(arr2d)
# [[ 1 20  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
```

### 3차원 배열 인덱싱

```python
arr3d = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]], 
                  [[9, 10], [11, 12]]])

# 단일 요소 접근
print(arr3d[0, 0, 0])  # 1
print(arr3d[1, 1, 0])  # 7

# 2차원 부분 배열 접근
print(arr3d[1])  # [[5 6] [7 8]]
print(arr3d[:, 0])  # [[1 2] [5 6] [9 10]]
```

## 슬라이싱

슬라이싱은 배열의 부분 집합을 선택하는 방법입니다. `start:stop:step` 형식을 사용합니다.

### 1차원 배열 슬라이싱

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 기본 슬라이싱
print(arr[2:6])  # [2 3 4 5]

# 시작부터 특정 인덱스까지
print(arr[:4])  # [0 1 2 3]

# 특정 인덱스부터 끝까지
print(arr[3:])  # [3 4 5 6 7 8 9]

# 간격 지정
print(arr[1:8:2])  # [1 3 5 7]

# 간격만 지정 (전체 배열에 적용)
print(arr[::3])  # [0 3 6 9]

# 역순
print(arr[::-1])  # [9 8 7 6 5 4 3 2 1 0]

# 값 수정
arr[2:5] = [20, 30, 40]
print(arr)  # [ 0  1 20 30 40  5  6  7  8  9]
```

### 2차원 배열 슬라이싱

```python
arr2d = np.array([[1, 2, 3, 4, 5], 
                  [6, 7, 8, 9, 10], 
                  [11, 12, 13, 14, 15], 
                  [16, 17, 18, 19, 20]])

# 행 슬라이싱
print(arr2d[1:3])
# [[ 6  7  8  9 10]
#  [11 12 13 14 15]]

# 열 슬라이싱
print(arr2d[:, 1:4])
# [[ 2  3  4]
#  [ 7  8  9]
#  [12 13 14]
#  [17 18 19]]

# 행과 열 모두 슬라이싱
print(arr2d[1:3, 2:4])
# [[ 8  9]
#  [13 14]]

# 간격 지정
print(arr2d[::2, ::2])
# [[ 1  3  5]
#  [11 13 15]]

# 값 수정
arr2d[1:3, 1:3] = 0
print(arr2d)
# [[ 1  2  3  4  5]
#  [ 6  0  0  9 10]
#  [11  0  0 14 15]
#  [16 17 18 19 20]]
```

## 팬시 인덱싱(Fancy Indexing)

팬시 인덱싱은 정수 배열을 사용하여 여러 요소를 동시에 선택하는 방법입니다.

### 1차원 배열 팬시 인덱싱

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# 인덱스 배열로 선택
indices = [0, 2, 4, 6]
print(arr[indices])  # [10 30 50 70]

# 순서에 상관없이 선택
indices = [6, 2, 8, 0]
print(arr[indices])  # [70 30 90 10]

# 불리언 배열로 선택
bool_indices = [True, False, True, False, True, False, True, False, True, False]
print(arr[bool_indices])  # [10 30 50 70 90]

# 조건을 만족하는 요소 선택
print(arr[arr > 50])  # [60 70 80 90 100]
print(arr[(arr > 20) & (arr < 80)])  # [30 40 50 60 70]
```

### 2차원 배열 팬시 인덱싱

```python
arr2d = np.array([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12], 
                  [13, 14, 15, 16]])

# 특정 행 선택
row_indices = [0, 2]
print(arr2d[row_indices])
# [[ 1  2  3  4]
#  [ 9 10 11 12]]

# 특정 행과 열 선택
row_indices = [0, 2, 3]
col_indices = [1, 3]
print(arr2d[row_indices, col_indices])  # [ 2 12 16]

# 다른 방식의 팬시 인덱싱
print(arr2d[[1, 3], [0, 2]])  # [5 15] (arr2d[1,0]과 arr2d[3,2])

# 불리언 인덱싱
mask = arr2d > 8
print(arr2d[mask])  # [ 9 10 11 12 13 14 15 16]
```

## 불리언 인덱싱

불리언 인덱싱은 조건을 기반으로 요소를 선택하는 강력한 방법입니다.

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 조건을 만족하는 요소 선택
print(arr[arr > 5])  # [ 6  7  8  9 10]
print(arr[arr % 2 == 0])  # [ 2  4  6  8 10]
print(arr[(arr > 3) & (arr < 8)])  # [4 5 6 7]

# 조건을 만족하는 요소 수정
arr[arr > 5] = 0
print(arr)  # [1 2 3 4 5 0 0 0 0 0]

# 2차원 배열 불리언 인덱싱
arr2d = np.array([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])

# 특정 조건을 만족하는 요소 선택
print(arr2d[arr2d > 7])  # [ 8  9 10 11 12]

# 특정 조건을 만족하는 요소 수정
arr2d[arr2d % 2 == 0] = -1
print(arr2d)
# [[ 1 -1  3 -1]
#  [ 5 -1  7 -1]
#  [ 9 -1 11 -1]]
```

## where 함수

`np.where()` 함수는 조건에 따라 값을 선택하는 데 사용됩니다.

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 조건이 참이면 x, 거짓이면 y
result = np.where(arr > 5, arr, 0)
print(result)  # [0 0 0 0 0 6 7 8 9 10]

# 2차원 배열
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result2d = np.where(arr2d % 2 == 0, '짝수', '홀수')
print(result2d)
# [['홀수' '짝수' '홀수']
#  ['짝수' '홀수' '짝수']
#  ['홀수' '짝수' '홀수']]
```

## 인덱싱과 슬라이싱의 뷰(View) vs 복사(Copy)

### 뷰(View) 동작

```python
arr = np.array([1, 2, 3, 4, 5])

# 슬라이싱은 뷰를 반환
slice_arr = arr[1:4]
print(slice_arr)  # [2 3 4]

# 뷰를 수정하면 원본도 수정됨
slice_arr[0] = 20
print(arr)  # [ 1 20  3  4  5]
```

### 복사(Copy) 동작

```python
arr = np.array([1, 2, 3, 4, 5])

# 팬시 인덱싱은 복사를 반환
fancy_arr = arr[[1, 3, 4]]
print(fancy_arr)  # [2 4 5]

# 복사를 수정해도 원본은 수정되지 않음
fancy_arr[0] = 20
print(arr)  # [1 2 3 4 5]

# 명시적으로 복사
copy_arr = arr[1:4].copy()
copy_arr[0] = 30
print(arr)  # [1 2 3 4 5]
```

## 고급 인덱싱 기법

### np.ix_ 함수

```python
arr2d = np.array([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12], 
                  [13, 14, 15, 16]])

# 여러 차원에서 동시에 인덱싱
rows = np.array([0, 2])
cols = np.array([1, 3])
print(arr2d[np.ix_(rows, cols)])
# [[ 2  4]
#  [10 12]]
```

### take 함수

```python
arr = np.array([10, 20, 30, 40, 50])

# 특정 인덱스의 값 가져오기
indices = [0, 2, 4]
print(np.take(arr, indices))  # [10 30 50]

# 축 지정
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(np.take(arr2d, [0, 2], axis=1))  # [[1 3] [4 6]]
```

### put 함수

```python
arr = np.array([10, 20, 30, 40, 50])

# 특정 인덱스에 값 설정
indices = [1, 3]
values = [200, 400]
np.put(arr, indices, values)
print(arr)  # [ 10 200  30 400  50]
```

## 다음 학습 내용

다음으로는 배열의 형태 변환(reshape)에 대해 알아보겠습니다. [`reshaping.md`](reshaping.md)를 참조하세요.