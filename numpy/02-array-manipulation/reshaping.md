# NumPy 배열 형태 변환(Reshaping)

## reshape 함수

`reshape()` 함수는 배열의 형태를 변경하는 가장 기본적인 방법입니다. 배열의 전체 요소 수는 유지되면서 차원과 크기를 변경합니다.

### 기본 reshape

```python
import numpy as np

# 1차원 배열 생성
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# 2차원으로 변환
arr_2d = arr.reshape(3, 4)
print(arr_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 3차원으로 변환
arr_3d = arr.reshape(2, 2, 3)
print(arr_3d)
# [[[ 0  1  2]
#   [ 3  4  5]]
# 
#  [[ 6  7  8]
#   [ 9 10 11]]]
```

### 자동 크기 계산 (-1 사용)

```python
arr = np.arange(12)

# 하나의 차원을 -1로 지정하면 자동으로 계산
arr_auto = arr.reshape(3, -1)
print(arr_auto.shape)  # (3, 4)

arr_auto2 = arr.reshape(-1, 6)
print(arr_auto2.shape)  # (2, 6)

# 여러 차원에서 -1 사용 (하나만 가능)
try:
    arr.reshape(-1, -1, 3)
except ValueError as e:
    print(e)  # can only specify one unknown dimension
```

## 차원 변경

### 1차원으로 변환 (flatten, ravel)

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# flatten: 항상 복사본 반환
flattened = arr_2d.flatten()
print(flattened)  # [1 2 3 4 5 6]

# ravel: 가능하면 뷰(view) 반환
raveled = arr_2d.ravel()
print(raveled)  # [1 2 3 4 5 6]

# 뷰인지 확인
raveled[0] = 100
print(arr_2d)  # [[100   2   3] [  4   5   6]] (원본이 변경됨)
```

### 차원 추가/제거

```python
arr = np.array([1, 2, 3])

# np.newaxis로 차원 추가
arr_2d = arr[np.newaxis, :]
print(arr_2d.shape)  # (1, 3)
print(arr_2d)  # [[1 2 3]]

arr_2d_col = arr[:, np.newaxis]
print(arr_2d_col.shape)  # (3, 1)
print(arr_2d_col)
# [[1]
#  [2]
#  [3]]

# expand_dim으로 차원 추가
arr_expanded = np.expand_dims(arr, axis=0)
print(arr_expanded.shape)  # (1, 3)

arr_expanded2 = np.expand_dims(arr, axis=1)
print(arr_expanded2.shape)  # (3, 1)

# squeeze로 크기가 1인 차원 제거
arr_squeezed = np.squeeze(arr_2d)
print(arr_squeezed.shape)  # (3,)
```

## 배열 전치(Transpose)

### 2차원 배열 전치

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)
# [[1 2 3]
#  [4 5 6]]

# T 속성으로 전치
transposed = arr.T
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]

# transpose 함수
transposed2 = np.transpose(arr)
print(transposed2)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### 다차원 배열 전치

```python
arr = np.arange(24).reshape(2, 3, 4)
print(arr.shape)  # (2, 3, 4)

# 축 순서 변경
transposed = np.transpose(arr, (1, 0, 2))
print(transposed.shape)  # (3, 2, 4)

# swapaxes로 두 축 교환
swapped = np.swapaxes(arr, 0, 2)
print(swapped.shape)  # (4, 3, 2)
```

## 배열 결합

### concatenate

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 수직 결합 (axis=0)
result_v = np.concatenate((arr1, arr2), axis=0)
print(result_v)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 수평 결합 (axis=1)
result_h = np.concatenate((arr1, arr2), axis=1)
print(result_h)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### vstack, hstack

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 수직 쌓기
v_result = np.vstack((arr1, arr2))
print(v_result)
# [[1 2 3]
#  [4 5 6]]

# 수평 쌓기
h_result = np.hstack((arr1, arr2))
print(h_result)  # [1 2 3 4 5 6]

# 2차원 배열
arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6], [7, 8]])

v_result_2d = np.vstack((arr1_2d, arr2_2d))
print(v_result_2d)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

h_result_2d = np.hstack((arr1_2d, arr2_2d))
print(h_result_2d)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### dstack

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 깊이 방향으로 쌓기
d_result = np.dstack((arr1, arr2))
print(d_result.shape)  # (2, 2, 2)
print(d_result)
# [[[1 5]
#   [2 6]]
# 
#  [[3 7]
#   [4 8]]]
```

## 배열 분할

### split

```python
arr = np.arange(12).reshape(4, 3)
print(arr)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

# 수평 분할
h_split = np.split(arr, 3, axis=1)
print(h_split[0])
# [[0]
#  [3]
#  [6]
#  [9]]

# 수직 분할
v_split = np.split(arr, 2, axis=0)
print(v_split[0])
# [[0 1 2]
#  [3 4 5]]
```

### hsplit, vsplit

```python
arr = np.arange(16).reshape(4, 4)

# 수평 분할
h_split = np.hsplit(arr, 2)
print(h_split[0])
# [[ 0  1]
#  [ 4  5]
#  [ 8  9]
#  [12 13]]

# 수직 분할
v_split = np.vsplit(arr, 2)
print(v_split[0])
# [[0 1 2 3]
#  [4 5 6 7]]
```

### array_split

균등하게 나누어지지 않을 때 사용

```python
arr = np.arange(10)

# 3개로 분할 (균등하지 않음)
split_result = np.array_split(arr, 3)
print(split_result)
# [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([8, 9])]
```

## 기타 형태 변환

### resize

```python
arr = np.array([1, 2, 3, 4])

# reshape와 달리 요소 수를 변경할 수 있음
resized = np.resize(arr, (2, 3))
print(resized)
# [[1 2 3]
#  [4 1 2]]

# 원본 배열의 resize 메서드
arr.resize(2, 3, refcheck=False)
print(arr)
# [[1 2 3]
#  [4 0 0]]
```

### pad

```python
arr = np.array([1, 2, 3])

# 상수 값으로 패딩
padded = np.pad(arr, (2, 3), 'constant', constant_values=(0, 0))
print(padded)  # [0 0 1 2 3 0 0 0]

# 엣지 값으로 패딩
padded_edge = np.pad(arr, (2, 2), 'edge')
print(padded_edge)  # [1 1 1 2 3 3 3]

# 반복으로 패딩
padded_repeat = np.pad(arr, (2, 2), 'reflect')
print(padded_repeat)  # [3 2 1 2 3 2 1]
```

### tile

```python
arr = np.array([1, 2, 3])

# 배열 반복
tiled = np.tile(arr, 3)
print(tiled)  # [1 2 3 1 2 3 1 2 3]

# 다차원 반복
arr_2d = np.array([[1, 2], [3, 4]])
tiled_2d = np.tile(arr_2d, (2, 3))
print(tiled_2d)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]
```

## 실용적인 형태 변환 예제

### 이미지 데이터 처리

```python
# 이미지 데이터를 (높이, 너비, 채널)에서 (채널, 높이, 너비)로 변환
image = np.random.rand(100, 200, 3)  # (높이, 너비, 채널)
image_transposed = np.transpose(image, (2, 0, 1))  # (채널, 높이, 너비)
print(image.shape, image_transposed.shape)  # (100, 200, 3) (3, 100, 200)
```

### 배치 데이터 처리

```python
# 여러 데이터를 하나의 배치로 결합
data1 = np.random.rand(10, 5)  # 10개 샘플, 5개 특성
data2 = np.random.rand(15, 5)  # 15개 샘플, 5개 특성

# 배치로 결합
batch = np.vstack((data1, data2))
print(batch.shape)  # (25, 5)

# 배치를 다시 작은 배치로 분할
mini_batches = np.array_split(batch, 5)
print([b.shape for b in mini_batches])  # [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
```

## 다음 학습 내용

다음으로는 배열의 결합과 분할에 대해 더 자세히 알아보겠습니다. [`joining-splitting.md`](joining-splitting.md)를 참조하세요.