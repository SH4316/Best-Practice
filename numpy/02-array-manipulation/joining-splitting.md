# NumPy 배열 결합과 분할

## 배열 결합 (Joining)

NumPy에서는 여러 배열을 하나로 결합하는 다양한 방법을 제공합니다.

### concatenate 함수

가장 기본적인 배열 결합 방법으로, 지정된 축을 따라 배열을 결합합니다.

```python
import numpy as np

# 1차원 배열 결합
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

result = np.concatenate((arr1, arr2, arr3))
print(result)  # [1 2 3 4 5 6 7 8 9]

# 2차원 배열 결합
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 수직 결합 (axis=0, 기본값)
result_v = np.concatenate((arr1, arr2))
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

# 3차원 배열 결합
arr1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
arr2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])

# 깊이 방향 결합 (axis=0)
result_d = np.concatenate((arr1, arr2), axis=0)
print(result_d.shape)  # (4, 2, 2)
```

### stack 함수

`stack`은 새로운 축을 추가하여 배열을 결합합니다.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 새로운 축으로 쌓기 (axis=0, 기본값)
result = np.stack((arr1, arr2))
print(result)
# [[1 2 3]
#  [4 5 6]]
print(result.shape)  # (2, 3)

# 다른 축으로 쌓기
result_axis1 = np.stack((arr1, arr2), axis=1)
print(result_axis1)
# [[1 4]
#  [2 5]
#  [3 6]]
print(result_axis1.shape)  # (3, 2)

# 2차원 배열 스택
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

result_2d = np.stack((arr1, arr2))
print(result_2d.shape)  # (2, 2, 2)
print(result_2d)
# [[[1 2]
#   [3 4]]
# 
#  [[5 6]
#   [7 8]]]
```

### vstack, hstack, dstack

더 직관적인 이름의 결합 함수들입니다.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 수직 쌓기 (vstack)
v_result = np.vstack((arr1, arr2))
print(v_result)
# [[1 2 3]
#  [4 5 6]]

# 수평 쌓기 (hstack)
h_result = np.hstack((arr1, arr2))
print(h_result)  # [1 2 3 4 5 6]

# 깊이 쌓기 (dstack)
d_result = np.dstack((arr1, arr2))
print(d_result.shape)  # (1, 3, 2)
print(d_result)
# [[[1 4]
#   [2 5]
#   [3 6]]]

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

d_result_2d = np.dstack((arr1_2d, arr2_2d))
print(d_result_2d.shape)  # (2, 2, 2)
print(d_result_2d)
# [[[1 5]
#   [2 6]]
# 
#  [[3 7]
#   [4 8]]]
```

### column_stack, row_stack

특정 축으로 배열을 쌓는 전문 함수들입니다.

```python
# 1차원 배열
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 열로 쌓기
col_result = np.column_stack((arr1, arr2))
print(col_result)
# [[1 4]
#  [2 5]
#  [3 6]]

# 행으로 쌓기 (vstack과 동일)
row_result = np.row_stack((arr1, arr2))
print(row_result)
# [[1 2 3]
#  [4 5 6]]

# 2차원 배열
arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6], [7, 8]])

col_result_2d = np.column_stack((arr1_2d, arr2_2d))
print(col_result_2d)
# [[1 2 5 6]
#  [3 4 7 8]]
```

## 배열 분할 (Splitting)

배열 분할은 결합의 반대 과정으로, 하나의 배열을 여러 부분으로 나눕니다.

### split 함수

```python
arr = np.arange(12)

# 균등 분할
result = np.split(arr, 3)
print(result)  # [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([8, 9, 10, 11])]

# 분할 지점 지정
result_points = np.split(arr, [3, 7])
print(result_points)  # [array([0, 1, 2]), array([3, 4, 5, 6]), array([ 7,  8,  9, 10, 11])]

# 2차원 배열 분할
arr_2d = np.arange(12).reshape(4, 3)
print(arr_2d)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

# 수직 분할 (axis=0)
v_split = np.split(arr_2d, 2, axis=0)
print(v_split[0])
# [[0 1 2]
#  [3 4 5]]

# 수평 분할 (axis=1)
h_split = np.split(arr_2d, 3, axis=1)
print(h_split[0])
# [[0]
#  [3]
#  [6]
#  [9]]
```

### array_split 함수

균등하게 나누어지지 않을 때 사용합니다.

```python
arr = np.arange(10)

# 3개로 분할 (균등하지 않음)
result = np.array_split(arr, 3)
print(result)
# [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([8, 9])]

# 2차원 배열
arr_2d = np.arange(10).reshape(5, 2)
result_2d = np.array_split(arr_2d, 3)
print([r.shape for r in result_2d])  # [(2, 2), (2, 2), (1, 2)]
```

### hsplit, vsplit, dsplit

```python
arr = np.arange(16).reshape(4, 4)
print(arr)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

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

# 3차원 배열
arr_3d = np.arange(24).reshape(2, 3, 4)
d_split = np.dsplit(arr_3d, 2)
print(d_split[0].shape)  # (2, 3, 2)
```

## 고급 결합과 분할 기법

### 블록 결합 (block)

```python
# 블록으로 배열 결합
A = np.eye(2) * 2
B = np.ones((2, 2))
C = np.zeros((2, 2))
D = np.eye(2) * 3

block_result = np.block([[A, B], [C, D]])
print(block_result)
# [[2. 0. 1. 1.]
#  [0. 2. 1. 1.]
#  [0. 0. 3. 0.]
#  [0. 0. 0. 3.]]
```

### 그리드 결합 (meshgrid)

```python
# 1차원 배열로부터 그리드 생성
x = np.array([1, 2, 3])
y = np.array([4, 5, 6, 7])

X, Y = np.meshgrid(x, y)
print(X)
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]
#  [1 2 3]]

print(Y)
# [[4 4 4]
#  [5 5 5]
#  [6 6 6]
#  [7 7 7]]

# 인덱싱 스타일 지정
X_sparse, Y_sparse = np.meshgrid(x, y, sparse=True)
print(X_sparse)
# [[1 2 3]]

print(Y_sparse)
# [[4]
#  [5]
#  [6]
#  [7]]
```

## 실용적인 예제

### 데이터 배치 처리

```python
# 여러 데이터 파일을 하나의 배치로 결합
data1 = np.random.rand(100, 10)  # 100개 샘플, 10개 특성
data2 = np.random.rand(80, 10)   # 80개 샘플, 10개 특성
data3 = np.random.rand(120, 10)  # 120개 샘플, 10개 특성

# 모든 데이터 결합
full_dataset = np.vstack((data1, data2, data3))
print(full_dataset.shape)  # (300, 10)

# 배치로 분할
batch_size = 50
batches = np.array_split(full_dataset, len(full_dataset) // batch_size)
print(len(batches))  # 6
print(batches[0].shape)  # (50, 10)
```

### 이미지 데이터 처리

```python
# 여러 이미지를 하나의 배열로 결합
image1 = np.random.rand(64, 64, 3)  # 64x64 RGB 이미지
image2 = np.random.rand(64, 64, 3)
image3 = np.random.rand(64, 64, 3)

# 배치로 결합 (배치 크기, 높이, 너비, 채널)
image_batch = np.stack((image1, image2, image3))
print(image_batch.shape)  # (3, 64, 64, 3)

# 배치 분할
individual_images = np.split(image_batch, image_batch.shape[0])
print(len(individual_images))  # 3
print(individual_images[0].shape)  # (1, 64, 64, 3)
```

### 시계열 데이터 처리

```python
# 시계열 데이터 생성
time_series = np.random.rand(1000, 5)  # 1000개 시간점, 5개 변수

# 훈련/검증/테스트 데이터 분할
train_size = int(0.7 * len(time_series))
val_size = int(0.2 * len(time_series))

train_data = time_series[:train_size]
val_data = time_series[train_size:train_size + val_size]
test_data = time_series[train_size + val_size:]

print(train_data.shape, val_data.shape, test_data.shape)  # (700, 5) (200, 5) (100, 5)

# 다시 결합
combined_data = np.concatenate((train_data, val_data, test_data))
print(combined_data.shape)  # (1000, 5)
```

## 성능 고려사항

### 메모리 효율성

```python
# 큰 배열을 결합할 때 메모리 문제 고려
large_arr1 = np.random.rand(10000, 1000)
large_arr2 = np.random.rand(10000, 1000)

# 미리 결과 배열을 할당하여 메모리 효율성 향상
result = np.empty((20000, 1000))
result[:10000] = large_arr1
result[10000:] = large_arr2
```

### 뷰 vs 복사

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# concatenate는 항상 새로운 배열 생성 (복사)
result = np.concatenate((arr1, arr2))
print(np.shares_memory(result, arr1))  # False

# stack도 항상 새로운 배열 생성 (복사)
stacked = np.stack((arr1, arr2))
print(np.shares_memory(stacked, arr1))  # False

# vstack, hstack도 마찬가지
v_result = np.vstack((arr1, arr2))
print(np.shares_memory(v_result, arr1))  # False
```

## 다음 학습 내용

다음으로는 수학적 연산과 유니버설 함수에 대해 알아보겠습니다. [`../03-mathematical-operations/universal-functions.md`](../03-mathematical-operations/universal-functions.md)를 참조하세요.