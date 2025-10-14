# NumPy 선형대수 연산

## 선형대수 모듈 소개

NumPy는 선형대수 연산을 위한 `numpy.linalg` 모듈을 제공합니다. 이 모듈을 사용하여 행렬 연산, 고유값 분해, 특이값 분석 등 다양한 선형대수 계산을 수행할 수 있습니다.

## 기본 행렬 연산

### 행렬 곱셈

```python
import numpy as np

# 행렬 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 행렬 곱셈 (dot 함수)
C = np.dot(A, B)
print(C)
# [[19 22]
#  [43 50]]

# @ 연산자 (Python 3.5+)
C2 = A @ B
print(C2)
# [[19 22]
#  [43 50]]

# matmul 함수
C3 = np.matmul(A, B)
print(C3)
# [[19 22]
#  [43 50]]

# 요소별 곱셈 (주의: 행렬 곱셈과 다름)
elementwise = A * B
print(elementwise)
# [[ 5 12]
#  [21 32]]
```

### 벡터 내적과 외적

```python
# 벡터 생성
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 내적 (dot product)
dot_product = np.dot(v1, v2)
print(dot_product)  # 32

# @ 연산자를 사용한 내적
dot_product2 = v1 @ v2
print(dot_product2)  # 32

# 외적 (cross product, 3차원 벡터)
cross_product = np.cross(v1, v2)
print(cross_product)  # [-3  6 -3]

# 외적 (2차원 벡터의 경우)
v1_2d = np.array([1, 2])
v2_2d = np.array([3, 4])
cross_2d = np.cross(v1_2d, v2_2d)
print(cross_2d)  # -2 (스칼라 값)
```

## 행렬 속성과 변환

### 전치 행렬

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# 전치 행렬
A_T = A.T
print(A_T)
# [[1 4]
#  [2 5]
#  [3 6]]

# transpose 함수
A_T2 = np.transpose(A)
print(A_T2)
# [[1 4]
#  [2 5]
#  [3 6]]

# 다차원 배열의 전치
B = np.arange(24).reshape(2, 3, 4)
B_T = np.transpose(B, (1, 0, 2))  # 축 순서 변경
print(B_T.shape)  # (3, 2, 4)
```

### 대각 행렬과 추적

```python
# 대각 행렬 생성
diag_matrix = np.diag([1, 2, 3])
print(diag_matrix)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

# 행렬의 대각 요소 추출
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diag_elements = np.diag(A)
print(diag_elements)  # [1 5 9]

# 추적 (trace): 대각 요소의 합
trace_val = np.trace(A)
print(trace_val)  # 15

# 오프셋을 이용한 대각 요소
diag_offset = np.diag(A, k=1)  # 주대각선 위쪽
print(diag_offset)  # [2 6]
```

### 삼각 행렬

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 하삼각 행렬
lower_tri = np.tril(A)
print(lower_tri)
# [[1 0 0]
#  [4 5 0]
#  [7 8 9]]

# 상삼각 행렬
upper_tri = np.triu(A)
print(upper_tri)
# [[1 2 3]
#  [0 5 6]
#  [0 0 9]]

# 오프셋을 이용한 삼각 행렬
lower_tri_offset = np.tril(A, k=1)
print(lower_tri_offset)
# [[1 2 0]
#  [4 5 6]
#  [7 8 9]]
```

## 행렬식과 역행렬

### 행렬식 (Determinant)

```python
A = np.array([[1, 2], [3, 4]])

# 행렬식 계산
det_A = np.linalg.det(A)
print(det_A)  # -2.0

# 3x3 행렬의 행렬식
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
det_B = np.linalg.det(B)
print(det_B)  # 0.0 (특이 행렬)
```

### 역행렬 (Inverse)

```python
A = np.array([[1, 2], [3, 4]])

# 역행렬 계산
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# 역행렬 확인 (A @ A_inv = I)
print(np.round(A @ A_inv))
# [[1. 0.]
#  [0. 1.]]

# 유사 역행렬 (Moore-Penrose)
B = np.array([[1, 2, 3], [4, 5, 6]])  # 정방 행렬이 아님
B_pinv = np.linalg.pinv(B)
print(B_pinv)
```

## 선형 시스템 해결

### 연립 방정식 해결

```python
# 연립 방정식: 2x + y = 5, 3x - y = 1
A = np.array([[2, 1], [3, -1]])  # 계수 행렬
b = np.array([5, 1])  # 상수 벡터

# solve 함수로 해결
x = np.linalg.solve(A, b)
print(x)  # [1.333... 2.333...]

# 역행렬을 이용한 해결 (비효율적)
x_inv = np.linalg.inv(A) @ b
print(x_inv)  # [1.333... 2.333...]
```

### 최소제곱법

```python
# 과결정 시스템 (방정식이 미지수보다 많음)
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([7, 8, 9])

# 최소제곱해
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print(x)  # [-1.  4.]
print(residuals)  # 잔차 제곱합
print(rank)  # 행렬의 랭크
print(s)  # 특이값
```

## 고유값과 고유벡터

### 고유값 분해

```python
A = np.array([[3, 1], [1, 3]])

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)  # [4. 2.]
print(eigenvectors)
# [[ 0.70710678 -0.70710678]
#  [ 0.70710678  0.70710678]]

# 고유값 분해 확인
reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print(np.round(reconstructed))
# [[3. 1.]
#  [1. 3.]]
```

### 대칭 행렬의 고유값 분해

```python
# 대칭 행렬
A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])

# 대칭 행렬을 위한 고유값 분해
eigenvalues, eigenvectors = np.linalg.eigh(A)
print(eigenvalues)  # [1.58578644 2.         5.41421356]
print(eigenvectors)
```

## 특이값 분해 (SVD)

### SVD 계산

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# 특이값 분해
U, s, Vt = np.linalg.svd(A)
print(U.shape, s.shape, Vt.shape)  # (2, 2) (2,) (3, 3)
print(U)
print(s)  # 특이값
print(Vt)

# 완전한 SVD 재구성
Sigma = np.zeros(A.shape)
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
reconstructed = U @ Sigma @ Vt
print(np.round(reconstructed))
# [[1. 2. 3.]
#  [4. 5. 6.]]
```

### 축소된 SVD

```python
# 축소된 SVD (계산 효율성)
U_reduced, s_reduced, Vt_reduced = np.linalg.svd(A, full_matrices=False)
print(U_reduced.shape, s_reduced.shape, Vt_reduced.shape)  # (2, 2) (2,) (2, 3)

# 축소된 SVD 재구성
reduced_reconstructed = U_reduced @ np.diag(s_reduced) @ Vt_reduced
print(np.round(reduced_reconstructed))
# [[1. 2. 3.]
#  [4. 5. 6.]]
```

## 노름과 거리

### 벡터 노름

```python
v = np.array([3, 4])

# L2 노름 (유클리드 노름)
l2_norm = np.linalg.norm(v)
print(l2_norm)  # 5.0

# L1 노름 (맨해튼 노름)
l1_norm = np.linalg.norm(v, ord=1)
print(l1_norm)  # 7.0

# L∞ 노름 (최대값 노름)
linf_norm = np.linalg.norm(v, ord=np.inf)
print(linf_norm)  # 4.0

# 일반 p-노름
p_norm = np.linalg.norm(v, ord=3)
print(p_norm)  # 4.497...
```

### 행렬 노름

```python
A = np.array([[1, 2], [3, 4]])

# Frobenius 노름
fro_norm = np.linalg.norm(A, ord='fro')
print(fro_norm)  # 5.477...

# 행렬 2-노름 (스펙트럼 노름)
matrix_2_norm = np.linalg.norm(A, ord=2)
print(matrix_2_norm)  # 5.464...

# 행렬 1-노름 (최대 열 합)
matrix_1_norm = np.linalg.norm(A, ord=1)
print(matrix_1_norm)  # 6.0

# 행렬 ∞-노름 (최대 행 합)
matrix_inf_norm = np.linalg.norm(A, ord=np.inf)
print(matrix_inf_norm)  # 7.0
```

## 행렬 분해

### QR 분해

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])

# QR 분해
Q, R = np.linalg.qr(A)
print(Q.shape, R.shape)  # (3, 3) (3, 3)
print(np.round(Q))  # 직교 행렬
print(np.round(R))  # 상삼각 행렬

# QR 분해 확인
reconstructed = Q @ R
print(np.round(reconstructed))
# [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8. 10.]]
```

### 촐레스키 분해

```python
# 양의 정부호 대칭 행렬
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])

# 촐레스키 분해
L = np.linalg.cholesky(A)
print(L)
# [[2.         0.         0.        ]
#  [1.         2.         0.        ]
#  [0.5        1.25       2.05480467]]

# 촐레스키 분해 확인
reconstructed = L @ L.T
print(np.round(reconstructed))
# [[4. 2. 1.]
#  [2. 5. 3.]
#  [1. 3. 6.]]
```

## 실용적인 선형대수 응용

### 주성분 분석 (PCA)

```python
# 데이터 생성
np.random.seed(42)
data = np.random.multivariate_normal([0, 0], [[3, 1], [1, 2]], 100)

# 데이터 중앙화
data_centered = data - np.mean(data, axis=0)

# 공분산 행렬 계산
cov_matrix = np.cov(data_centered, rowvar=False)

# 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 주성분 (고유값이 큰 순서로 정렬)
sorted_indices = np.argsort(eigenvalues)[::-1]
principal_components = eigenvectors[:, sorted_indices]

print(principal_components)
```

### 선형 회귀 분석

```python
# 데이터 생성
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2 * x + 1 + np.random.normal(0, 1, 50)

# 설계 행렬 생성
X = np.vstack([x, np.ones(len(x))]).T

# 최소제곱법으로 회귀 계수 계산
coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
slope, intercept = coefficients

print(f"기울기: {slope:.4f}, 절편: {intercept:.4f}")
```

### 이미지 압축 (SVD 활용)

```python
# 예제 이미지 행렬
image = np.random.rand(100, 100)

# SVD 수행
U, s, Vt = np.linalg.svd(image)

# 상위 k개 특이값만 사용하여 압축
k = 20
compressed_image = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

print(f"원본 크기: {image.size}, 압축 크기: {U[:, :k].size + s[:k].size + Vt[:k, :size}")
```

## 다음 학습 내용

다음으로는 성능 최적화 기법에 대해 알아보겠습니다. [`../04-performance-optimization/vectorization.md`](../04-performance-optimization/vectorization.md)를 참조하세요.