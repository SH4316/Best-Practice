# NumPy 브로드캐스팅(Broadcasting)

## 브로드캐스팅이란?

브로드캐스팅은 NumPy에서 형태가 다른 배열 간의 산술 연산을 가능하게 하는 메커니즘입니다. 작은 배열이 큰 배열의 형태로 "확장"되어 연산이 가능하게 됩니다.

## 브로드캐스팅 규칙

NumPy의 브로드캐스팅은 다음 규칙을 따릅니다:

1. **차원 수 맞추기**: 차원 수가 적은 배열의 왼쪽에 1을 추가하여 차원 수를 맞춤
2. **차원 크기 비교**: 각 차원에서 크기가 같거나 하나가 1이면 호환됨
3. **크기 1인 차원 확장**: 크기가 1인 차원은 다른 배열의 해당 차원 크기로 확장됨

### 기본 브로드캐스팅 예제

```python
import numpy as np

# 스칼라와 배열
arr = np.array([1, 2, 3, 4, 5])
scalar = 2
result = arr + scalar
print(f"스칼라와 배열: {result}")
# 스칼라가 [2, 2, 2, 2, 2]로 확장됨

# 1차원 배열과 2차원 배열
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])
result = arr_2d + arr_1d
print(f"1차원과 2차원:\n{result}")
# arr_1d가 [[10, 20, 30], [10, 20, 30]]로 확장됨

# 크기가 1인 차원
arr_2d = np.array([[1], [2], [3]])
arr_1d = np.array([10, 20, 30])
result = arr_2d + arr_1d
print(f"크기 1인 차원:\n{result}")
# arr_2d가 [[1, 1, 1], [2, 2, 2], [3, 3, 3]]로 확장됨
# arr_1d가 [[10, 20, 30], [10, 20, 30], [10, 20, 30]]로 확장됨
```

## 브로드캐스팅 시각화

### 차원별 브로드캐스팅

```python
# 다양한 형태의 배열 브로드캐스팅
# 형태: (3, 1) + (1, 4) -> (3, 4)
a = np.array([[1], [2], [3]])  # 형태: (3, 1)
b = np.array([10, 20, 30, 40])  # 형태: (4,) -> (1, 4)
result = a + b
print(f"결과 형태: {result.shape}")
print(result)
# a가 [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]로 확장됨
# b가 [[10, 20, 30, 40], [10, 20, 30, 40], [10, 20, 30, 40]]로 확장됨
```

### 브로드캐스팅 불가능한 경우

```python
# 브로드캐스팅이 불가능한 예제
try:
    a = np.array([[1, 2, 3], [4, 5, 6]])  # 형태: (2, 3)
    b = np.array([10, 20])  # 형태: (2,)
    result = a + b  # 오류 발생
except ValueError as e:
    print(f"브로드캐스팅 오류: {e}")
    # (2, 3)과 (2,)는 브로드캐스팅 규칙에 맞지 않음
```

## 브로드캐스팅 활용 기법

### 데이터 정규화

```python
# 데이터 정규화에 브로드캐스팅 활용
data = np.random.rand(5, 3)  # 5개 샘플, 3개 특성

# 특성별 정규화 (열별 정규화)
feature_means = np.mean(data, axis=0)  # 형태: (3,)
feature_stds = np.std(data, axis=0)    # 형태: (3,)

normalized_data = (data - feature_means) / feature_stds
print(f"원본 데이터 형태: {data.shape}")
print(f"평균 형태: {feature_means.shape}")
print(f"정규화된 데이터 형태: {normalized_data.shape}")

# 샘플별 정규화 (행별 정규화)
sample_means = np.mean(data, axis=1, keepdims=True)  # 형태: (5, 1)
sample_stds = np.std(data, axis=1, keepdims=True)    # 형태: (5, 1)

normalized_samples = (data - sample_means) / sample_stds
print(f"샘플 평균 형태: {sample_means.shape}")
print(f"샘플별 정규화된 데이터 형태: {normalized_samples.shape}")
```

### 거리 계산

```python
# 브로드캐스팅을 이용한 효율적인 거리 계산
points = np.array([[1, 2], [3, 4], [5, 6]])  # 3개 점
center = np.array([0, 0])  # 중심점

# 각 점에서 중심점까지의 거리
distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
print(f"거리: {distances}")

# 모든 점 쌍 간의 거리 행렬
def pairwise_distances(points):
    """모든 점 쌍 간의 거리 계산"""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # 형태: (3, 1, 2) - (1, 3, 2) -> (3, 3, 2)
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances

dist_matrix = pairwise_distances(points)
print(f"거리 행렬:\n{dist_matrix}")
```

### 그리드 생성

```python
# 브로드캐스팅을 이용한 그리드 생성
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# 2D 그리드 생성
X, Y = np.meshgrid(x, y)
print(f"X 형태: {X.shape}, Y 형태: {Y.shape}")

# 2D 가우시안 함수
Z = np.exp(-(X**2 + Y**2) / 10)
print(f"Z 형태: {Z.shape}")

# 브로드캐스팅을 직접 이용한 그리드 생성
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X_direct = x[np.newaxis, :]  # 형태: (1, 100)
Y_direct = y[:, np.newaxis]  # 형태: (100, 1)
Z_direct = np.exp(-(X_direct**2 + Y_direct**2) / 10)
print(f"직접 생성한 Z 형태: {Z_direct.shape}")
```

## 고급 브로드캐스팅 기법

### 다차원 브로드캐스팅

```python
# 3차원 브로드캐스팅
# 형태: (2, 1, 3) + (1, 4, 1) -> (2, 4, 3)
a = np.array([[[1, 2, 3]], [[4, 5, 6]]])  # 형태: (2, 1, 3)
b = np.array([[[10], [20], [30], [40]]])  # 형태: (1, 4, 1)
result = a + b
print(f"결과 형태: {result.shape}")
print(result)
```

### np.broadcast_to 명시적 브로드캐스팅

```python
# 명시적 브로드캐스팅
arr = np.array([1, 2, 3])
broadcasted = np.broadcast_to(arr, (4, 3))
print(f"원본 형태: {arr.shape}")
print(f"브로드캐스팅된 형태: {broadcasted.shape}")
print(broadcasted)

# 브로드캐스팅된 배열은 읽기 전용
try:
    broadcasted[0, 0] = 100
except ValueError as e:
    print(f"수정 오류: {e}")

# 쓰기 가능한 브로드캐스팅을 위해서는 복사 필요
writable = np.broadcast_to(arr, (4, 3)).copy()
writable[0, 0] = 100
print(f"수정 가능한 배열:\n{writable}")
```

### np.broadcast 객체

```python
# 브로드캐스팅 정보 확인
a = np.array([[1], [2], [3]])  # 형태: (3, 1)
b = np.array([10, 20, 30])     # 형태: (3,)

# 브로드캐스팅 객체 생성
broad = np.broadcast(a, b)
print(f"브로드캐스팅된 형태: {broad.shape}")
print(f"브로드캐스팅된 차원 수: {broad.nd}")
print(f"각 배열의 브로드캐스팅된 형태:")
for i, arr in enumerate([a, b]):
    print(f"  배열 {i}: {broad.iters[i].shape}")
```

## 브로드캐스팅 최적화

### 메모리 효율적인 브로드캐스팅

```python
import time

# 큰 배열에서의 브로드캐스팅 성능
size = 10000
a = np.random.rand(size, 1)  # 형태: (10000, 1)
b = np.random.rand(1, size)  # 형태: (1, 10000)

# 브로드캐스팅 연산
start = time.time()
result = a + b  # 형태: (10000, 10000)
broadcast_time = time.time() - start
print(f"브로드캐스팅 시간: {broadcast_time:.6f}초")
print(f"결과 형태: {result.shape}")

# 명시적 확장과 비교
start = time.time()
a_expanded = np.repeat(a, size, axis=1)  # 메모리 사용량 증가
b_expanded = np.repeat(b, size, axis=0)  # 메모리 사용량 증가
result_explicit = a_expanded + b_expanded
explicit_time = time.time() - start
print(f"명시적 확장 시간: {explicit_time:.6f}초")
print(f"메모리 사용량 비교: 브로드캐스팅이 훨씬 효율적")
```

### keepdims 매개변수 활용

```python
# keepdims를 이용한 차원 유지
data = np.random.rand(4, 6)

# keepdims=False (기본값)
row_means = np.mean(data, axis=1)  # 형태: (4,)
col_means = np.mean(data, axis=0)  # 형태: (6,)

# keepdims=True
row_means_keep = np.mean(data, axis=1, keepdims=True)  # 형태: (4, 1)
col_means_keep = np.mean(data, axis=0, keepdims=True)  # 형태: (1, 6)

# keepdims=True가 브로드캐스팅에 더 편리
normalized_rows = data - row_means_keep  # 자동 브로드캐스팅
normalized_cols = data - col_means_keep  # 자동 브로드캐스팅

# keepdims=False는 수동으로 차원 추가 필요
normalized_rows_manual = data - row_means[:, np.newaxis]  # 수차원 추가
normalized_cols_manual = data - col_means[np.newaxis, :]  # 수차원 추가
```

## 실용적인 브로드캐스팅 예제

### 이미지 처리

```python
# 브로드캐스팅을 이용한 이미지 처리
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# 밝기 조정 (모든 채널에 동일한 값 적용)
brightness_factor = 1.2
brightened = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

# 채널별 다른 조정 (브로드캐스팅)
channel_factors = np.array([1.1, 1.2, 0.9])  # R, G, B 채널별 다른 계수
adjusted = np.clip(image * channel_factors, 0, 255).astype(np.uint8)

print(f"원본 이미지 형태: {image.shape}")
print(f"채널 계수 형태: {channel_factors.shape}")
print(f"조정된 이미지 형태: {adjusted.shape}")
```

### 신호 처리

```python
# 브로드캐스팅을 이용한 신호 처리
t = np.linspace(0, 1, 1000)  # 시간 축
frequencies = np.array([5, 10, 15])  # 여러 주파수

# 여러 주파수의 정현파 생성
signals = np.sin(2 * np.pi * frequencies[:, np.newaxis] * t[np.newaxis, :])
# 형태: (3,) -> (3, 1), (1000,) -> (1, 1000) -> (3, 1000)

print(f"시간 축 형태: {t.shape}")
print(f"주파수 형태: {frequencies.shape}")
print(f"신호 형태: {signals.shape}")

# 신호 합성
combined_signal = np.sum(signals, axis=0)
print(f"합성된 신호 형태: {combined_signal.shape}")
```

### 머신러닝

```python
# 브로드캐스팅을 이용한 배치 연산
# 배치 입력 데이터 (batch_size, input_dim)
batch_size = 32
input_dim = 10
X = np.random.rand(batch_size, input_dim)

# 가중치와 편향
weights = np.random.rand(input_dim, 5)  # (input_dim, output_dim)
bias = np.random.rand(5)  # (output_dim,)

# 선형 변환: X @ weights + bias
# (32, 10) @ (10, 5) + (5,) -> (32, 5) + (5,) -> (32, 5) + (1, 5) -> (32, 5)
output = X @ weights + bias
print(f"입력 형태: {X.shape}")
print(f가중치 형태: {weights.shape}")
print(f"편향 형태: {bias.shape}")
print(f"출력 형태: {output.shape}")

# 활성화 함수 적용 (브로드캐스팅)
activated = 1 / (1 + np.exp(-output))  # 시그모이드
print(f"활성화된 출력 형태: {activated.shape}")
```

## 브로드캐스팅 디버깅

### 브로드캐스팅 오류 해결

```python
# 브로드캐스팅 오류 예제와 해결책
try:
    # 오류 발생 코드
    a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20])  # (2,)
    result = a + b
except ValueError as e:
    print(f"오류: {e}")
    
    # 해결책 1: 차원 추가
    b_fixed = b[:, np.newaxis]  # (2,) -> (2, 1)
    result1 = a + b_fixed  # (2, 3) + (2, 1) -> (2, 3)
    print(f"해결책 1 결과 형태: {result1.shape}")
    
    # 해결책 2: reshape
    b_fixed2 = b.reshape(2, 1)  # (2,) -> (2, 1)
    result2 = a + b_fixed2  # (2, 3) + (2, 1) -> (2, 3)
    print(f"해결책 2 결과 형태: {result2.shape}")
```

### 브로드캐스팅 시각화 도구

```python
def visualize_broadcasting(shape1, shape2):
    """브로드캐스팅 과정 시각화"""
    try:
        # 가상 배열 생성
        a = np.zeros(shape1)
        b = np.zeros(shape2)
        
        # 브로드캐스팅 시도
        c = a + b
        
        print(f"배열 1 형태: {shape1}")
        print(f"배열 2 형태: {shape2}")
        print(f"브로드캐스팅된 형태: {c.shape}")
        print("브로드캐스팅 성공!")
        
        # 브로드캐스팅 과정 설명
        max_dims = max(len(shape1), len(shape2))
        padded_shape1 = (1,) * (max_dims - len(shape1)) + shape1
        padded_shape2 = (1,) * (max_dims - len(shape2)) + shape2
        
        print("\n브로드캐스팅 과정:")
        print(f"  배열 1: {padded_shape1}")
        print(f"  배열 2: {padded_shape2}")
        print(f"  결과:   {c.shape}")
        
    except ValueError as e:
        print(f"브로드캐스팅 실패: {e}")

# 예제 실행
visualize_broadcasting((3, 1), (1, 4))
visualize_broadcasting((2, 3), (2,))
visualize_broadcasting((2, 3), (3,))
```

## 브로드캐스팅 모범 사례

1. **차원 이해**: 브로드캐스팅 규칙을 명확히 이해하고 활용
2. **keepdims 활용**: 차원을 유지하여 자동 브로드캐스팅 활용
3. **명시적 차원 추가**: np.newaxis나 reshape로 명시적으로 차원 추가
4. **메모리 효율성**: 브로드캐스팅은 메모리를 효율적으로 사용
5. **디버깅**: 브로드캐스팅 오류 시 형태를 확인하고 차원을 조정

## 다음 학습 내용

다음으로는 고급 기법에 대해 알아보겠습니다. [`../05-advanced-techniques/structured-arrays.md`](../05-advanced-techniques/structured-arrays.md)를 참조하세요.