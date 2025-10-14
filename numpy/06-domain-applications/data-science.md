# NumPy 데이터 과학 응용

## 데이터 과학에서의 NumPy

NumPy는 데이터 과학의 기본 라이브러리로, 데이터 전처리, 통계 분석, 기계 학습 등 다양한 작업에 사용됩니다. Pandas, Scikit-learn, Matplotlib 등 다른 데이터 과학 라이브러리의 기반이 됩니다.

## 데이터 전처리

### 결측값 처리

```python
import numpy as np

# 결측값이 포함된 데이터 생성
data = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, np.nan])
print(f"원본 데이터: {data}")

# 결측값 확인
print(f"결측값 개수: {np.sum(np.isnan(data))}")
print(f"결측값 위치: {np.where(np.isnan(data))[0]}")

# 결측값 제거
cleaned_data = data[~np.isnan(data)]
print(f"결측값 제거: {cleaned_data}")

# 결측값 대체 (평균값)
mean_val = np.nanmean(data)  # NaN을 무시한 평균
filled_data = np.where(np.isnan(data), mean_val, data)
print(f"평균값으로 대체: {filled_data}")

# 결측값 대체 (중앙값)
median_val = np.nanmedian(data)  # NaN을 무시한 중앙값
filled_median = np.where(np.isnan(data), median_val, data)
print(f"중앙값으로 대체: {filled_median}")

# 선형 보간으로 결측값 대체
def linear_interpolation(data):
    """선형 보간으로 결측값 대체"""
    mask = ~np.isnan(data)
    if not np.any(mask):
        return data
    
    # 유효한 인덱스와 값
    valid_indices = np.where(mask)[0]
    valid_values = data[mask]
    
    # 보간
    interpolated = np.interp(np.arange(len(data)), valid_indices, valid_values)
    
    # 원래 NaN 위치에만 보간된 값 적용
    result = data.copy()
    result[np.isnan(data)] = interpolated[np.isnan(data)]
    
    return result

interpolated_data = linear_interpolation(data)
print(f"선형 보간으로 대체: {interpolated_data}")
```

### 이상치 탐지 및 처리

```python
# 이상치 탐지를 위한 데이터 생성
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)

# 이상치 추가
outliers = np.array([10, -8, 12, -10])  # 명백한 이상치
data_with_outliers = np.concatenate([normal_data, outliers])

# Z-score를 이용한 이상치 탐지
z_scores = np.abs((data_with_outliers - np.mean(data_with_outliers)) / np.std(data_with_outliers))
outlier_mask = z_scores > 3  # Z-score가 3 초과인 값
outliers_detected = data_with_outliers[outlier_mask]

print(f"이상치 개수 (Z-score): {np.sum(outlier_mask)}")
print(f"탐지된 이상치: {outliers_detected}")

# IQR을 이용한 이상치 탐지
q1, q3 = np.percentile(data_with_outliers, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask_iqr = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)
outliers_iqr = data_with_outliers[outlier_mask_iqr]

print(f"이상치 개수 (IQR): {np.sum(outlier_mask_iqr)}")
print(f"탐지된 이상치: {outliers_iqr}")

# 이상치 처리 (제거)
cleaned_data = data_with_outliers[~outlier_mask]
print(f"이상치 제거 후 데이터 크기: {len(cleaned_data)}")

# 이상치 처리 (대체)
# 이상치를 경계값으로 대체
capped_data = data_with_outliers.copy()
capped_data[capped_data < lower_bound] = lower_bound
capped_data[capped_data > upper_bound] = upper_bound
```

### 데이터 정규화와 표준화

```python
# 데이터 생성
data = np.random.randn(1000) * 10 + 50  # 평균 50, 표준편차 10

# Min-Max 정규화 (0-1 스케일링)
min_val = np.min(data)
max_val = np.max(data)
normalized_data = (data - min_val) / (max_val - min_val)

print(f"Min-Max 정규화:")
print(f"  최소값: {np.min(normalized_data):.6f}")
print(f"  최대값: {np.max(normalized_data):.6f}")

# Z-score 표준화 (평균 0, 표준편차 1)
mean_val = np.mean(data)
std_val = np.std(data)
standardized_data = (data - mean_val) / std_val

print(f"\nZ-score 표준화:")
print(f"  평균: {np.mean(standardized_data):.6f}")
print(f"  표준편차: {np.std(standardized_data):.6f}")

# Robust 스케일링 (중앙값과 IQR 사용)
median_val = np.median(data)
iqr_val = np.percentile(data, 75) - np.percentile(data, 25)
robust_scaled = (data - median_val) / iqr_val

print(f"\nRobust 스케일링:")
print(f"  중앙값: {np.median(robust_scaled):.6f}")
print(f"  IQR: {np.percentile(robust_scaled, 75) - np.percentile(robust_scaled, 25):.6f}")

# 다차원 데이터 정규화
data_2d = np.random.randn(100, 5)  # 100개 샘플, 5개 특성

# 특성별 정규화 (열별)
min_vals = np.min(data_2d, axis=0)
max_vals = np.max(data_2d, axis=0)
normalized_2d = (data_2d - min_vals) / (max_vals - min_vals)

# 샘플별 정규화 (행별)
sample_norms = np.linalg.norm(data_2d, axis=1, keepdims=True)
normalized_samples = data_2d / sample_norms

print(f"\n다차원 데이터 정규화:")
print(f"  원본 형태: {data_2d.shape}")
print(f"  특성별 정규화 형태: {normalized_2d.shape}")
print(f"  샘플별 정규화 형태: {normalized_samples.shape}")
```

## 통계 분석

### 기술 통계

```python
# 샘플 데이터 생성
np.random.seed(42)
data = np.random.normal(100, 15, 1000)  # 평균 100, 표준편차 15

# 기본 기술 통계
print(f"기본 기술 통계:")
print(f"  샘플 수: {len(data)}")
print(f"  평균: {np.mean(data):.2f}")
print(f"  중앙값: {np.median(data):.2f}")
print(f"  표준편차: {np.std(data):.2f}")
print(f"  분산: {np.var(data):.2f}")
print(f"  최소값: {np.min(data):.2f}")
print(f"  최대값: {np.max(data):.2f}")
print(f"  범위: {np.ptp(data):.2f}")  # peak-to-peak

# 분위수
percentiles = [0, 25, 50, 75, 100]
values = np.percentile(data, percentiles)
print(f"\n분위수:")
for p, v in zip(percentiles, values):
    print(f"  {p}%: {v:.2f}")

# 왜도와 첨도 (계산 필요)
def skewness(data):
    """왜도 계산"""
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 3)

def kurtosis(data):
    """첨도 계산"""
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 4) - 3  # 초과 첨도

print(f"\n분포 모양:")
print(f"  왜도: {skewness(data):.4f}")
print(f"  첨도: {kurtosis(data):.4f}")
```

### 상관관계 분석

```python
# 다변량 데이터 생성
np.random.seed(42)
n_samples = 1000

# 상관관계가 있는 데이터 생성
x = np.random.randn(n_samples)
y = 0.5 * x + 0.5 * np.random.randn(n_samples)  # x와 양의 상관관계
z = -0.3 * x + 0.7 * np.random.randn(n_samples)  # x와 음의 상관관계

# 상관계수 행렬
data_matrix = np.column_stack([x, y, z])
corr_matrix = np.corrcoef(data_matrix.T)  # 열별 상관계수

print(f"상관계수 행렬:")
print(corr_matrix)

# 개별 상관계수
corr_xy = np.corrcoef(x, y)[0, 1]
corr_xz = np.corrcoef(x, z)[0, 1]
corr_yz = np.corrcoef(y, z)[0, 1]

print(f"\n개별 상관계수:")
print(f"  x-y: {corr_xy:.4f}")
print(f"  x-z: {corr_xz:.4f}")
print(f"  y-z: {corr_yz:.4f}")

# 공분산 행렬
cov_matrix = np.cov(data_matrix.T)
print(f"\n공분산 행렬:")
print(cov_matrix)
```

### 가설 검정

```python
# 두 그룹 간 평균 차이 검정 (t-검정)
np.random.seed(42)
group1 = np.random.normal(100, 15, 100)  # 평균 100, 표준편차 15
group2 = np.random.normal(105, 15, 100)  # 평균 105, 표준편차 15

# 독립 표본 t-검정 (수동 계산)
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)

# 풀링된 표준편차
pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))

# t-통계량
t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))

# 자유도
df = n1 + n2 - 2

print(f"독립 표본 t-검정:")
print(f"  그룹 1 평균: {mean1:.2f}")
print(f"  그룹 2 평균: {mean2:.2f}")
print(f"  t-통계량: {t_stat:.4f}")
print(f"  자유도: {df}")

# 카이제곱 검정 (적합도 검정)
observed = np.array([20, 30, 25, 15, 10])  # 관측 빈도
expected = np.array([25, 25, 25, 15, 10])  # 기대 빈도

# 카이제곱 통계량
chi2_stat = np.sum((observed - expected)**2 / expected)

print(f"\n카이제곱 검정:")
print(f"  관측 빈도: {observed}")
print(f"  기대 빈도: {expected}")
print(f"  카이제곱 통계량: {chi2_stat:.4f}")
```

## 기계 학습 기초

### 선형 회귀

```python
# 선형 회귀 구현
def linear_regression(X, y):
    """최소제곱법을 이용한 선형 회귀"""
    # 절편 항 추가
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # 정규方程식: β = (X'X)^(-1)X'y
    XtX = np.dot(X_with_intercept.T, X_with_intercept)
    Xty = np.dot(X_with_intercept.T, y)
    
    # 역행렬 계산
    try:
        XtX_inv = np.linalg.inv(XtX)
        coefficients = np.dot(XtX_inv, Xty)
    except np.linalg.LinAlgError:
        # 특이 행렬인 경우 유사 역행렬 사용
        XtX_inv = np.linalg.pinv(XtX)
        coefficients = np.dot(XtX_inv, Xty)
    
    return coefficients

# 예측 함수
def predict(X, coefficients):
    """선형 회귀 예측"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    return np.dot(X_with_intercept, coefficients)

# 평가 지표
def mse(y_true, y_pred):
    """평균 제곱 오차"""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """결정 계수"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# 데이터 생성
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 1) * 10  # 0-10 사이의 값
true_slope = 2.5
true_intercept = 1.5
y = true_intercept + true_slope * X[:, 0] + np.random.randn(n_samples) * 2  # 노이즈 추가

# 선형 회귀 모델 학습
coefficients = linear_regression(X, y)
intercept, slope = coefficients

print(f"선형 회귀 결과:")
print(f"  실제 절편: {true_intercept:.2f}, 추정 절편: {intercept:.2f}")
print(f"  실제 기울기: {true_slope:.2f}, 추정 기울기: {slope:.2f}")

# 예측 및 평가
y_pred = predict(X, coefficients)
mse_value = mse(y, y_pred)
r2_value = r2_score(y, y_pred)

print(f"\n모델 평가:")
print(f"  MSE: {mse_value:.4f}")
print(f"  R²: {r2_value:.4f}")
```

### 로지스틱 회귀

```python
# 시그모이드 함수
def sigmoid(z):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-z))

# 로지스틱 회귀 구현
def logistic_regression(X, y, learning_rate=0.01, n_iterations=1000):
    """경사 하강법을 이용한 로지스틱 회귀"""
    n_samples, n_features = X.shape
    
    # 절편 항 추가
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # 가중치 초기화
    weights = np.zeros(n_features + 1)
    
    # 경사 하강법
    for _ in range(n_iterations):
        # 선형 예측
        linear_pred = np.dot(X_with_intercept, weights)
        
        # 시그모이드 적용
        y_pred = sigmoid(linear_pred)
        
        # 그래디언트 계산
        gradient = np.dot(X_with_intercept.T, (y_pred - y)) / n_samples
        
        # 가중치 업데이트
        weights -= learning_rate * gradient
    
    return weights

# 예측 함수
def predict_logistic(X, weights, threshold=0.5):
    """로지스틱 회귀 예측"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    probabilities = sigmoid(np.dot(X_with_intercept, weights))
    return (probabilities >= threshold).astype(int)

# 정확도 계산
def accuracy(y_true, y_pred):
    """정확도 계산"""
    return np.mean(y_true == y_pred)

# 데이터 생성
np.random.seed(42)
n_samples = 200

# 클래스 0 데이터
X0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
y0 = np.zeros(n_samples // 2)

# 클래스 1 데이터
X1 = np.random.randn(n_samples // 2, 2) + np.array([6, 6])
y1 = np.ones(n_samples // 2)

# 데이터 결합
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# 로지스틱 회귀 학습
weights = logistic_regression(X, y, learning_rate=0.1, n_iterations=1000)

# 예측 및 평가
y_pred = predict_logistic(X, weights)
acc = accuracy(y, y_pred)

print(f"로지스틱 회귀 결과:")
print(f"  가중치: {weights}")
print(f"  정확도: {acc:.4f}")
```

### K-최근접 이웃 (K-NN)

```python
# 유클리드 거리 계산
def euclidean_distance(a, b):
    """두 점 사이의 유클리드 거리"""
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# K-NN 구현
def knn_predict(X_train, y_train, X_test, k=5):
    """K-최근접 이웃 예측"""
    y_pred = np.zeros(len(X_test))
    
    for i, x_test in enumerate(X_test):
        # 모든 훈련 데이터와의 거리 계산
        distances = euclidean_distance(X_train, x_test)
        
        # 가장 가까운 k개 이웃의 인덱스
        k_nearest_indices = np.argsort(distances)[:k]
        
        # k개 이웃의 레이블
        k_nearest_labels = y_train[k_nearest_indices]
        
        # 다수결 투표
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        y_pred[i] = unique_labels[np.argmax(counts)]
    
    return y_pred

# 데이터 생성
np.random.seed(42)
n_samples_per_class = 50

# 클래스 0 데이터
X0 = np.random.randn(n_samples_per_class, 2) + np.array([2, 2])
y0 = np.zeros(n_samples_per_class)

# 클래스 1 데이터
X1 = np.random.randn(n_samples_per_class, 2) + np.array([6, 6])
y1 = np.ones(n_samples_per_class)

# 데이터 결합
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# 훈련/테스트 분할
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# K-NN 예측
k = 5
y_pred = knn_predict(X_train, y_train, X_test, k=k)

# 정확도 계산
acc = accuracy(y_test, y_pred)

print(f"K-NN 결과 (k={k}):")
print(f"  테스트 정확도: {acc:.4f}")
```

## 데이터 시각화 기초

### 히스토그램과 분포

```python
# 히스토그램 데이터 생성
data = np.random.normal(0, 1, 1000)

# 히스토그램 계산
hist, bin_edges = np.histogram(data, bins=20)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

print(f"히스토그램 정보:")
print(f"  빈도수: {hist[:5]}...")  # 처음 5개만 표시
print(f"  구간 경계: {bin_edges[:6]}...")  # 처음 6개만 표시

# 밀도 추정 (커널 밀도 추정 간단 버전)
def simple_kde(data, points, bandwidth=0.2):
    """간단한 커널 밀도 추정"""
    n = len(data)
    density = np.zeros_like(points)
    
    for i, point in enumerate(points):
        # 가우시안 커널
        kernel_values = np.exp(-0.5 * ((data - point) / bandwidth) ** 2)
        density[i] = np.sum(kernel_values) / (n * bandwidth * np.sqrt(2 * np.pi))
    
    return density

# 밀도 추정
x_points = np.linspace(np.min(data), np.max(data), 100)
density = simple_kde(data, x_points)

print(f"\n밀도 추정:")
print(f"  최대 밀도: {np.max(density):.4f}")
print(f"  최대 밀도 위치: {x_points[np.argmax(density)]:.4f}")
```

## 다음 학습 내용

다음으로는 과학 계산 응용에 대해 알아보겠습니다. [`scientific-computing.md`](scientific-computing.md)를 참조하세요.