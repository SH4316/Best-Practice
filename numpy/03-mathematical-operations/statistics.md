# NumPy 통계 연산

## 기본 통계 함수

NumPy는 데이터 분석에 필요한 다양한 통계 함수를 제공합니다.

### 중심 경향성 (Central Tendency)

```python
import numpy as np

# 데이터 생성
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 평균 (mean)
mean_val = np.mean(data)
print(mean_val)  # 5.5

# 중앙값 (median)
median_val = np.median(data)
print(median_val)  # 5.5

# 가중 평균
weights = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # 뒤쪽 값에 더 높은 가중치
weighted_mean = np.average(data, weights=weights)
print(weighted_mean)  # 6.333...

# 다차원 배열
data_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 전체 평균
print(np.mean(data_2d))  # 5.0

# 축별 평균
print(np.mean(data_2d, axis=0))  # [4. 5. 6.] (열 평균)
print(np.mean(data_2d, axis=1))  # [2. 5. 8.] (행 평균)
```

### 산포도 (Dispersion)

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 범위 (range)
range_val = np.ptp(data)  # peak-to-peak
print(range_val)  # 9 (최대값 - 최소값)

# 분산 (variance)
var_val = np.var(data)  # 표본 분산 (기본값 ddof=0)
print(var_val)  # 8.25

var_unbiased = np.var(data, ddof=1)  # 불편 분산
print(var_unbiased)  # 9.166...

# 표준편차 (standard deviation)
std_val = np.std(data)  # 표본 표준편차 (기본값 ddof=0)
print(std_val)  # 2.872...

std_unbiased = np.std(data, ddof=1)  # 불편 표준편차
print(std_unbiased)  # 3.027...

# 사분위수 (quantiles)
q1 = np.percentile(data, 25)  # 제1사분위수
q2 = np.percentile(data, 50)  # 제2사분위수 (중앙값과 동일)
q3 = np.percentile(data, 75)  # 제3사분위수

print(q1, q2, q3)  # 3.25 5.5 7.75

# 사분위 범위 (IQR)
iqr = q3 - q1
print(iqr)  # 4.5

# 중간 절대 편차 (Median Absolute Deviation)
median_val = np.median(data)
mad = np.median(np.abs(data - median_val))
print(mad)  # 2.5
```

### 최소값과 최대값

```python
data = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

# 최소값과 최대값
min_val = np.min(data)
max_val = np.max(data)
print(min_val, max_val)  # 1 9

# 최소값과 최대값의 인덱스
min_idx = np.argmin(data)
max_idx = np.argmax(data)
print(min_idx, max_idx)  # 1 5

# 다차원 배열
data_2d = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])

# 전체 최소값과 최대값
print(np.min(data_2d), np.max(data_2d))  # 1 9

# 축별 최소값과 최대값
print(np.min(data_2d, axis=0))  # [1 1 4] (열별 최소값)
print(np.max(data_2d, axis=1))  # [4 9 6] (행별 최대값)

# 축별 최소값과 최대값의 인덱스
print(np.argmin(data_2d, axis=0))  # [1 0 0] (열별 최소값 인덱스)
print(np.argmax(data_2d, axis=1))  # [2 2 1] (행별 최대값 인덱스)
```

## 상관관계와 공분산

### 공분산 (Covariance)

```python
# 두 변수 데이터 생성
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 공분산 행렬
cov_matrix = np.cov(x, y)
print(cov_matrix)
# [[2.5  1. ]
#  [1.   1.5]]

# x와 y의 공분산
cov_xy = cov_matrix[0, 1]
print(cov_xy)  # 1.0

# 다변량 데이터
data_multi = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
cov_matrix_multi = np.cov(data_multi)
print(cov_matrix_multi)
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]
```

### 상관관계 (Correlation)

```python
# 두 변수 데이터 생성
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 상관계수 행렬
corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
# [[1.         0.51639778]
#  [0.51639778 1.        ]]

# x와 y의 상관계수
corr_xy = corr_matrix[0, 1]
print(corr_xy)  # 0.516...

# 다변량 데이터
data_multi = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
corr_matrix_multi = np.corrcoef(data_multi)
print(corr_matrix_multi)
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]
```

## 히스토그램과 빈도 분석

### 히스토그램

```python
# 데이터 생성
data = np.random.normal(0, 1, 1000)  # 평균 0, 표준편차 1인 정규분포

# 히스토그램 계산
hist, bin_edges = np.histogram(data, bins=10)
print(hist)  # 각 구간의 빈도수
print(bin_edges)  # 구간 경계값

# 균등하지 않은 구간
custom_bins = [-3, -2, -1, 0, 1, 2, 3]
hist_custom, bin_edges_custom = np.histogram(data, bins=custom_bins)
print(hist_custom)
print(bin_edges_custom)

# 밀도 히스토그램 (정규화)
hist_density, _ = np.histogram(data, bins=10, density=True)
print(hist_density)  # 전체 면적이 1이 되도록 정규화된 빈도
```

### 고유값과 빈도

```python
# 데이터 생성
data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])

# 고유값과 빈도
unique_values, counts = np.unique(data, return_counts=True)
print(unique_values)  # [1 2 3 4 5]
print(counts)  # [1 2 3 4 5]

# 가장 자주 나타나는 값
most_frequent_idx = np.argmax(counts)
most_frequent_val = unique_values[most_frequent_idx]
print(most_frequent_val)  # 5

# 다차원 배열의 고유값
data_2d = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
unique_rows, row_counts = np.unique(data_2d, axis=0, return_counts=True)
print(unique_rows)
# [[1 2]
#  [1 3]
#  [2 2]
#  [2 3]]
print(row_counts)  # [1 1 1 1]
```

## 누적 통계

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 누적 합계
cumsum = np.cumsum(data)
print(cumsum)  # [ 1  3  6 10 15 21 28 36 45 55]

# 누적 곱
cumprod = np.cumprod(data)
print(cumprod)  # [1 2 6 24 120 720 5040 40320 362880 3628800]

# 누적 최대값
cummax = np.maximum.accumulate(data)
print(cummax)  # [ 1  2  3  4  5  6  7  8  9 10]

# 누적 최소값
cummin = np.minimum.accumulate(data)
print(cummin)  # [1 1 1 1 1 1 1 1 1 1]
```

## 통계적 검정

### t-검정

```python
# 두 그룹 데이터 생성
group1 = np.random.normal(5, 2, 100)  # 평균 5, 표준편차 2
group2 = np.random.normal(6, 2, 100)  # 평균 6, 표준편차 2

# 독립 표본 t-검정 (수동 계산)
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)

# 풀링된 표준편차
pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))

# t-통계량
t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))
print(t_stat)
```

### 카이제곱 검정

```python
# 관측 빈도와 기대 빈도
observed = np.array([10, 15, 20, 25, 30])
expected = np.array([20, 20, 20, 20, 20])

# 카이제곱 통계량
chi2_stat = np.sum((observed - expected)**2 / expected)
print(chi2_stat)
```

## 실용적인 통계 분석 예제

### 이상치 탐지

```python
# 데이터 생성
np.random.seed(42)
data = np.random.normal(0, 1, 100)
data[10] = 5  # 이상치 추가
data[20] = -4  # 이상치 추가

# Z-score를 이용한 이상치 탐지
z_scores = np.abs((data - np.mean(data)) / np.std(data))
outliers = data[z_scores > 3]  # Z-score가 3을 초과하는 값
print(outliers)  # [5. -4.]

# IQR을 이용한 이상치 탐지
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
print(outliers_iqr)  # [5. -4.]
```

### 데이터 정규화

```python
# 데이터 생성
data = np.random.normal(10, 3, 100)

# Z-score 정규화 (표준화)
normalized_data = (data - np.mean(data)) / np.std(data)
print(np.mean(normalized_data), np.std(normalized_data))  # 약 0, 1

# Min-Max 정규화 (0-1 사이로 변환)
min_val, max_val = np.min(data), np.max(data)
minmax_normalized = (data - min_val) / (max_val - min_val)
print(np.min(minmax_normalized), np.max(minmax_normalized))  # 0.0, 1.0

# 로버스트 스케일링 (중앙값과 IQR 사용)
median_val = np.median(data)
iqr_val = np.percentile(data, 75) - np.percentile(data, 25)
robust_scaled = (data - median_val) / iqr_val
```

### 시계열 데이터의 이동 평균

```python
# 시계열 데이터 생성
time_series = np.random.normal(0, 1, 100)
time_series = np.cumsum(time_series) + 50  # 랜덤 워크

# 이동 평균 계산
window_size = 10
moving_avg = np.convolve(time_series, np.ones(window_size)/window_size, mode='valid')

# 이동 표준편차
moving_std = np.array([np.std(time_series[i:i+window_size]) for i in range(len(time_series)-window_size+1)])

print(time_series.shape, moving_avg.shape)  # (100,) (91,)
```

## 다음 학습 내용

다음으로는 선형대수 연산에 대해 알아보겠습니다. [`linear-algebra.md`](linear-algebra.md)를 참조하세요.