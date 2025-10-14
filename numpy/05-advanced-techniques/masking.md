# NumPy 마스킹(Masking)

## 마스킹이란?

마스킹은 배열의 특정 요소를 선택하거나 제외하는 기법입니다. 불리언 마스크를 사용하여 조건에 맞는 요소만 선택하거나, 결측값이나 이상치를 처리할 때 유용합니다.

## 불리언 마스킹

### 기본 불리언 마스킹

```python
import numpy as np

# 기본 배열 생성
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 불리언 마스크 생성
mask = arr > 5
print(f"원본 배열: {arr}")
print(f"마스크: {mask}")

# 마스크를 사용한 요소 선택
filtered = arr[mask]
print(f"필터링된 배열: {filtered}")

# 다양한 조건의 마스크
mask_even = arr % 2 == 0
print(f"짝수 마스크: {mask_even}")
print(f"짝수만: {arr[mask_even]}")

mask_range = (arr >= 3) & (arr <= 7)
print(f"범위 마스크: {mask_range}")
print(f"3-7 사이 값: {arr[mask_range]}")
```

### 다차원 배열 마스킹

```python
# 2차원 배열 마스킹
arr_2d = np.array([[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])

# 조건에 맞는 요소 선택
mask_2d = arr_2d > 6
print(f"2차원 배열:\n{arr_2d}")
print(f"2차원 마스크:\n{mask_2d}")
print(f"마스킹된 결과: {arr_2d[mask_2d]}")

# 행별 마스킹
row_mask = np.array([True, False, True])
print(f"행 마스크: {row_mask}")
print(f"선택된 행:\n{arr_2d[row_mask]}")

# 열별 마스킹
col_mask = np.array([True, False, True, False])
print(f"열 마스크: {col_mask}")
print(f"선택된 열:\n{arr_2d[:, col_mask]}")
```

## 마스크를 이용한 값 수정

### 조건부 값 수정

```python
# 원본 배열
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 조건에 맞는 값 수정
arr[arr > 5] = 0
print(f"5보다 큰 값을 0으로 변경: {arr}")

# 복잡한 조건의 값 수정
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr[(arr >= 3) & (arr <= 7)] = -1
print(f"3-7 사이 값을 -1로 변경: {arr}")

# 다차원 배열의 값 수정
arr_2d = np.array([[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])
arr_2d[arr_2d % 2 == 0] = 100  # 짝수를 100으로 변경
print(f"짝수를 100으로 변경:\n{arr_2d}")
```

### where 함수를 이용한 조건부 수정

```python
# np.where를 이용한 조건부 수정
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 조건이 참이면 100, 거짓이면 원래 값
result = np.where(arr > 5, 100, arr)
print(f"5보다 크면 100, 아니면 원래 값: {result}")

# 조건이 참이면 100, 거짓이면 0
result = np.where(arr > 5, 100, 0)
print(f"5보다 크면 100, 아니면 0: {result}")

# 다차원 배열에서의 where
arr_2d = np.array([[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])
result_2d = np.where(arr_2d > 6, arr_2d * 2, arr_2d)
print(f"6보다 크면 2배, 아니면 원래 값:\n{result_2d}")
```

## 마스크된 배열(MaskedArray)

### MaskedArray 기본

```python
import numpy.ma as ma

# 마스크된 배열 생성
arr = np.array([1, 2, 3, 4, 5])
mask = np.array([False, True, False, True, False])

masked_arr = ma.masked_array(arr, mask=mask)
print(f"원본 배열: {arr}")
print(f"마스크: {mask}")
print(f"마스크된 배열: {masked_arr}")

# 마스크된 값은 계산에서 제외
print(f"평균 (마스크 적용): {masked_arr.mean()}")
print(f"평균 (마스크 무시): {np.mean(arr)}")

# 마스크된 값은 표시되지 않음
print(f"마스크된 배열 출력: {masked_arr}")
```

### 결측값 처리

```python
# 결측값(NaN) 처리
data = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])

# NaN을 마스크로 처리
masked_data = ma.masked_invalid(data)  # NaN을 마스크
print(f"원본 데이터: {data}")
print(f"마스크된 데이터: {masked_data}")
print(f"마스크된 데이터 평균: {masked_data.mean()}")

# 특정 값 마스킹
data = np.array([1, 2, 999, 4, 5, 999, 7, 8, 9, 10])
masked_data = ma.masked_equal(data, 999)  # 999를 마스크
print(f"999를 마스크: {masked_data}")
print(f"마스크된 데이터 평균: {masked_data.mean()}")
```

### 마스크 동적 관리

```python
# 마스크 동적 관리
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
masked_data = ma.masked_array(data)

# 마스크 추가
masked_data.mask[2:5] = True  # 인덱스 2-4를 마스크
print(f"마스크 추가: {masked_data}")

# 마스크 제거
masked_data.mask[2:5] = False  # 인덱스 2-4 마스크 제거
print(f"마스크 제거: {masked_data}")

# 조건부 마스킹
masked_data = ma.masked_where(data > 7, data)  # 7보다 큰 값 마스크
print(f"조건부 마스킹: {masked_data}")
```

## 고급 마스킹 기법

### 복잡한 조건의 마스킹

```python
# 복잡한 조건의 마스킹
data = np.random.randn(100)  # 정규분포 난수

# 표준편차를 벗어나는 값 마스킹 (이상치 제거)
mean_val = np.mean(data)
std_val = np.std(data)
outlier_mask = np.abs(data - mean_val) > 2 * std_val  # 2표준편차 벗어나는 값

cleaned_data = data[~outlier_mask]  # 이상치 제거
print(f"원본 데이터 크기: {len(data)}")
print(f"이상치 개수: {np.sum(outlier_mask)}")
print(f"정제된 데이터 크기: {len(cleaned_data)}")

# 다중 조건 마스킹
data = np.random.randint(0, 100, 100)
mask = (data < 20) | (data > 80)  # 20 미만 또는 80 초과
filtered_data = data[mask]
print(f"20 미만 또는 80 초과인 값: {filtered_data[:10]}...")  # 처음 10개만 표시
```

### 구조화된 배열 마스킹

```python
# 구조화된 배열 마스킹
dtype = [('name', 'U10'), ('age', 'i4'), ('score', 'f4')]
people = np.array([('Alice', 25, 85.5), 
                  ('Bob', 30, 72.0), 
                  ('Charlie', 35, 91.5), 
                  ('Diana', 28, 68.0)], dtype=dtype)

# 조건부 마스킹
age_mask = people['age'] >= 30
score_mask = people['score'] >= 80

# 나이가 30 이상이거나 점수가 80 이상인 사람
combined_mask = age_mask | score_mask
filtered_people = people[combined_mask]

print(f"나이 30+ 또는 점수 80+:")
for person in filtered_people:
    print(f"  {person['name']}: 나이 {person['age']}, 점수 {person['score']}")
```

### 이미지 마스킹

```python
# 이미지 마스킹 예제
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# 특정 색상 영역 마스킹
red_mask = image[:, :, 0] > 200  # 빨간색 채널이 200 초과
green_mask = image[:, :, 1] < 50  # 초록색 채널이 50 미만
blue_mask = image[:, :, 2] < 50   # 파란색 채널이 50 미만

# 붉은색 영역 마스크
red_region_mask = red_mask & green_mask & blue_mask

# 마스크된 영역 검은색으로 변경
masked_image = image.copy()
masked_image[red_region_mask] = [0, 0, 0]

print(f"원본 이미지 형태: {image.shape}")
print(f"마스크된 영역 픽셀 수: {np.sum(red_region_mask)}")
```

## 마스킹 성능 최적화

### 대용량 데이터 마스킹

```python
import time

# 대용량 데이터 마스킹 성능 비교
size = 10000000
data = np.random.randn(size)

# 일반적인 마스킹
start = time.time()
mask = data > 2.0
filtered = data[mask]
normal_time = time.time() - start

# np.where를 이용한 마스킹
start = time.time()
filtered_where = np.where(data > 2.0, data, np.nan)
where_time = time.time() - start

# 마스크된 배열
start = time.time()
masked_data = ma.masked_where(data <= 2.0, data)
masked_time = time.time() - start

print(f"일반 마스킹 시간: {normal_time:.6f}초")
print(f"where 마스킹 시간: {where_time:.6f}초")
print(f"마스크된 배열 시간: {masked_time:.6f}초")
```

### 메모리 효율적인 마스킹

```python
# 메모리 효율적인 마스킹
# 큰 배열의 일부만 필요한 경우
large_array = np.random.randn(10000, 10000)

# 전체 마스크 생성 (메모리 많이 사용)
full_mask = large_array > 2.0
print(f"전체 마스크 크기: {full_mask.nbytes} 바이트")

# 필요한 부분만 마스킹 (메모리 효율적)
row_mask = np.any(large_array > 2.0, axis=1)  # 행별로 조건 확인
filtered_rows = large_array[row_mask]  # 조건에 맞는 행만 선택
print(f"필터링된 배열 형태: {filtered_rows.shape}")
print(f"메모리 절약: {full_mask.nbytes / (row_mask.nbytes + filtered_rows.nbytes):.2f}배")
```

## 실용적인 마스킹 예제

### 시계열 데이터 정제

```python
# 시계열 데이터 정제
# 시계열 데이터 생성 (결측값 포함)
np.random.seed(42)
dates = np.arange('2023-01-01', '2023-01-31', dtype='datetime64[D]')
values = np.random.randn(30) * 10 + 50

# 결측값 추가
values[5] = np.nan
values[12] = np.nan
values[18] = np.nan

# 이상치 추가
values[8] = 150  # 비정상적으로 높은 값
values[22] = -20  # 비정상적으로 낮은 값

# 마스크를 이용한 결측값 처리
cleaned_values = ma.masked_invalid(values)  # NaN 마스크

# 이상치 마스킹 (3표준편차 벗어나는 값)
mean_val = ma.mean(cleaned_values)
std_val = ma.std(cleaned_values)
outlier_mask = np.abs(values - mean_val) > 3 * std_val
cleaned_values.mask[outlier_mask] = True  # 이상치 마스크

print(f"원본 데이터 (처음 15개): {values[:15]}")
print(f"정제된 데이터 (처음 15개): {cleaned_values[:15]}")
print(f"결측값 개수: {np.sum(cleaned_values.mask)}")
print(f"정제된 평균: {ma.mean(cleaned_values):.2f}")

# 마스크된 값 보간
from scipy import interpolate

try:
    # 유효한 인덱스와 값으로 보간
    valid_indices = np.where(~cleaned_values.mask)[0]
    valid_values = cleaned_values.compressed()
    
    # 선형 보간
    f = interpolate.interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')
    interpolated_values = f(np.arange(len(values)))
    
    print(f"보간된 데이터 (처음 15개): {interpolated_values[:15]}")
    
except ImportError:
    print("Scipy가 설치되지 않았습니다. 'pip install scipy'로 설치하세요.")
```

### 과학 데이터 분석

```python
# 과학 데이터 분석 (실험 데이터)
# 실험 데이터 생성
np.random.seed(42)
experiment_data = np.random.randn(1000) * 5 + 20

# 실험 오류 추가 (측정 오류)
error_indices = np.random.choice(1000, 50, replace=False)
experiment_data[error_indices] += np.random.randn(50) * 20  # 큰 오류 추가

# 실험 조건 (온도)
temperature = np.linspace(20, 80, 1000)

# 마스킹을 이용한 데이터 분석
# 유효한 데이터 범위 (0-50)
valid_range_mask = (experiment_data >= 0) & (experiment_data <= 50)
valid_data = experiment_data[valid_range_mask]
valid_temp = temperature[valid_range_mask]

# 온도 구간별 분석
temp_ranges = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]

for i, (temp_min, temp_max) in enumerate(temp_ranges):
    temp_mask = (valid_temp >= temp_min) & (valid_temp < temp_max)
    range_data = valid_data[temp_mask]
    
    if len(range_data) > 0:
        print(f"온도 {temp_min}-{temp_max}°C: 평균 {np.mean(range_data):.2f}, 표준편차 {np.std(range_data):.2f}, 샘플 수 {len(range_data)}")
    else:
        print(f"온도 {temp_min}-{temp_max}°C: 유효한 데이터 없음")
```

## 마스킹 모범 사례

1. **불리언 연산 최적화**: `&`, `|` 대신 `np.logical_and`, `np.logical_or` 사용
2. **마스크 재사용**: 반복 사용되는 마스크는 변수로 저장
3. **메모리 관리**: 대용량 배열은 필요한 부분만 마스킹
4. **결측값 처리**: `np.ma` 모듈 활용하여 결측값 효율적 처리
5. **성능 고려**: 대용량 데이터는 `np.where`가 불리언 인덱싱보다 효율적일 수 있음

## 다음 학습 내용

다음으로는 커스텀 함수에 대해 알아보겠습니다. [`custom-functions.md`](custom-functions.md)를 참조하세요.