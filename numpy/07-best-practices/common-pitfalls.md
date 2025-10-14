# NumPy 일반적인 실수와 피해야 할 사항

## 소개

NumPy는 강력한 라이브러리지만, 잘못 사용하면 성능 저하, 메모리 문제, 예기치 않은 결과 등이 발생할 수 있습니다. 이 문서에서는 NumPy 사용 시 흔히 발생하는 실수와 이를 피하는 방법을 설명합니다.

## 데이터 타입 관련 실수

### 암시적 데이터 타입 변환

```python
import numpy as np

# 실수와 정수 연산 시 데이터 타입 변환
arr_int = np.array([1, 2, 3, 4, 5], dtype=np.int32)
arr_float = np.array([1.1, 2.2, 3.3], dtype=np.float32)

# 결과는 더 높은 정밀도의 타입으로 변환됨
result = arr_int[:3] + arr_float
print(f"정수 배열 타입: {arr_int.dtype}")
print(f"실수 배열 타입: {arr_float.dtype}")
print(f"연산 결과 타입: {result.dtype}")

# 의도치 않은 정밀도 손실
large_int = np.array([123456789, 987654321], dtype=np.int64)
small_float = np.array([0.1, 0.2], dtype=np.float32)
result = large_int + small_float
print(f"\n큰 정수 + 작은 실수 결과 타입: {result.dtype}")
print(f"결과: {result}")  # 정밀도 손실 가능성

# 해결책: 명시적 데이터 타입 지정
result_fixed = large_int.astype(np.float64) + small_float.astype(np.float64)
print(f"명시적 타입 변환 결과: {result_fixed}")
```

### 정수 오버플로우

```python
# 정수 오버플로우
arr_uint8 = np.array([250, 251, 252, 253, 254, 255], dtype=np.uint8)
result = arr_uint8 + 10
print(f"uint8 오버플로우: {result}")  # 255를 넘어가면 0으로 돌아감

# 음수 처리
arr_int8 = np.array([-125, -126, -127, -128], dtype=np.int8)
result = arr_int8 - 10
print(f"int8 언더플로우: {result}")  # -128을 넘어가면 127로 돌아감

# 해결책: 더 큰 데이터 타입 사용
result_fixed = arr_uint8.astype(np.uint16) + 10
print(f"uint16으로 변환 후 연산: {result_fixed}")
```

## 브로드캐스팅 관련 실수

### 브로드캐스팅 규칙 오해

```python
# 브로드캐스팅 오류
try:
    a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20])  # (2,)
    result = a + b  # 오류 발생
except ValueError as e:
    print(f"브로드캐스팅 오류: {e}")

# 해결책 1: 차원 명시적 추가
b_fixed = b[:, np.newaxis]  # (2,) -> (2, 1)
result = a + b_fixed  # (2, 3) + (2, 1) -> (2, 3)
print(f"차원 추가 후 브로드캐스팅:\n{result}")

# 해결책 2: reshape 사용
b_fixed2 = b.reshape(2, 1)  # (2,) -> (2, 1)
result2 = a + b_fixed2
print(f"reshape 후 브로드캐스팅:\n{result2}")
```

### 의도치 않은 브로드캐스팅

```python
# 의도치 않은 브로드캐스팅
a = np.array([1, 2, 3, 4, 5])
b = np.array([10])  # 스칼라와 유사하지만 배열

result = a + b  # 브로드캐스팅 발생
print(f"배열 + 원소 배열: {result}")

# 의도와 다른 연산
matrix = np.array([[1, 2], [3, 4]])
vector = np.array([10, 20])

# 행렬의 각 행에 벡터를 더하려고 했지만, 열 방향으로 더해짐
result = matrix + vector  # (2, 2) + (2,) -> (2, 2)
print(f"\n행렬 + 벡터 (의도와 다름):\n{result}")

# 해결책: 명시적 차원 추가
vector_fixed = vector[np.newaxis, :]  # (2,) -> (1, 2)
result_fixed = matrix + vector_fixed  # (2, 2) + (1, 2) -> (2, 2)
print(f"명시적 브로드캐스팅:\n{result_fixed}")

# 또는 다른 방향으로 브로드캐스팅
vector_fixed2 = vector[:, np.newaxis]  # (2,) -> (2, 1)
result_fixed2 = matrix + vector_fixed2  # (2, 2) + (2, 1) -> (2, 2)
print(f"다른 방향 브로드캐스팅:\n{result_fixed2}")
```

## 메모리 관련 실수

### 뷰와 복사 혼동

```python
# 뷰와 복사 혼동
original = np.array([1, 2, 3, 4, 5])

# 슬라이싱은 뷰를 반환
view = original[1:4]
view[0] = 100
print(f"뷰 수정 후 원본: {original}")  # 원본도 변경됨

# 팬시 인덱싱은 복사를 반환
copy = original[[0, 2, 4]]
copy[0] = 200
print(f"복사 수정 후 원본: {original}")  # 원본은 변경되지 않음

# 해결책: 명시적 복사
explicit_copy = original[1:4].copy()
explicit_copy[0] = 300
print(f"명시적 복사 수정 후 원본: {original}")  # 원본은 변경되지 않음
```

### 메모리 누수

```python
# 메모리 누수가 발생할 수 있는 코드
def memory_leak_example():
    """메모리 누수 예제"""
    arrays = []
    for i in range(1000):
        # 매번 새 배열 생성하고 리스트에 추가
        large_array = np.random.rand(1000, 1000)
        arrays.append(large_array)
    return arrays

# 메모리 효율적인 대안
def memory_efficient_example():
    """메모리 효율적인 예제"""
    # 하나의 배열을 재사용
    result = np.empty((1000, 1000, 1000))
    for i in range(1000):
        # 계산 결과만 저장
        result[i] = np.random.rand(1000, 1000)
    return result
```

## 인덱싱 관련 실수

### 경계를 벗어난 인덱싱

```python
# 경계를 벗어난 인덱싱
arr = np.array([1, 2, 3, 4, 5])

# Python 리스트와 달리 오류가 발생하지 않음
try:
    result = arr[10]  # IndexError 발생
except IndexError as e:
    print(f"경계를 벗어난 인덱싱: {e}")

# 슬라이싱에서는 경계를 벗어나도 자동 조정
result = arr[3:10]  # 경계를 벗어나도 자동 조정
print(f"경계를 벗어난 슬라이싱: {result}")

# 다차원 배열에서의 경계 문제
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
try:
    result = arr_2d[2, 0]  # IndexError 발생
except IndexError as e:
    print(f"2차원 배열 경계 오류: {e}")
```

### 불리언 인덱싱 실수

```python
# 불리언 인덱싱 실수
arr = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True])  # 길이가 다름

try:
    result = arr[mask]  # 오류 발생
except IndexError as e:
    print(f"길이가 다른 불리언 마스크: {e}")

# 해결책: 마스크 길이 확인
if len(mask) == len(arr):
    result = arr[mask]
    print(f"올바른 불리언 인덱싱: {result}")

# 다차원 배열에서의 불리언 인덱싱
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask_2d = np.array([[True, False, True], [False, True, False]])  # 형태가 다름

try:
    result = arr_2d[mask_2d]  # 오류 발생
except IndexError as e:
    print(f"형태가 다른 불리언 마스크: {e}")
```

## 성능 관련 실수

### 반복문 오남용

```python
import time

# 반복문 오남용 (성능 저하)
size = 100000
arr1 = np.random.rand(size)
arr2 = np.random.rand(size)

# 나쁜 예제: Python 반복문 사용
start = time.time()
result = np.empty(size)
for i in range(size):
    result[i] = arr1[i] + arr2[i]
loop_time = time.time() - start

# 좋은 예제: NumPy 벡터화 연산
start = time.time()
result_vectorized = arr1 + arr2
vectorized_time = time.time() - start

print(f"반복문 시간: {loop_time:.6f}초")
print(f"벡터화 시간: {vectorized_time:.6f}초")
print(f"성능 향상: {loop_time/vectorized_time:.1f}배")
```

### 불필요한 배열 복사

```python
# 불필요한 배열 복사
large_array = np.random.rand(10000, 10000)

# 나쁜 예제: 불필요한 복사
start = time.time()
copied_array = large_array.copy()  # 불필요한 복사
result = copied_array * 2
copy_time = time.time() - start

# 좋은 예제: in-place 연산
start = time.time()
result_inplace = large_array * 2  # 복사 없이 연산
inplace_time = time.time() - start

print(f"복사 후 연산 시간: {copy_time:.6f}초")
print(f"in-place 연산 시간: {inplace_time:.6f}초")
print(f"성능 향상: {copy_time/inplace_time:.1f}배")
```

## 수치 정확성 관련 실수

### 부동소수점 비교

```python
# 부동소수점 비교 실수
a = np.array([0.1 + 0.2])
b = np.array([0.3])

print(f"0.1 + 0.2 = {a[0]}")
print(f"0.3 = {b[0]}")
print(f"직접 비교: {a == b}")  # False
print(f"차이: {a - b}")  # 작은 차이 존재

# 해결책: np.allclose 사용
print(f"allclose 비교: {np.allclose(a, b)}")

# 허용 오차 지정
print(f"허용 오차 비교: {np.allclose(a, b, rtol=1e-10)}")
```

### 수치 안정성

```python
# 수치 안정성 문제
# 나쁜 예제: 큰 값과 작은 값의 차이
large = 1e16
small = 1.0
result = large + small - large
print(f"수치 안정성 문제: {result}")  # 0이 나와야 하지만 0이 아닐 수 있음

# 해결책: 알고리즘 개선
def stable_sum(values):
    """수치 안정적인 합계 계산"""
    sorted_values = np.sort(values)
    result = 0.0
    for val in sorted_values:
        result += val
    return result

values = np.array([large, small, -large])
stable_result = stable_sum(values)
print(f"안정적인 합계: {stable_result}")
```

## 디버깅 관련 실수

### 디버깅 어려운 코드

```python
# 디버깅 어려운 코드
# 나쁜 예제: 한 줄에 너무 많은 연산
result = np.sqrt(np.sum((np.random.rand(1000) - np.mean(np.random.rand(1000))) ** 2))

# 좋은 예제: 단계별로 나누기
data = np.random.rand(1000)
mean_val = np.mean(data)
deviations = data - mean_val
squared_deviations = deviations ** 2
sum_squared = np.sum(squared_deviations)
std_dev = np.sqrt(sum_squared)
print(f"표준편차: {std_dev}")
```

### 경고 무시

```python
# 경고 무시 (나쁜 습관)
import warnings

# 나쁜 예제: 모든 경고 무시
warnings.filterwarnings('ignore')
result = np.sqrt(np.array([-1, 0, 1]))  # 경고가 발생하지만 무시됨

# 좋은 예제: 특정 경고만 처리
warnings.filterwarnings('default')  # 경고 복원
try:
    result = np.sqrt(np.array([-1, 0, 1]))
except RuntimeWarning as w:
    print(f"경고 처리: {w}")
    # 적절한 처리
    result = np.sqrt(np.abs(np.array([-1, 0, 1])))
    print(f"처리된 결과: {result}")
```

## 일반적인 실수 요약

### 피해야 할 사항

1. **암시적 데이터 타입 변환에 의존**: 명시적으로 데이터 타입 지정
2. **뷰와 복사 혼동**: 어떤 연산이 뷰를 반환하는지 이해
3. **브로드캐스팅 규칙 오해**: 명시적으로 차원 추가
4. **반복문 오남용**: 벡터화 연산 우선
5. **불필요한 배열 복사**: in-place 연산 고려
6. **부동소수점 직접 비교**: np.allclose 사용
7. **경고 무시**: 경고의 원인을 이해하고 처리

### 권장 사항

1. **명시적 데이터 타입 지정**: `dtype` 매개변수 활용
2. **메모리 효율성 고려**: 대용량 데이터는 청크 단위 처리
3. **벡터화 연산**: 반복문 대신 NumPy 내장 함수 사용
4. **코드 가독성**: 복잡한 연산은 단계별로 나누기
5. **수치 안정성**: 알고리즘의 수치적 안정성 고려
6. **경고 처리**: 경고의 원인을 이해하고 적절히 처리

## 다음 학습 내용

다음으로는 최적화 팁에 대해 알아보겠습니다. [`optimization-tips.md`](optimization-tips.md)를 참조하세요.