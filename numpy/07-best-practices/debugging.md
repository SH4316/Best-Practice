# NumPy 디버깅 기법

## 소개

NumPy 코드를 디버깅하는 것은 때로는 까다로울 수 있습니다. 특히 대용량 배열이나 복잡한 수학적 연산을 다룰 때는 더욱 그렇습니다. 이 문서에서는 NumPy 코드를 효과적으로 디버깅하는 다양한 기법과 도구를 소개합니다.

## 기본 디버깅 기법

### 배열 상태 확인

```python
import numpy as np

# 배열 상태 확인을 위한 함수
def print_array_info(arr, name="Array"):
    """배열의 기본 정보 출력"""
    print(f"{name} 정보:")
    print(f"  형태: {arr.shape}")
    print(f"  차원: {arr.ndim}")
    print(f"  데이터 타입: {arr.dtype}")
    print(f"  크기: {arr.size}")
    print(f"  메모리 사용량: {arr.nbytes} 바이트")
    print(f"  최소값: {np.min(arr)}")
    print(f"  최대값: {np.max(arr)}")
    print(f"  평균값: {np.mean(arr):.6f}")
    print(f"  표준편차: {np.std(arr):.6f}")
    print(f"  NaN 개수: {np.sum(np.isnan(arr))}")
    print(f"  무한대 개수: {np.sum(np.isinf(arr))}")
    
    # 일부 값 출력
    if arr.size > 0:
        if arr.ndim == 1:
            print(f"  처음 5개 값: {arr[:5]}")
            if arr.size > 5:
                print(f"  마지막 5개 값: {arr[-5:]}")
        else:
            print(f"  처음 2x2 블록:\n{arr[:2, :2]}")

# 테스트 배열
arr = np.array([1, 2, np.nan, 4, 5, np.inf])
print_array_info(arr, "테스트 배열")

# 다차원 배열
arr_2d = np.random.rand(5, 5)
print_array_info(arr_2d, "2차원 배열")
```

### 배열 내용 시각화

```python
# 배열 내용을 시각적으로 확인하는 함수
def visualize_array(arr):
    """배열 내용을 시각적으로 출력"""
    if arr.ndim == 1:
        print("1차원 배열:")
        for i, val in enumerate(arr):
            print(f"  [{i}] {val}")
    elif arr.ndim == 2:
        print("2차원 배열:")
        for i, row in enumerate(arr):
            print(f"  [{i}] {row}")
    elif arr.ndim == 3:
        print("3차원 배열:")
        for i, matrix in enumerate(arr):
            print(f"  [{i}]")
            for j, row in enumerate(matrix):
                print(f"    [{j}] {row}")
    else:
        print(f"{arr.ndim}차원 배열 (너무 복잡하여 표시 불가)")

# 테스트
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

visualize_array(arr_1d)
visualize_array(arr_2d)
visualize_array(arr_3d)
```

## 데이터 정확성 검증

### 수치적 안정성 확인

```python
# 수치적 안정성 검증 함수
def check_numerical_stability(result, expected=None, tolerance=1e-10):
    """수치적 안정성 검증"""
    if expected is not None:
        diff = np.abs(result - expected)
        max_diff = np.max(diff)
        
        print(f"수치적 안정성 검증:")
        print(f"  최대 차이: {max_diff:.2e}")
        print(f"  허용 오차: {tolerance:.2e}")
        
        if max_diff < tolerance:
            print("  결과: 안정적 ✓")
        else:
            print("  결과: 불안정적 ✗")
            print(f"  불안정한 요소 수: {np.sum(diff > tolerance)}")
    
    # NaN과 무한대 확인
    nan_count = np.sum(np.isnan(result))
    inf_count = np.sum(np.isinf(result))
    
    if nan_count > 0:
        print(f"  NaN 개수: {nan_count}")
    if inf_count > 0:
        print(f"  무한대 개수: {inf_count}")

# 테스트
# 수치적으로 불안정한 계산
x = np.array([1e10, 1, -1e10])
result1 = x + 1e-10  # 큰 값과 작은 값의 덧셈
expected1 = np.array([1e10 + 1e-10, 1 + 1e-10, -1e10 + 1e-10])

check_numerical_stability(result1, expected1)

# 수치적으로 안정적인 계산
x = np.array([1e10, 1, -1e10])
result2 = np.sort(x)  # 정렬 후 연산
result2 = result2 + 1e-10
expected2 = np.array([-1e10 + 1e-10, 1 + 1e-10, 1e10 + 1e-10])

check_numerical_stability(result2, expected2)
```

### 데이터 타입 일관성 검증

```python
# 데이터 타입 일관성 검증
def check_data_type_consistency(arrays, expected_dtype=None):
    """여러 배열의 데이터 타입 일관성 검증"""
    print("데이터 타입 일관성 검증:")
    
    if expected_dtype is not None:
        all_consistent = True
        for i, arr in enumerate(arrays):
            is_consistent = arr.dtype == expected_dtype
            status = "일치 ✓" if is_consistent else "불일치 ✗"
            print(f"  배열 {i}: {arr.dtype} (기대: {expected_dtype}) {status}")
            if not is_consistent:
                all_consistent = False
        
        if all_consistent:
            print("  결과: 모든 배열이 기대 타입과 일치 ✓")
        else:
            print("  결과: 일부 배열이 기대 타입과 불일치 ✗")
    else:
        # 배열 간 타입 비교
        first_dtype = arrays[0].dtype
        all_consistent = all(arr.dtype == first_dtype for arr in arrays)
        
        for i, arr in enumerate(arrays):
            is_consistent = arr.dtype == first_dtype
            status = "일치 ✓" if is_consistent else "불일치 ✗"
            print(f"  배열 {i}: {arr.dtype} (기준: {first_dtype}) {status}")
        
        if all_consistent:
            print("  결과: 모든 배열의 타입이 일치 ✓")
        else:
            print("  결과: 일부 배열의 타입이 불일치 ✗")

# 테스트
arr1 = np.array([1, 2, 3], dtype=np.int32)
arr2 = np.array([4, 5, 6], dtype=np.int32)
arr3 = np.array([7, 8, 9], dtype=np.float32)

check_data_type_consistency([arr1, arr2], np.int32)
check_data_type_consistency([arr1, arr2, arr3])
```

## 성능 디버깅

### 실행 시간 측정

```python
import time

# 함수 실행 시간 측정 데코레이터
def timing_decorator(func):
    """함수 실행 시간 측정 데코레이터"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 실행 시간: {elapsed:.6f}초")
        return result
    return wrapper

# 테스트 함수
@timing_decorator
def slow_function(n):
    """의도적으로 느린 함수"""
    result = 0
    for i in range(n):
        result += i
    return result

@timing_decorator
def fast_function(n):
    """빠른 NumPy 함수"""
    return np.sum(np.arange(n))

# 성능 비교
n = 1000000
slow_result = slow_function(n)
fast_result = fast_function(n)

print(f"결과 일치: {slow_result == fast_result}")
```

### 메모리 사용량 모니터링

```python
# 메모리 사용량 모니터링
def memory_usage_monitor(func, *args, **kwargs):
    """함수의 메모리 사용량 모니터링"""
    import sys
    import gc
    
    # 실행 전 메모리 상태
    gc.collect()  # 가비지 컬렉션
    mem_before = sys.getsizeof(args[0]) if args else 0
    
    # 함수 실행
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    
    # 실행 후 메모리 상태
    mem_after = sys.getsizeof(result) if result is not None else 0
    
    print(f"{func.__name__} 메모리 사용량:")
    print(f"  실행 전: {mem_before} 바이트")
    print(f"  실행 후: {mem_after} 바이트")
    print(f"  차이: {mem_after - mem_before} 바이트")
    print(f"  실행 시간: {elapsed:.6f}초")
    
    return result

# 테스트 함수
def memory_intensive_function(size):
    """메모리 집약적인 함수"""
    return np.random.rand(size, size)

# 메모리 사용량 테스트
result = memory_usage_monitor(memory_intensive_function, 1000)
```

## 로깅과 추적

### 간단한 로깅 시스템

```python
# 간단한 로깅 시스템
class NumpyLogger:
    """NumPy 배열 연산을 위한 간단한 로깅 시스템"""
    def __init__(self, log_file="numpy_debug.log"):
        self.log_file = log_file
        self.logs = []
    
    def log(self, message, level="INFO"):
        """로그 메시지 기록"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        self.logs.append(log_entry)
        
        # 파일에 기록
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
        
        # 콘솔에 출력
        print(log_entry)
    
    def log_array_operation(self, operation, input_arrays, output_array=None):
        """배열 연산 로그 기록"""
        self.log(f"연산: {operation}")
        
        for i, arr in enumerate(input_arrays):
            self.log(f"  입력 {i}: 형태={arr.shape}, 타입={arr.dtype}")
        
        if output_array is not None:
            self.log(f"  출력: 형태={output_array.shape}, 타입={output_array.dtype}")

# 로거 사용 예제
logger = NumpyLogger()

# 배열 연산 로깅
arr1 = np.random.rand(100, 100)
arr2 = np.random.rand(100, 100)

logger.log_array_operation("행렬 곱셈", [arr1, arr2])
result = np.dot(arr1, arr2)
logger.log_array_operation("행렬 곱셈", [arr1, arr2], result)
```

### 배열 변화 추적

```python
# 배열 변화 추적 클래스
class ArrayTracker:
    """배열의 변화를 추적하는 클래스"""
    def __init__(self, name, array):
        self.name = name
        self.history = []
        self.track("초기화", array)
    
    def track(self, operation, array):
        """배열 상태 변경 추적"""
        import copy
        state = {
            'operation': operation,
            'shape': array.shape,
            'dtype': array.dtype,
            'min': float(np.min(array)) if array.size > 0 else None,
            'max': float(np.max(array)) if array.size > 0 else None,
            'mean': float(np.mean(array)) if array.size > 0 else None,
            'has_nan': bool(np.any(np.isnan(array))) if array.size > 0 else False,
            'has_inf': bool(np.any(np.isinf(array))) if array.size > 0 else False,
        }
        
        self.history.append(state)
    
    def print_history(self):
        """변화 히스토리 출력"""
        print(f"'{self.name}' 배열 변화 히스토리:")
        for i, state in enumerate(self.history):
            print(f"  {i+1}. {state['operation']}")
            print(f"     형태: {state['shape']}, 타입: {state['dtype']}")
            print(f"     범위: [{state['min']}, {state['max']}], 평균: {state['mean']:.6f}")
            print(f"     NaN: {state['has_nan']}, 무한대: {state['has_inf']}")

# 테스트
tracker = ArrayTracker("테스트 배열", np.array([1, 2, 3]))

tracker.track("제곱", tracker.history[-1]['shape'])  # 예시
```

## 고급 디버깅 기법

### 배열 비교 디버깅

```python
# 배열 비교 디버깅 함수
def debug_array_comparison(arr1, arr2, name1="Array1", name2="Array2"):
    """두 배열의 상세 비교"""
    print(f"{name1} vs {name2} 비교:")
    
    # 기본 속성 비교
    print(f"  형태: {name1}={arr1.shape}, {name2}={arr2.shape}")
    print(f"  데이터 타입: {name1}={arr1.dtype}, {name2}={arr2.dtype}")
    
    # 형태 비교
    if arr1.shape != arr2.shape:
        print("  결과: 형태가 다름 ✗")
        return
    
    # 데이터 타입 비교
    if arr1.dtype != arr2.dtype:
        print("  결과: 데이터 타입이 다름 ✗")
        return
    
    # 값 비교
    if np.array_equal(arr1, arr2):
        print("  결과: 모든 요소가 일치 ✓")
        return
    
    # 차이 분석
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    
    print(f"  차이 정보:")
    print(f"    최대 차이: {np.max(abs_diff):.6f}")
    print(f"    평균 차이: {np.mean(abs_diff):.6f}")
    print(f"    차이가 0인 요소 수: {np.sum(abs_diff == 0)}/{arr1.size}")
    print(f"    차이가 허용 오차(1e-10) 이내인 요소 수: {np.sum(abs_diff < 1e-10)}/{arr1.size}")
    
    # 다른 요소 위치 찾기
    diff_indices = np.where(abs_diff >= 1e-10)
    if len(diff_indices[0]) > 0:
        print(f"    주요 차이 위치 (처음 10개):")
        for i in range(min(10, len(diff_indices[0]))):
            idx = tuple(dim[i] for dim in diff_indices)
            print(f"      위치 {idx}: {name1}={arr1[idx]}, {name2}={arr2[idx]}, 차이={diff[idx]:.6f}")

# 테스트
arr1 = np.array([1.0, 2.0, 3.0, 4.0])
arr2 = np.array([1.0, 2.0000001, 2.9999999, 4.0])

debug_array_comparison(arr1, arr2, "원본", "수정")
```

### 스택 트레이스와 배열 상태

```python
# 스택 트레이스와 배열 상태 출력
def debug_with_trace(array, operation=None):
    """배열 상태와 스택 트레이스 출력"""
    import traceback
    
    print(f"디버그 정보:")
    if operation:
        print(f"  연산: {operation}")
    
    print(f"  배열 상태:")
    print(f"    형태: {array.shape}")
    print(f"    데이터 타입: {array.dtype}")
    print(f"    범위: [{np.min(array)}, {np.max(array)}]")
    print(f"    평균: {np.mean(array):.6f}")
    
    print(f"  스택 트레이스:")
    stack = traceback.extract_stack()[-3:-1]  # 마지막 2개 프레임
    for frame in stack:
        print(f"    {frame.filename}:{frame.lineno} in {frame.name}")
        print(f"      {frame.line}")

# 테스트 함수
def test_function(arr):
    debug_with_trace(arr, "테스트 연산")
    return arr * 2

# 테스트
test_array = np.array([1, 2, 3, 4, 5])
result = test_function(test_array)
```

## 외부 도구 활용

### IPython 디버거 활용

```python
# IPython 디버거 사용 예제
def problematic_function(arr):
    """문제가 발생할 수 있는 함수"""
    # 디버거 설정
    from IPython import get_ipython
    ip = get_ipython()
    
    if ip:
        # IPython 환경인 경우
        ip.run_line_magic('debug', '')
    
    # 문제 발생 코드
    result = arr / np.mean(arr)  # 0으로 나눌 수 있음
    return result

# 테스트
try:
    test_arr = np.array([1, 2, 3, 4, 5])
    result = problematic_function(test_arr)
except Exception as e:
    print(f"오류 발생: {e}")
```

### pdb 디버거 활용

```python
# pdb 디버거 사용 예제
def debug_with_pdb(arr):
    """pdb를 이용한 디버깅"""
    import pdb
    
    # 브레이크포인트 설정
    pdb.set_trace()
    
    # 디버깅할 코드
    result = np.sum(arr)
    return result

# 테스트 (실제로는 주석 해제하고 실행)
# test_arr = np.array([1, 2, 3, 4, 5])
# result = debug_with_pdb(test_arr)
```

## 디버깅 모범 사례

### 디버깅 체크리스트

1. **배열 상태 확인**: 형태, 데이터 타입, 값 범위 확인
2. **데이터 정확성 검증**: NaN, 무한대, 수치적 안정성 확인
3. **단계별 실행**: 복잡한 연산은 단계별로 나누어 확인
4. **로깅**: 중요 연산은 로그로 기록
5. **비교**: 예상 결과와 실제 결과를 상세히 비교
6. **시각화**: 가능한 경우 데이터를 시각화하여 확인
7. **프로파일링**: 성능 문제는 프로파일러로 병목 확인

### 디버깅 팁

1. **작은 데이터로 테스트**: 문제를 재현하기 위해 작은 데이터로 시작
2. **단순한 예제**: 복잡한 문제는 단순한 예제로 단순화
3. **중간 결과 확인**: 연산의 중간 결과를 출력하여 확인
4. **경계 조건 테스트**: 최소/최대 값, 빈 배열 등 경계 조건 테스트
5. **버전 확인**: NumPy와 관련 라이브러리 버전 확인

## 다음 학습 내용

이제 NumPy 베스트 프랙티스 강의자료가 완성되었습니다. 다음으로는 코드 예제와 연습 문제를 만들어보겠습니다.