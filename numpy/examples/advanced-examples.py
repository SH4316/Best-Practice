"""
NumPy 고급 예제 코드

이 파일은 NumPy의 고급 기능인 구조화된 배열, 마스킹, 커스텀 함수 등을 다루는 예제 코드를 포함합니다.
강의자료와 함께 학습하면 NumPy의 전문적인 기능을 마스터하는 데 도움이 됩니다.
"""

import numpy as np
import time

def example_structured_arrays():
    """구조화된 배열 예제"""
    print("=== 구조화된 배열 예제 ===")
    
    # 기본 구조화된 배열 생성
    dtype = [('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
    people = np.array([('Alice', 25, 55.0), 
                      ('Bob', 30, 70.5), 
                      ('Charlie', 35, 65.2)], dtype=dtype)
    
    print("구조화된 배열:")
    print(people)
    print(f"데이터 타입: {people.dtype}")
    
    # 필드 접근
    print(f"\n이름: {people['name']}")
    print(f"나이: {people['age']}")
    print(f"몸무게: {people['weight']}")
    
    # 필드 수정
    people['age'] = [26, 31, 36]
    print(f"\n수정된 나이: {people['age']}")
    
    # 복잡한 구조화된 배열
    complex_dtype = [
        ('id', 'i4'),
        ('name', 'U20'),
        ('scores', 'f4', (3,)),
        ('active', '?'),
        ('registered', 'datetime64[D]')
    ]
    
    students = np.array([
        (1, 'Alice', [85.5, 90.0, 78.5], True, '2023-09-01'),
        (2, 'Bob', [70.0, 65.5, 80.0], True, '2023-09-01'),
        (3, 'Charlie', [95.0, 92.5, 88.0], False, '2023-09-02')
    ], dtype=complex_dtype)
    
    print("\n복잡한 구조화된 배열:")
    print(students)
    print(f"데이터 타입: {students.dtype}")
    
    # 필드별 정렬
    sorted_by_age = np.sort(people, order='age')
    print(f"\n나이순 정렬: {sorted_by_age}")
    
    print()

def example_masked_arrays():
    """마스크된 배열 예제"""
    print("=== 마스크된 배열 예제 ===")
    
    # 마스크된 배열 생성
    import numpy.ma as ma
    
    # NaN 값이 포함된 데이터
    data = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, np.nan])
    masked_data = ma.masked_invalid(data)
    
    print("원본 데이터:", data)
    print("마스크된 데이터:", masked_data)
    print("마스크:", masked_data.mask)
    
    # 마스크된 값은 계산에서 제외
    print(f"평균 (마스크 적용): {ma.mean(masked_data):.6f}")
    print(f"평균 (마스크 무시): {np.mean(data):.6f}")
    
    # 특정 값 마스킹
    data = np.array([1, 2, 999, 4, 5, 999, 7, 8, 9, 999])
    masked_data = ma.masked_equal(data, 999)
    
    print("\n특정 값 마스킹:")
    print("원본 데이터:", data)
    print("마스크된 데이터:", masked_data)
    print("마스크:", masked_data.mask)
    
    # 마스크 동적 관리
    masked_data = ma.masked_array([1, 2, 3, 4, 5])
    
    # 마스크 추가
    masked_data.mask[2:5] = True
    print("\n마스크 추가:", masked_data)
    
    # 마스크 제거
    masked_data.mask[2:5] = False
    print("마스크 제거:", masked_data)
    
    # 조건부 마스킹
    data = np.random.randn(100)
    masked_data = ma.masked_where(data > 2, data)
    
    print(f"\n조건부 마스킹: {masked_data}")
    print(f"마스크된 요소 수: {ma.count(masked_data)}")
    print(f"유효한 요소 수: {ma.count(masked_data.compressed())}")
    
    print()

def example_custom_functions():
    """커스텀 함수 예제"""
    print("=== 커스텀 함수 예제 ===")
    
    # vectorize 함수
    def my_function(x, y):
        """두 값 중 더 큰 값의 제곱을 반환"""
        if x > y:
            return x ** 2
        else:
            return y ** 2
    
    vectorized_func = np.vectorize(my_function)
    
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])
    
    result = vectorized_func(a, b)
    print("vectorize 결과:", result)
    
    # frompyfunc 함수
    def my_operation(x, y):
        """복잡한 연산"""
        return (x + y) * (x - y) / 2
    
    ufunc_operation = np.frompyfunc(my_operation, 2, 1)
    result = ufunc_operation(a, b)
    print("frompyfunc 결과:", result)
    
    # 성능 비교
    size = 100000
    a = np.random.rand(size)
    b = np.random.rand(size)
    
    # vectorize 성능
    start = time.time()
    result_vec = vectorized_func(a, b)
    vec_time = time.time() - start
    
    # frompyfunc 성능
    start = time.time()
    result_pyfunc = ufunc_operation(a, b)
    pyfunc_time = time.time() - start
    
    # NumPy 벡터화
    start = time.time()
    result_numpy = np.where(a > b, a**2, b**2)
    numpy_time = time.time() - start
    
    print(f"\n성능 비교 (size={size}):")
    print(f"vectorize 시간: {vec_time:.6f}초")
    print(f"frompyfunc 시간: {pyfunc_time:.6f}초")
    print(f"NumPy 벡터화 시간: {numpy_time:.6f}초")
    print(f"결과 동일성: {np.allclose(result_vec.astype(float), result_numpy)}")
    
    # Numba JIT 컴파일 (설치 필요)
    try:
        from numba import jit
        
        @jit(nopython=True)
        def numba_function(x, y):
            """Numba 최적화 함수"""
            result = np.empty_like(x)
            for i in range(len(x)):
                if x[i] > y[i]:
                    result[i] = x[i] ** 2
                else:
                    result[i] = y[i] ** 2
            return result
        
        # 첫 실행 (컴파일 포함)
        start = time.time()
        result_numba = numba_function(a, b)
        first_run_time = time.time() - start
        
        # 두 번째 실행 (컴파일된 코드)
        start = time.time()
        result_numba = numba_function(a, b)
        second_run_time = time.time() - start
        
        print(f"\nNumba 첫 실행 시간: {first_run_time:.6f}초")
        print(f"Numba 두 번째 실행 시간: {second_run_time:.6f}초")
        print(f"결과 동일성: {np.allclose(result_numba, result_numpy)}")
        
    except ImportError:
        print("\nNumba가 설치되지 않았습니다. 'pip install numba'로 설치하세요.")
    
    print()

def example_performance_optimization():
    """성능 최적화 예제"""
    print("=== 성능 최적화 예제 ===")
    
    # 스트라이드 트릭을 이용한 이동 평균
    def moving_average_strides(arr, window_size):
        """스트라이드 트릭을 이용한 효율적인 이동 평균"""
        shape = (arr.shape[0] - window_size + 1, window_size)
        strides = (arr.strides[0], arr.strides[0])
        return np.mean(np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides), axis=1)
    
    def moving_average_loop(arr, window_size):
        """반복문을 이용한 이동 평균"""
        result = np.empty(len(arr) - window_size + 1)
        for i in range(len(result)):
            result[i] = np.mean(arr[i:i+window_size])
        return result
    
    # 성능 비교
    data = np.random.rand(100000)
    window_size = 100
    
    start = time.time()
    result_strides = moving_average_strides(data, window_size)
    strides_time = time.time() - start
    
    start = time.time()
    result_loop = moving_average_loop(data, window_size)
    loop_time = time.time() - start
    
    print(f"스트라이드 트릭 시간: {strides_time:.6f}초")
    print(f"반복문 시간: {loop_time:.6f}초")
    print(f"성능 향상: {loop_time/strides_time:.1f}배")
    print(f"결과 동일성: {np.allclose(result_strides, result_loop)}")
    
    # 메모리 효율적인 연산
    large_array = np.random.rand(10000, 10000)
    
    # 일반 연산 (새 배열 생성)
    start = time.time()
    result = large_array * 2 + 1
    normal_time = time.time() - start
    
    # in-place 연산 (기존 배열 수정)
    large_array_copy = large_array.copy()
    start = time.time()
    large_array_copy *= 2
    large_array_copy += 1
    inplace_time = time.time() - start
    
    print(f"\n일반 연산 시간: {normal_time:.6f}초")
    print(f"in-place 연산 시간: {inplace_time:.6f}초")
    print(f"성능 향상: {normal_time/inplace_time:.2f}배")
    
    # 데이터 타입 최적화
    data = np.random.rand(1000000)
    
    # float64
    data_f64 = data.astype(np.float64)
    start = time.time()
    result_f64 = data_f64 * 2
    f64_time = time.time() - start
    
    # float32
    data_f32 = data.astype(np.float32)
    start = time.time()
    result_f32 = data_f32 * 2
    f32_time = time.time() - start
    
    print(f"\nfloat64 연산 시간: {f64_time:.6f}초")
    print(f"float32 연산 시간: {f32_time:.6f}초")
    print(f"성능 향상: {f64_time/f32_time:.2f}배")
    print(f"메모리 절약: {data_f64.nbytes/data_f32.nbytes:.1f}배")
    
    print()

def example_domain_applications():
    """도메인 응용 예제"""
    print("=== 도메인 응용 예제 ===")
    
    # 데이터 과학: 선형 회귀
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
    
    # 테스트 데이터 생성
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    true_slope = 2.5
    true_intercept = 1.5
    y = true_intercept + true_slope * X[:, 0] + np.random.randn(100) * 2
    
    # 선형 회귀 모델 학습
    coefficients = linear_regression(X, y)
    intercept, slope = coefficients
    
    print("선형 회귀 결과:")
    print(f"실제 절편: {true_intercept:.2f}, 추정 절편: {intercept:.2f}")
    print(f"실제 기울기: {true_slope:.2f}, 추정 기울기: {slope:.2f}")
    
    # 과학 계산: 미분 방정식 (오일러 방법)
    def euler_method(f, t0, y0, t_end, h):
        """오일러 방법을 이용한 미분 방정식 수치 해석"""
        n_steps = int((t_end - t0) / h) + 1
        t = np.linspace(t0, t_end, n_steps)
        y = np.zeros(n_steps)
        y[0] = y0
        
        for i in range(n_steps - 1):
            y[i + 1] = y[i] + h * f(t[i], y[i])
        
        return t, y
    
    # 테스트 미분 방정식: dy/dt = -2t * y², y(0) = 1
    def f(t, y):
        return -2 * t * y ** 2
    
    # 수치 해석
    t0, y0 = 0, 1
    t_end = 2
    h = 0.1
    
    t_num, y_num = euler_method(f, t0, y0, t_end, h)
    
    # 해석해: y(t) = 1 / (1 + t²)
    def analytical_solution(t):
        return 1 / (1 + t ** 2)
    
    y_analytical = analytical_solution(t_num)
    error = np.abs(y_num - y_analytical)
    max_error = np.max(error)
    
    print("\n미분 방정식 해석:")
    print(f"최대 오차: {max_error:.6f}")
    
    # 이미지 처리: 컨볼루션 필터
    def apply_filter(image, kernel):
        """컨볼루션 필터 적용"""
        if len(image.shape) == 3:  # 컬러 이미지
            height, width, channels = image.shape
            filtered = np.zeros_like(image)
            
            for c in range(channels):
                filtered[:, :, c] = apply_filter_grayscale(image[:, :, c], kernel)
            
            return filtered
        else:  # 흑백 이미지
            return apply_filter_grayscale(image, kernel)
    
    def apply_filter_grayscale(image, kernel):
        """흑백 이미지에 컨볼루션 필터 적용"""
        height, width = image.shape
        k_height, k_width = kernel.shape
        pad_h, pad_w = k_height // 2, k_width // 2
        
        # 경계 처리를 위한 패딩
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # 결과 이미지
        filtered = np.zeros_like(image)
        
        # 컨볼루션 연산
        for y in range(height):
            for x in range(width):
                region = padded[y:y+k_height, x:x+k_width]
                filtered[y, x] = np.sum(region * kernel)
        
        # 결과를 유효한 범위로 클리핑
        filtered = np.clip(filtered, 0, 255).astype(image.dtype)
        
        return filtered
    
    # 테스트 이미지 생성
    test_image = np.zeros((50, 50), dtype=np.uint8)
    test_image[10:40, 10:40] = 255  # 중앙에 흰색 사각형
    
    # 가우시안 블러 필터
    gaussian_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16
    
    # 필터 적용
    filtered_image = apply_filter(test_image, gaussian_kernel)
    
    print("\n이미지 필터 적용:")
    print(f"원본 이미지 픽셀 값 범위: {np.min(test_image)} - {np.max(test_image)}")
    print(f"필터링된 이미지 픽셀 값 범위: {np.min(filtered_image)} - {np.max(filtered_image)}")
    
    print()

def main():
    """모든 예제 실행"""
    print("NumPy 고급 예제 실행")
    print("=" * 50)
    
    example_structured_arrays()
    example_masked_arrays()
    example_custom_functions()
    example_performance_optimization()
    example_domain_applications()
    
    print("모든 예제 실행 완료!")

if __name__ == "__main__":
    main()