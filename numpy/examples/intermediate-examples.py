"""
NumPy 중급 예제 코드

이 파일은 NumPy의 배열 조작, 수학적 연산, 성능 최적화 등 중급 개념을 다루는 예제 코드를 포함합니다.
강의자료와 함께 학습하면 NumPy의 심화된 기능을 이해하는 데 도움이 됩니다.
"""

import numpy as np
import time

def example_array_manipulation():
    """배열 조작 예제"""
    print("=== 배열 조작 예제 ===")
    
    # 슬라이싱과 뷰
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    slice_arr = arr[2:8]
    print(f"원본 배열: {arr}")
    print(f"슬라이스: {slice_arr}")
    
    # 뷰 수정 시 원본 변경
    slice_arr[0] = 100
    print(f"슬라이스 수정 후 원본: {arr}")
    
    # 팬시 인덱싱
    fancy_arr = arr[[0, 2, 4, 6, 8]]
    print(f"팬시 인덱싱: {fancy_arr}")
    
    # 불리언 인덱싱
    bool_arr = arr[arr > 5]
    print(f"불리언 인덱싱: {bool_arr}")
    
    # 배열 결합
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    # 수직 결합
    vstack = np.vstack((a, b))
    print(f"수직 결합:\n{vstack}")
    
    # 수평 결합
    hstack = np.hstack((a, b))
    print(f"수평 결합:\n{hstack}")
    
    # 배열 분할
    arr = np.arange(16).reshape(4, 4)
    print(f"분할할 배열:\n{arr}")
    
    # 수평 분할
    h_split = np.hsplit(arr, 2)
    print(f"수평 분할:")
    for i, part in enumerate(h_split):
        print(f"  부분 {i}:\n{part}")
    
    # 수직 분할
    v_split = np.vsplit(arr, 2)
    print(f"수직 분할:")
    for i, part in enumerate(v_split):
        print(f"  부분 {i}:\n{part}")
    
    print()

def example_broadcasting():
    """브로드캐스팅 예제"""
    print("=== 브로드캐스팅 예제 ===")
    
    # 스칼라와 배열
    arr = np.array([1, 2, 3, 4, 5])
    scalar = 2
    result = arr + scalar
    print(f"배열 + 스칼라: {result}")
    
    # 1차원과 2차원 배열
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_1d = np.array([10, 20, 30])
    result = arr_2d + arr_1d
    print(f"2차원 + 1차원:\n{result}")
    
    # 크기가 1인 차원
    a = np.array([[1], [2], [3]])  # (3, 1)
    b = np.array([10, 20, 30])     # (3,)
    result = a + b
    print(f"크기 1인 차원 브로드캐스팅:\n{result}")
    
    # 브로드캐스팅 규칙 시각화
    a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20])  # (2,)
    try:
        result = a + b
        print(f"브로드캐스팅 가능:\n{result}")
    except ValueError as e:
        print(f"브로드캐스팅 오류: {e}")
        
        # 해결책
        b_fixed = b[:, np.newaxis]  # (2,) -> (2, 1)
        result = a + b_fixed
        print(f"수정된 브로드캐스팅:\n{result}")
    
    print()

def example_vectorization():
    """벡터화 예제"""
    print("=== 벡터화 예제 ===")
    
    # 큰 데이터 생성
    size = 1000000
    a = np.random.rand(size)
    b = np.random.rand(size)
    
    # Python 반복문 (느림)
    start = time.time()
    result_loop = np.empty(size)
    for i in range(size):
        result_loop[i] = a[i] + b[i]
    loop_time = time.time() - start
    
    # NumPy 벡터화 (빠름)
    start = time.time()
    result_vectorized = a + b
    vectorized_time = time.time() - start
    
    print(f"반복문 시간: {loop_time:.6f}초")
    print(f"벡터화 시간: {vectorized_time:.6f}초")
    print(f"성능 향상: {loop_time/vectorized_time:.1f}배")
    
    # 복잡한 연산 벡터화
    x = np.linspace(0, 2*np.pi, 1000)
    
    # Python 반복문
    start = time.time()
    result_loop = np.empty_like(x)
    for i in range(len(x)):
        result_loop[i] = np.sin(x[i]) * np.cos(x[i])
    loop_time = time.time() - start
    
    # NumPy 벡터화
    start = time.time()
    result_vectorized = np.sin(x) * np.cos(x)
    vectorized_time = time.time() - start
    
    print(f"\n복잡한 연산 반복문 시간: {loop_time:.6f}초")
    print(f"복잡한 연산 벡터화 시간: {vectorized_time:.6f}초")
    print(f"성능 향상: {loop_time/vectorized_time:.1f}배")
    
    print()

def example_memory_management():
    """메모리 관리 예제"""
    print("=== 메모리 관리 예제 ===")
    
    # 뷰와 복사
    arr = np.array([1, 2, 3, 4, 5])
    
    # 뷰 생성
    view = arr[1:4]
    print(f"원본 배열: {arr}")
    print(f"뷰: {view}")
    print(f"뷰와 원본 메모리 공유: {np.shares_memory(view, arr)}")
    
    # 복사 생성
    copy = arr[1:4].copy()
    print(f"복사: {copy}")
    print(f"복사와 원본 메모리 공유: {np.shares_memory(copy, arr)}")
    
    # in-place 연산
    arr = np.random.rand(1000000)
    
    # 일반 연산 (새 배열 생성)
    start = time.time()
    result = arr * 2 + 1
    normal_time = time.time() - start
    
    # in-place 연산 (기존 배열 수정)
    arr_copy = arr.copy()
    start = time.time()
    arr_copy *= 2
    arr_copy += 1
    inplace_time = time.time() - start
    
    print(f"\n일반 연산 시간: {normal_time:.6f}초")
    print(f"in-place 연산 시간: {inplace_time:.6f}초")
    print(f"성능 향상: {normal_time/inplace_time:.2f}배")
    
    # 데이터 타입 최적화
    data = np.random.rand(100000)
    
    # float64
    data_f64 = data.astype(np.float64)
    print(f"\nfloat64 메모리: {data_f64.nbytes} 바이트")
    
    # float32
    data_f32 = data.astype(np.float32)
    print(f"float32 메모리: {data_f32.nbytes} 바이트")
    print(f"메모리 절약: {data_f64.nbytes/data_f32.nbytes:.1f}배")
    
    print()

def example_linear_algebra():
    """선형대수 예제"""
    print("=== 선형대수 예제 ===")
    
    # 행렬 곱셈
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    # 요소별 곱셈
    elementwise = a * b
    print(f"요소별 곱셈:\n{elementwise}")
    
    # 행렬 곱셈
    matrix_product = a @ b
    print(f"행렬 곱셈:\n{matrix_product}")
    
    # 역행렬
    square_matrix = np.array([[1, 2], [3, 4]])
    inv_matrix = np.linalg.inv(square_matrix)
    print(f"원본 행렬:\n{square_matrix}")
    print(f"역행렬:\n{inv_matrix}")
    
    # 확인
    identity = square_matrix @ inv_matrix
    print(f"곱셈 결과 (단위 행렬):\n{np.round(identity)}")
    
    # 고유값과 고유벡터
    eigenvalues, eigenvectors = np.linalg.eig(square_matrix)
    print(f"\n고유값: {eigenvalues}")
    print(f"고유벡터:\n{eigenvectors}")
    
    # 연립 방정식 해결
    # 2x + y = 5
    # x + 3y = 7
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 7])
    
    x = np.linalg.solve(A, b)
    print(f"\n연립 방정식 해: {x}")
    
    # 확인
    print(f"확인: Ax = {A @ x}")
    
    print()

def example_statistics():
    """통계 연산 예제"""
    print("=== 통계 연산 예제 ===")
    
    # 기본 통계
    data = np.random.normal(0, 1, 1000)
    
    print(f"평균: {np.mean(data):.6f}")
    print(f"중앙값: {np.median(data):.6f}")
    print(f"표준편차: {np.std(data):.6f}")
    print(f"분산: {np.var(data):.6f}")
    print(f"최소값: {np.min(data):.6f}")
    print(f"최대값: {np.max(data):.6f}")
    
    # 분위수
    percentiles = [25, 50, 75]
    values = np.percentile(data, percentiles)
    for p, v in zip(percentiles, values):
        print(f"{p}% 분위수: {v:.6f}")
    
    # 상관계수
    x = np.random.rand(100)
    y = 0.5 * x + 0.5 * np.random.rand(100)
    
    correlation = np.corrcoef(x, y)[0, 1]
    print(f"\n상관계수: {correlation:.6f}")
    
    # 히스토그램
    hist, bin_edges = np.histogram(data, bins=10)
    print(f"\n히스토그램 (구간 수: 10)")
    for i in range(len(hist)):
        print(f"  구간 {i+1}: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}) - 빈도: {hist[i]}")
    
    print()

def example_advanced_indexing():
    """고급 인덱싱 예제"""
    print("=== 고급 인덱싱 예제 ===")
    
    # 다차원 배열 인덱싱
    arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"3차원 배열:\n{arr_3d}")
    print(f"arr_3d[0, 1, 0]: {arr_3d[0, 1, 0]}")
    print(f"arr_3d[1, :, 1]: {arr_3d[1, :, 1]}")
    
    # 정수 배열 인덱싱
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rows = np.array([0, 2])
    cols = np.array([0, 2])
    result = arr[rows, cols]
    print(f"\n정수 배열 인덱싱: {result}")
    
    # 불리언 배열 인덱싱
    mask = arr > 5
    print(f"\n불리언 마스크:\n{mask}")
    print(f"마스크 적용 결과: {arr[mask]}")
    
    # np.ix_를 이용한 다차원 인덱싱
    rows = np.array([0, 2])
    cols = np.array([0, 1])
    result = arr[np.ix_(rows, cols)]
    print(f"\nnp.ix_ 인덱싱:\n{result}")
    
    # where 함수
    arr = np.array([1, 2, 3, 4, 5])
    result = np.where(arr > 3, arr, 0)
    print(f"\nwhere 함수: {result}")
    
    # take 함수
    indices = [0, 2, 4]
    result = np.take(arr, indices)
    print(f"take 함수: {result}")
    
    print()

def main():
    """모든 예제 실행"""
    print("NumPy 중급 예제 실행")
    print("=" * 50)
    
    example_array_manipulation()
    example_broadcasting()
    example_vectorization()
    example_memory_management()
    example_linear_algebra()
    example_statistics()
    example_advanced_indexing()
    
    print("모든 예제 실행 완료!")

if __name__ == "__main__":
    main()