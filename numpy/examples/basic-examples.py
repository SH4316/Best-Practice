"""
NumPy 기초 예제 코드

이 파일은 NumPy의 기본 개념과 연산을 다루는 예제 코드를 포함합니다.
강의자료와 함께 학습하면 NumPy의 기초를 이해하는 데 도움이 됩니다.
"""

import numpy as np

def example_array_creation():
    """배열 생성 예제"""
    print("=== 배열 생성 예제 ===")
    
    # 리스트로부터 배열 생성
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"1차원 배열: {arr1}")
    
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"2차원 배열:\n{arr2}")
    
    # 특정 값으로 초기화된 배열 생성
    zeros_arr = np.zeros((3, 4))
    print(f"0으로 초기화된 배열:\n{zeros_arr}")
    
    ones_arr = np.ones((2, 3))
    print(f"1로 초기화된 배열:\n{ones_arr}")
    
    full_arr = np.full((2, 3), 7)
    print(f"7로 채워진 배열:\n{full_arr}")
    
    # 단위 행렬
    eye_arr = np.eye(3)
    print(f"단위 행렬:\n{eye_arr}")
    
    # 순차적인 값으로 배열 생성
    arange_arr = np.arange(0, 10, 2)
    print(f"arange 배열: {arange_arr}")
    
    linspace_arr = np.linspace(0, 1, 5)
    print(f"linspace 배열: {linspace_arr}")
    
    # 난수 배열
    random_arr = np.random.rand(3, 3)
    print(f"난수 배열:\n{random_arr}")
    
    print()

def example_array_attributes():
    """배열 속성 예제"""
    print("=== 배열 속성 예제 ===")
    
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    
    print(f"배열:\n{arr}")
    print(f"차원 수 (ndim): {arr.ndim}")
    print(f"형태 (shape): {arr.shape}")
    print(f"전체 요소 수 (size): {arr.size}")
    print(f"데이터 타입 (dtype): {arr.dtype}")
    print(f"요소당 바이트 수 (itemsize): {arr.itemsize}")
    print(f"전체 바이트 수 (nbytes): {arr.nbytes}")
    
    print()

def example_array_operations():
    """배열 기본 연산 예제"""
    print("=== 배열 기본 연산 예제 ===")
    
    # 배열과 스칼라 연산
    arr = np.array([1, 2, 3, 4, 5])
    print(f"원본 배열: {arr}")
    print(f"배열 + 2: {arr + 2}")
    print(f"배열 * 3: {arr * 3}")
    print(f"배열 ** 2: {arr ** 2}")
    
    # 배열 간 연산
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([5, 4, 3, 2, 1])
    print(f"배열1: {arr1}")
    print(f"배열2: {arr2}")
    print(f"배열1 + 배열2: {arr1 + arr2}")
    print(f"배열1 * 배열2: {arr1 * arr2}")
    
    # 2차원 배열 연산
    arr_2d1 = np.array([[1, 2], [3, 4]])
    arr_2d2 = np.array([[5, 6], [7, 8]])
    print(f"2차원 배열1:\n{arr_2d1}")
    print(f"2차원 배열2:\n{arr_2d2}")
    print(f"덧셈:\n{arr_2d1 + arr_2d2}")
    print(f"곱셈 (요소별):\n{arr_2d1 * arr_2d2}")
    print(f"행렬 곱셈:\n{arr_2d1 @ arr_2d2}")
    
    print()

def example_indexing_slicing():
    """인덱싱과 슬라이싱 예제"""
    print("=== 인덱싱과 슬라이싱 예제 ===")
    
    # 1차원 배열 인덱싱
    arr1d = np.array([10, 20, 30, 40, 50])
    print(f"1차원 배열: {arr1d}")
    print(f"arr1d[0]: {arr1d[0]}")
    print(f"arr1d[2]: {arr1d[2]}")
    print(f"arr1d[-1]: {arr1d[-1]}")
    
    # 1차원 배열 슬라이싱
    print(f"arr1d[1:4]: {arr1d[1:4]}")
    print(f"arr1d[:3]: {arr1d[:3]}")
    print(f"arr1d[2:]: {arr1d[2:]}")
    print(f"arr1d[::2]: {arr1d[::2]}")
    
    # 2차원 배열 인덱싱
    arr2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"2차원 배열:\n{arr2d}")
    print(f"arr2d[0, 0]: {arr2d[0, 0]}")
    print(f"arr2d[1, 2]: {arr2d[1, 2]}")
    print(f"arr2d[2, -1]: {arr2d[2, -1]}")
    
    # 2차원 배열 슬라이싱
    print(f"arr2d[1, :]: {arr2d[1, :]}")
    print(f"arr2d[:, 2]: {arr2d[:, 2]}")
    print(f"arr2d[0:2, 1:3]:\n{arr2d[0:2, 1:3]}")
    
    print()

def example_math_functions():
    """수학 함수 예제"""
    print("=== 수학 함수 예제 ===")
    
    arr = np.array([1, 4, 9, 16, 25])
    print(f"원본 배열: {arr}")
    
    # 기본 수학 함수
    print(f"제곱근: {np.sqrt(arr)}")
    print(f"제곱: {np.square(arr)}")
    print(f"절대값: {np.abs(np.array([-1, -2, 3, 4]))}")
    
    # 삼각함수
    angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    print(f"사인: {np.sin(angles)}")
    print(f"코사인: {np.cos(angles)}")
    
    # 지수와 로그 함수
    exp_arr = np.array([1, 2, 3])
    print(f"지수 함수: {np.exp(exp_arr)}")
    print(f"자연 로그: {np.log(exp_arr)}")
    
    print()

def example_aggregation():
    """집계 함수 예제"""
    print("=== 집계 함수 예제 ===")
    
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"배열:\n{arr}")
    
    # 전체 배열에 대한 집계
    print(f"전체 합계: {np.sum(arr)}")
    print(f"전체 평균: {np.mean(arr)}")
    print(f"전체 표준편차: {np.std(arr)}")
    print(f"전체 최소값: {np.min(arr)}")
    print(f"전체 최대값: {np.max(arr)}")
    
    # 축별 집계
    print(f"행별 합계 (axis=1): {np.sum(arr, axis=1)}")
    print(f"열별 합계 (axis=0): {np.sum(arr, axis=0)}")
    print(f"행별 평균 (axis=1): {np.mean(arr, axis=1)}")
    print(f"열별 평균 (axis=0): {np.mean(arr, axis=0)}")
    
    print()

def example_boolean_operations():
    """불리언 연산 예제"""
    print("=== 불리언 연산 예제 ===")
    
    arr = np.array([1, 2, 3, 4, 5])
    print(f"원본 배열: {arr}")
    
    # 비교 연산
    print(f"arr > 3: {arr > 3}")
    print(f"arr < 3: {arr < 3}")
    print(f"arr == 3: {arr == 3}")
    
    # 불리언 인덱싱
    print(f"arr[arr > 3]: {arr[arr > 3]}")
    
    # 논리 연산
    mask = (arr > 2) & (arr < 5)
    print(f"(arr > 2) & (arr < 5): {mask}")
    print(f"arr[mask]: {arr[mask]}")
    
    print()

def example_reshaping():
    """형태 변환 예제"""
    print("=== 형태 변환 예제 ===")
    
    # 1차원 배열을 2차원으로 변환
    arr1d = np.arange(12)
    print(f"1차원 배열: {arr1d}")
    
    arr2d = arr1d.reshape(3, 4)
    print(f"2차원으로 변환 (3x4):\n{arr2d}")
    
    arr2d_2 = arr1d.reshape(4, 3)
    print(f"2차원으로 변환 (4x3):\n{arr2d_2}")
    
    # -1을 사용한 자동 크기 계산
    arr_auto = arr1d.reshape(3, -1)
    print(f"자동 크기 계산 (3x?):\n{arr_auto}")
    
    # 배열 평탄화
    flattened = arr2d.flatten()
    print(f"평탄화된 배열: {flattened}")
    
    print()

def main():
    """모든 예제 실행"""
    print("NumPy 기초 예제 실행")
    print("=" * 50)
    
    example_array_creation()
    example_array_attributes()
    example_array_operations()
    example_indexing_slicing()
    example_math_functions()
    example_aggregation()
    example_boolean_operations()
    example_reshaping()
    
    print("모든 예제 실행 완료!")

if __name__ == "__main__":
    main()