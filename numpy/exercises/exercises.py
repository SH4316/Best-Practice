"""
NumPy 연습 문제

이 파일은 NumPy의 다양한 개념을 연습할 수 있는 문제들을 포함합니다.
각 문제를 풀어보면서 NumPy의 기능을 익히고 실력을 향상시켜 보세요.
"""

import numpy as np

def exercise_array_creation():
    """배열 생성 연습"""
    print("=== 배열 생성 연습 ===")
    
    # 문제 1: 0부터 9까지의 정수를 포함하는 1차원 배열 생성
    arr1 = np.arange(10)
    print(f"문제 1: {arr1}")
    
    # 문제 2: 1부터 10까지의 정수를 포함하는 1차원 배열 생성
    arr2 = np.arange(1, 11)
    print(f"문제 2: {arr2}")
    
    # 문제 3: 0으로 채워진 3x3 크기의 2차원 배열 생성
    arr3 = np.zeros((3, 3))
    print(f"문제 3:\n{arr3}")
    
    # 문제 4: 1로 채워진 2x4 크기의 2차원 배열 생성
    arr4 = np.ones((2, 4))
    print(f"문제 4:\n{arr4}")
    
    # 문제 5: 5로 채워진 2x2 크기의 2차원 배열 생성
    arr5 = np.full((2, 2), 5)
    print(f"문제 5:\n{arr5}")
    
    # 문제 6: 3x3 단위 행렬 생성
    arr6 = np.eye(3)
    print(f"문제 6:\n{arr6}")
    
    # 문제 7: 0부터 1까지 5개의 등간격 숫자를 포함하는 배열 생성
    arr7 = np.linspace(0, 1, 5)
    print(f"문제 7: {arr7}")
    
    # 문제 8: 평균이 0이고 표준편차가 1인 정규분포를 따르는 2x3 난수 배열 생성
    arr8 = np.random.randn(2, 3)
    print(f"문제 8:\n{arr8}")
    
    print()

def exercise_array_operations():
    """배열 연산 연습"""
    print("=== 배열 연산 연습 ===")
    
    # 문제 1: 두 배열의 요소별 합계 계산
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])
    result1 = a + b
    print(f"문제 1: {result1}")
    
    # 문제 2: 배열의 모든 요소에 10을 곱하기
    result2 = a * 10
    print(f"문제 2: {result2}")
    
    # 문제 3: 배열의 모든 요소를 제곱하기
    result3 = a ** 2
    print(f"문제 3: {result3}")
    
    # 문제 4: 배열의 요소 중 3보다 큰 값만 선택하기
    result4 = a[a > 3]
    print(f"문제 4: {result4}")
    
    # 문제 5: 배열의 요소 중 짝수만 선택하기
    result5 = a[a % 2 == 0]
    print(f"문제 5: {result5}")
    
    # 문제 6: 두 배열의 요소별 곱셈 계산
    result6 = a * b
    print(f"문제 6: {result6}")
    
    # 문제 7: 배열의 모든 요소의 합계 계산
    result7 = np.sum(a)
    print(f"문제 7: {result7}")
    
    # 문제 8: 배열의 모든 요소의 평균 계산
    result8 = np.mean(a)
    print(f"문제 8: {result8}")
    
    # 문제 9: 배열의 모든 요소의 표준편차 계산
    result9 = np.std(a)
    print(f"문제 9: {result9}")
    
    # 문제 10: 배열의 최대값과 최소값 계산
    result10_max = np.max(a)
    result10_min = np.min(a)
    print(f"문제 10: 최대값={result10_max}, 최소값={result10_min}")
    
    print()

def exercise_indexing_slicing():
    """인덱싱과 슬라이싱 연습"""
    print("=== 인덱싱과 슬라이싱 연습 ===")
    
    # 문제 1: 배열의 3번째 요소 가져오기
    arr = np.array([10, 20, 30, 40, 50])
    result1 = arr[2]
    print(f"문제 1: {result1}")
    
    # 문제 2: 배열의 마지막 요소 가져오기
    result2 = arr[-1]
    print(f"문제 2: {result2}")
    
    # 문제 3: 배열의 2번째부터 4번째까지의 요소 가져오기
    result3 = arr[1:4]
    print(f"문제 3: {result3}")
    
    # 문제 4: 배열의 처음부터 3번째까지의 요소 가져오기
    result4 = arr[:3]
    print(f"문제 4: {result4}")
    
    # 문제 5: 배열의 2번째부터 끝까지의 요소 가져오기
    result5 = arr[1:]
    print(f"문제 5: {result5}")
    
    # 문제 6: 배열의 모든 요소를 역순으로 가져오기
    result6 = arr[::-1]
    print(f"문제 6: {result6}")
    
    # 문제 7: 2차원 배열의 (1, 2) 위치의 요소 가져오기
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result7 = arr_2d[1, 2]
    print(f"문제 7: {result7}")
    
    # 문제 8: 2차원 배열의 2번째 행 가져오기
    result8 = arr_2d[1, :]
    print(f"문제 8: {result8}")
    
    # 문제 9: 2차원 배열의 3번째 열 가져오기
    result9 = arr_2d[:, 2]
    print(f"문제 9: {result9}")
    
    # 문제 10: 2차원 배열의 1번째부터 2번째 행과 1번째부터 2번째 열의 부분 배열 가져오기
    result10 = arr_2d[0:2, 0:2]
    print(f"문제 10:\n{result10}")
    
    print()

def exercise_reshaping():
    """형태 변환 연습"""
    print("=== 형태 변환 연습 ===")
    
    # 문제 1: 1차원 배열을 2x5 크기의 2차원 배열로 변환
    arr = np.arange(10)
    result1 = arr.reshape(2, 5)
    print(f"문제 1:\n{result1}")
    
    # 문제 2: 1차원 배열을 5x2 크기의 2차원 배열로 변환
    result2 = arr.reshape(5, 2)
    print(f"문제 2:\n{result2}")
    
    # 문제 3: 2차원 배열을 1차원 배열로 변환
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    result3 = arr_2d.flatten()
    print(f"문제 3: {result3}")
    
    # 문제 4: 2차원 배열을 전치하기
    result4 = arr_2d.T
    print(f"문제 4:\n{result4}")
    
    # 문제 5: 1차원 배열을 3차원 배열로 변환
    result5 = arr.reshape(2, 5, 1)
    print(f"문제 5:\n{result5}")
    
    # 문제 6: 2차원 배열을 1차원 배열로 변환 (다른 방법)
    result6 = arr_2d.ravel()
    print(f"문제 6: {result6}")
    
    # 문제 7: 1차원 배열을 2x2x2 크기의 3차원 배열로 변환 (불가능한 경우)
    try:
        result7 = arr.reshape(2, 2, 2)
        print(f"문제 7:\n{result7}")
    except ValueError as e:
        print(f"문제 7: 오류 - {e}")
    
    # 문제 8: 2차원 배열을 1차원 배열로 변환 (order='F' 사용)
    result8 = arr_2d.flatten(order='F')
    print(f"문제 8: {result8}")
    
    # 문제 9: 1차원 배열을 2x3 크기의 2차원 배열로 변환하고 남는 요소는 버리기
    result9 = arr[:6].reshape(2, 3)
    print(f"문제 9:\n{result9}")
    
    # 문제 10: 2차원 배열을 3x2 크기의 2차원 배열로 변환 (불가능한 경우)
    try:
        result10 = arr_2d.reshape(3, 2)
        print(f"문제 10:\n{result10}")
    except ValueError as e:
        print(f"문제 10: 오류 - {e}")
    
    print()

def exercise_math_functions():
    """수학 함수 연습"""
    print("=== 수학 함수 연습 ===")
    
    # 문제 1: 배열의 모든 요소의 제곱근 계산
    arr = np.array([1, 4, 9, 16, 25])
    result1 = np.sqrt(arr)
    print(f"문제 1: {result1}")
    
    # 문제 2: 배열의 모든 요소의 지수 함수 값 계산
    result2 = np.exp(arr)
    print(f"문제 2: {result2}")
    
    # 문제 3: 배열의 모든 요소의 자연 로그 계산
    result3 = np.log(arr)
    print(f"문제 3: {result3}")
    
    # 문제 4: 배열의 모든 요소의 상용 로그 계산
    result4 = np.log10(arr)
    print(f"문제 4: {result4}")
    
    # 문제 5: 배열의 모든 요소의 사인 값 계산
    angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    result5 = np.sin(angles)
    print(f"문제 5: {result5}")
    
    # 문제 6: 배열의 모든 요소의 코사인 값 계산
    result6 = np.cos(angles)
    print(f"문제 6: {result6}")
    
    # 문제 7: 배열의 모든 요소의 탄젠트 값 계산
    result7 = np.tan(angles)
    print(f"문제 7: {result7}")
    
    # 문제 8: 배열의 모든 요소의 절대값 계산
    arr_with_neg = np.array([-1, -2, 3, 4, -5])
    result8 = np.abs(arr_with_neg)
    print(f"문제 8: {result8}")
    
    # 문제 9: 배열의 모든 요소를 반올림하기
    arr_with_decimals = np.array([1.2, 2.5, 3.7, 4.1])
    result9 = np.round(arr_with_decimals)
    print(f"문제 9: {result9}")
    
    # 문제 10: 배열의 모든 요소를 올림하기
    result10 = np.ceil(arr_with_decimals)
    print(f"문제 10: {result10}")
    
    print()

def exercise_statistics():
    """통계 연습"""
    print("=== 통계 연습 ===")
    
    # 문제 1: 배열의 평균 계산
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result1 = np.mean(arr)
    print(f"문제 1: {result1}")
    
    # 문제 2: 배열의 중앙값 계산
    result2 = np.median(arr)
    print(f"문제 2: {result2}")
    
    # 문제 3: 배열의 표준편차 계산
    result3 = np.std(arr)
    print(f"문제 3: {result3}")
    
    # 문제 4: 배열의 분산 계산
    result4 = np.var(arr)
    print(f"문제 4: {result4}")
    
    # 문제 5: 배열의 최소값 계산
    result5 = np.min(arr)
    print(f"문제 5: {result5}")
    
    # 문제 6: 배열의 최대값 계산
    result6 = np.max(arr)
    print(f"문제 6: {result6}")
    
    # 문제 7: 배열의 25% 분위수 계산
    result7 = np.percentile(arr, 25)
    print(f"문제 7: {result7}")
    
    # 문제 8: 배열의 75% 분위수 계산
    result8 = np.percentile(arr, 75)
    print(f"문제 8: {result8}")
    
    # 문제 9: 배열의 사분위 범위(IQR) 계산
    result9 = np.percentile(arr, 75) - np.percentile(arr, 25)
    print(f"문제 9: {result9}")
    
    # 문제 10: 배열의 표준화 (평균 0, 표준편차 1)
    result10 = (arr - np.mean(arr)) / np.std(arr)
    print(f"문제 10: {result10}")
    
    print()

def exercise_linear_algebra():
    """선형대수 연습"""
    print("=== 선형대수 연습 ===")
    
    # 문제 1: 두 행렬의 곱셈 계산
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result1 = a @ b
    print(f"문제 1:\n{result1}")
    
    # 문제 2: 행렬의 전치 계산
    result2 = a.T
    print(f"문제 2:\n{result2}")
    
    # 문제 3: 행렬의 역행렬 계산
    try:
        result3 = np.linalg.inv(a)
        print(f"문제 3:\n{result3}")
    except np.linalg.LinAlgError:
        print("문제 3: 역행렬이 존재하지 않음")
    
    # 문제 4: 행렬의 행렬식 계산
    result4 = np.linalg.det(a)
    print(f"문제 4: {result4}")
    
    # 문제 5: 행렬의 대각합 계산
    result5 = np.trace(a)
    print(f"문제 5: {result5}")
    
    # 문제 6: 행렬의 고유값과 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eig(a)
    print(f"문제 6:")
    print(f"  고유값: {eigenvalues}")
    print(f"  고유벡터:\n{eigenvectors}")
    
    # 문제 7: 연립 방정식 해결
    # 2x + y = 5
    # x + 3y = 7
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 7])
    result7 = np.linalg.solve(A, b)
    print(f"문제 7: {result7}")
    
    # 문제 8: 두 벡터의 내적 계산
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    result8 = np.dot(v1, v2)
    print(f"문제 8: {result8}")
    
    # 문제 9: 두 벡터의 외적 계산
    result9 = np.cross(v1, v2)
    print(f"문제 9: {result9}")
    
    # 문제 10: 벡터의 노름 계산
    result10 = np.linalg.norm(v1)
    print(f"문제 10: {result10}")
    
    print()

def exercise_challenges():
    """도전 과제"""
    print("=== 도전 과제 ===")
    
    # 문제 1: 1부터 100까지의 정수 중 소수만 포함하는 배열 생성
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    numbers = np.arange(1, 101)
    primes = numbers[np.array([is_prime(n) for n in numbers])]
    print(f"문제 1: 소수 배열 (처음 10개): {primes[:10]}")
    
    # 문제 2: 피보나치 수열의 처음 20개 요소를 포함하는 배열 생성
    def fibonacci(n):
        fib = np.zeros(n, dtype=int)
        fib[0] = 1
        if n > 1:
            fib[1] = 1
            for i in range(2, n):
                fib[i] = fib[i-1] + fib[i-2]
        return fib
    
    fib_sequence = fibonacci(20)
    print(f"문제 2: 피보나치 수열: {fib_sequence}")
    
    # 문제 3: 1000x1000 크기의 랜덤 행렬의 고유값 중 가장 큰 값 찾기
    large_matrix = np.random.rand(100, 100)  # 100x100으로 축소
    eigenvalues = np.linalg.eigvals(large_matrix)
    max_eigenvalue = np.max(eigenvalues)
    print(f"문제 3: 최대 고유값: {max_eigenvalue:.6f}")
    
    # 문제 4: 두 행렬의 코사인 유사도 계산
    def cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    matrix1 = np.random.rand(10, 10)
    matrix2 = np.random.rand(10, 10)
    
    # 행렬을 벡터로 변환
    vec1 = matrix1.flatten()
    vec2 = matrix2.flatten()
    
    similarity = cosine_similarity(vec1, vec2)
    print(f"문제 4: 코사인 유사도: {similarity:.6f}")
    
    # 문제 5: 주어진 배열에서 이동 평균 계산 (윈도우 크기=3)
    def moving_average(arr, window_size=3):
        result = np.empty(len(arr) - window_size + 1)
        for i in range(len(result)):
            result[i] = np.mean(arr[i:i+window_size])
        return result
    
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ma = moving_average(data)
    print(f"문제 5: 이동 평균: {ma}")
    
    print()

def check_solutions():
    """솔루션 확인 함수"""
    print("=== 솔루션 확인 ===")
    
    # 정답 배열
    answers = {
        "exercise_array_creation": [
            np.arange(10),
            np.arange(1, 11),
            np.zeros((3, 3)),
            np.ones((2, 4)),
            np.full((2, 2), 5),
            np.eye(3),
            np.linspace(0, 1, 5),
            "난수 배열은 실행할 때마다 다름"
        ],
        "exercise_array_operations": [
            np.array([6, 6, 6, 6, 6]),
            np.array([10, 20, 30, 40, 50]),
            np.array([1, 4, 9, 16, 25]),
            np.array([4, 5]),
            np.array([2, 4]),
            np.array([5, 8, 9, 8, 5]),
            15,
            3.0,
            1.4142135623730951,
            (5, 1)
        ]
    }
    
    # 일부 솔루션 확인
    print("배열 생성 연습 솔루션 확인:")
    print(f"문제 1: {np.array_equal(np.arange(10), answers['exercise_array_creation'][0])}")
    print(f"문제 2: {np.array_equal(np.arange(1, 11), answers['exercise_array_creation'][1])}")
    
    print("\n배열 연산 연습 솔루션 확인:")
    print(f"문제 1: {np.array_equal(np.array([1, 2, 3, 4, 5]) + np.array([5, 4, 3, 2, 1]), answers['exercise_array_operations'][0])}")
    print(f"문제 7: {np.sum(np.array([1, 2, 3, 4, 5])) == answers['exercise_array_operations'][6]}")
    
    print()

def main():
    """모든 연습 실행"""
    print("NumPy 연습 문제")
    print("=" * 50)
    
    exercise_array_creation()
    exercise_array_operations()
    exercise_indexing_slicing()
    exercise_reshaping()
    exercise_math_functions()
    exercise_statistics()
    exercise_linear_algebra()
    exercise_challenges()
    
    # 솔루션 확인
    check_solutions()
    
    print("모든 연습 완료!")

if __name__ == "__main__":
    main()