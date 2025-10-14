# NumPy 완벽 가이드: 인덱스 및 개요

## 프로젝트 소개

이 프로젝트는 NumPy의 기초부터 고급 기법까지 체계적으로 학습할 수 있는 완벽한 가이드입니다. 초보자부터 전문가까지 모든 수준의 사용자에게 유용한 자료를 제공합니다.

## 학습 경로

### 1. NumPy 기초 (Basics)
NumPy를 처음 사용하는 분들을 위한 기초 개념과 설치 방법을 다룹니다.

- **[설치 가이드](01-basics/installation.md)**: NumPy 설치 및 환경 설정
- **[배열 기초](01-basics/arrays.md)**: NumPy 배열의 개념과 생성 방법
- **[기본 연산](01-basics/basic-operations.md)**: 배열의 기본 산술 및 논리 연산

### 2. 배열 조작 (Array Manipulation)
배열의 형태를 변경하고 데이터를 조작하는 다양한 방법을 배웁니다.

- **[인덱싱과 슬라이싱](02-array-manipulation/indexing-slicing.md)**: 배열 요소에 접근하는 방법
- **[형태 변환](02-array-manipulation/reshaping.md)**: 배열의 형태를 변경하는 방법
- **[배열 결합과 분할](02-array-manipulation/joining-splitting.md)**: 여러 배열을 결합하거나 분할하는 방법

### 3. 수학적 연산 (Mathematical Operations)
NumPy의 강력한 수학적 기능과 통계 연산을 학습합니다.

- **[유니버설 함수](03-mathematical-operations/universal-functions.md)**: 벡터화된 수학 연산
- **[통계 연산](03-mathematical-operations/statistics.md)**: 데이터 분석을 위한 통계 함수
- **[선형대수](03-mathematical-operations/linear-algebra.md)**: 행렬 연산과 선형대수 문제 해결

### 4. 성능 최적화 (Performance Optimization)
NumPy 코드의 성능을 향상시키는 고급 기법을 배웁니다.

- **[벡터화](04-performance-optimization/vectorization.md)**: 반복문 없는 효율적인 연산
- **[메모리 관리](04-performance-optimization/memory-management.md)**: 메모리 효율적인 사용 방법
- **[브로드캐스팅](04-performance-optimization/broadcasting.md)**: 형태가 다른 배열 간의 연산

### 5. 고급 기법 (Advanced Techniques)
NumPy의 전문적인 기능과 응용 방법을 다룹니다.

- **[구조화된 배열](05-advanced-techniques/structured-arrays.md)**: 다양한 데이터 타입을 다루는 배열
- **[마스킹](05-advanced-techniques/masking.md)**: 결측값이나 특정 조건의 데이터 처리
- **[커스텀 함수](05-advanced-techniques/custom-functions.md)**: 사용자 정의 함수와 최적화

### 6. 도메인 응용 (Domain Applications)
실제 문제에 NumPy를 적용하는 방법을 배웁니다.

- **[데이터 과학](06-domain-applications/data-science.md)**: 데이터 전처리와 분석
- **[과학 계산](06-domain-applications/scientific-computing.md)**: 수치 해석과 미분 방정식
- **[이미지 처리](06-domain-applications/image-processing.md)**: 이미지 데이터 처리와 분석

### 7. 베스트 프랙티스 (Best Practices)
NumPy 코드를 효율적으로 작성하고 디버깅하는 방법을 배웁니다.

- **[일반적인 실수](07-best-practices/common-pitfalls.md)**: 피해야 할 일반적인 실수
- **[최적화 팁](07-best-practices/optimization-tips.md)**: 성능 최적화를 위한 팁
- **[디버깅 기법](07-best-practices/debugging.md)**: 효과적인 디버깅 방법

## 코드 예제 및 연습 문제

실습을 통해 학습 내용을巩固할 수 있는 다양한 자료를 제공합니다.

- **[기본 예제](examples/basic-examples.py)**: NumPy 기초 개념 예제 코드
- **[중급 예제](examples/intermediate-examples.py)**: 배열 조작과 수학 연산 예제
- **[고급 예제](examples/advanced-examples.py)**: 고급 기법과 응용 예제
- **[연습 문제](exercises/exercises.py)**: 다양한 난이도의 연습 문제

## 빠른 참조

### 자주 사용하는 배열 생성 함수

| 함수 | 설명 | 예제 |
|------|------|------|
| `np.array()` | 리스트로부터 배열 생성 | `np.array([1, 2, 3])` |
| `np.zeros()` | 0으로 채워진 배열 생성 | `np.zeros((3, 3))` |
| `np.ones()` | 1로 채워진 배열 생성 | `np.ones((2, 4))` |
| `np.arange()` | 순차적인 값으로 배열 생성 | `np.arange(0, 10, 2)` |
| `np.linspace()` | 등간격 값으로 배열 생성 | `np.linspace(0, 1, 5)` |
| `np.random.rand()` | 0-1 사이의 난수 배열 생성 | `np.random.rand(3, 3)` |

### 자주 사용하는 배열 연산

| 연산 | 설명 | 예제 |
|------|------|------|
| `+` | 배열 덧셈 | `arr1 + arr2` |
| `*` | 요소별 곱셈 | `arr1 * arr2` |
| `@` | 행렬 곱셈 | `arr1 @ arr2` |
| `np.dot()` | 내적 계산 | `np.dot(arr1, arr2)` |
| `np.sum()` | 합계 계산 | `np.sum(arr)` |
| `np.mean()` | 평균 계산 | `np.mean(arr)` |
| `np.std()` | 표준편차 계산 | `np.std(arr)` |

### 자주 사용하는 인덱싱 기법

| 기법 | 설명 | 예제 |
|------|------|------|
| `arr[i]` | 1차원 인덱싱 | `arr[0]` |
| `arr[i, j]` | 2차원 인덱싱 | `arr[0, 1]` |
| `arr[i:j]` | 슬라이싱 | `arr[1:4]` |
| `arr[i:j, k:l]` | 2차원 슬라이싱 | `arr[0:2, 1:3]` |
| `arr[arr > 5]` | 불리언 인덱싱 | `arr[arr > 5]` |
| `arr[[0, 2, 4]]` | 팬시 인덱싱 | `arr[[0, 2, 4]]` |

## 학습 팁

1. **실습 중심 학습**: 이론을 학습한 후 반드시 코드 예제를 실행해보세요.
2. **점진적 어려움**: 기초부터 시작하여 점차적으로 어려운 내용으로 나아가세요.
3. **문제 해결**: 연습 문제를 직접 풀어보면서 개념을巩固하세요.
4. **코드 수정**: 예제 코드를 수정하며 결과가 어떻게 변하는지 확인하세요.
5. **실제 프로젝트**: 학습한 내용을 바탕으로 작은 프로젝트를 만들어보세요.

## 추가 자료

### 공식 문서
- [NumPy 공식 문서](https://numpy.org/doc/stable/)
- [NumPy 사용자 가이드](https://numpy.org/doc/stable/user/index.html)

### 추천 도서
- "Python for Data Analysis" by Wes McKinney
- "Python Data Science Handbook" by Jake VanderPlas
- "Elegant SciPy" by Juan Nunez-Iglesias, Stéfan van der Walt, and Harriet Dashnow

### 온라인 강의
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [SciPy Lecture Notes](https://scipy-lectures.org/)

## 기여

이 프로젝트는 커뮤니티의 기여를 환영합니다. 오타를 발견하거나 개선할 내용이 있다면 GitHub를 통해 이슈를 제출하거나 풀 리퀘스트를 보내주세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

---

**NumPy 완벽 가이드와 함께 효율적인 데이터 과학과 수치 계산의 세계를 탐험해보세요!**