# NumPy 구조화된 배열(Structured Arrays)

## 구조화된 배열이란?

구조화된 배열은 하나의 배열에 서로 다른 데이터 타입을 가진 여러 필드를 저장할 수 있는 NumPy의 고급 기능입니다. 데이터베이스의 테이블이나 C의 구조체와 유사한 개념입니다.

## 구조화된 배열 생성

### 기본 구조화된 배열

```python
import numpy as np

# 데이터 타입 정의
dtype = [('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]

# 구조화된 배열 생성
people = np.array([('Alice', 25, 55.0), 
                  ('Bob', 30, 70.5), 
                  ('Charlie', 35, 65.2)], dtype=dtype)

print(people)
print(f"데이터 타입: {people.dtype}")
print(f"형태: {people.shape}")
```

### 다양한 데이터 타입

```python
# 다양한 데이터 타입을 가진 구조화된 배열
dtype = [
    ('id', 'i4'),           # 32비트 정수
    ('name', 'U20'),        # 20자 유니코드 문자열
    ('scores', 'f4', (3,)), # 3개 요소를 가진 float32 배열
    ('active', '?'),        # 불리언
    ('registered', 'datetime64[D]')  # 날짜
]

# 데이터 생성
students = np.array([
    (1, 'Alice', [85.5, 90.0, 78.5], True, '2023-09-01'),
    (2, 'Bob', [70.0, 65.5, 80.0], True, '2023-09-01'),
    (3, 'Charlie', [95.0, 92.5, 88.0], False, '2023-09-02')
], dtype=dtype)

print(students)
print(students.dtype)
```

### 중첩된 구조화된 배열

```python
# 중첩된 구조화된 배열
person_dtype = [
    ('name', 'U10'),
    ('address', [
        ('street', 'U20'),
        ('city', 'U15'),
        ('zipcode', 'U5')
    ])
]

people = np.array([
    ('Alice', ('123 Main St', 'New York', '10001')),
    ('Bob', ('456 Oak Ave', 'Boston', '02108'))
], dtype=person_dtype)

print(people)
print(people.dtype)
```

## 구조화된 배열 조작

### 필드 접근

```python
# 기본 필드 접근
dtype = [('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
people = np.array([('Alice', 25, 55.0), 
                  ('Bob', 30, 70.5), 
                  ('Charlie', 35, 65.2)], dtype=dtype)

# 필드별 접근
print("이름:", people['name'])
print("나이:", people['age'])
print("몸무게:", people['weight'])

# 단일 요소 접근
print("첫 번째 사람:", people[0])
print("첫 번째 사람 이름:", people[0]['name'])
```

### 필드 수정

```python
# 필드 값 수정
people['age'] = [26, 31, 36]  # 전체 나이 수정
print("수정된 나이:", people['age'])

# 특정 요소의 필드 수정
people[0]['weight'] = 56.5
print("첫 번째 사람 수정 후:", people[0])

# 조건부 수정
people[people['age'] > 30]['weight'] = 75.0
print("30세 이상 몸무게 수정:", people['weight'])
```

### 필드 추가 및 삭제

```python
# 새 필드 추가 (새 배열 생성)
new_dtype = people.dtype.descr + [('height', 'f4')]
new_people = np.zeros(people.shape, dtype=new_dtype)

# 기존 데이터 복사
for field in people.dtype.names:
    new_people[field] = people[field]

# 새 필드 값 설정
new_people['height'] = [165.0, 175.0, 180.0]

print("새 배열:", new_people)
print("새 데이터 타입:", new_people.dtype)
```

## 고급 필드 연산

### 필드별 정렬

```python
# 특정 필드로 정렬
sorted_by_age = np.sort(people, order='age')
print("나이순 정렬:", sorted_by_age)

sorted_by_weight = np.sort(people, order='weight')
print("몸무게순 정렬:", sorted_by_weight)

# 다중 필드로 정렬
sorted_multi = np.sort(people, order=['age', 'weight'])
print("나이, 몸무게순 정렬:", sorted_multi)
```

### 필드별 집계

```python
# 수치 필드에 대한 집계 연산
print("평균 나이:", np.mean(people['age']))
print("평균 몸무게:", np.mean(people['weight']))

# 조건부 집계
adults = people[people['age'] >= 30]
print("성인 평균 몸무게:", np.mean(adults['weight']))
```

### 필드별 수학 연산

```python
# 필드별 수학 연산
# BMI 계산
height_m = np.array([1.65, 1.75, 1.80])  # 키 (미터)
bmi = people['weight'] / (height_m ** 2)
print("BMI:", bmi)

# 새 필드로 BMI 추가
bmi_dtype = people.dtype.descr + [('bmi', 'f4')]
people_with_bmi = np.zeros(people.shape, dtype=bmi_dtype)

for field in people.dtype.names:
    people_with_bmi[field] = people[field]

people_with_bmi['bmi'] = bmi
print("BMI가 포함된 배열:", people_with_bmi)
```

## 구조화된 배열과 파일 입출력

### CSV 파일 읽기/쓰기

```python
# 구조화된 배열을 CSV 파일로 저장
import csv

filename = 'people.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(people.dtype.names)  # 헤더 쓰기
    for person in people:
        writer.writerow(person)

# CSV 파일에서 구조화된 배열로 읽기
def read_structured_csv(filename, dtype):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 헤더 건너뛰기
        data = []
        for row in reader:
            # 데이터 타입에 맞게 변환
            converted_row = []
            for i, (value, (name, dt)) in enumerate(zip(row, dtype)):
                if 'i' in dt:  # 정수
                    converted_row.append(int(value))
                elif 'f' in dt:  # 실수
                    converted_row.append(float(value))
                else:  # 문자열
                    converted_row.append(value)
            data.append(tuple(converted_row))
    
    return np.array(data, dtype=dtype)

# 읽기 테스트
loaded_people = read_structured_csv(filename, people.dtype)
print("파일에서 읽은 데이터:", loaded_people)
```

### 바이너리 파일 입출력

```python
# 구조화된 배열을 바이너리 파일로 저장
binary_filename = 'people.bin'
people.tofile(binary_filename)

# 바이너리 파일에서 구조화된 배열로 읽기
loaded_binary = np.fromfile(binary_filename, dtype=people.dtype)
print("바이너리 파일에서 읽은 데이터:", loaded_binary)
```

## 구조화된 배열과 Pandas

### NumPy 구조화된 배열을 Pandas DataFrame으로 변환

```python
try:
    import pandas as pd
    
    # NumPy 구조화된 배열을 Pandas DataFrame으로 변환
    df = pd.DataFrame(people)
    print("DataFrame:")
    print(df)
    
    # DataFrame을 NumPy 구조화된 배열로 변환
    numpy_array = df.to_records(index=False)
    print("다시 NumPy 배열:")
    print(numpy_array)
    
except ImportError:
    print("Pandas가 설치되지 않았습니다. 'pip install pandas'로 설치하세요.")
```

## 실용적인 구조화된 배열 예제

### 센서 데이터 처리

```python
# 센서 데이터를 위한 구조화된 배열
sensor_dtype = [
    ('timestamp', 'datetime64[s]'),  # 타임스탬프
    ('sensor_id', 'i4'),              # 센서 ID
    ('temperature', 'f4'),            # 온도
    ('humidity', 'f4'),               # 습도
    ('pressure', 'f4')                # 압력
]

# 가상 센서 데이터 생성
np.random.seed(42)
num_readings = 100

sensor_data = np.zeros(num_readings, dtype=sensor_dtype)
sensor_data['timestamp'] = np.arange('2023-01-01', '2023-01-01T01:40:00', dtype='datetime64[s]')
sensor_data['sensor_id'] = np.random.randint(1, 5, num_readings)
sensor_data['temperature'] = 20 + 5 * np.random.rand(num_readings)
sensor_data['humidity'] = 40 + 20 * np.random.rand(num_readings)
sensor_data['pressure'] = 1000 + 10 * np.random.randn(num_readings)

print("센서 데이터 (처음 5개):")
print(sensor_data[:5])

# 센서별 분석
for sensor_id in np.unique(sensor_data['sensor_id']):
    mask = sensor_data['sensor_id'] == sensor_id
    sensor_readings = sensor_data[mask]
    
    print(f"\n센서 {sensor_id} 분석:")
    print(f"  평균 온도: {np.mean(sensor_readings['temperature']):.2f}°C")
    print(f"  평균 습도: {np.mean(sensor_readings['humidity']):.2f}%")
    print(f"  평균 압력: {np.mean(sensor_readings['pressure']):.2f} hPa")
```

### 금융 데이터 분석

```python
# 주식 데이터를 위한 구조화된 배열
stock_dtype = [
    ('date', 'datetime64[D]'),  # 날짜
    ('symbol', 'U5'),            # 종목 코드
    ('open', 'f4'),              # 시가
    ('high', 'f4'),              # 고가
    ('low', 'f4'),               # 저가
    ('close', 'f4'),             # 종가
    ('volume', 'i8')             # 거래량
]

# 가상 주식 데이터 생성
np.random.seed(42)
symbols = ['AAPL', 'GOOGL', 'MSFT']
num_days = 30

stock_data = np.zeros(len(symbols) * num_days, dtype=stock_dtype)

# 데이터 채우기
idx = 0
for symbol in symbols:
    base_price = 100 + np.random.rand() * 200
    for day in range(num_days):
        stock_data[idx]['date'] = np.datetime64('2023-01-01') + np.timedelta64(day, 'D')
        stock_data[idx]['symbol'] = symbol
        
        # 가격 생성 (랜덤 워크)
        price_change = np.random.randn() * 5
        open_price = base_price + price_change
        high_price = open_price + abs(np.random.randn() * 3)
        low_price = open_price - abs(np.random.randn() * 3)
        close_price = low_price + np.random.rand() * (high_price - low_price)
        volume = np.random.randint(1000000, 10000000)
        
        stock_data[idx]['open'] = open_price
        stock_data[idx]['high'] = high_price
        stock_data[idx]['low'] = low_price
        stock_data[idx]['close'] = close_price
        stock_data[idx]['volume'] = volume
        
        base_price = close_price  # 다음 날의 기준가
        idx += 1

print("주식 데이터 (처음 5개):")
print(stock_data[:5])

# 종목별 분석
for symbol in np.unique(stock_data['symbol']):
    mask = stock_data['symbol'] == symbol
    symbol_data = stock_data[mask]
    
    # 일일 수익률 계산
    daily_returns = (symbol_data['close'][1:] - symbol_data['close'][:-1]) / symbol_data['close'][:-1]
    
    print(f"\n{symbol} 분석:")
    print(f"  평균 종가: {np.mean(symbol_data['close']):.2f}")
    print(f"  평균 일일 수익률: {np.mean(daily_returns)*100:.2f}%")
    print(f"  수익률 표준편차: {np.std(daily_returns)*100:.2f}%")
    print(f"  총 거래량: {np.sum(symbol_data['volume']):,}")
```

## 구조화된 배열 최적화

### 메모리 레이아웃 최적화

```python
# 메모리 정렬 최적화
# 필드를 크기 순서대로 정렬하여 메모리 패딩 최소화
dtype_unoptimized = [
    ('name', 'U10'),  # 40 바이트
    ('age', 'i1'),    # 1 바이트
    ('weight', 'f8'), # 8 바이트
    ('active', '?')   # 1 바이트
]

dtype_optimized = [
    ('weight', 'f8'), # 8 바이트
    ('name', 'U10'),  # 40 바이트
    ('age', 'i1'),    # 1 바이트
    ('active', '?')   # 1 바이트
]

data_unoptimized = np.zeros(1000, dtype=dtype_unoptimized)
data_optimized = np.zeros(1000, dtype=dtype_optimized)

print(f"최적화되지 않은 배열 크기: {data_unoptimized.nbytes} 바이트")
print(f"최적화된 배열 크기: {data_optimized.nbytes} 바이트")
```

### 데이터 타입 최적화

```python
# 적절한 데이터 타입 선택으로 메모리 절약
dtype_large = [
    ('id', 'i8'),      # 64비트 정수
    ('value', 'f8'),   # 64비트 실수
    ('flag', 'i4')     # 32비트 정수 (불리언 대신)
]

dtype_optimized = [
    ('id', 'i4'),      # 32비트 정수 (충분한 경우)
    ('value', 'f4'),   # 32비트 실수 (정밀도가 낮아도 되는 경우)
    ('flag', '?')      # 1비트 불리언
]

data_large = np.zeros(10000, dtype=dtype_large)
data_optimized = np.zeros(10000, dtype=dtype_optimized)

print(f"큰 데이터 타입 배열 크기: {data_large.nbytes} 바이트")
print(f"최적화된 데이터 타입 배열 크기: {data_optimized.nbytes} 바이트")
print(f"메모리 절약: {data_large.nbytes / data_optimized.nbytes:.2f}배")
```

## 구조화된 배열 모범 사례

1. **적절한 데이터 타입 선택**: 실제 데이터 범위에 맞는 최소 크기의 타입 사용
2. **필드 순서 최적화**: 메모리 정렬을 위해 크기가 큰 필드부터 배치
3. **필드 이름 명확성**: 의미 있는 필드 이름 사용
4. **데이터 접근 최적화**: 자주 접근하는 필드를 앞쪽에 배치
5. **문서화**: 복잡한 구조는 주석으로 문서화

## 다음 학습 내용

다음으로는 마스킹과 팬시 인덱싱에 대해 알아보겠습니다. [`masking.md`](masking.md)를 참조하세요.