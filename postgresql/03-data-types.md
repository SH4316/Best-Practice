# 데이터 타입과 스키마 설계 가이드라인

## 학습 목표

- PostgreSQL의 다양한 데이터 타입 이해
- 적절한 데이터 타입 선택 방법 학습
- 배열, JSON, JSONB와 같은 고급 데이터 타입 활용법 습득
- 도메인과 사용자 정의 타입 생성 방법 이해

## 기본 데이터 타입

### 수치 타입

```sql
-- 정수 타입
CREATE TABLE numeric_types (
    id SERIAL PRIMARY KEY,
    smallint_col SMALLINT,    -- 2바이트 (-32,768 ~ 32,767)
    integer_col INTEGER,      -- 4바이트 (-2,147,483,648 ~ 2,147,483,647)
    bigint_col BIGINT,        -- 8바이트 (매우 큰 정수)
    
    -- 고정 소수점 타입 (금융 데이터에 권장)
    decimal_col DECIMAL(10, 2),  -- 전체 10자리, 소수점 이하 2자리
    numeric_col NUMERIC(15, 4),  -- 전체 15자리, 소수점 이하 4자리
    
    -- 부동 소수점 타입 (과학 계산에 적합)
    real_col REAL,            -- 4바이트 부동 소수점
    double_precision_col DOUBLE PRECISION,  -- 8바이트 부동 소수점
    
    -- 자동 증가 타입
    serial_col SERIAL,        -- SERIAL = INTEGER + 자동 증가
    bigserial_col BIGSERIAL   -- BIGSERIAL = BIGINT + 자동 증가
);

-- 예제 데이터 삽입
INSERT INTO numeric_types (smallint_col, integer_col, bigint_col, 
                          decimal_col, numeric_col, real_col, double_precision_col)
VALUES (100, 1000000, 1000000000000, 12345.67, 123456789.1234, 3.14, 3.14159265359);
```

### 문자열 타입

```sql
CREATE TABLE string_types (
    id SERIAL PRIMARY KEY,
    
    -- 고정 길이 문자열
    char_col CHAR(10),        -- 항상 10자리, 부족하면 공백으로 채움
    
    -- 가변 길이 문자열
    varchar_col VARCHAR(255), -- 최대 255자리
    text_col TEXT,            -- 길이 제한 없는 텍스트
    
    -- 특수 목적 문자열 타입
    name_col NAME,            -- 객체 식별자용, 최대 63자리
    cidr_col CIDR,            -- IP 주소
    inet_col INET,            -- IP 주소와 네트워크
    macaddr_col MACADDR       -- MAC 주소
);

-- 예제 데이터 삽입
INSERT INTO string_types (char_col, varchar_col, text_col, name_col, cidr_col, inet_col, macaddr_col)
VALUES ('fixed     ', 'variable length', 'This is a long text that can contain multiple sentences...', 
        'table_name', '192.168.1.0/24', '192.168.1.100', '08:00:2b:01:02:03');
```

### 날짜와 시간 타입

```sql
CREATE TABLE datetime_types (
    id SERIAL PRIMARY KEY,
    
    -- 날짜와 시간
    timestamp_col TIMESTAMP,              -- 날짜와 시간
    timestamp_with_tz_col TIMESTAMP WITH TIME ZONE,  -- 타임존 포함
    date_col DATE,                       -- 날짜만
    time_col TIME,                       -- 시간만
    time_with_tz_col TIME WITH TIME ZONE, -- 타임존 포함 시간
    
    -- 간격 타입
    interval_col INTERVAL                -- 시간 간격
);

-- 예제 데이터 삽입
INSERT INTO datetime_types (timestamp_col, timestamp_with_tz_col, date_col, 
                           time_col, time_with_tz_col, interval_col)
VALUES (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_DATE, 
        CURRENT_TIME, CURRENT_TIME, INTERVAL '1 day 2 hours 30 minutes');

-- 날짜/시간 함수 예제
SELECT 
    CURRENT_TIMESTAMP,
    CURRENT_DATE,
    CURRENT_TIME,
    NOW(),
    EXTRACT(YEAR FROM timestamp_col) as year,
    EXTRACT(MONTH FROM timestamp_col) as month,
    EXTRACT(DAY FROM timestamp_col) as day
FROM datetime_types
WHERE id = 1;
```

## 고급 데이터 타입

### 배열 타입

```sql
CREATE TABLE array_examples (
    id SERIAL PRIMARY KEY,
    
    -- 정수 배열
    scores INTEGER[],
    
    -- 텍스트 배열
    tags TEXT[],
    
    -- 다차원 배열
    matrix INTEGER[][],
    
    -- 배열 크기 지정
    fixed_numbers INTEGER[5]
);

-- 배열 데이터 삽입
INSERT INTO array_examples (scores, tags, matrix, fixed_numbers)
VALUES 
(ARRAY[95, 87, 92, 88], ARRAY['postgresql', 'database', 'sql'], 
 ARRAY[[1, 2, 3], [4, 5, 6]], ARRAY[1, 2, 3, 4, 5]);

-- 배열 조회
SELECT 
    scores,
    scores[1] as first_score,           -- 1-based 인덱스
    scores[2:3] as middle_scores,       -- 슬라이스
    array_length(scores, 1) as length,  -- 배열 길이
    tags,
    matrix[1][2] as matrix_value        -- 다차원 배열 접근
FROM array_examples;

-- 배열 검색
SELECT * FROM array_examples 
WHERE 92 = ANY(scores);  -- 배열에 92가 포함된 행

SELECT * FROM array_examples 
WHERE 'postgresql' = ANY(tags);  -- 태그에 'postgresql'이 포함된 행

-- 배열 함수
SELECT 
    unnest(scores) as individual_score,  -- 배열을 행으로 확장
    array_append(scores, 100) as new_scores,  -- 배열에 요소 추가
    array_prepend(90, scores) as prepended_scores,  -- 배열 앞에 요소 추가
    array_cat(scores, ARRAY[100, 101]) as concatenated_scores  -- 배열 연결
FROM array_examples
WHERE id = 1;
```

### JSON과 JSONB 타입

```sql
CREATE TABLE json_examples (
    id SERIAL PRIMARY KEY,
    
    -- JSON 타입 (원본 텍스트 저장)
    config JSON,
    
    -- JSONB 타입 (이진 형식 저장, 인덱싱 가능)
    metadata JSONB
);

-- JSON 데이터 삽입
INSERT INTO json_examples (config, metadata)
VALUES 
('{"theme": "dark", "notifications": true, "timeout": 30}', 
 '{"user": {"id": 123, "name": "John"}, "permissions": ["read", "write"], "active": true}');

-- JSONB 데이터 삽입
INSERT INTO json_examples (metadata)
VALUES 
('{"product": {"id": 1, "name": "Laptop", "price": 999.99}, "tags": ["electronics", "computer"]}');

-- JSON 조회
SELECT 
    config,
    config->>'theme' as theme,           -- JSON 연산자: 키 값 추출 (텍스트)
    config->'notifications' as notifications,  -- JSON 연산자: 키 값 추출 (JSON)
    metadata,
    metadata->'user' as user_info,        -- 중첩 객체 접근
    metadata->>'user'->>'name' as user_name,  -- 중첩 키 값 추출
    metadata->'permissions' as permissions   -- 배열 접근
FROM json_examples;

-- JSONB 함수
SELECT 
    metadata,
    jsonb_typeof(metadata) as type,                    -- JSONB 타입 확인
    jsonb_extract_path(metadata, 'user', 'name') as name,  -- 경로로 값 추출
    jsonb_array_length(metadata->'permissions') as perm_count,  -- 배열 길이
    jsonb_keys(metadata) as all_keys                   -- 모든 키
FROM json_examples
WHERE id = 2;

-- JSONB 검색
SELECT * FROM json_examples 
WHERE metadata @> '{"user": {"id": 123}}';  -- 포함 관계 검색

SELECT * FROM json_examples 
WHERE metadata ? 'product';  -- 키 존재 여부 확인

SELECT * FROM json_examples 
WHERE metadata ?| array['user', 'product'];  -- 여러 키 중 하나라도 존재

-- JSONB 수정
UPDATE json_examples 
SET metadata = jsonb_set(metadata, '{user, name}', '"Jane"', true)  -- 값 수정
WHERE id = 1;

UPDATE json_examples 
SET metadata = metadata || '{"last_login": "2023-01-15"}'  -- 키-값 쌍 추가
WHERE id = 1;

UPDATE json_examples 
SET metadata = metadata - 'active'  -- 키 삭제
WHERE id = 1;
```

### 열거형(Enum) 타입

```sql
-- 열거형 타입 생성
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended', 'deleted');
CREATE TYPE priority AS ENUM ('low', 'medium', 'high', 'urgent');

CREATE TABLE enum_examples (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    status user_status DEFAULT 'active',
    task_priority priority DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 열거형 데이터 삽입
INSERT INTO enum_examples (username, status, task_priority)
VALUES 
('john_doe', 'active', 'high'),
('jane_smith', 'inactive', 'low'),
('bob_wilson', 'suspended', 'urgent');

-- 열거형 조회
SELECT 
    username,
    status,
    status::text as status_text,  -- 텍스트로 변환
    task_priority,
    CASE task_priority
        WHEN 'low' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'high' THEN 3
        WHEN 'urgent' THEN 4
    END as priority_number
FROM enum_examples;

-- 열거형 수정
ALTER TYPE user_status ADD VALUE 'pending' AFTER 'inactive';  -- 새 값 추가
```

### 범위(Range) 타입

```sql
CREATE TABLE range_examples (
    id SERIAL PRIMARY KEY,
    
    -- 숫자 범위
    price_range NUMRANGE,
    
    -- 날짜 범위
    date_range DATERANGE,
    
    -- 타임스탬프 범위
    tsrange_col TSRANGE
);

-- 범위 데이터 삽입
INSERT INTO range_examples (price_range, date_range, tsrange_col)
VALUES 
('[100, 500)', '[2023-01-01, 2023-12-31]', '[2023-01-01 09:00:00, 2023-01-01 18:00:00)'),
('(50, 200]', '(2023-06-01, 2023-06-30)', '(2023-06-01 10:00:00, 2023-06-01 17:00:00]');

-- 범위 조회
SELECT 
    price_range,
    lower(price_range) as min_price,
    upper(price_range) as max_price,
    lower_inc(price_range) as include_min,  -- 하한 포함 여부
    upper_inc(price_range) as include_max,  -- 상한 포함 여부
    date_range,
    tsrange_col
FROM range_examples;

-- 범위 검색
SELECT * FROM range_examples 
WHERE price_range @> 250;  -- 범위에 250이 포함되는지

SELECT * FROM range_examples 
WHERE date_range && '[2023-06-15, 2023-06-20]';  -- 범위 겹침 확인

SELECT * FROM range_examples 
WHERE tsrange_col <@ '[2023-01-01, 2023-12-31]';  -- 범위가 다른 범위에 포함되는지
```

## 도메인(Domain) 타입

도메인은 기존 데이터 타입에 제약 조건을 추가한 사용자 정의 타입입니다.

```sql
-- 이메일 도메인 생성
CREATE DOMAIN email_domain AS VARCHAR(255)
CHECK (VALUE ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

-- 양수 도메인 생성
CREATE DOMAIN positive_number AS DECIMAL(10, 2)
CHECK (VALUE > 0);

-- 나이 도메인 생성
CREATE DOMAIN age_domain AS INTEGER
CHECK (VALUE >= 0 AND VALUE <= 150);

CREATE TABLE domain_examples (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email email_domain,
    salary positive_number,
    age age_domain
);

-- 도메인 제약 조건 테스트
INSERT INTO domain_examples (name, email, salary, age)
VALUES 
('John Doe', 'john@example.com', 50000.00, 30),  -- 성공
('Jane Smith', 'jane@example.com', 60000.00, 25); -- 성공

-- 아래 쿼리는 도메인 제약 조건 위반으로 실패
-- INSERT INTO domain_examples (name, email, salary, age)
-- VALUES ('Invalid Email', 'invalid-email', 50000.00, 30);
-- INSERT INTO domain_examples (name, email, salary, age)
-- VALUES ('Negative Salary', 'test@example.com', -1000.00, 30);
-- INSERT INTO domain_examples (name, email, salary, age)
-- VALUES ('Invalid Age', 'test@example.com', 50000.00, 200);
```

## 사용자 정의 타입(Composite Type)

```sql
-- 주소 타입 정의
CREATE TYPE address_type AS (
    street VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(50)
);

-- 연락처 타입 정의
CREATE TYPE contact_type AS (
    phone VARCHAR(20),
    email VARCHAR(255),
    website VARCHAR(255)
);

CREATE TABLE custom_type_examples (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address address_type,
    contact contact_type,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 정의 타입 데이터 삽입
INSERT INTO custom_type_examples (name, address, contact)
VALUES 
('Acme Corporation', 
 ROW('123 Main St', 'New York', 'NY', '10001', 'USA'),
 ROW('+1-555-1234', 'info@acme.com', 'https://www.acme.com')),
('Global Tech', 
 ROW('456 Oak Ave', 'San Francisco', 'CA', '94102', 'USA'),
 ROW('+1-555-5678', 'contact@globaltech.com', 'https://www.globaltech.com'));

-- 사용자 정의 타입 조회
SELECT 
    name,
    (address).street as street,
    (address).city as city,
    (address).postal_code as postal_code,
    (contact).email as email,
    (contact).phone as phone
FROM custom_type_examples;

-- 사용자 정의 타입 검색
SELECT * FROM custom_type_examples 
WHERE (address).city = 'New York';

SELECT * FROM custom_type_examples 
WHERE (contact).email LIKE '%@acme.com';
```

## 데이터 타입 선택 가이드라인

### 수치 데이터 선택

```sql
-- 권장 사례
CREATE TABLE product_prices (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    
    -- 수량: 항상 정수이고 크지 않음
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    
    -- 가격: 소수점이 필요하고 정확성이 중요 (금융 데이터)
    price DECIMAL(10, 2) NOT NULL CHECK (price > 0),
    
    -- 할인율: 소수점이 필요하지만 정확성보다 성능이 중요
    discount_rate REAL CHECK (discount_rate >= 0 AND discount_rate <= 1),
    
    -- 재고: 매우 큰 수가 될 수 있음
    stock_count BIGINT DEFAULT 0
);
```

### 문자열 데이터 선택

```sql
-- 권장 사례
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    
    -- 고정 길이 코드: CHAR 사용
    country_code CHAR(2),  -- ISO 국가 코드
    
    -- 가변 길이 텍스트: VARCHAR 사용
    username VARCHAR(50) NOT NULL,
    email VARCHAR(255) NOT NULL,
    
    -- 긴 텍스트: TEXT 사용
    bio TEXT,
    
    -- 객체 식별자: NAME 사용
    table_name NAME
);
```

### 날짜/시간 데이터 선택

```sql
-- 권장 사례
CREATE TABLE event_logs (
    id SERIAL PRIMARY KEY,
    
    -- 이벤트 발생 시간: TIMESTAMP WITH TIME ZONE 사용
    event_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- 이벤트 날짜만 필요: DATE 사용
    event_date DATE GENERATED ALWAYS AS (event_time::date) STORED,
    
    -- 처리 시간: INTERVAL 사용
    processing_time INTERVAL
);
```

## 실습: 데이터 타입 활용 연습

온라인 상점의 제품 카탈로그를 위한 테이블을 설계해 봅시다.

```sql
-- 제품 카테고리 열거형
CREATE TYPE product_category AS ENUM (
    'electronics', 'clothing', 'books', 'home', 'sports', 'toys'
);

-- 제품 상태 열거형
CREATE TYPE product_status AS ENUM (
    'active', 'inactive', 'discontinued', 'out_of_stock'
);

-- 가격 범위 타입
CREATE DOMAIN price_domain AS DECIMAL(10, 2)
CHECK (VALUE >= 0);

-- 평점 도메인
CREATE DOMAIN rating_domain AS DECIMAL(3, 2)
CHECK (VALUE >= 0 AND VALUE <= 5);

-- 제품 테이블
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category product_category NOT NULL,
    status product_status DEFAULT 'active',
    
    -- 가격 정보
    price price_domain NOT NULL,
    sale_price price_domain,
    
    -- 재고 정보
    stock_quantity INTEGER DEFAULT 0 CHECK (stock_quantity >= 0),
    
    -- 평점 정보
    average_rating rating_domain,
    review_count INTEGER DEFAULT 0 CHECK (review_count >= 0),
    
    -- 제품 속성 (JSONB)
    attributes JSONB,
    
    -- 제품 태그 (배열)
    tags TEXT[],
    
    -- 크기 정보 (범위)
    weight_range NUMRANGE,
    
    -- 시간 정보
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    discontinued_at TIMESTAMP WITH TIME ZONE
);

-- 제품 데이터 삽입
INSERT INTO products (name, description, category, price, sale_price, 
                     stock_quantity, attributes, tags, weight_range)
VALUES 
('스마트워치', '최신 스마트워치, 심박수 측정, GPS 내장', 'electronics', 
 299.99, 249.99, 50, 
 '{"brand": "TechCorp", "model": "SW-2023", "color": "black", "screen_size": "1.4"}',
 ARRAY['wearable', 'fitness', 'gps'], '[30.0, 50.0]'),

('코트', '겨울용 따뜻한 코트, 방수 기능', 'clothing', 
 159.99, NULL, 30, 
 '{"brand": "FashionHub", "material": "wool", "size": "L", "color": "navy"}',
 ARRAY['winter', 'waterproof', 'warm'], '[800.0, 1200.0]');

-- 제품 조회
SELECT 
    name,
    category,
    price,
    sale_price,
    attributes->>'brand' as brand,
    attributes->>'color' as color,
    tags,
    stock_quantity
FROM products
WHERE status = 'active' AND stock_quantity > 0;
```

## 요약

PostgreSQL의 데이터 타입 선택을 위한 핵심 원칙:

1. **적절한 기본 타입 선택**: 데이터의 특성에 맞는 타입 사용
2. **고급 타입 활용**: 배열, JSONB, 열거형 등으로 복잡한 데이터 효율적 처리
3. **도메인으로 데이터 무결성 보장**: 재사용 가능한 제약 조건 정의
4. **사용자 정의 타입으로 복잡한 구조 표현**: 관련 데이터를 그룹화
5. **성능 고려**: 저장 공간과 쿼리 성능 간의 균형

다음 섹션에서는 PostgreSQL의 인덱싱 전략과 베스트 프랙티스에 대해 알아보겠습니다.

## 추가 자료

- [PostgreSQL 문서: 데이터 타입](https://www.postgresql.org/docs/current/datatype.html)
- [PostgreSQL JSON 함수 문서](https://www.postgresql.org/docs/current/functions-json.html)
- [PostgreSQL 배열 함수 문서](https://www.postgresql.org/docs/current/functions-array.html)