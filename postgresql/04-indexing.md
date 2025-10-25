# 인덱싱 전략과 베스트 프랙티스

## 학습 목표

- 인덱스의 원리와 중요성 이해
- PostgreSQL의 다양한 인덱스 타입과 사용법 학습
- 효과적인 인덱스 설계 전략 습득
- 인덱스 성능 분석 및 최적화 방법 이해

## 인덱스 기본 개념

### 인덱스란?

인덱스는 데이터베이스 테이블의 검색 속도를 향상시키기 위한 데이터 구조입니다. 책의 색인과 유사하게 특정 데이터를 빠르게 찾을 수 있게 해줍니다.

```sql
-- 인덱스 없는 테이블
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 대량 데이터 삽입
INSERT INTO users (username, email)
SELECT 
    'user_' || i,
    'user_' || i || '@example.com'
FROM generate_series(1, 100000) i;

-- 인덱스 없이 검색 (느림)
EXPLAIN ANALYZE SELECT * FROM users WHERE username = 'user_50000';

-- 인덱스 생성
CREATE INDEX idx_users_username ON users(username);

-- 인덱스로 검색 (빠름)
EXPLAIN ANALYZE SELECT * FROM users WHERE username = 'user_50000';
```

### 인덱스의 장단점

**장점:**
- 검색 속도 향상
- 정렬 및 그룹화 성능 향상
- 조인 연산 최적화

**단점:**
- 추가 저장 공간 필요
- 삽입, 수정, 삭제 시 성능 저하
- 인덱스 관리 오버헤드

## B-Tree 인덱스

B-Tree(Balanced Tree)는 PostgreSQL의 기본 인덱스 타입으로, 대부분의 경우에 적합합니다.

```sql
-- B-Tree 인덱스 생성
CREATE INDEX idx_products_name ON products(name);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- 복합 인덱스
CREATE INDEX idx_order_items_order_product ON order_items(order_id, product_id);

-- 고유 인덱스
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- 부분 인덱스 (특정 조건을 만족하는 행만 인덱싱)
CREATE INDEX idx_active_users ON users(id) WHERE is_active = true;
CREATE INDEX idx_recent_orders ON orders(created_at) WHERE created_at > CURRENT_DATE - INTERVAL '30 days';

-- 함수 기반 인덱스
CREATE INDEX idx_users_lower_email ON users(LOWER(email));
CREATE INDEX idx_users_email_domain ON users(SUBSTRING(email FROM POSITION('@' IN email) + 1));

-- 표현식 기반 인덱스
CREATE INDEX idx_products_price_category ON products((price * 1.1)) WHERE category = 'electronics';
```

### 복합 인덱스의 컬럼 순서

```sql
-- 테이블 생성
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    region VARCHAR(50) NOT NULL,
    sale_date DATE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL
);

-- 복합 인덱스 생성
CREATE INDEX idx_sales_region_date ON sales(region, sale_date);

-- 쿼리와 인덱스 사용
-- 이 쿼리는 인덱스를 효과적으로 사용
EXPLAIN SELECT * FROM sales 
WHERE region = 'Seoul' AND sale_date BETWEEN '2023-01-01' AND '2023-01-31';

-- 이 쿼리도 인덱스를 사용하지만 덜 효율적
EXPLAIN SELECT * FROM sales 
WHERE sale_date BETWEEN '2023-01-01' AND '2023-01-31';

-- 이 쿼리는 인덱스를 거의 사용하지 못함
EXPLAIN SELECT * FROM sales 
WHERE sale_date = '2023-01-15';

-- 컬럼 순서가 중요한 이유
-- 좋은 예: 카디널리티가 높은 컬럼을 앞에 배치
CREATE INDEX idx_sales_date_region ON sales(sale_date, region);

-- 나쁜 예: 카디널리티가 낮은 컬럼을 앞에 배치
CREATE INDEX idx_sales_region_date ON sales(region, sale_date);
```

## GIN 인덱스

GIN(Generalized Inverted Index)은 배열, JSONB, 전체 텍스트 검색에 적합합니다.

```sql
-- 배열 인덱스
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tags TEXT[],
    attributes JSONB
);

-- GIN 인덱스 생성
CREATE INDEX idx_products_tags ON products USING GIN(tags);
CREATE INDEX idx_products_attributes ON products USING GIN(attributes);

-- 배열 검색
SELECT * FROM products WHERE 'electronics' = ANY(tags);
SELECT * FROM products WHERE tags @> ARRAY['electronics', 'computer'];

-- JSONB 검색
SELECT * FROM products WHERE attributes @> '{"brand": "Apple"}';
SELECT * FROM products WHERE attributes ? 'color';
SELECT * FROM products WHERE attributes ?| array['color', 'size'];

-- GIN 인덱스 옵션
CREATE INDEX idx_products_tags_fastupdate ON products USING GIN(tags) WITH (fastupdate = off);
```

## GiST 인덱스

GiST(Generalized Search Tree)는 지리 공간 데이터, 전체 텍스트 검색에 사용됩니다.

```sql
-- PostGIS 확장 설치
CREATE EXTENSION IF NOT EXISTS postgis;

-- 지리 데이터 테이블
CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    geom GEOMETRY(POINT, 4326)  -- 위도/경도
);

-- GiST 인덱스 생성
CREATE INDEX idx_locations_geom ON locations USING GIST(geom);

-- 지리 공간 검색
SELECT * FROM locations 
WHERE ST_DWithin(geom, ST_MakePoint(126.9780, 37.5665), 1000);  -- 1km 반경 내 검색

-- 전체 텍스트 검색을 위한 GiST 인덱스
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    search_vector TSVECTOR
);

-- 검색 벡터 생성
UPDATE articles SET search_vector = to_tsvector('korean', title || ' ' || content);

-- GiST 인덱스 생성
CREATE INDEX idx_articles_search ON articles USING GIST(search_vector);

-- 전체 텍스트 검색
SELECT * FROM articles 
WHERE search_vector @@ to_tsquery('korean', 'PostgreSQL & 데이터베이스');
```

## BRIN 인덱스

BRIN(Block Range Index)은 대용량 정렬된 데이터에 적합합니다.

```sql
-- 대용량 로그 테이블
CREATE TABLE logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    level VARCHAR(10) NOT NULL,
    message TEXT
);

-- BRIN 인덱스 생성 (타임스탬프로 정렬된 데이터에 적합)
CREATE INDEX idx_logs_timestamp ON logs USING BRIN(timestamp);

-- BRIN 인덱스 옵션
CREATE INDEX idx_logs_timestamp_pages ON logs USING BRIN(timestamp) WITH (pages_per_range = 128);
```

## 해시 인덱스

해시 인덱스는 정확히 일치하는 검색에만 사용됩니다.

```sql
-- 해시 인덱스 생성
CREATE INDEX idx_users_username_hash ON users USING HASH(username);

-- 해시 인덱스는 등호 검색에만 효과적
SELECT * FROM users WHERE username = 'john_doe';  -- 인덱스 사용

-- 해시 인덱스는 범위 검색에 사용되지 않음
SELECT * FROM users WHERE username > 'john';  -- 인덱스 사용 안됨
```

## 부분 인덱스

부분 인덱스는 특정 조건을 만족하는 행만 인덱싱하여 저장 공간을 절약하고 성능을 향상시킵니다.

```sql
-- 활성 사용자만 인덱싱
CREATE INDEX idx_active_users_email ON users(email) WHERE is_active = true;

-- 최근 주문만 인덱싱
CREATE INDEX idx_recent_orders ON orders(created_at) 
WHERE created_at > CURRENT_DATE - INTERVAL '90 days';

-- 특정 카테고리 제품만 인덱싱
CREATE INDEX idx_electronics_products ON products(name, price) 
WHERE category = 'electronics';

-- NULL이 아닌 값만 인덱싱
CREATE INDEX idx_non_null_phone ON customers(phone) WHERE phone IS NOT NULL;

-- 부분 인덱스 활용
-- 이 쿼리는 부분 인덱스를 효과적으로 사용
SELECT * FROM users WHERE email = 'test@example.com' AND is_active = true;

-- 이 쿼리는 부분 인덱스를 사용하지 못함
SELECT * FROM users WHERE email = 'test@example.com' AND is_active = false;
```

## 함수 기반 인덱스

함수 기반 인덱스는 컬럼 값을 변환한 결과를 인덱싱합니다.

```sql
-- 대소문자 무시 검색
CREATE INDEX idx_users_lower_username ON users(LOWER(username));
SELECT * FROM users WHERE LOWER(username) = 'john_doe';

-- 이메일 도메인 검색
CREATE INDEX idx_users_email_domain ON users(SUBSTRING(email FROM POSITION('@' IN email) + 1));
SELECT * FROM users WHERE SUBSTRING(email FROM POSITION('@' IN email) + 1) = 'gmail.com';

-- 날짜 부분 검색
CREATE INDEX idx_orders_month_year ON orders(EXTRACT(MONTH FROM created_at), EXTRACT(YEAR FROM created_at));
SELECT * FROM orders 
WHERE EXTRACT(MONTH FROM created_at) = 1 AND EXTRACT(YEAR FROM created_at) = 2023;

-- 문자열 일부 검색
CREATE INDEX idx_products_name_first3 ON products(LEFT(name, 3));
SELECT * FROM products WHERE LEFT(name, 3) = 'Sam';

-- 계산된 값 인덱싱
CREATE INDEX idx_products_discounted_price ON products((price * 0.9)) 
WHERE category = 'electronics' AND sale_price IS NULL;
```

## 인덱스 성능 분석

### 실행 계획 확인

```sql
-- 실행 계획 확인
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';

-- 상세 실행 계획 (실제 실행 시간 포함)
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- 버퍼 사용량 확인
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM users WHERE email = 'test@example.com';

-- 형식화된 실행 계획
EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM users WHERE email = 'test@example.com';
```

### 인덱스 사용 통계

```sql
-- 인덱스 사용 통계 확인
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 테이블 통계 확인
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch
FROM pg_stat_user_tables
ORDER BY seq_scan DESC;

-- 인덱스 크기 확인
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_indexes
WHERE schemaname = 'public';
```

### 인덱스 효율성 분석

```sql
-- 인덱스가 사용되지 않는 쿼리 찾기
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
WHERE query LIKE '%users%' AND query NOT LIKE '%EXPLAIN%'
ORDER BY total_time DESC;

-- 인덱스 재구성이 필요한 인덱스 찾기
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_stat_user_indexes
WHERE idx_scan = 0  -- 한 번도 사용되지 않은 인덱스
ORDER BY pg_relation_size(indexname::regclass) DESC;
```

## 인덱스 최적화 전략

### 인덱스 설계 원칙

```sql
-- 1. 자주 검색되는 컬럼에 인덱스 생성
CREATE INDEX idx_orders_customer_id ON orders(customer_id);

-- 2. 조인 조건에 사용되는 컬럼에 인덱스 생성
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- 3. 정렬에 사용되는 컬럼에 인덱스 생성
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);

-- 4. 카디널리티가 높은 컬럼을 우선적으로 인덱싱
-- 좋은 예: email (고유값에 가까움)
CREATE INDEX idx_users_email ON users(email);

-- 나쁜 예: gender (카디널리티가 낮음)
CREATE INDEX idx_users_gender ON users(gender);  -- 효과가 적음

-- 5. 복합 인덱스에서는 선택도가 높은 컬럼을 앞에 배치
CREATE INDEX idx_sales_date_amount ON sales(sale_date, amount);  -- 좋음
CREATE INDEX idx_sales_amount_date ON sales(amount, sale_date);  -- 나쁨
```

### 인덱스 유지보수

```sql
-- 인덱스 재구성
REINDEX INDEX idx_users_email;

-- 테이블의 모든 인덱스 재구성
REINDEX TABLE users;

-- 인덱스 동시 재구성 (잠금 최소화)
REINDEX INDEX CONCURRENTLY idx_users_email;

-- 인덱스 삭제
DROP INDEX idx_users_email;

-- 인덱스 이름 변경
ALTER INDEX idx_users_email RENAME TO idx_users_email_new;
```

## 실습: 인덱스 설계 연습

온라인 상점의 검색 성능을 최적화하는 인덱스를 설계해 봅시다.

```sql
-- 상점 테이블
CREATE TABLE shops (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    city VARCHAR(100),
    district VARCHAR(100),
    rating DECIMAL(3, 2) CHECK (rating >= 0 AND rating <= 5),
    review_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 상품 테이블
CREATE TABLE shop_products (
    id SERIAL PRIMARY KEY,
    shop_id INTEGER REFERENCES shops(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2),
    category VARCHAR(100),
    tags TEXT[],
    attributes JSONB,
    is_available BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 대량 데이터 삽입
INSERT INTO shops (name, category, city, district, rating, review_count)
SELECT 
    'Shop ' || i,
    ARRAY['Restaurant', 'Cafe', 'Retail', 'Service'][floor(random() * 4) + 1],
    ARRAY['Seoul', 'Busan', 'Incheon', 'Daegu'][floor(random() * 4) + 1],
    'District ' || (floor(random() * 20) + 1),
    round((random() * 4 + 1)::numeric, 2),
    floor(random() * 1000)
FROM generate_series(1, 10000) i;

-- 인덱스 설계
-- 1. 상점 검색을 위한 인덱스
CREATE INDEX idx_shops_category ON shops(category);
CREATE INDEX idx_shops_city_district ON shops(city, district);
CREATE INDEX idx_shops_rating ON shops(rating DESC);
CREATE INDEX idx_shops_active ON shops(id) WHERE is_active = true;

-- 2. 상품 검색을 위한 인덱스
CREATE INDEX idx_shop_products_shop_id ON shop_products(shop_id);
CREATE INDEX idx_shop_products_category ON shop_products(category);
CREATE INDEX idx_shop_products_price ON shop_products(price);
CREATE INDEX idx_shop_products_available ON shop_products(id) WHERE is_available = true;

-- 3. 배열 및 JSONB 검색을 위한 인덱스
CREATE INDEX idx_shop_products_tags ON shop_products USING GIN(tags);
CREATE INDEX idx_shop_products_attributes ON shop_products USING GIN(attributes);

-- 4. 함수 기반 인덱스
CREATE INDEX idx_shops_lower_name ON shops(LOWER(name));
CREATE INDEX idx_shop_products_lower_name ON shop_products(LOWER(name));

-- 5. 복합 인덱스
CREATE INDEX idx_shops_category_rating ON shops(category, rating DESC);
CREATE INDEX idx_shop_products_shop_category ON shop_products(shop_id, category);

-- 쿼리와 인덱스 사용 확인
EXPLAIN ANALYZE 
SELECT s.*, sp.name as product_name, sp.price
FROM shops s
JOIN shop_products sp ON s.id = sp.shop_id
WHERE s.city = 'Seoul' 
  AND s.category = 'Restaurant'
  AND s.rating >= 4.0
  AND sp.is_available = true
ORDER BY s.rating DESC, sp.price ASC
LIMIT 20;

-- 전체 텍스트 검색을 위한 인덱스
ALTER TABLE shops ADD COLUMN search_vector TSVECTOR;
UPDATE shops SET search_vector = to_tsvector('korean', name || ' ' || COALESCE(description, ''));
CREATE INDEX idx_shops_search ON shops USING GIN(search_vector);

-- 전체 텍스트 검색
SELECT * FROM shops 
WHERE search_vector @@ to_tsquery('korean', '서울 & 레스토랑')
  AND is_active = true;
```

## 요약

효과적인 인덱싱을 위한 핵심 원칙:

1. **적절한 인덱스 타입 선택**: B-Tree, GIN, GiST, BRIN, 해시
2. **복합 인덱스 최적화**: 컬럼 순서와 선택도 고려
3. **부분 인덱스 활용**: 특정 조건을 만족하는 데이터만 인덱싱
4. **함수 기반 인덱스**: 변환된 값으로 검색 최적화
5. **정기적인 성능 분석**: 사용하지 않는 인덱스 식별 및 제거
6. **인덱스 유지보수**: 정기적인 재구성으로 성능 유지

다음 섹션에서는 PostgreSQL 쿼리 최적화 기법에 대해 알아보겠습니다.

## 추가 자료

- [PostgreSQL 문서: 인덱스](https://www.postgresql.org/docs/current/indexes.html)
- [PostgreSQL 인덱스 타입 가이드](https://www.postgresql.org/docs/current/indexes-types.html)
- [PostgreSQL 성능 튜닝 가이드](https://wiki.postgresql.org/wiki/Tuning_Your_PostgreSQL_Server)