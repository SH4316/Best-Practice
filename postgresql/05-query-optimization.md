# 쿼리 최적화 기법

## 학습 목표

- PostgreSQL 쿼리 실행 계획 이해
- 쿼리 성능 저하 원인 분석 방법 학습
- 효율적인 쿼리 작성 기법 습득
- 고급 최적화 기법과 도구 활용법 이해

## 쿼리 실행 계획 분석

### EXPLAIN 명령어

EXPLAIN 명령어는 PostgreSQL이 쿼리를 어떻게 실행할지 보여주는 실행 계획을 제공합니다.

```sql
-- 기본 실행 계획
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';

-- 상세 실행 계획 (실제 실행 시간 포함)
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- 버퍼 사용량 포함
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM users WHERE email = 'test@example.com';

-- 형식화된 출력
EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM users WHERE email = 'test@example.com';

-- 비용 정보 포함
EXPLAIN (ANALYZE, VERBOSE, COSTS) SELECT * FROM users WHERE email = 'test@example.com';
```

### 실행 계획 읽는 법

```sql
-- 예제 테이블 생성
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(20) NOT NULL
);

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    city VARCHAR(100)
);

-- 인덱스 생성
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_customers_city ON customers(city);

-- 조인 쿼리 실행 계획
EXPLAIN ANALYZE 
SELECT c.name, c.email, COUNT(o.id) as order_count, SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE c.city = 'Seoul'
  AND o.order_date >= '2023-01-01'
GROUP BY c.id, c.name, c.email
HAVING COUNT(o.id) > 5
ORDER BY total_spent DESC;
```

### 주요 실행 노드 이해

```sql
-- Seq Scan: 순차 스캔 (전체 테이블 스캔)
EXPLAIN SELECT * FROM orders WHERE total_amount > 100000;

-- Index Scan: 인덱스 스캔
EXPLAIN SELECT * FROM orders WHERE customer_id = 123;

-- Index Only Scan: 인덱스만 스캔 (테이블 접근 없음)
EXPLAIN SELECT id FROM orders WHERE customer_id = 123;

-- Bitmap Heap Scan + Bitmap Index Scan: 비트맵 스캔
EXPLAIN SELECT * FROM orders WHERE customer_id = 123 AND order_date >= '2023-01-01';

-- Hash Join: 해시 조인
EXPLAIN SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id;

-- Nested Loop: 중첩 루프 조인
EXPLAIN SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.city = 'Seoul';

-- Merge Join: 병합 조인
EXPLAIN SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id 
ORDER BY c.id, o.id;
```

## 쿼리 작성 최적화

### SELECT 절 최적화

```sql
-- 나쁜 예: 불필요한 컬럼 선택
SELECT * FROM users WHERE id = 1;

-- 좋은 예: 필요한 컬럼만 선택
SELECT id, username, email FROM users WHERE id = 1;

-- 나쁜 예: 불필요한 계산 반복
SELECT *, price * 1.1 as price_with_tax FROM products;

-- 좋은 예: 필요한 계산만 수행
SELECT id, name, price, price * 1.1 as price_with_tax FROM products;

-- 나쁜 예: DISTINCT 불필요 사용
SELECT DISTINCT category FROM products WHERE price > 1000;

-- 좋은 예: GROUP BY 사용
SELECT category FROM products WHERE price > 1000 GROUP BY category;
```

### WHERE 절 최적화

```sql
-- 나쁜 예: 함수 사용으로 인덱스 미사용
SELECT * FROM users WHERE LOWER(email) = 'test@example.com';

-- 좋은 예: 함수 기반 인덱스 또는 원본 값 비교
-- 함수 기반 인덱스 생성: CREATE INDEX idx_users_lower_email ON users(LOWER(email));
SELECT * FROM users WHERE email = 'test@example.com';

-- 나쁜 예: OR 조건으로 인덱스 미사용
SELECT * FROM products WHERE category = 'electronics' OR price > 1000;

-- 좋은 예: UNION 사용 또는 복합 인덱스 활용
SELECT * FROM products WHERE category = 'electronics'
UNION
SELECT * FROM products WHERE price > 1000;

-- 나쁜 예: LIKE 앞에 와일드카드 사용
SELECT * FROM products WHERE name LIKE '%laptop%';

-- 좋은 예: 전체 텍스트 검색 또는 접두사 검색
SELECT * FROM products WHERE name LIKE 'laptop%';
-- 또는 전체 텍스트 검색 인덱스 사용
SELECT * FROM products WHERE search_vector @@ to_tsquery('laptop');

-- 나쁜 예: NULL 비교
SELECT * FROM customers WHERE phone = NULL;

-- 좋은 예: IS NULL 사용
SELECT * FROM customers WHERE phone IS NULL;
```

### 조인 최적화

```sql
-- 나쁜 예: CROSS JOIN 후 WHERE 필터링
SELECT c.name, o.total_amount
FROM customers c, orders o
WHERE c.id = o.customer_id AND c.city = 'Seoul';

-- 좋은 예: INNER JOIN 사용
SELECT c.name, o.total_amount
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
WHERE c.city = 'Seoul';

-- 나쁜 예: 서브쿼리 사용
SELECT * FROM customers 
WHERE id IN (SELECT customer_id FROM orders WHERE total_amount > 1000);

-- 좋은 예: JOIN 사용
SELECT DISTINCT c.* 
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.total_amount > 1000;

-- LATERAL JOIN 사용 예제
-- 각 고객별 최근 주문 정보 가져오기
SELECT c.name, c.email, recent_order.*
FROM customers c
LEFT JOIN LATERAL (
    SELECT * FROM orders 
    WHERE customer_id = c.id 
    ORDER BY order_date DESC 
    LIMIT 1
) recent_order ON true;
```

### 서브쿼리 최적화

```sql
-- 나쁜 예: 상관 서브쿼리
SELECT c.name, 
       (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) as order_count
FROM customers c;

-- 좋은 예: JOIN 사용
SELECT c.name, COUNT(o.id) as order_count
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name;

-- 나쁜 예: 비상관 서브쿼리
SELECT * FROM products 
WHERE price > (SELECT AVG(price) FROM products);

-- 좋은 예: WITH 절 사용
WITH avg_price AS (
    SELECT AVG(price) as avg FROM products
)
SELECT p.* 
FROM products p, avg_price ap
WHERE p.price > ap.avg;

-- CTE (Common Table Expression) 사용
WITH customer_stats AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        SUM(total_amount) as total_spent
    FROM orders
    GROUP BY customer_id
)
SELECT c.name, cs.order_count, cs.total_spent
FROM customers c
JOIN customer_stats cs ON c.id = cs.customer_id
WHERE cs.order_count > 5;
```

## 데이터 집계 최적화

### GROUP BY 최적화

```sql
-- 나쁜 예: 불필요한 컬럼 포함
SELECT c.id, c.name, c.email, COUNT(o.id) as order_count
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.email;

-- 좋은 예: 필요한 컬럼만 그룹화
SELECT c.id, c.name, COUNT(o.id) as order_count
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name;

-- 나쁜 예: HAVING 절에서 함수 사용
SELECT category, AVG(price) as avg_price
FROM products
GROUP BY category
HAVING AVG(price) > 1000;

-- 좋은 예: WHERE 절에서 필터링
SELECT category, AVG(price) as avg_price
FROM products
WHERE price > 1000
GROUP BY category;
```

### 윈도우 함수 최적화

```sql
-- 윈도우 함수를 사용한 순위 계산
SELECT 
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank_in_dept,
    LAG(salary) OVER (PARTITION BY department ORDER BY salary DESC) as prev_salary
FROM employees;

-- 윈도우 함수와 집계 결합
SELECT 
    department,
    name,
    salary,
    AVG(salary) OVER (PARTITION BY department) as dept_avg,
    salary - AVG(salary) OVER (PARTITION BY department) as diff_from_avg
FROM employees;
```

## 고급 최적화 기법

### 파티셔닝 활용

```sql
-- 파티션 테이블 생성
CREATE TABLE orders_partitioned (
    id SERIAL,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(20) NOT NULL
) PARTITION BY RANGE (order_date);

-- 파티션 생성
CREATE TABLE orders_2023_q1 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2023-01-01') TO ('2023-04-01');

CREATE TABLE orders_2023_q2 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2023-04-01') TO ('2023-07-01');

CREATE TABLE orders_2023_q3 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2023-07-01') TO ('2023-10-01');

CREATE TABLE orders_2023_q4 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2023-10-01') TO ('2024-01-01');

-- 파티션 프루닝 활용 쿼리
EXPLAIN SELECT * FROM orders_partitioned 
WHERE order_date BETWEEN '2023-02-01' AND '2023-02-28';
```

### 머티리얼라이즈드 뷰 활용

```sql
-- 머티리얼라이즈드 뷰 생성
CREATE MATERIALIZED VIEW customer_summary AS
SELECT 
    c.id,
    c.name,
    c.email,
    COUNT(o.id) as order_count,
    SUM(o.total_amount) as total_spent,
    MAX(o.order_date) as last_order_date
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.email
WITH DATA;

-- 머티리얼라이즈드 뷰 조회
SELECT * FROM customer_summary WHERE order_count > 10;

-- 머티리얼라이즈드 뷰 갱신
REFRESH MATERIALIZED VIEW customer_summary;

-- 동시 갱신 (잠금 최소화)
REFRESH MATERIALIZED VIEW CONCURRENTLY customer_summary;

-- 고유 인덱스 생성 (동시 갱신을 위해 필요)
CREATE UNIQUE INDEX idx_customer_summary_id ON customer_summary(id);
```

### 테이블 힌트 사용

```sql
-- 인덱스 힌트 사용 (PostgreSQL 9.1+)
-- 특정 인덱스 강제 사용
SELECT * FROM orders /*+ Index(orders idx_orders_date) */
WHERE order_date >= '2023-01-01';

-- 조인 순서 힌트
SELECT /*+ Leading(c o) */ c.name, o.total_amount
FROM customers c
JOIN orders o ON c.id = o.customer_id;

-- 조인 방법 힌트
SELECT /*+ HashJoin(c o) */ c.name, o.total_amount
FROM customers c
JOIN orders o ON c.id = o.customer_id;
```

## 쿼리 성능 모니터링

### pg_stat_statements 활용

```sql
-- pg_stat_statements 확장 활성화
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 쿼리 통계 확인
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- 특정 테이블 관련 쿼리 확인
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
WHERE query LIKE '%orders%'
ORDER BY total_time DESC;
```

### 성능 분석 도구

```sql
-- 느린 쿼리 로그 설정
-- postgresql.conf 파일에서 설정
-- log_min_duration_statement = 1000  -- 1초 이상 걸리는 쿼리 로그
-- log_statement = 'all'              -- 모든 쿼리 로그

-- 자동 백그라운드 분석 활성화
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 쿼리 플랜 캐시 확인
SELECT 
    query,
    calls,
    plans,
    total_plan_time,
    mean_plan_time
FROM pg_stat_statements
WHERE plans > 0
ORDER BY total_plan_time DESC;
```

## 실습: 쿼리 최적화 연습

복잡한 비즈니스 쿼리를 최적화하는 연습을 해봅시다.

```sql
-- 예제 데이터베이스
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(50) NOT NULL,
    position VARCHAR(50),
    salary DECIMAL(10, 2),
    hire_date DATE,
    manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12, 2),
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE employee_projects (
    employee_id INTEGER REFERENCES employees(id),
    project_id INTEGER REFERENCES projects(id),
    role VARCHAR(50),
    hours_worked INTEGER,
    PRIMARY KEY (employee_id, project_id)
);

-- 인덱스 생성
CREATE INDEX idx_employees_department ON employees(department);
CREATE INDEX idx_employees_manager ON employees(manager_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_employee_projects_employee ON employee_projects(employee_id);
CREATE INDEX idx_employee_projects_project ON employee_projects(project_id);

-- 최적화 전 쿼리
EXPLAIN ANALYZE
SELECT 
    e.name as employee_name,
    e.department,
    e.position,
    e.salary,
    COUNT(DISTINCT ep.project_id) as project_count,
    SUM(ep.hours_worked) as total_hours,
    AVG(p.budget) as avg_project_budget
FROM employees e
LEFT JOIN employee_projects ep ON e.id = ep.employee_id
LEFT JOIN projects p ON ep.project_id = p.id
WHERE e.hire_date >= '2020-01-01'
  AND (p.status = 'active' OR p.status IS NULL)
GROUP BY e.id, e.name, e.department, e.position, e.salary
HAVING COUNT(DISTINCT ep.project_id) > 2
ORDER BY total_hours DESC;

-- 최적화 후 쿼리 (CTE 사용)
EXPLAIN ANALYZE
WITH active_projects AS (
    SELECT id, budget FROM projects WHERE status = 'active'
),
employee_stats AS (
    SELECT 
        e.id,
        e.name,
        e.department,
        e.position,
        e.salary,
        COUNT(DISTINCT ep.project_id) as project_count,
        SUM(ep.hours_worked) as total_hours
    FROM employees e
    LEFT JOIN employee_projects ep ON e.id = ep.employee_id
    LEFT JOIN active_projects ap ON ep.project_id = ap.id
    WHERE e.hire_date >= '2020-01-01'
    GROUP BY e.id, e.name, e.department, e.position, e.salary
)
SELECT 
    es.*,
    AVG(ap.budget) as avg_project_budget
FROM employee_stats es
LEFT JOIN employee_projects ep ON es.id = ep.employee_id
LEFT JOIN active_projects ap ON ep.project_id = ap.id
WHERE es.project_count > 2
GROUP BY es.id, es.name, es.department, es.position, es.salary, es.project_count, es.total_hours
ORDER BY es.total_hours DESC;
```

## 쿼리 최적화 체크리스트

### 기본 검토 항목

1. **실행 계획 확인**: `EXPLAIN ANALYZE`로 비효율적인 부분 식별
2. **인덱스 활용**: 적절한 인덱스가 사용되는지 확인
3. **데이터 타입**: 일치하는 데이터 타입으로 비교
4. **함수 사용**: 인덱스를 방해하는 함수 사용 최소화

### 고급 검토 항목

1. **조인 전략**: 적절한 조인 방법 선택
2. **서브쿼리 vs JOIN**: 상황에 맞는 최적화 방법 선택
3. **파티셔닝**: 대용량 데이터에 파티셔닝 활용
4. **머티리얼라이즈드 뷰**: 복잡한 집계 쿼리에 활용

## 요약

효과적인 쿼리 최적화를 위한 핵심 원칙:

1. **실행 계획 분석**: 쿼리 실행 과정 정확히 이해
2. **인덱스 활용**: 적절한 인덱스 생성과 활용
3. **효율적인 조인**: 상황에 맞는 조인 전략 선택
4. **불필요한 작업 제거**: 불필요한 컬럼, 계산, 정렬 제거
5. **정기적인 모니터링**: 성능 저하 쿼리 식별 및 개선
6. **고급 기법 활용**: 파티셔닝, 머티리얼라이즈드 뷰 등

다음 섹션에서는 PostgreSQL 트랜잭션 관리와 동시성 제어에 대해 알아보겠습니다.

## 추가 자료

- [PostgreSQL 문서: 성능 튜닝](https://www.postgresql.org/docs/current/performance-tips.html)
- [PostgreSQL 실행 계획 가이드](https://www.postgresql.org/docs/current/using-explain.html)
- [pg_stat_statements 문서](https://www.postgresql.org/docs/current/pgstatstatements.html)