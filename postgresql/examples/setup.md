# 예제 데이터베이스 설정 가이드

## 개요

이 가이드는 PostgreSQL 베스트 프랙티스 강의자료에서 사용하는 예제 데이터베이스를 설정하는 방법을 설명합니다. 예제 데이터베이스는 전자상거래 시스템을 모델링하며, 다양한 PostgreSQL 기능과 베스트 프랙티스를 실습하는 데 사용됩니다.

## 시스템 요구사항

- PostgreSQL 13 이상
- 최소 2GB RAM
- 최소 5GB 디스크 공간
- psql 클라이언트 도구

## 데이터베이스 생성

### 1. 데이터베이스 생성

```bash
# PostgreSQL에 접속
psql -U postgres

# 데이터베이스 생성
CREATE DATABASE lecture_db;

# 데이터베이스 접속
\c lecture_db

# 데이터베이스 목록 확인
\l
```

### 2. 확장 모듈 활성화

```sql
-- 필요한 확장 모듈 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 확장 모듈 확인
\dx
```

## 스키마 및 데이터 로드

### 1. 스키마 생성

```bash
# 스키마 파일 실행
psql -U postgres -d lecture_db -f schema.sql

# 또는 psql 내에서 실행
\i schema.sql
```

### 2. 샘플 데이터 로드

```bash
# 샘플 데이터 파일 실행
psql -U postgres -d lecture_db -f sample-data.sql

# 또는 psql 내에서 실행
\i sample-data.sql
```

### 3. 데이터 확인

```sql
-- 테이블 목록 확인
\dt

-- 데이터 확인 쿼리
SELECT 
    'Users' as table_name, COUNT(*) as record_count FROM users
UNION ALL
SELECT 'Categories', COUNT(*) FROM categories
UNION ALL
SELECT 'Products', COUNT(*) FROM products
UNION ALL
SELECT 'Orders', COUNT(*) FROM orders
UNION ALL
SELECT 'Order Items', COUNT(*) FROM order_items
UNION ALL
SELECT 'Payments', COUNT(*) FROM payments
UNION ALL
SELECT 'Product Reviews', COUNT(*) FROM product_reviews
ORDER BY table_name;
```

## 데이터베이스 구조

### 주요 테이블

1. **users**: 사용자 정보
2. **categories**: 제품 카테고리
3. **products**: 제품 정보
4. **orders**: 주문 정보
5. **order_items**: 주문 항목
6. **payments**: 결제 정보
7. **product_reviews**: 제품 리뷰
8. **shopping_cart**: 장바구니
9. **wishlist**: 위시리스트
10. **coupons**: 쿠폰 정보

### 주요 뷰

1. **product_details**: 제품 상세 정보
2. **order_details**: 주문 상세 정보
3. **user_statistics**: 사용자 통계
4. **product_statistics**: 제품 통계

### 주요 함수

1. **check_product_stock()**: 제품 재고 확인
2. **get_user_orders()**: 사용자 주문 내역 조회
3. **get_popular_products()**: 인기 제품 조회
4. **get_category_statistics()**: 카테고리별 통계
5. **update_order_status()**: 주문 상태 업데이트

## 실습 시나리오

### 1. 기본 쿼리 실습

```sql
-- 모든 사용자 조회
SELECT * FROM users LIMIT 10;

-- 활성 사용자만 조회
SELECT * FROM users WHERE is_active = true;

-- 특정 카테고리 제품 조회
SELECT p.name, p.price, c.name as category
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE c.name = 'electronics';
```

### 2. 조인 실습

```sql
-- 사용자별 주문 내역
SELECT u.username, o.order_number, o.total_amount, o.status
FROM users u
JOIN orders o ON u.id = o.user_id
ORDER BY o.created_at DESC;

-- 제품별 리뷰 평균
SELECT p.name, COUNT(pr.id) as review_count, AVG(pr.rating) as avg_rating
FROM products p
LEFT JOIN product_reviews pr ON p.id = pr.product_id
GROUP BY p.id, p.name
ORDER BY avg_rating DESC;
```

### 3. 집계 함수 실습

```sql
-- 카테고리별 제품 수
SELECT c.name, COUNT(p.id) as product_count
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
GROUP BY c.id, c.name;

-- 월별 매출 통계
SELECT 
    DATE_TRUNC('month', created_at) as month,
    COUNT(*) as order_count,
    SUM(total_amount) as total_revenue
FROM orders
WHERE status = 'delivered'
GROUP BY DATE_TRUNC('month', created_at)
ORDER BY month;
```

### 4. 서브쿼리 실습

```sql
-- 평균보다 높은 가격의 제품
SELECT name, price
FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- 리뷰가 있는 제품만 조회
SELECT name, price
FROM products
WHERE id IN (SELECT DISTINCT product_id FROM product_reviews);
```

### 5. 윈도우 함수 실습

```sql
-- 제품별 판매 순위
SELECT 
    p.name,
    COUNT(oi.id) as sales_count,
    RANK() OVER (ORDER BY COUNT(oi.id) DESC) as sales_rank
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name
ORDER BY sales_rank;

-- 사용자별 구매 금액 순위
SELECT 
    u.username,
    SUM(o.total_amount) as total_spent,
    RANK() OVER (ORDER BY SUM(o.total_amount) DESC) as spending_rank
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.status = 'delivered'
GROUP BY u.id, u.username
ORDER BY spending_rank;
```

## 권한 설정

### 1. 읽기 전용 사용자

```sql
-- 읽기 전용 사용자 생성
CREATE USER readonly_user WITH PASSWORD 'readonly_password';

-- 권한 부여
GRANT CONNECT ON DATABASE lecture_db TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readonly_user;

-- 기본 권한 설정
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;
```

### 2. 애플리케이션 사용자

```sql
-- 애플리케이션 사용자 생성
CREATE USER app_user WITH PASSWORD 'app_password';

-- 권한 부여
GRANT CONNECT ON DATABASE lecture_db TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- 기본 권한 설정
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO app_user;
```

## 백업 및 복구

### 1. 백업

```bash
# 전체 데이터베이스 백업
pg_dump -U postgres -d lecture_db > lecture_db_backup.sql

# 커스텀 형식 백업
pg_dump -U postgres -d lecture_db -Fc > lecture_db_backup.dump

# 특정 테이블만 백업
pg_dump -U postgres -d lecture_db -t users > users_backup.sql
```

### 2. 복구

```bash
# SQL 파일로 복구
psql -U postgres -d lecture_db < lecture_db_backup.sql

# 커스텀 형식으로 복구
pg_restore -U postgres -d lecture_db lecture_db_backup.dump

-- 데이터베이스 삭제 후 재생성
DROP DATABASE IF EXISTS lecture_db;
CREATE DATABASE lecture_db;
psql -U postgres -d lecture_db < lecture_db_backup.sql
```

## 성능 튜닝

### 1. 인덱스 확인

```sql
-- 인덱스 목록 확인
\di

-- 인덱스 사용 통계
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### 2. 쿼리 성능 분석

```sql
-- 실행 계획 확인
EXPLAIN ANALYZE 
SELECT u.username, COUNT(o.id) as order_count
FROM users u
JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username
ORDER BY order_count DESC;

-- 느린 쿼리 확인
SELECT 
    query,
    calls,
    total_time,
    mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## 문제 해결

### 1. 일반적인 오류

**오류**: `permission denied for relation`
**해결**: 적절한 권한 부여
```sql
GRANT SELECT ON table_name TO user_name;
```

**오류**: `function does not exist`
**해결**: 확장 모듈 활성화
```sql
CREATE EXTENSION IF NOT EXISTS "extension_name";
```

**오류**: `duplicate key value violates unique constraint`
**해결**: 중복 데이터 확인 및 처리
```sql
SELECT column_name, COUNT(*) 
FROM table_name 
GROUP BY column_name 
HAVING COUNT(*) > 1;
```

### 2. 성능 문제

**문제**: 쿼리 실행이 느림
**해결**: 인덱스 생성 및 실행 계획 분석
```sql
EXPLAIN ANALYZE SELECT ...;
CREATE INDEX index_name ON table_name(column_name);
```

**문제**: 메모리 부족
**해결**: work_mem 설정 증가
```sql
SET work_mem = '16MB';
```

## 추가 리소스

- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
- [psql 명령어 참조](https://www.postgresql.org/docs/current/app-psql.html)
- [SQL 문법 참조](https://www.postgresql.org/docs/current/sql-syntax.html)

## 연락처

문제가 있거나 질문이 있으시면 강의자료의 GitHub 저장소에 이슈를 등록해주세요.