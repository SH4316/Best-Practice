# PostgreSQL 기초 실습 문제

## 개요

이 실습 문제들은 PostgreSQL의 기본 기능과 베스트 프랙티스를 연습하기 위해 설계되었습니다. 예제 데이터베이스를 이미 설정했다고 가정하고 진행합니다.

## 실습 환경 설정

```sql
-- 실습을 위해 필요한 확장 모듈 확인
SELECT * FROM pg_extension;

-- 예제 데이터베이스 연결
\c lecture_db

-- 테이블 목록 확인
\dt

-- 데이터 확인
SELECT 'Users' as table_name, COUNT(*) as record_count FROM users
UNION ALL
SELECT 'Products', COUNT(*) FROM products
UNION ALL
SELECT 'Orders', COUNT(*) FROM orders
UNION ALL
SELECT 'Categories', COUNT(*) FROM categories;
```

## 실습 1: 기본 쿼리

### 문제 1.1: 기본 SELECT 문

**목표**: 기본적인 SELECT 문을 사용하여 데이터 조회하기

**과제**:
1. 모든 사용자 정보를 조회하세요.
2. 활성 사용자만 조회하세요.
3. 특정 역할(admin)을 가진 사용자만 조회하세요.

**예상 결과**:
```sql
-- 1. 모든 사용자 정보
SELECT * FROM users;

-- 2. 활성 사용자만
SELECT * FROM users WHERE is_active = true;

-- 3. 관리자 사용자만
SELECT * FROM users WHERE role = 'admin';
```

### 문제 1.2: 정렬과 필터링

**목표**: ORDER BY와 WHERE 절을 사용하여 데이터 정렬 및 필터링

**과제**:
1. 가장 최근에 가입한 사용자 5명을 조회하세요.
2. 가격이 높은 제품 10개를 조회하세요.
3. 2023년에 생성된 주문을 조회하세요.

**예상 결과**:
```sql
-- 1. 최근 가입자 5명
SELECT username, email, created_at 
FROM users 
ORDER BY created_at DESC 
LIMIT 5;

-- 2. 가격이 높은 제품 10개
SELECT name, price, category_id 
FROM products 
ORDER BY price DESC 
LIMIT 10;

-- 3. 2023년 주문
SELECT order_number, total_amount, created_at 
FROM orders 
WHERE created_at >= '2023-01-01' AND created_at < '2024-01-01'
ORDER BY created_at;
```

### 문제 1.3: 집계 함수

**목표**: COUNT, SUM, AVG 등 집계 함수 사용하기

**과제**:
1. 전체 사용자 수를 계산하세요.
2. 전체 제품의 평균 가격을 계산하세요.
3. 각 카테고리별 제품 수를 계산하세요.

**예상 결과**:
```sql
-- 1. 전체 사용자 수
SELECT COUNT(*) as total_users FROM users;

-- 2. 제품 평균 가격
SELECT AVG(price) as average_price FROM products;

-- 3. 카테고리별 제품 수
SELECT c.name, COUNT(p.id) as product_count
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
GROUP BY c.id, c.name
ORDER BY product_count DESC;
```

## 실습 2: 조인

### 문제 2.1: INNER JOIN

**목표**: INNER JOIN을 사용하여 여러 테이블의 데이터 결합

**과제**:
1. 모든 주문과 해당 주문을 한 사용자 정보를 조회하세요.
2. 모든 제품과 해당 카테고리 정보를 조회하세요.
3. 모든 주문 항목과 해당 제품 정보를 조회하세요.

**예상 결과**:
```sql
-- 1. 주문과 사용자 정보
SELECT o.order_number, o.total_amount, u.username, u.email
FROM orders o
INNER JOIN users u ON o.user_id = u.id
ORDER BY o.created_at DESC;

-- 2. 제품과 카테고리 정보
SELECT p.name, p.price, c.name as category_name
FROM products p
INNER JOIN categories c ON p.category_id = c.id
ORDER BY c.name, p.name;

-- 3. 주문 항목과 제품 정보
SELECT oi.quantity, oi.unit_price, p.name as product_name
FROM order_items oi
INNER JOIN products p ON oi.product_id = p.id
ORDER BY oi.created_at DESC;
```

### 문제 2.2: LEFT JOIN

**목표**: LEFT JOIN을 사용하여 한쪽 테이블의 모든 데이터 조회

**과제**:
1. 모든 사용자와 해당 사용자의 주문 수를 조회하세요.
2. 모든 제품과 해당 제품의 리뷰 수를 조회하세요.
3. 모든 카테고리와 해당 카테고리의 제품 수를 조회하세요.

**예상 결과**:
```sql
-- 1. 사용자별 주문 수
SELECT u.username, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username, u.email
ORDER BY order_count DESC;

-- 2. 제품별 리뷰 수
SELECT p.name, COUNT(pr.id) as review_count
FROM products p
LEFT JOIN product_reviews pr ON p.id = pr.product_id
GROUP BY p.id, p.name
ORDER BY review_count DESC;

-- 3. 카테고리별 제품 수
SELECT c.name, COUNT(p.id) as product_count
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
GROUP BY c.id, c.name
ORDER BY product_count DESC;
```

## 실습 3: 데이터 타입과 함수

### 문제 3.1: 문자열 함수

**목표**: PostgreSQL 문자열 함수 사용하기

**과제**:
1. 사용자 이름을 대문자로 변환하여 조회하세요.
2. 제품 이름의 길이를 계산하여 조회하세요.
3. 이메일 도메인만 추출하여 조회하세요.

**예상 결과**:
```sql
-- 1. 사용자 이름 대문자 변환
SELECT username, UPPER(username) as upper_username, email
FROM users;

-- 2. 제품 이름 길이
SELECT name, LENGTH(name) as name_length
FROM products
ORDER BY name_length DESC;

-- 3. 이메일 도메인 추출
SELECT email, SUBSTRING(email FROM POSITION('@' IN email) + 1) as domain
FROM users
ORDER BY domain;
```

### 문제 3.2: 날짜/시간 함수

**목표**: 날짜/시간 함수를 사용하여 데이터 분석

**과제**:
1. 각 사용자의 가입 기간을 계산하세요.
2. 월별 주문 수를 계산하세요.
3. 최근 30일 이내에 생성된 주문을 조회하세요.

**예상 결과**:
```sql
-- 1. 사용자 가입 기간
SELECT username, created_at, 
       CURRENT_DATE - created_at::date as membership_days
FROM users
ORDER BY created_at;

-- 2. 월별 주문 수
SELECT DATE_TRUNC('month', created_at) as month,
       COUNT(*) as order_count,
       SUM(total_amount) as total_revenue
FROM orders
GROUP BY DATE_TRUNC('month', created_at)
ORDER BY month;

-- 3. 최근 30일 이내 주문
SELECT order_number, total_amount, created_at
FROM orders
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY created_at DESC;
```

### 문제 3.3: 조건부 표현식

**목표**: CASE 문을 사용하여 조건부 데이터 처리

**과제**:
1. 제품 가격에 따라 등급을 부여하세요.
2. 주문 상태에 따라 한글 설명을 추가하세요.
3. 사용자 역할에 따라 권한 레벨을 표시하세요.

**예상 결과**:
```sql
-- 1. 제품 가격 등급
SELECT name, price,
       CASE 
           WHEN price < 50000 THEN '저가'
           WHEN price < 200000 THEN '중가'
           ELSE '고가'
       END as price_level
FROM products
ORDER BY price DESC;

-- 2. 주문 상태 한글 설명
SELECT order_number, status,
       CASE status
           WHEN 'pending' THEN '대기중'
           WHEN 'processing' THEN '처리중'
           WHEN 'shipped' THEN '배송중'
           WHEN 'delivered' THEN '배송완료'
           WHEN 'cancelled' THEN '취소됨'
           ELSE '알 수 없음'
       END as status_korean
FROM orders
ORDER BY created_at DESC;

-- 3. 사용자 권한 레벨
SELECT username, role,
       CASE role
           WHEN 'admin' THEN 100
           WHEN 'vendor' THEN 50
           WHEN 'customer' THEN 10
           ELSE 0
       END as permission_level
FROM users
ORDER BY permission_level DESC;
```

## 실습 4: 서브쿼리

### 문제 4.1: 스칼라 서브쿼리

**목표**: 단일 값을 반환하는 서브쿼리 사용하기

**과제**:
1. 평균 가격보다 비싼 제품을 조회하세요.
2. 가장 많은 주문을 한 사용자를 조회하세요.
3. 가장 최근에 주문한 사용자를 조회하세요.

**예상 결과**:
```sql
-- 1. 평균 가격보다 비싼 제품
SELECT name, price
FROM products
WHERE price > (SELECT AVG(price) FROM products)
ORDER BY price DESC;

-- 2. 가장 많은 주문을 한 사용자
SELECT username, email
FROM users
WHERE id = (
    SELECT user_id
    FROM orders
    GROUP BY user_id
    ORDER BY COUNT(*) DESC
    LIMIT 1
);

-- 3. 가장 최근에 주문한 사용자
SELECT username, email
FROM users
WHERE id = (
    SELECT user_id
    FROM orders
    ORDER BY created_at DESC
    LIMIT 1
);
```

### 문제 4.2: IN 서브쿼리

**목표**: IN 연산자와 서브쿼리 사용하기

**과제**:
1. 리뷰가 있는 제품을 조회하세요.
2. 주문을 한 적이 있는 사용자를 조회하세요.
3. 판매된 제품을 조회하세요.

**예상 결과**:
```sql
-- 1. 리뷰가 있는 제품
SELECT name, price
FROM products
WHERE id IN (SELECT DISTINCT product_id FROM product_reviews)
ORDER BY name;

-- 2. 주문을 한 적이 있는 사용자
SELECT username, email
FROM users
WHERE id IN (SELECT DISTINCT user_id FROM orders)
ORDER BY username;

-- 3. 판매된 제품
SELECT name, price
FROM products
WHERE id IN (SELECT DISTINCT product_id FROM order_items)
ORDER BY name;
```

## 실습 5: 데이터 수정

### 문제 5.1: INSERT

**목표**: 새로운 데이터 삽입하기

**과제**:
1. 새로운 카테고리를 추가하세요.
2. 새로운 제품을 추가하세요.
3. 새로운 사용자를 추가하세요.

**예상 결과**:
```sql
-- 1. 새로운 카테고리 추가
INSERT INTO categories (id, name, description)
VALUES ('550e8400-e29b-41d4-a716-446655440007', 'beauty', '뷰티 및 화장품');

-- 2. 새로운 제품 추가
INSERT INTO products (id, name, description, category_id, sku, price, stock_quantity, tags, is_active)
VALUES (
    '310e8400-e29b-41d4-a716-446655440011', 
    '화장품 세트', 
    '기초 화장품 5종 세트', 
    '550e8400-e29b-41d4-a716-446655440007', 
    'BEAUTY_SET001', 
    50000, 
    100, 
    ARRAY['beauty', 'skincare', 'cosmetics'], 
    true
);

-- 3. 새로운 사용자 추가
INSERT INTO users (id, username, email, password_hash, first_name, last_name, role, is_active, email_verified)
VALUES (
    '110e8400-e29b-41d4-a716-446655440008', 
    'new_user', 
    'newuser@example.com', 
    crypt('newpassword123', gen_salt('bf')), 
    'New', 
    'User', 
    'customer', 
    true, 
    true
);
```

### 문제 5.2: UPDATE

**목표**: 기존 데이터 수정하기

**과제**:
1. 특정 제품의 가격을 10% 인상하세요.
2. 특정 사용자의 이메일 인증 상태를 변경하세요.
3. 특정 주문의 상태를 변경하세요.

**예상 결과**:
```sql
-- 1. 제품 가격 10% 인상
UPDATE products 
SET price = price * 1.1, updated_at = CURRENT_TIMESTAMP
WHERE name = '화장품 세트';

-- 2. 사용자 이메일 인증 상태 변경
UPDATE users 
SET email_verified = true, updated_at = CURRENT_TIMESTAMP
WHERE username = 'new_user';

-- 3. 주문 상태 변경
UPDATE orders 
SET status = 'processing', updated_at = CURRENT_TIMESTAMP
WHERE order_number = 'ORD-20230405-0001';
```

### 문제 5.3: DELETE

**목표**: 데이터 삭제하기

**과제**:
1. 특정 장바구니 항목을 삭제하세요.
2. 특정 위시리스트 항목을 삭제하세요.
3. 비활성 사용자를 삭제하세요 (주의: 실제 운영 환경에서는 하드 삭제보다 소프트 삭제 권장).

**예상 결과**:
```sql
-- 1. 특정 장바구니 항목 삭제
DELETE FROM shopping_cart 
WHERE user_id = '110e8400-e29b-41d4-a716-446655440008' 
  AND product_id = '310e8400-e29b-41d4-a716-446655440011';

-- 2. 특정 위시리스트 항목 삭제
DELETE FROM wishlist 
WHERE user_id = '110e8400-e29b-41d4-a716-446655440008' 
  AND product_id = '310e8400-e29b-41d4-a716-446655440011';

-- 3. 비활성 사용자 삭제 (주의 필요)
DELETE FROM users 
WHERE is_active = false 
  AND created_at < CURRENT_DATE - INTERVAL '1 year';
```

## 실습 6: 데이터 무결성

### 문제 6.1: 제약 조건

**목표**: 제약 조건을 이해하고 활용하기

**과제**:
1. 중복 이메일을 가진 사용자를 찾으세요.
2. NULL 값을 가진 컬럼을 찾으세요.
3. 외래 키 제약 조건을 위반하는 데이터를 찾으세요.

**예상 결과**:
```sql
-- 1. 중복 이메일 찾기
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- 2. NULL 값 가진 컬럼 찾기
SELECT 
    'phone' as column_name, 
    COUNT(*) as null_count
FROM users 
WHERE phone IS NULL
UNION ALL
SELECT 
    'last_name' as column_name, 
    COUNT(*) as null_count
FROM users 
WHERE last_name IS NULL;

-- 3. 외래 키 제약 조건 위반 확인
SELECT oi.id as order_item_id, oi.product_id
FROM order_items oi
LEFT JOIN products p ON oi.product_id = p.id
WHERE p.id IS NULL;
```

### 문제 6.2: 트랜잭션

**목표**: 트랜잭션을 사용하여 데이터 일관성 유지하기

**과제**:
1. 제품 재고를 감소시키고 주문을 생성하는 트랜잭션을 작성하세요.
2. 사용자 정보를 업데이트하고 로그를 기록하는 트랜잭션을 작성하세요.
3. 롤백이 필요한 상황을 시뮬레이션하세요.

**예상 결과**:
```sql
-- 1. 제품 재고 감소와 주문 생성
BEGIN;
UPDATE products 
SET stock_quantity = stock_quantity - 1 
WHERE id = '310e8400-e29b-41d4-a716-446655440011';
INSERT INTO orders (user_id, order_number, status, subtotal, total_amount, shipping_address, billing_address)
VALUES ('110e8400-e29b-41d4-a716-446655440008', 'ORD-20231024-0001', 'pending', 50000, 53000, 
        '{"street": "123 Test St", "city": "Seoul"}', '{"street": "123 Test St", "city": "Seoul"}');
COMMIT;

-- 2. 사용자 정보 업데이트와 로그 기록
BEGIN;
UPDATE users 
SET last_login = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
WHERE id = '110e8400-e29b-41d4-a716-446655440008';
INSERT INTO audit_logs (table_name, operation, record_id, new_values, user_id)
VALUES ('users', 'UPDATE', '110e8400-e29b-41d4-a716-446655440008', 
        '{"last_login": "' || CURRENT_TIMESTAMP || '""}', '110e8400-e29b-41d4-a716-446655440008');
COMMIT;

-- 3. 롤백 시뮬레이션
BEGIN;
UPDATE products SET price = price * 2; -- 실수로 모든 가격을 2배로 설정
-- 실수 발견 후 롤백
ROLLBACK;
```

## 실습 7: 성능 최적화

### 문제 7.1: 인덱스

**목표**: 인덱스를 생성하고 효과 확인하기

**과제**:
1. 자주 조회되는 컬럼에 인덱스를 생성하세요.
2. 복합 인덱스를 생성하세요.
3. 인덱스 사용 여부를 확인하세요.

**예상 결과**:
```sql
-- 1. 자주 조회되는 컬럼에 인덱스 생성
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- 2. 복합 인덱스 생성
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
CREATE INDEX idx_order_items_order_product ON order_items(order_id, product_id);

-- 3. 인덱스 사용 여부 확인
EXPLAIN ANALYZE 
SELECT * FROM users WHERE email = 'john.doe@example.com';

EXPLAIN ANALYZE 
SELECT * FROM orders 
WHERE user_id = '110e8400-e29b-41d4-a716-446655440001' 
  AND status = 'delivered';
```

### 문제 7.2: 쿼리 최적화

**목표**: 비효율적인 쿼리를 최적화하기

**과제**:
1. 서브쿼리를 JOIN으로 변환하세요.
2. 불필요한 컬럼 제거하기
3. 적절한 조건 순서로 쿼리 재작성하기

**예상 결과**:
```sql
-- 1. 서브쿼리를 JOIN으로 변환
-- 비효율적 쿼리
SELECT u.username, 
       (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count
FROM users u;

-- 최적화된 쿼리
SELECT u.username, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

-- 2. 불필요한 컬럼 제거
-- 비효율적 쿼리
SELECT * FROM products WHERE category_id = '550e8400-e29b-41d4-a716-446655440011';

-- 최적화된 쿼리
SELECT id, name, price FROM products WHERE category_id = '550e8400-e29b-41d4-a716-446655440011';

-- 3. 적절한 조건 순서
-- 비효율적 쿼리
SELECT * FROM orders 
WHERE total_amount > 100000 
  AND created_at >= '2023-01-01';

-- 최적화된 쿼리 (선택도 높은 조건 먼저)
SELECT * FROM orders 
WHERE created_at >= '2023-01-01'
  AND total_amount > 100000;
```

## 실습 8: 보안

### 문제 8.1: 권한 관리

**목표**: 사용자 권한을 안전하게 관리하기

**과제**:
1. 읽기 전용 사용자를 생성하고 권한을 부여하세요.
2. 특정 테이블에 대한 접근을 제한하세요.
3. 롤을 사용하여 권한을 그룹화하세요.

**예상 결과**:
```sql
-- 1. 읽기 전용 사용자 생성
CREATE USER readonly_user WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE lecture_db TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- 2. 특정 테이블 접근 제한
REVOKE ALL ON users FROM readonly_user;
GRANT SELECT(id, username, created_at) ON users TO readonly_user;

-- 3. 롤을 사용한 권한 그룹화
CREATE ROLE customer_service;
GRANT SELECT ON users TO customer_service;
GRANT SELECT ON orders TO customer_service;
GRANT UPDATE ON orders TO customer_service;
GRANT customer_service TO cs_user;
```

### 문제 8.2: 데이터 암호화

**목표**: 민감한 데이터를 암호화하기

**과제**:
1. 비밀번호를 안전하게 해싱하세요.
2. 민감한 정보를 암호화하여 저장하세요.
3. 암호화된 데이터를 복호화하여 조회하세요.

**예상 결과**:
```sql
-- 1. 비밀번호 해싱
SELECT crypt('newpassword', gen_salt('bf', 12));

-- 2. 민감 정보 암호화
CREATE TABLE sensitive_data (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    encrypted_info BYTEA
);

INSERT INTO sensitive_data (user_id, encrypted_info)
VALUES ('110e8400-e29b-41d4-a716-446655440001', 
        encrypt('민감한 정보', 'encryption_key', 'aes'));

-- 3. 암호화된 데이터 복호화
SELECT user_id, 
       convert_from(decrypt(encrypted_info, 'encryption_key', 'aes'), 'UTF-8') as decrypted_info
FROM sensitive_data
WHERE user_id = '110e8400-e29b-41d4-a716-446655440001';
```

## 정답 확인

각 실습 문제의 정답은 `exercises/solutions/` 디렉토리에서 확인할 수 있습니다. 자신의 해결책과 비교하여 학습 효과를 높이세요.

## 추가 도전 과제

1. 복잡한 보고서 쿼리 작성하기
2. 트리거를 사용하여 자동화된 비즈니스 로직 구현하기
3. 저장 프로시저를 사용하여 복잡한 작업 캡슐화하기
4. 파티셔닝을 사용하여 대용량 데이터 최적화하기

이 추가 과제들을 통해 PostgreSQL의 고급 기능들을 더 깊이 있게 탐구해 보세요.