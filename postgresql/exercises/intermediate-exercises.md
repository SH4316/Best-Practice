# PostgreSQL 중급 실습 문제

## 개요

이 실습 문제들은 PostgreSQL의 중급 기능과 고급 베스트 프랙티스를 연습하기 위해 설계되었습니다. 기초 실습을 완료했다고 가정하고 진행합니다.

## 실습 환경 설정

```sql
-- 실습을 위해 필요한 확장 모듈 확인
SELECT * FROM pg_extension;

-- 예제 데이터베이스 연결
\c lecture_db

-- 성능 분석을 위한 확장 활성화
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 통계 정보 업데이트
ANALYZE;
```

## 실습 1: 고급 조인과 서브쿼리

### 문제 1.1: 복잡한 조인

**목표**: 여러 테이블을 조인하여 복잡한 보고서 생성

**과제**:
1. 각 사용자별 구매한 제품 카테고리별 총 지출액을 계산하세요.
2. 각 카테고리별 평균 리뷰 평점과 리뷰 수를 계산하세요.
3. 리뷰를 작성하지 않은 사용자와 그들이 구매한 제품을 조회하세요.

**예상 결과**:
```sql
-- 1. 사용자별 카테고리별 지출액
SELECT 
    u.username,
    c.name as category_name,
    SUM(oi.total_price) as total_spent,
    COUNT(DISTINCT oi.product_id) as unique_products
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
JOIN categories c ON p.category_id = c.id
WHERE o.status = 'delivered'
GROUP BY u.id, u.username, c.id, c.name
ORDER BY u.username, total_spent DESC;

-- 2. 카테고리별 평균 리뷰 평점
SELECT 
    c.name as category_name,
    COUNT(pr.id) as review_count,
    AVG(pr.rating) as average_rating,
    COUNT(DISTINCT pr.product_id) as reviewed_products
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
LEFT JOIN product_reviews pr ON p.id = pr.product_id
GROUP BY c.id, c.name
ORDER BY average_rating DESC NULLS LAST;

-- 3. 리뷰를 작성하지 않은 사용자
SELECT 
    u.username,
    u.email,
    COUNT(DISTINCT o.id) as order_count,
    COUNT(DISTINCT oi.product_id) as purchased_products
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
LEFT JOIN product_reviews pr ON u.id = pr.user_id
WHERE o.status = 'delivered'
  AND pr.id IS NULL
GROUP BY u.id, u.username, u.email
ORDER BY order_count DESC;
```

### 문제 1.2: 상관 서브쿼리

**목표**: 상관 서브쿼리를 사용하여 복잡한 데이터 분석

**과제**:
1. 각 카테고리에서 평균 가격보다 비싼 제품을 조회하세요.
2. 각 사용자의 평균 주문 금액보다 큰 주문을 조회하세요.
3. 각 제품의 평균 리뷰 평점보다 높은 리뷰를 조회하세요.

**예상 결과**:
```sql
-- 1. 카테고리별 평균보다 비싼 제품
SELECT 
    p.name,
    p.price,
    c.name as category_name,
    avg_cat.avg_price,
    (p.price - avg_cat.avg_price) as price_difference
FROM products p
JOIN categories c ON p.category_id = c.id
JOIN (
    SELECT 
        category_id, 
        AVG(price) as avg_price
    FROM products
    GROUP BY category_id
) avg_cat ON p.category_id = avg_cat.category_id
WHERE p.price > avg_cat.avg_price
ORDER BY price_difference DESC;

-- 2. 사용자별 평균보다 큰 주문
SELECT 
    o.order_number,
    u.username,
    o.total_amount,
    user_avg.avg_order_amount,
    (o.total_amount - user_avg.avg_order_amount) as amount_difference
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN (
    SELECT 
        user_id, 
        AVG(total_amount) as avg_order_amount
    FROM orders
    WHERE status = 'delivered'
    GROUP BY user_id
) user_avg ON o.user_id = user_avg.user_id
WHERE o.total_amount > user_avg.avg_order_amount
  AND o.status = 'delivered'
ORDER BY amount_difference DESC;

-- 3. 제품별 평균보다 높은 리뷰
SELECT 
    pr.id as review_id,
    pr.rating,
    pr.title,
    p.name as product_name,
    prod_avg.avg_rating,
    (pr.rating - prod_avg.avg_rating) as rating_difference
FROM product_reviews pr
JOIN products p ON pr.product_id = p.id
JOIN (
    SELECT 
        product_id, 
        AVG(rating) as avg_rating
    FROM product_reviews
    GROUP BY product_id
) prod_avg ON pr.product_id = prod_avg.product_id
WHERE pr.rating > prod_avg.avg_rating
ORDER BY rating_difference DESC;
```

## 실습 2: 윈도우 함수

### 문제 2.1: 순위 함수

**목표**: 윈도우 함수를 사용하여 데이터 순위 매기기

**과제**:
1. 제품을 판매량 순위로 조회하세요.
2. 사용자를 총 지출액 순위로 조회하세요.
3. 카테고리를 평균 리뷰 평점 순위로 조회하세요.

**예상 결과**:
```sql
-- 1. 제품 판매량 순위
SELECT 
    p.name,
    c.name as category_name,
    SUM(oi.quantity) as total_sold,
    SUM(oi.total_price) as total_revenue,
    RANK() OVER (ORDER BY SUM(oi.quantity) DESC) as sales_rank,
    DENSE_RANK() OVER (ORDER BY SUM(oi.total_price) DESC) as revenue_rank
FROM products p
JOIN categories c ON p.category_id = c.id
LEFT JOIN order_items oi ON p.id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.id AND o.status = 'delivered'
GROUP BY p.id, p.name, c.name
ORDER BY sales_rank;

-- 2. 사용자 총 지출액 순위
SELECT 
    u.username,
    u.email,
    COUNT(DISTINCT o.id) as order_count,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value,
    RANK() OVER (ORDER BY SUM(o.total_amount) DESC) as spending_rank,
    ROW_NUMBER() OVER (ORDER BY SUM(o.total_amount) DESC, u.username) as row_num
FROM users u
LEFT JOIN orders o ON u.id = o.user_id AND o.status = 'delivered'
GROUP BY u.id, u.username, u.email
HAVING COUNT(o.id) > 0
ORDER BY spending_rank;

-- 3. 카테고리 평균 리뷰 평점 순위
SELECT 
    c.name as category_name,
    COUNT(DISTINCT p.id) as product_count,
    COUNT(pr.id) as review_count,
    AVG(pr.rating) as avg_rating,
    RANK() OVER (ORDER BY AVG(pr.rating) DESC NULLS LAST) as rating_rank,
    NTILE(4) OVER (ORDER BY AVG(pr.rating) DESC NULLS LAST) as rating_quartile
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
LEFT JOIN product_reviews pr ON p.id = pr.product_id
GROUP BY c.id, c.name
ORDER BY rating_rank;
```

### 문제 2.2: 집계 윈도우 함수

**목표**: 집계 윈도우 함수를 사용하여 누적 값과 이동 평균 계산

**과제**:
1. 월별 누적 매출을 계산하세요.
2. 제품별 3개월 이동 평균 판매량을 계산하세요.
3. 사용자별 주문 간격을 계산하세요.

**예상 결과**:
```sql
-- 1. 월별 누적 매출
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', created_at) as month,
        SUM(total_amount) as monthly_revenue,
        COUNT(*) as order_count
    FROM orders
    WHERE status = 'delivered'
    GROUP BY DATE_TRUNC('month', created_at)
)
SELECT 
    month,
    monthly_revenue,
    order_count,
    SUM(monthly_revenue) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumulative_revenue,
    SUM(order_count) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumulative_orders,
    ROUND(monthly_revenue * 100.0 / LAG(monthly_revenue) OVER (ORDER BY month), 2) as revenue_growth_pct
FROM monthly_sales
ORDER BY month;

-- 2. 제품별 3개월 이동 평균 판매량
WITH monthly_product_sales AS (
    SELECT 
        DATE_TRUNC('month', o.created_at) as month,
        p.id as product_id,
        p.name as product_name,
        SUM(oi.quantity) as monthly_quantity
    FROM products p
    JOIN order_items oi ON p.id = oi.product_id
    JOIN orders o ON oi.order_id = o.id AND o.status = 'delivered'
    GROUP BY DATE_TRUNC('month', o.created_at), p.id, p.name
)
SELECT 
    month,
    product_name,
    monthly_quantity,
    AVG(monthly_quantity) OVER (
        PARTITION BY product_id 
        ORDER BY month 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as moving_avg_3months,
    SUM(monthly_quantity) OVER (
        PARTITION BY product_id 
        ORDER BY month 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_quantity
FROM monthly_product_sales
ORDER BY product_name, month;

-- 3. 사용자별 주문 간격
WITH user_order_dates AS (
    SELECT 
        u.id as user_id,
        u.username,
        o.id as order_id,
        o.created_at,
        LAG(o.created_at) OVER (PARTITION BY u.id ORDER BY o.created_at) as prev_order_date
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE o.status = 'delivered'
)
SELECT 
    username,
    order_id,
    created_at,
    prev_order_date,
    EXTRACT(DAYS FROM created_at - prev_order_date) as days_between_orders,
    AVG(EXTRACT(DAYS FROM created_at - prev_order_date)) OVER (
        PARTITION BY user_id 
        ORDER BY created_at 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as avg_days_between_orders
FROM user_order_dates
WHERE prev_order_date IS NOT NULL
ORDER BY username, created_at;
```

## 실습 3: CTE (Common Table Expressions)

### 문제 3.1: 재귀 CTE

**목표**: 재귀 CTE를 사용하여 계층적 데이터 처리

**과제**:
1. 카테고리 계층 구조를 조회하세요.
2. 각 카테고리의 하위 카테고리 수를 계산하세요.
3. 카테고리 경로를 생성하세요.

**예상 결과**:
```sql
-- 1. 카테고리 계층 구조
WITH RECURSIVE category_hierarchy AS (
    -- 기본 쿼리: 최상위 카테고리
    SELECT 
        id,
        name,
        parent_id,
        0 as level,
        name as path
    FROM categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    -- 재귀 쿼리: 하위 카테고리
    SELECT 
        c.id,
        c.name,
        c.parent_id,
        ch.level + 1,
        ch.path || ' > ' || c.name
    FROM categories c
    JOIN category_hierarchy ch ON c.parent_id = ch.id
)
SELECT 
    id,
    name,
    parent_id,
    level,
    path,
    LPAD(' ', level * 2, ' ') || name as indented_name
FROM category_hierarchy
ORDER BY path;

-- 2. 각 카테고리의 하위 카테고리 수
WITH RECURSIVE category_tree AS (
    SELECT 
        id,
        name,
        parent_id,
        id as root_id
    FROM categories
    
    UNION ALL
    
    SELECT 
        c.id,
        c.name,
        c.parent_id,
        ct.root_id
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT 
    c.name as category_name,
    COUNT(ct.id) - 1 as subcategory_count
FROM categories c
LEFT JOIN category_tree ct ON c.id = ct.root_id
GROUP BY c.id, c.name
ORDER BY subcategory_count DESC;

-- 3. 카테고리 전체 경로
WITH RECURSIVE category_path AS (
    SELECT 
        id,
        name,
        parent_id,
        ARRAY[name] as path_array
    FROM categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    SELECT 
        c.id,
        c.name,
        c.parent_id,
        cp.path_array || c.name
    FROM categories c
    JOIN category_path cp ON c.parent_id = cp.id
)
SELECT 
    cp.id,
    cp.name,
    cp.path_array,
    ARRAY_TO_STRING(cp.path_array, ' > ') as full_path
FROM category_path cp
ORDER BY cp.path_array;
```

### 문제 3.2: 복잡한 CTE

**목표**: 여러 CTE를 조합하여 복잡한 분석 쿼리 작성

**과제**:
1. 고객 세분화 분석을 수행하세요.
2. 제품 성장 분석을 수행하세요.
3. 시간대별 판매 패턴을 분석하세요.

**예상 결과**:
```sql
-- 1. 고객 세분화 분석
WITH customer_stats AS (
    SELECT 
        u.id,
        u.username,
        COUNT(DISTINCT o.id) as order_count,
        SUM(o.total_amount) as total_spent,
        AVG(o.total_amount) as avg_order_value,
        MIN(o.created_at) as first_order,
        MAX(o.created_at) as last_order,
        EXTRACT(DAYS FROM MAX(o.created_at) - MIN(o.created_at)) as customer_lifetime_days
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id AND o.status = 'delivered'
    GROUP BY u.id, u.username
),
customer_segments AS (
    SELECT 
        *,
        CASE 
            WHEN total_spent > 1000000 AND order_count > 5 THEN 'VIP'
            WHEN total_spent > 500000 AND order_count > 3 THEN '골드'
            WHEN total_spent > 200000 OR order_count > 2 THEN '실버'
            WHEN total_spent > 0 THEN '브론즈'
            ELSE '신규'
        END as segment
    FROM customer_stats
)
SELECT 
    segment,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    AVG(total_spent) as avg_segment_spending,
    AVG(order_count) as avg_segment_orders
FROM customer_segments
GROUP BY segment
ORDER BY 
    CASE segment
        WHEN 'VIP' THEN 1
        WHEN '골드' THEN 2
        WHEN '실버' THEN 3
        WHEN '브론즈' THEN 4
        WHEN '신규' THEN 5
    END;

-- 2. 제품 성장 분석
WITH monthly_product_stats AS (
    SELECT 
        DATE_TRUNC('month', o.created_at) as month,
        p.id as product_id,
        p.name as product_name,
        SUM(oi.quantity) as monthly_quantity,
        SUM(oi.total_price) as monthly_revenue
    FROM products p
    JOIN order_items oi ON p.id = oi.product_id
    JOIN orders o ON oi.order_id = o.id AND o.status = 'delivered'
    GROUP BY DATE_TRUNC('month', o.created_at), p.id, p.name
),
product_growth AS (
    SELECT 
        product_id,
        product_name,
        month,
        monthly_quantity,
        monthly_revenue,
        LAG(monthly_quantity) OVER (PARTITION BY product_id ORDER BY month) as prev_quantity,
        LAG(monthly_revenue) OVER (PARTITION BY product_id ORDER BY month) as prev_revenue
    FROM monthly_product_stats
)
SELECT 
    product_name,
    month,
    monthly_quantity,
    monthly_revenue,
    COALESCE(prev_quantity, 0) as prev_quantity,
    COALESCE(prev_revenue, 0) as prev_revenue,
    CASE 
        WHEN prev_quantity > 0 THEN ROUND((monthly_quantity - prev_quantity) * 100.0 / prev_quantity, 2)
        ELSE NULL
    END as quantity_growth_pct,
    CASE 
        WHEN prev_revenue > 0 THEN ROUND((monthly_revenue - prev_revenue) * 100.0 / prev_revenue, 2)
        ELSE NULL
    END as revenue_growth_pct
FROM product_growth
ORDER BY product_name, month;

-- 3. 시간대별 판매 패턴
WITH hourly_sales AS (
    SELECT 
        EXTRACT(HOUR FROM created_at) as hour_of_day,
        EXTRACT(DOW FROM created_at) as day_of_week,
        COUNT(*) as order_count,
        SUM(total_amount) as total_revenue
    FROM orders
    WHERE status = 'delivered'
    GROUP BY EXTRACT(HOUR FROM created_at), EXTRACT(DOW FROM created_at)
),
hourly_patterns AS (
    SELECT 
        hour_of_day,
        AVG(order_count) as avg_orders,
        AVG(total_revenue) as avg_revenue,
        STDDEV(order_count) as stddev_orders
    FROM hourly_sales
    GROUP BY hour_of_day
)
SELECT 
    hs.hour_of_day,
    TO_CHAR(hs.hour_of_day, 'HH:MI') as time_slot,
    hs.avg_orders,
    hs.avg_revenue,
    hs.stddev_orders,
    CASE 
        WHEN hs.avg_orders > hp.avg_orders THEN 'Peak'
        WHEN hs.avg_orders < hp.avg_orders * 0.5 THEN 'Low'
        ELSE 'Normal'
    END as activity_level
FROM hourly_sales hs
JOIN hourly_patterns hp ON hs.hour_of_day = hp.hour_of_day
WHERE hs.day_of_week BETWEEN 1 AND 5  -- 주중만
GROUP BY hs.hour_of_day, hp.avg_orders
ORDER BY hs.hour_of_day;
```

## 실습 4: JSON과 배열 처리

### 문제 4.1: JSON 데이터 처리

**목표**: JSONB 데이터 타입을 사용하여 복잡한 데이터 구조 처리

**과제**:
1. 제품 속성에서 특정 정보를 추출하세요.
2. 제품 속성을 기반으로 검색 기능을 구현하세요.
3. 제품 속성을 집계하여 보고서를 생성하세요.

**예상 결과**:
```sql
-- 1. 제품 속성에서 특정 정보 추출
SELECT 
    name,
    price,
    attributes->>'brand' as brand,
    attributes->>'model' as model,
    attributes->>'color' as color,
    attributes->>'storage' as storage,
    attributes->>'screen_size' as screen_size,
    CASE 
        WHEN attributes->>'brand' = 'Apple' THEN '애플 제품'
        WHEN attributes->>'brand' = 'Samsung' THEN '삼성 제품'
        ELSE '기타 브랜드'
    END as brand_category
FROM products
WHERE attributes IS NOT NULL
ORDER BY brand, model;

-- 2. 제품 속성 기반 검색
-- 브랜드가 Apple이고 색상이 검은색인 제품
SELECT 
    name,
    price,
    attributes
FROM products
WHERE attributes @> '{"brand": "Apple"}'
  AND attributes @> '{"color": "black"}'
ORDER BY price;

-- 스토리지 용량이 128GB 이상인 제품
SELECT 
    name,
    price,
    attributes->>'storage' as storage
FROM products
WHERE attributes->>'storage' ~ '^\d+GB$'
  AND (REPLACE(attributes->>'storage', 'GB', '')::INTEGER) >= 128
ORDER BY price;

-- 3. 제품 속성 집계 보고서
SELECT 
    attributes->>'brand' as brand,
    COUNT(*) as product_count,
    AVG(price) as avg_price,
    MIN(price) as min_price,
    MAX(price) as max_price,
    STRING_AGG(DISTINCT attributes->>'color', ', ') as available_colors,
    STRING_AGG(DISTINCT attributes->>'storage', ', ') as available_storage
FROM products
WHERE attributes IS NOT NULL
  AND attributes->>'brand' IS NOT NULL
GROUP BY attributes->>'brand'
ORDER BY product_count DESC;
```

### 문제 4.2: 배열 데이터 처리

**목표**: 배열 데이터 타입을 사용하여 다중 값 속성 처리

**과제**:
1. 제품 태그를 기반으로 검색 기능을 구현하세요.
2. 태그별 제품 수를 집계하세요.
3. 태그 유사성을 기반으로 관련 제품을 찾으세요.

**예상 결과**:
```sql
-- 1. 제품 태그 기반 검색
-- 특정 태그를 가진 제품
SELECT 
    name,
    price,
    tags
FROM products
WHERE 'smartphone' = ANY(tags)
ORDER BY price;

-- 여러 태그를 모두 가진 제품
SELECT 
    name,
    price,
    tags
FROM products
WHERE tags @> ARRAY['apple', 'smartphone']
ORDER BY price;

-- 태그 중 하나라도 포함하는 제품
SELECT 
    name,
    price,
    tags
FROM products
WHERE tags && ARRAY['laptop', 'computer']
ORDER BY price;

-- 2. 태그별 제품 수 집계
SELECT 
    unnest(tags) as tag,
    COUNT(*) as product_count,
    AVG(price) as avg_price,
    STRING_AGG(name, ', ') as products
FROM products
WHERE tags IS NOT NULL
GROUP BY unnest(tags)
ORDER BY product_count DESC;

-- 3. 태그 유사성 기반 관련 제품
WITH target_product_tags AS (
    SELECT tags
    FROM products
    WHERE name = 'iPhone 14 Pro'
),
tag_similarity AS (
    SELECT 
        p.id,
        p.name,
        p.price,
        p.tags,
        array_length(p.tags, 1) as tag_count,
        array_length(
            p.tags & (SELECT tags FROM target_product_tags),
            1
        ) as common_tags
    FROM products p
    WHERE p.name != 'iPhone 14 Pro'
      AND p.tags IS NOT NULL
)
SELECT 
    name,
    price,
    tags,
    common_tags,
    tag_count,
    ROUND(
        common_tags::numeric / 
        NULLIF(LEAST(tag_count, (SELECT array_length(tags, 1) FROM target_product_tags)), 0) * 100, 
        2
    ) as similarity_percentage
FROM tag_similarity
WHERE common_tags > 0
ORDER BY similarity_percentage DESC, common_tags DESC;
```

## 실습 5: 성능 최적화

### 문제 5.1: 실행 계획 분석

**목표**: 실행 계획을 분석하여 성능 병목 식별

**과제**:
1. 복잡한 쿼리의 실행 계획을 분석하세요.
2. 인덱스 사용 여부를 확인하세요.
3. 성능 개선 방안을 제시하세요.

**예상 결과**:
```sql
-- 1. 복잡한 쿼리 실행 계획 분석
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    u.username,
    c.name as category_name,
    p.name as product_name,
    COUNT(oi.id) as purchase_count,
    SUM(oi.total_price) as total_spent
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
JOIN categories c ON p.category_id = c.id
WHERE o.status = 'delivered'
  AND o.created_at >= '2023-01-01'
GROUP BY u.id, u.username, c.id, c.name, p.id, p.name
HAVING SUM(oi.total_price) > 100000
ORDER BY total_spent DESC
LIMIT 20;

-- 2. 인덱스 사용 여부 확인
EXPLAIN (ANALYZE)
SELECT *
FROM products
WHERE attributes->>'brand' = 'Apple'
  AND price > 1000000
ORDER BY price DESC;

-- 3. 성능 개선을 위한 인덱스 생성
CREATE INDEX idx_products_brand ON products USING GIN ((attributes->>'brand'));
CREATE INDEX idx_products_price ON products(price DESC);
CREATE INDEX idx_products_brand_price ON products USING GIN ((attributes->>'brand')) WHERE price > 1000000;

-- 개선 후 실행 계획 확인
EXPLAIN (ANALYZE)
SELECT *
FROM products
WHERE attributes->>'brand' = 'Apple'
  AND price > 1000000
ORDER BY price DESC;
```

### 문제 5.2: 쿼리 리팩토링

**목표**: 비효율적인 쿼리를 효율적으로 리팩토링

**과제**:
1. 서브쿼리를 JOIN으로 변환하세요.
2. 상관 서브쿼리를 비상관 서브쿼리로 변환하세요.
3. 불필요한 정렬을 제거하세요.

**예상 결과**:
```sql
-- 1. 서브쿼리를 JOIN으로 변환
-- 비효율적 쿼리
SELECT 
    u.username,
    (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id AND o.status = 'delivered') as order_count,
    (SELECT SUM(o.total_amount) FROM orders o WHERE o.user_id = u.id AND o.status = 'delivered') as total_spent
FROM users u
WHERE u.is_active = true;

-- 효율적 쿼리
SELECT 
    u.username,
    COUNT(o.id) as order_count,
    SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id AND o.status = 'delivered'
WHERE u.is_active = true
GROUP BY u.id, u.username;

-- 2. 상관 서브쿼리를 비상관 서브쿼리로 변환
-- 비효율적 쿼리
SELECT 
    p.name,
    p.price,
    (SELECT AVG(pr.rating) FROM product_reviews pr WHERE pr.product_id = p.id) as avg_rating
FROM products p
WHERE p.category_id = '550e8400-e29b-41d4-a716-446655440011';

-- 효율적 쿼리
WITH product_ratings AS (
    SELECT 
        product_id,
        AVG(rating) as avg_rating
    FROM product_reviews
    GROUP BY product_id
)
SELECT 
    p.name,
    p.price,
    pr.avg_rating
FROM products p
LEFT JOIN product_ratings pr ON p.id = pr.product_id
WHERE p.category_id = '550e8400-e29b-41d4-a716-446655440011';

-- 3. 불필요한 정렬 제거
-- 비효율적 쿼리
SELECT DISTINCT u.username
FROM users u
JOIN orders o ON u.id = o.user_id
ORDER BY u.username;

-- 효율적 쿼리
SELECT u.username
FROM users u
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
ORDER BY u.username;
```

## 실습 6: 트랜잭션과 동시성

### 문제 6.1: 동시성 제어

**목표**: 동시성 문제를 이해하고 해결 방법을 학습

**과제**:
1. 데드락 상황을 시뮬레이션하세요.
2. 잠금 충돌을 해결하세요.
3. 격리 수준에 따른 동작 차이를 확인하세요.

**예상 결과**:
```sql
-- 1. 데드락 시뮬레이션
-- 세션 1
BEGIN;
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = '310e8400-e29b-41d4-a716-446655440001';
-- 잠시 대기
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = '310e8400-e29b-41d4-a716-446655440002';
COMMIT;

-- 세션 2 (동시 실행)
BEGIN;
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = '310e8400-e29b-41d4-a716-446655440002';
-- 잠시 대기
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = '310e8400-e29b-41d4-a716-446655440001';
COMMIT;

-- 2. 잠금 충돌 해결
-- 일관된 순서로 잠금
BEGIN;
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = '310e8400-e29b-41d4-a716-446655440001';
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = '310e8400-e29b-41d4-a716-446655440002';
COMMIT;

-- 3. 격리 수준 비교
-- READ COMMITTED
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
SELECT balance FROM accounts WHERE id = 1;
-- 다른 세션에서 업데이트
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
COMMIT;
SELECT balance FROM accounts WHERE id = 1;
COMMIT;

-- REPEATABLE READ
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT balance FROM accounts WHERE id = 1;
-- 다른 세션에서 업데이트
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
COMMIT;
SELECT balance FROM accounts WHERE id = 1;
COMMIT;
```

### 문제 6.2: 고급 트랜잭션 패턴

**목표**: 복잡한 비즈니스 로직을 트랜잭션으로 구현

**과제**:
1. 재고 관리 트랜잭션을 구현하세요.
2. 주문 취소 트랜잭션을 구현하세요.
3. 보상 트랜잭션을 구현하세요.

**예상 결과**:
```sql
-- 1. 재고 관리 트랜잭션
CREATE OR REPLACE FUNCTION update_inventory(
    product_uuid UUID,
    quantity_change INTEGER,
    reason TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    current_stock INTEGER;
    new_stock INTEGER;
BEGIN
    -- 현재 재고 확인
    SELECT stock_quantity INTO current_stock
    FROM products
    WHERE id = product_uuid
    FOR UPDATE;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION '제품을 찾을 수 없습니다';
        RETURN FALSE;
    END IF;
    
    new_stock := current_stock + quantity_change;
    
    -- 재고 부족 확인
    IF new_stock < 0 THEN
        RAISE EXCEPTION '재고가 부족합니다. 현재: %, 요청: %', current_stock, -quantity_change;
        RETURN FALSE;
    END IF;
    
    -- 재고 업데이트
    UPDATE products 
    SET stock_quantity = new_stock
    WHERE id = product_uuid;
    
    -- 재고 로그 기록
    INSERT INTO inventory_logs (
        product_id, 
        change_type, 
        quantity_change, 
        previous_quantity, 
        new_quantity,
        reason
    ) VALUES (
        product_uuid,
        CASE WHEN quantity_change > 0 THEN 'purchase' ELSE 'sale' END,
        quantity_change,
        current_stock,
        new_stock,
        reason
    );
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- 2. 주문 취소 트랜잭션
CREATE OR REPLACE FUNCTION cancel_order(
    order_uuid UUID,
    reason TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    order_status TEXT;
BEGIN
    -- 주문 상태 확인
    SELECT status INTO order_status
    FROM orders
    WHERE id = order_uuid
    FOR UPDATE;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION '주문을 찾을 수 없습니다';
        RETURN FALSE;
    END IF;
    
    -- 취소 가능 상태 확인
    IF order_status IN ('shipped', 'delivered') THEN
        RAISE EXCEPTION '배송된 주문은 취소할 수 없습니다';
        RETURN FALSE;
    END IF;
    
    IF order_status = 'cancelled' THEN
        RAISE EXCEPTION '이미 취소된 주문입니다';
        RETURN FALSE;
    END IF;
    
    -- 주문 상태 업데이트
    UPDATE orders 
    SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
    WHERE id = order_uuid;
    
    -- 재고 복원
    PERFORM update_inventory(oi.product_id, oi.quantity, '주문 취소: ' || reason)
    FROM order_items oi
    WHERE oi.order_id = order_uuid;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- 3. 보상 트랜잭션
CREATE OR REPLACE FUNCTION process_order_with_compensation(
    user_uuid UUID,
    product_items JSONB,
    payment_info JSONB
) RETURNS UUID AS $$
DECLARE
    order_uuid UUID;
    payment_uuid UUID;
    item_record JSONB;
    product_uuid UUID;
    quantity INTEGER;
BEGIN
    -- 주문 생성
    INSERT INTO orders (user_id, status, subtotal, total_amount, shipping_address, billing_address)
    VALUES (
        user_uuid,
        'pending',
        (SELECT SUM((item->>'quantity')::INTEGER * (item->>'price')::DECIMAL) FROM jsonb_array_elements(product_items) as item),
        (SELECT SUM((item->>'quantity')::INTEGER * (item->>'price')::DECIMAL) FROM jsonb_array_elements(product_items) as item),
        payment_info->>'shipping_address',
        payment_info->>'billing_address'
    ) RETURNING id INTO order_uuid;
    
    BEGIN
        -- 주문 항목 생성 및 재고 차감
        FOR item_record IN SELECT * FROM jsonb_array_elements(product_items) as item LOOP
            product_uuid := (item_record->>'product_id')::UUID;
            quantity := (item_record->>'quantity')::INTEGER;
            
            -- 주문 항목 생성
            INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price, product_snapshot)
            VALUES (
                order_uuid,
                product_uuid,
                quantity,
                (item_record->>'price')::DECIMAL,
                quantity * (item_record->>'price')::DECIMAL,
                item_record
            );
            
            -- 재고 차감
            PERFORM update_inventory(product_uuid, -quantity, '주문 처리: ' || order_uuid::TEXT);
        END LOOP;
        
        -- 결제 처리
        INSERT INTO payments (order_id, payment_method, amount, status, payment_details)
        VALUES (
            order_uuid,
            payment_info->>'payment_method',
            (SELECT SUM(total_price) FROM order_items WHERE order_id = order_uuid),
            'completed',
            payment_info
        ) RETURNING id INTO payment_uuid;
        
        -- 주문 상태 업데이트
        UPDATE orders 
        SET status = 'processing', updated_at = CURRENT_TIMESTAMP
        WHERE id = order_uuid;
        
        RETURN order_uuid;
        
    EXCEPTION
        WHEN OTHERS THEN
            -- 보상 트랜잭션: 재고 복원
            FOR item_record IN SELECT * FROM jsonb_array_elements(product_items) as item LOOP
                product_uuid := (item_record->>'product_id')::UUID;
                quantity := (item_record->>'quantity')::INTEGER;
                
                PERFORM update_inventory(product_uuid, quantity, '주문 실패 보상: ' || order_uuid::TEXT);
            END LOOP;
            
            -- 주문 상태 업데이트
            UPDATE orders 
            SET status = 'failed', updated_at = CURRENT_TIMESTAMP
            WHERE id = order_uuid;
            
            RAISE EXCEPTION '주문 처리 실패: %', SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;
```

## 실습 7: 보안 고급

### 문제 7.1: 행 수준 보안

**목표**: Row Level Security를 사용하여 데이터 접근 제어

**과제**:
1. 사용자별 데이터 접근 정책을 구현하세요.
2. 역할 기반 접근 제어를 구현하세요.
3. 동적 필터링을 구현하세요.

**예상 결과**:
```sql
-- 1. 사용자별 데이터 접근 정책
-- 사용자 테이블에 자신의 정보만 접근하도록 정책 설정
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_self_access ON users
    FOR ALL
    USING (id = current_setting('app.current_user_id')::UUID)
    WITH CHECK (id = current_setting('app.current_user_id')::UUID);

-- 2. 역할 기반 접근 제어
-- 주문 테이블에 역할 기반 접근 정책
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY customer_order_access ON orders
    FOR ALL
    USING (
        CASE current_setting('app.user_role')
            WHEN 'customer' THEN user_id = current_setting('app.current_user_id')::UUID
            WHEN 'admin' THEN true
            ELSE false
        END
    );

-- 3. 동적 필터링
-- 제품 테이블에 동적 필터링 정책
ALTER TABLE products ENABLE ROW LEVEL SECURITY;

CREATE POLICY product_active_filter ON products
    FOR SELECT
    USING (
        CASE current_setting('app.user_role')
            WHEN 'customer' THEN is_active = true
            WHEN 'vendor' THEN created_by = current_setting('app.current_user_id')::UUID
            WHEN 'admin' THEN true
            ELSE false
        END
    );

-- 정책 테스트
SET app.current_user_id = '110e8400-e29b-41d4-a716-446655440001';
SET app.user_role = 'customer';

SELECT * FROM users;  -- 자신의 정보만 보임
SELECT * FROM orders;  -- 자신의 주문만 보임
SELECT * FROM products;  -- 활성 제품만 보임
```

### 문제 7.2: 감사 로깅

**목표**: 포괄적인 감사 로깅 시스템 구현

**과제**:
1. 테이블별 감사 트리거를 구현하세요.
2. 민감한 데이터 접근을 로깅하세요.
3. 감사 보고서를 생성하세요.

**예상 결과**:
```sql
-- 1. 테이블별 감사 트리거
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    audit_data JSONB;
BEGIN
    audit_data := jsonb_build_object(
        'table_name', TG_TABLE_NAME,
        'operation', TG_OP,
        'user_id', current_setting('app.current_user_id'),
        'ip_address', inet_client_addr(),
        'timestamp', CURRENT_TIMESTAMP
    );
    
    IF TG_OP = 'DELETE' THEN
        audit_data := audit_data || 
            jsonb_build_object('old_values', row_to_json(OLD));
        INSERT INTO audit_logs (table_name, operation, record_id, old_values, user_id, ip_address)
        VALUES (TG_TABLE_NAME, TG_OP, OLD.id, row_to_json(OLD), 
                current_setting('app.current_user_id')::UUID, inet_client_addr());
    ELSIF TG_OP = 'UPDATE' THEN
        audit_data := audit_data || 
            jsonb_build_object('old_values', row_to_json(OLD), 'new_values', row_to_json(NEW));
        INSERT INTO audit_logs (table_name, operation, record_id, old_values, new_values, user_id, ip_address)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(OLD), row_to_json(NEW),
                current_setting('app.current_user_id')::UUID, inet_client_addr());
    ELSIF TG_OP = 'INSERT' THEN
        audit_data := audit_data || 
            jsonb_build_object('new_values', row_to_json(NEW));
        INSERT INTO audit_logs (table_name, operation, record_id, new_values, user_id, ip_address)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(NEW),
                current_setting('app.current_user_id')::UUID, inet_client_addr());
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- 2. 민감한 데이터 접근 로깅
CREATE OR REPLACE FUNCTION log_sensitive_access()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_logs (table_name, operation, record_id, user_id, ip_address, created_at)
    VALUES (
        TG_TABLE_NAME,
        'SELECT',
        OLD.id,
        current_setting('app.current_user_id')::UUID,
        inet_client_addr(),
        CURRENT_TIMESTAMP
    );
    
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- 민감한 데이터 접근 로깅 트리거
CREATE TRIGGER log_sensitive_user_access
    AFTER SELECT ON users
    FOR EACH ROW EXECUTE FUNCTION log_sensitive_access();

-- 3. 감사 보고서 생성
-- 사용자 활동 보고서
CREATE VIEW user_activity_report AS
SELECT 
    u.username,
    al.table_name,
    al.operation,
    COUNT(*) as operation_count,
    MIN(al.created_at) as first_access,
    MAX(al.created_at) as last_access
FROM audit_logs al
JOIN users u ON al.user_id = u.id
WHERE al.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY u.id, u.username, al.table_name, al.operation
ORDER BY operation_count DESC;

-- 테이블 변경 활동 보고서
CREATE VIEW table_change_report AS
SELECT 
    table_name,
    operation,
    COUNT(*) as change_count,
    COUNT(DISTINCT user_id) as unique_users,
    DATE_TRUNC('day', created_at) as change_date
FROM audit_logs
WHERE operation IN ('INSERT', 'UPDATE', 'DELETE')
  AND created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY table_name, operation, DATE_TRUNC('day', created_at)
ORDER BY change_date DESC, change_count DESC;
```

## 정답 확인

각 실습 문제의 정답은 `exercises/solutions/` 디렉토리에서 확인할 수 있습니다. 자신의 해결책과 비교하여 학습 효과를 높이세요.

## 추가 도전 과제

1. 파티셔닝을 사용한 대용량 데이터 처리
2. 복제를 이용한 고가용성 구현
3. 외부 데이터 웨어하우스와 연동
4. 머신러닝 모델을 위한 데이터 파이프라인 구축

이 추가 과제들을 통해 PostgreSQL의 엔터프라이즈급 기능들을 더 깊이 있게 탐구해 보세요.