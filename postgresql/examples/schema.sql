-- PostgreSQL 베스트 프랙티스 강의자료 예제 데이터베이스 스키마
-- 생성일: 2023-10-24
-- 설명: 전자상거래 시스템 예제 데이터베이스

-- 확장 모듈 생성
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 데이터베이스 생성 (필요시)
-- CREATE DATABASE lecture_db;
-- \c lecture_db;

-- 사용자 정의 타입 생성
CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped', 'delivered', 'cancelled');
CREATE TYPE payment_method AS ENUM ('credit_card', 'bank_transfer', 'mobile_payment', 'cash_on_delivery');
CREATE TYPE user_role AS ENUM ('customer', 'admin', 'vendor');
CREATE TYPE product_category AS ENUM ('electronics', 'clothing', 'books', 'home', 'sports', 'toys');

-- 도메인 타입 생성
CREATE DOMAIN email_domain AS VARCHAR(255)
CHECK (VALUE ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

CREATE DOMAIN positive_integer AS INTEGER
CHECK (VALUE > 0);

CREATE DOMAIN price_domain AS DECIMAL(10, 2)
CHECK (VALUE >= 0);

CREATE DOMAIN rating_domain AS DECIMAL(3, 2)
CHECK (VALUE >= 0 AND VALUE <= 5);

-- 사용자 테이블
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email email_domain UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    role user_role DEFAULT 'customer',
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- 사용자 주소 테이블
CREATE TABLE user_addresses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    address_type VARCHAR(20) NOT NULL CHECK (address_type IN ('billing', 'shipping')),
    street_address VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100),
    postal_code VARCHAR(20) NOT NULL,
    country VARCHAR(50) NOT NULL,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 카테고리 테이블
CREATE TABLE categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name product_category UNIQUE NOT NULL,
    description TEXT,
    parent_id UUID REFERENCES categories(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 제품 테이블
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category_id UUID NOT NULL REFERENCES categories(id),
    sku VARCHAR(100) UNIQUE NOT NULL,
    price price_domain NOT NULL,
    sale_price price_domain,
    cost_price price_domain,
    stock_quantity positive_integer DEFAULT 0,
    min_stock_level positive_integer DEFAULT 10,
    weight DECIMAL(8, 2),
    dimensions JSONB,
    attributes JSONB,
    tags TEXT[],
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 제품 이미지 테이블
CREATE TABLE product_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    image_url VARCHAR(500) NOT NULL,
    alt_text VARCHAR(255),
    is_primary BOOLEAN DEFAULT false,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 제품 리뷰 테이블
CREATE TABLE product_reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating rating_domain NOT NULL CHECK (rating >= 1 AND rating <= 5),
    title VARCHAR(255),
    content TEXT,
    is_verified_purchase BOOLEAN DEFAULT false,
    helpful_count positive_integer DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, user_id)
);

-- 주문 테이블
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    order_number VARCHAR(50) UNIQUE NOT NULL,
    status order_status DEFAULT 'pending',
    subtotal price_domain NOT NULL,
    tax_amount price_domain DEFAULT 0,
    shipping_amount price_domain DEFAULT 0,
    discount_amount price_domain DEFAULT 0,
    total_amount price_domain NOT NULL,
    currency VARCHAR(3) DEFAULT 'KRW',
    shipping_address JSONB NOT NULL,
    billing_address JSONB NOT NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    shipped_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE
);

-- 주문 항목 테이블
CREATE TABLE order_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id),
    quantity positive_integer NOT NULL,
    unit_price price_domain NOT NULL,
    total_price price_domain NOT NULL,
    product_snapshot JSONB NOT NULL, -- 주문 시점의 제품 정보 스냅샷
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 결제 테이블
CREATE TABLE payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id),
    payment_method payment_method NOT NULL,
    amount price_domain NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'completed', 'failed', 'refunded')),
    transaction_id VARCHAR(255),
    payment_details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- 장바구니 테이블
CREATE TABLE shopping_cart (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    quantity positive_integer NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, product_id)
);

-- 위시리스트 테이블
CREATE TABLE wishlist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, product_id)
);

-- 쿠폰 테이블
CREATE TABLE coupons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    discount_type VARCHAR(20) NOT NULL CHECK (discount_type IN ('percentage', 'fixed_amount')),
    discount_value DECIMAL(10, 2) NOT NULL,
    minimum_amount DECIMAL(10, 2),
    usage_limit INTEGER,
    used_count INTEGER DEFAULT 0,
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 주문 쿠폰 사용 테이블
CREATE TABLE order_coupons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    coupon_id UUID NOT NULL REFERENCES coupons(id),
    discount_amount DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 재고 이력 테이블
CREATE TABLE inventory_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id),
    change_type VARCHAR(20) NOT NULL CHECK (change_type IN ('purchase', 'sale', 'adjustment', 'return')),
    quantity_change INTEGER NOT NULL,
    previous_quantity INTEGER NOT NULL,
    new_quantity INTEGER NOT NULL,
    reason TEXT,
    reference_id UUID, -- 주문 ID 등 관련 레코드
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- 로그 테이블
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(20) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    record_id UUID NOT NULL,
    old_values JSONB,
    new_values JSONB,
    user_id UUID REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 시스템 설정 테이블
CREATE TABLE system_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(100) UNIQUE NOT NULL,
    value TEXT,
    description TEXT,
    data_type VARCHAR(20) DEFAULT 'string' CHECK (data_type IN ('string', 'integer', 'boolean', 'json')),
    is_public BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
-- 사용자 관련 인덱스
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(id) WHERE is_active = true;
CREATE INDEX idx_users_created_at ON users(created_at);

-- 주소 관련 인덱스
CREATE INDEX idx_user_addresses_user_id ON user_addresses(user_id);
CREATE INDEX idx_user_addresses_type ON user_addresses(address_type);

-- 제품 관련 인덱스
CREATE INDEX idx_products_category_id ON products(category_id);
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_products_active ON products(id) WHERE is_active = true;
CREATE INDEX idx_products_stock ON products(stock_quantity) WHERE stock_quantity <= min_stock_level;
CREATE INDEX idx_products_tags ON products USING GIN(tags);
CREATE INDEX idx_products_attributes ON products USING GIN(attributes);
CREATE INDEX idx_products_created_at ON products(created_at);

-- 제품 이미지 관련 인덱스
CREATE INDEX idx_product_images_product_id ON product_images(product_id);
CREATE INDEX idx_product_images_primary ON product_images(product_id) WHERE is_primary = true;

-- 리뷰 관련 인덱스
CREATE INDEX idx_product_reviews_product_id ON product_reviews(product_id);
CREATE INDEX idx_product_reviews_user_id ON product_reviews(user_id);
CREATE INDEX idx_product_reviews_rating ON product_reviews(rating);
CREATE INDEX idx_product_reviews_created_at ON product_reviews(created_at);

-- 주문 관련 인덱스
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_orders_total_amount ON orders(total_amount);
CREATE INDEX idx_orders_number ON orders(order_number);

-- 주문 항목 관련 인덱스
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- 결제 관련 인덱스
CREATE INDEX idx_payments_order_id ON payments(order_id);
CREATE INDEX idx_payments_status ON payments(status);
CREATE INDEX idx_payments_method ON payments(payment_method);

-- 장바구니 관련 인덱스
CREATE INDEX idx_shopping_cart_user_id ON shopping_cart(user_id);
CREATE INDEX idx_shopping_cart_product_id ON shopping_cart(product_id);

-- 위시리스트 관련 인덱스
CREATE INDEX idx_wishlist_user_id ON wishlist(user_id);
CREATE INDEX idx_wishlist_product_id ON wishlist(product_id);

-- 쿠폰 관련 인덱스
CREATE INDEX idx_coupons_code ON coupons(code);
CREATE INDEX idx_coupons_valid_period ON coupons(valid_from, valid_until);
CREATE INDEX idx_coupons_active ON coupons(id) WHERE is_active = true;

-- 재고 로그 관련 인덱스
CREATE INDEX idx_inventory_logs_product_id ON inventory_logs(product_id);
CREATE INDEX idx_inventory_logs_created_at ON inventory_logs(created_at);
CREATE INDEX idx_inventory_logs_change_type ON inventory_logs(change_type);

-- 감사 로그 관련 인덱스
CREATE INDEX idx_audit_logs_table_name ON audit_logs(table_name);
CREATE INDEX idx_audit_logs_operation ON audit_logs(operation);
CREATE INDEX idx_audit_logs_record_id ON audit_logs(record_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

-- 시스템 설정 관련 인덱스
CREATE INDEX idx_system_settings_key ON system_settings(key);
CREATE INDEX idx_system_settings_public ON system_settings(id) WHERE is_public = true;

-- 트리거 함수 생성
-- updated_at 자동 업데이트 함수
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- updated_at 트리거 적용
CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_user_addresses_updated_at
    BEFORE UPDATE ON user_addresses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_products_updated_at
    BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_product_reviews_updated_at
    BEFORE UPDATE ON product_reviews
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_shopping_cart_updated_at
    BEFORE UPDATE ON shopping_cart
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_system_settings_updated_at
    BEFORE UPDATE ON system_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 감사 로그 트리거 함수
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (table_name, operation, record_id, old_values)
        VALUES (TG_TABLE_NAME, TG_OP, OLD.id, row_to_json(OLD));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (table_name, operation, record_id, old_values, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (table_name, operation, record_id, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(NEW));
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 감사 로그 트리거 적용 (중요 테이블에만)
CREATE TRIGGER trigger_users_audit
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER trigger_orders_audit
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER trigger_payments_audit
    AFTER INSERT OR UPDATE OR DELETE ON payments
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- 재고 로그 트리거 함수
CREATE OR REPLACE FUNCTION inventory_log_trigger()
RETURNS TRIGGER AS $$
DECLARE
    stock_change INTEGER;
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- 주문 항목 추가 시 재고 차감
        stock_change := -NEW.quantity;
    ELSIF TG_OP = 'UPDATE' THEN
        -- 주문 항목 수정 시 재고 조정
        stock_change := OLD.quantity - NEW.quantity;
    ELSIF TG_OP = 'DELETE' THEN
        -- 주문 항목 삭제 시 재고 복원
        stock_change := OLD.quantity;
    END IF;
    
    -- 재고 업데이트
    UPDATE products 
    SET stock_quantity = stock_quantity + stock_change
    WHERE id = COALESCE(NEW.product_id, OLD.product_id);
    
    -- 재고 로그 기록
    INSERT INTO inventory_logs (
        product_id, 
        change_type, 
        quantity_change, 
        previous_quantity, 
        new_quantity,
        reason,
        reference_id
    )
    SELECT 
        COALESCE(NEW.product_id, OLD.product_id),
        'sale',
        stock_change,
        stock_quantity - stock_change,
        stock_quantity,
        'Order item ' || TG_OP,
        COALESCE(NEW.id, OLD.id)
    FROM products 
    WHERE id = COALESCE(NEW.product_id, OLD.product_id);
    
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 재고 로그 트리거 적용
CREATE TRIGGER trigger_order_items_inventory
    AFTER INSERT OR UPDATE OR DELETE ON order_items
    FOR EACH ROW EXECUTE FUNCTION inventory_log_trigger();

-- 제품 검색 벡터 업데이트 함수
CREATE OR REPLACE FUNCTION update_product_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    -- 검색 벡터 업데이트 (전체 텍스트 검색을 위해)
    NEW.search_vector := to_tsvector('korean', 
        COALESCE(NEW.name, '') || ' ' || 
        COALESCE(NEW.description, '') || ' ' ||
        COALESCE(array_to_string(NEW.tags, ' '), '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 제품 테이블에 검색 벡터 컬럼 추가
ALTER TABLE products ADD COLUMN search_vector TSVECTOR;

-- 검색 벡터 트리거 적용
CREATE TRIGGER trigger_products_search_vector
    BEFORE INSERT OR UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_product_search_vector();

-- 검색 벡터 인덱스 생성
CREATE INDEX idx_products_search_vector ON products USING GIN(search_vector);

-- 뷰 생성
-- 제품 상세 뷰
CREATE VIEW product_details AS
SELECT 
    p.id,
    p.name,
    p.description,
    p.category_id,
    c.name as category_name,
    p.sku,
    p.price,
    p.sale_price,
    p.stock_quantity,
    p.attributes,
    p.tags,
    p.is_active,
    p.created_at,
    p.updated_at,
    (SELECT AVG(rating) FROM product_reviews pr WHERE pr.product_id = p.id) as average_rating,
    (SELECT COUNT(*) FROM product_reviews pr WHERE pr.product_id = p.id) as review_count,
    (SELECT image_url FROM product_images pi WHERE pi.product_id = p.id AND pi.is_primary = true LIMIT 1) as primary_image
FROM products p
LEFT JOIN categories c ON p.category_id = c.id;

-- 주문 상세 뷰
CREATE VIEW order_details AS
SELECT 
    o.id,
    o.user_id,
    u.username,
    u.email,
    o.order_number,
    o.status,
    o.subtotal,
    o.tax_amount,
    o.shipping_amount,
    o.discount_amount,
    o.total_amount,
    o.created_at,
    o.updated_at,
    COUNT(oi.id) as item_count,
    SUM(oi.quantity) as total_quantity
FROM orders o
JOIN users u ON o.user_id = u.id
LEFT JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.id, u.username, u.email;

-- 사용자 통계 뷰
CREATE VIEW user_statistics AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.created_at,
    COUNT(DISTINCT o.id) as order_count,
    COALESCE(SUM(o.total_amount), 0) as total_spent,
    COALESCE(AVG(o.total_amount), 0) as average_order_value,
    MAX(o.created_at) as last_order_date,
    COUNT(DISTINCT pr.id) as review_count,
    COALESCE(AVG(pr.rating), 0) as average_rating_given
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
LEFT JOIN product_reviews pr ON u.id = pr.user_id
GROUP BY u.id, u.username, u.email, u.created_at;

-- 제품 통계 뷰
CREATE VIEW product_statistics AS
SELECT 
    p.id,
    p.name,
    p.category_id,
    c.name as category_name,
    p.price,
    p.stock_quantity,
    COUNT(DISTINCT oi.order_id) as order_count,
    COALESCE(SUM(oi.quantity), 0) as total_sold,
    COALESCE(SUM(oi.total_price), 0) as total_revenue,
    COUNT(DISTINCT pr.id) as review_count,
    COALESCE(AVG(pr.rating), 0) as average_rating,
    MIN(o.created_at) as first_order_date,
    MAX(o.created_at) as last_order_date
FROM products p
LEFT JOIN categories c ON p.category_id = c.id
LEFT JOIN order_items oi ON p.id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.id
LEFT JOIN product_reviews pr ON p.id = pr.product_id
GROUP BY p.id, p.name, p.category_id, c.name, p.price, p.stock_quantity;

-- 함수 생성
-- 제품 재고 확인 함수
CREATE OR REPLACE FUNCTION check_product_stock(product_uuid UUID, required_quantity INTEGER)
RETURNS BOOLEAN AS $$
DECLARE
    current_stock INTEGER;
BEGIN
    SELECT stock_quantity INTO current_stock
    FROM products
    WHERE id = product_uuid;
    
    RETURN current_stock >= required_quantity;
END;
$$ LANGUAGE plpgsql;

-- 사용자 주문 내역 조회 함수
CREATE OR REPLACE FUNCTION get_user_orders(user_uuid UUID, limit_count INTEGER DEFAULT 10, offset_count INTEGER DEFAULT 0)
RETURNS TABLE (
    order_id UUID,
    order_number VARCHAR,
    status order_status,
    total_amount price_domain,
    created_at TIMESTAMP WITH TIME ZONE,
    item_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        o.id,
        o.order_number,
        o.status,
        o.total_amount,
        o.created_at,
        COUNT(oi.id)
    FROM orders o
    LEFT JOIN order_items oi ON o.id = oi.order_id
    WHERE o.user_id = user_uuid
    GROUP BY o.id, o.order_number, o.status, o.total_amount, o.created_at
    ORDER BY o.created_at DESC
    LIMIT limit_count
    OFFSET offset_count;
END;
$$ LANGUAGE plpgsql;

-- 인기 제품 조회 함수
CREATE OR REPLACE FUNCTION get_popular_products(days_back INTEGER DEFAULT 30, limit_count INTEGER DEFAULT 10)
RETURNS TABLE (
    product_id UUID,
    product_name VARCHAR,
    category_name VARCHAR,
    total_sold BIGINT,
    total_revenue price_domain,
    average_rating rating_domain
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id,
        p.name,
        c.name,
        COALESCE(SUM(oi.quantity), 0) as total_sold,
        COALESCE(SUM(oi.total_price), 0) as total_revenue,
        COALESCE(AVG(pr.rating), 0) as average_rating
    FROM products p
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN order_items oi ON p.id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.id AND o.created_at >= CURRENT_DATE - INTERVAL '1 day' * days_back
    LEFT JOIN product_reviews pr ON p.id = pr.product_id
    WHERE p.is_active = true
    GROUP BY p.id, p.name, c.name
    HAVING COALESCE(SUM(oi.quantity), 0) > 0
    ORDER BY total_sold DESC, total_revenue DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- 카테고리별 제품 통계 함수
CREATE OR REPLACE FUNCTION get_category_statistics()
RETURNS TABLE (
    category_id UUID,
    category_name product_category,
    product_count BIGINT,
    total_stock BIGINT,
    average_price price_domain,
    total_sold BIGINT,
    total_revenue price_domain
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.name,
        COUNT(p.id) as product_count,
        COALESCE(SUM(p.stock_quantity), 0) as total_stock,
        COALESCE(AVG(p.price), 0) as average_price,
        COALESCE(SUM(oi.quantity), 0) as total_sold,
        COALESCE(SUM(oi.total_price), 0) as total_revenue
    FROM categories c
    LEFT JOIN products p ON c.id = p.category_id AND p.is_active = true
    LEFT JOIN order_items oi ON p.id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.id
    GROUP BY c.id, c.name
    ORDER BY total_revenue DESC;
END;
$$ LANGUAGE plpgsql;

-- 주문 상태 업데이트 함수
CREATE OR REPLACE FUNCTION update_order_status(order_uuid UUID, new_status order_status)
RETURNS BOOLEAN AS $$
DECLARE
    current_status order_status;
BEGIN
    -- 현재 상태 확인
    SELECT status INTO current_status
    FROM orders
    WHERE id = order_uuid;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION '주문을 찾을 수 없습니다';
        RETURN false;
    END IF;
    
    -- 상태 전환 유효성 검사
    IF current_status = 'cancelled' AND new_status != 'cancelled' THEN
        RAISE EXCEPTION '취소된 주문은 상태를 변경할 수 없습니다';
        RETURN false;
    END IF;
    
    IF current_status = 'delivered' AND new_status != 'delivered' THEN
        RAISE EXCEPTION '배송완료된 주문은 상태를 변경할 수 없습니다';
        RETURN false;
    END IF;
    
    -- 상태 업데이트
    UPDATE orders 
    SET 
        status = new_status,
        updated_at = CURRENT_TIMESTAMP,
        shipped_at = CASE WHEN new_status = 'shipped' THEN CURRENT_TIMESTAMP ELSE shipped_at END,
        delivered_at = CASE WHEN new_status = 'delivered' THEN CURRENT_TIMESTAMP ELSE delivered_at END
    WHERE id = order_uuid;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- 주문 번호 생성 함수
CREATE OR REPLACE FUNCTION generate_order_number()
RETURNS TEXT AS $$
DECLARE
    order_number TEXT;
    order_count BIGINT;
BEGIN
    -- 날짜 기반 주문 번호 생성: ORD-YYYYMMDD-NNNN
    SELECT COUNT(*) + 1 INTO order_count
    FROM orders
    WHERE DATE(created_at) = CURRENT_DATE;
    
    order_number := 'ORD-' || TO_CHAR(CURRENT_DATE, 'YYYYMMDD') || '-' || LPAD(order_count::TEXT, 4, '0');
    
    -- 중복 확인 (동시성 문제 방지)
    WHILE EXISTS (SELECT 1 FROM orders WHERE order_number = order_number) LOOP
        order_count := order_count + 1;
        order_number := 'ORD-' || TO_CHAR(CURRENT_DATE, 'YYYYMMDD') || '-' || LPAD(order_count::TEXT, 4, '0');
    END LOOP;
    
    RETURN order_number;
END;
$$ LANGUAGE plpgsql;

-- 주문 생성 트리거 함수
CREATE OR REPLACE FUNCTION set_order_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.order_number IS NULL THEN
        NEW.order_number := generate_order_number();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 주문 번호 자동 생성 트리거
CREATE TRIGGER trigger_orders_set_order_number
    BEFORE INSERT ON orders
    FOR EACH ROW EXECUTE FUNCTION set_order_number();

-- 데이터베이스 설정
-- 기본값 설정
INSERT INTO system_settings (key, value, description, data_type, is_public) VALUES
('site_name', 'PostgreSQL 베스트 프랙티스 상점', '사이트 이름', 'string', true),
('site_description', 'PostgreSQL 학습을 위한 예제 전자상거래 사이트', '사이트 설명', 'string', true),
('default_currency', 'KRW', '기본 통화', 'string', true),
('tax_rate', '0.10', '부가세율', 'string', false),
('shipping_cost', '3000', '기본 배송비', 'string', false),
('free_shipping_threshold', '50000', '무료 배송 최소 주문 금액', 'string', false),
('max_review_length', '1000', '리뷰 최대 길이', 'integer', false),
('low_stock_threshold', '10', '재고 부족 알림 임계값', 'integer', false);

COMMENT ON DATABASE lecture_db IS 'PostgreSQL 베스트 프랙티스 강의자료 예제 데이터베이스';