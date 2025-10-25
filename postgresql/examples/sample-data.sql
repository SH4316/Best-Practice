-- PostgreSQL 베스트 프랙티스 강의자료 예제 데이터베이스 샘플 데이터
-- 생성일: 2023-10-24
-- 설명: 전자상거래 시스템 예제 데이터

-- 카테고리 데이터 삽입
INSERT INTO categories (id, name, description) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'electronics', '전자제품 및 가전제품'),
('550e8400-e29b-41d4-a716-446655440002', 'clothing', '의류 및 패션 아이템'),
('550e8400-e29b-41d4-a716-446655440003', 'books', '도서 및 출판물'),
('550e8400-e29b-41d4-a716-446655440004', 'home', '가정용품 및 가구'),
('550e8400-e29b-41d4-a716-446655440005', 'sports', '스포츠 용품 및 장비'),
('550e8400-e29b-41d4-a716-446655440006', 'toys', '장난감 및 취미용품');

-- 하위 카테고리 삽입
INSERT INTO categories (id, name, description, parent_id) VALUES
('550e8400-e29b-41d4-a716-446655440011', 'smartphones', '스마트폰 및 휴대폰', '550e8400-e29b-41d4-a716-446655440001'),
('550e8400-e29b-41d4-a716-446655440012', 'laptops', '노트북 및 컴퓨터', '550e8400-e29b-41d4-a716-446655440001'),
('550e8400-e29b-41d4-a716-446655440013', 'audio', '오디오 장비', '550e8400-e29b-41d4-a716-446655440001'),
('550e8400-e29b-41d4-a716-446655440021', 'mens_clothing', '남성 의류', '550e8400-e29b-41d4-a716-446655440002'),
('550e8400-e29b-41d4-a716-446655440022', 'womens_clothing', '여성 의류', '550e8400-e29b-41d4-a716-446655440002'),
('550e8400-e29b-41d4-a716-446655440031', 'fiction', '소설', '550e8400-e29b-41d4-a716-446655440003'),
('550e8400-e29b-41d4-a716-446655440032', 'non_fiction', '비소설', '550e8400-e29b-41d4-a716-446655440003');

-- 사용자 데이터 삽입
INSERT INTO users (id, username, email, password_hash, first_name, last_name, phone, role, is_active, email_verified, created_at) VALUES
('110e8400-e29b-41d4-a716-446655440001', 'john_doe', 'john.doe@example.com', crypt('password123', gen_salt('bf')), 'John', 'Doe', '010-1234-5678', 'customer', true, true, '2023-01-15 10:30:00'),
('110e8400-e29b-41d4-a716-446655440002', 'jane_smith', 'jane.smith@example.com', crypt('password456', gen_salt('bf')), 'Jane', 'Smith', '010-2345-6789', 'customer', true, true, '2023-02-20 14:15:00'),
('110e8400-e29b-41d4-a716-446655440003', 'bob_wilson', 'bob.wilson@example.com', crypt('password789', gen_salt('bf')), 'Bob', 'Wilson', '010-3456-7890', 'customer', true, true, '2023-03-10 09:45:00'),
('110e8400-e29b-41d4-a716-446655440004', 'alice_brown', 'alice.brown@example.com', crypt('passwordabc', gen_salt('bf')), 'Alice', 'Brown', '010-4567-8901', 'customer', true, true, '2023-04-05 16:20:00'),
('110e8400-e29b-41d4-a716-446655440005', 'charlie_davis', 'charlie.davis@example.com', crypt('passworddef', gen_salt('bf')), 'Charlie', 'Davis', '010-5678-9012', 'customer', true, false, '2023-05-12 11:30:00'),
('110e8400-e29b-41d4-a716-446655440006', 'admin_user', 'admin@example.com', crypt('admin123', gen_salt('bf')), 'Admin', 'User', '010-0000-0000', 'admin', true, true, '2023-01-01 00:00:00'),
('110e8400-e29b-41d4-a716-446655440007', 'vendor_user', 'vendor@example.com', crypt('vendor123', gen_salt('bf')), 'Vendor', 'User', '010-1111-1111', 'vendor', true, true, '2023-02-01 00:00:00');

-- 사용자 주소 데이터 삽입
INSERT INTO user_addresses (id, user_id, address_type, street_address, city, state, postal_code, country, is_default) VALUES
('210e8400-e29b-41d4-a716-446655440001', '110e8400-e29b-41d4-a716-446655440001', 'shipping', '123 Main St', 'Seoul', 'Seoul', '12345', 'South Korea', true),
('210e8400-e29b-41d4-a716-446655440002', '110e8400-e29b-41d4-a716-446655440001', 'billing', '123 Main St', 'Seoul', 'Seoul', '12345', 'South Korea', true),
('210e8400-e29b-41d4-a716-446655440003', '110e8400-e29b-41d4-a716-446655440002', 'shipping', '456 Oak Ave', 'Busan', 'Busan', '23456', 'South Korea', true),
('210e8400-e29b-41d4-a716-446655440004', '110e8400-e29b-41d4-a716-446655440002', 'billing', '456 Oak Ave', 'Busan', 'Busan', '23456', 'South Korea', true),
('210e8400-e29b-41d4-a716-446655440005', '110e8400-e29b-41d4-a716-446655440003', 'shipping', '789 Pine Rd', 'Incheon', 'Incheon', '34567', 'South Korea', true),
('210e8400-e29b-41d4-a716-446655440006', '110e8400-e29b-41d4-a716-446655440003', 'billing', '789 Pine Rd', 'Incheon', 'Incheon', '34567', 'South Korea', true);

-- 제품 데이터 삽입
INSERT INTO products (id, name, description, category_id, sku, price, sale_price, cost_price, stock_quantity, min_stock_level, weight, dimensions, attributes, tags, is_active, created_at) VALUES
('310e8400-e29b-41d4-a716-446655440001', 'iPhone 14 Pro', '최신 Apple 스마트폰', '550e8400-e29b-41d4-a716-446655440011', 'IPHONE14PRO001', 1200000, 1150000, 900000, 50, 10, 206.0, '{"length": 147.5, "width": 71.5, "height": 7.85, "unit": "mm"}', '{"brand": "Apple", "model": "iPhone 14 Pro", "color": "Deep Purple", "storage": "128GB", "screen_size": "6.1"}', ARRAY['smartphone', 'apple', 'ios', '5g'], true, '2023-01-15 10:00:00'),
('310e8400-e29b-41d4-a716-446655440002', 'Samsung Galaxy S23', '최신 Samsung 스마트폰', '550e8400-e29b-41d4-a716-446655440011', 'GALAXYS23001', 1100000, NULL, 800000, 75, 15, 168.0, '{"length": 157.8, "width": 74.8, "height": 7.6, "unit": "mm"}', '{"brand": "Samsung", "model": "Galaxy S23", "color": "Phantom Black", "storage": "256GB", "screen_size": "6.1"}', ARRAY['smartphone', 'samsung', 'android', '5g'], true, '2023-01-20 11:00:00'),
('310e8400-e29b-41d4-a716-446655440003', 'MacBook Pro 14"', '고성능 노트북', '550e8400-e29b-41d4-a716-446655440012', 'MACBOOKPRO14001', 2500000, 2400000, 1800000, 30, 5, 1600.0, '{"length": 311.5, "width": 221.2, "height": 15.5, "unit": "mm"}', '{"brand": "Apple", "model": "MacBook Pro 14\"", "processor": "M2 Pro", "ram": "16GB", "storage": "512GB SSD"}', ARRAY['laptop', 'apple', 'macos', 'm2'], true, '2023-02-01 09:00:00'),
('310e8400-e29b-41d4-a716-446655440004', 'Dell XPS 15', '윈도우 고성능 노트북', '550e8400-e29b-41d4-a716-446655440012', 'DELLXPS15001', 2000000, NULL, 1500000, 25, 5, 1800.0, '{"length": 354.0, "width": 235.0, "height": 18.0, "unit": "mm"}', '{"brand": "Dell", "model": "XPS 15", "processor": "Intel i7", "ram": "16GB", "storage": "1TB SSD"}', ARRAY['laptop', 'dell', 'windows', 'intel'], true, '2023-02-05 14:00:00'),
('310e8400-e29b-41d4-a716-446655440005', 'Sony WH-1000XM4', '노이즈 캔슬링 헤드폰', '550e8400-e29b-41d4-a716-446655440013', 'SONYWH1000XM4', 350000, 320000, 250000, 100, 20, 254.0, '{"length": 254.0, "width": 203.0, "height": 43.0, "unit": "mm"}', '{"brand": "Sony", "model": "WH-1000XM4", "type": "Over-ear", "battery_life": "30 hours", "noise_cancelling": true}', ARRAY['headphones', 'sony', 'noise_cancelling', 'wireless'], true, '2023-02-10 16:00:00'),
('310e8400-e29b-41d4-a716-446655440006', 'Men''s T-Shirt', '면 소재 기본 티셔츠', '550e8400-e29b-41d4-a716-446655440021', 'MENS_TSHIRT001', 25000, NULL, 10000, 200, 50, 150.0, '{"size": "L", "material": "Cotton 100%", "color": "White"}', '{"brand": "Basic Wear", "material": "Cotton", "fit": "Regular"}', ARRAY['clothing', 'mens', 'tshirt', 'cotton'], true, '2023-03-01 10:00:00'),
('310e8400-e29b-41d4-a716-446655440007', 'Women''s Jeans', '여성용 청바지', '550e8400-e29b-41d4-a716-446655440022', 'WOMENS_JEANS001', 80000, 72000, 40000, 80, 20, 400.0, '{"size": "M", "material": "Denim", "color": "Blue"}', '{"brand": "Fashion Denim", "material": "Denim", "fit": "Slim"}', ARRAY['clothing', 'womens', 'jeans', 'denim'], true, '2023-03-05 13:00:00'),
('310e8400-e29b-41d4-a716-446655440008', 'PostgreSQL Guide', 'PostgreSQL 데이터베이스 가이드북', '550e8400-e29b-41d4-a716-446655440031', 'BOOK_PG_GUIDE', 45000, NULL, 25000, 150, 30, 500.0, '{"pages": 500, "language": "Korean", "format": "Paperback"}', '{"author": "Database Expert", "publisher": "Tech Books", "isbn": "978-1234567890"}', ARRAY['book', 'postgresql', 'database', 'programming'], true, '2023-03-10 15:00:00'),
('310e8400-e29b-41d4-a716-446655440009', 'SQL Best Practices', 'SQL 베스트 프랙티스 가이드', '550e8400-e29b-41d4-a716-446655440032', 'BOOK_SQL_BEST', 50000, 45000, 30000, 120, 25, 600.0, '{"pages": 600, "language": "Korean", "format": "Hardcover"}', '{"author": "SQL Master", "publisher": "Programming Press", "isbn": "978-0987654321"}', ARRAY['book', 'sql', 'database', 'programming'], true, '2023-03-15 11:00:00'),
('310e8400-e29b-41d4-a716-446655440010', 'Coffee Maker', '자동 커피 머신', '550e8400-e29b-41d4-a716-446655440004', 'COFFEE_MAKER001', 150000, 135000, 90000, 40, 10, 3000.0, '{"length": 300.0, "width": 250.0, "height": 400.0, "unit": "mm"}', '{"brand": "BrewMaster", "model": "AutoBrew 2000", "capacity": "1.5L", "power": "1000W"}', ARRAY['home', 'kitchen', 'coffee', 'appliance'], true, '2023-04-01 09:00:00');

-- 제품 이미지 데이터 삽입
INSERT INTO product_images (id, product_id, image_url, alt_text, is_primary, sort_order) VALUES
('410e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440001', 'https://example.com/images/iphone14pro_front.jpg', 'iPhone 14 Pro 앞면', true, 1),
('410e8400-e29b-41d4-a716-446655440002', '310e8400-e29b-41d4-a716-446655440001', 'https://example.com/images/iphone14pro_back.jpg', 'iPhone 14 Pro 뒷면', false, 2),
('410e8400-e29b-41d4-a716-446655440003', '310e8400-e29b-41d4-a716-446655440002', 'https://example.com/images/galaxys23_front.jpg', 'Galaxy S23 앞면', true, 1),
('410e8400-e29b-41d4-a716-446655440004', '310e8400-e29b-41d4-a716-446655440003', 'https://example.com/images/macbookpro14.jpg', 'MacBook Pro 14"', true, 1),
('410e8400-e29b-41d4-a716-446655440005', '310e8400-e29b-41d4-a716-446655440005', 'https://example.com/images/sony_wh1000xm4.jpg', 'Sony WH-1000XM4 헤드폰', true, 1),
('410e8400-e29b-41d4-a716-446655440006', '310e8400-e29b-41d4-a716-446655440006', 'https://example.com/images/mens_tshirt_white.jpg', '남성용 흰색 티셔츠', true, 1),
('410e8400-e29b-41d4-a716-446655440007', '310e8400-e29b-41d4-a716-446655440007', 'https://example.com/images/womens_jeans_blue.jpg', '여성용 청바지', true, 1),
('410e8400-e29b-41d4-a716-446655440008', '310e8400-e29b-41d4-a716-446655440008', 'https://example.com/images/postgresql_guide.jpg', 'PostgreSQL 가이드북 표지', true, 1),
('410e8400-e29b-41d4-a716-446655440009', '310e8400-e29b-41d4-a716-446655440009', 'https://example.com/images/sql_best_practices.jpg', 'SQL 베스트 프랙티스 표지', true, 1),
('410e8400-e29b-41d4-a716-446655440010', '310e8400-e29b-41d4-a716-446655440010', 'https://example.com/images/coffee_maker.jpg', '자동 커피 머신', true, 1);

-- 제품 리뷰 데이터 삽입
INSERT INTO product_reviews (id, product_id, user_id, rating, title, content, is_verified_purchase, helpful_count, created_at) VALUES
('510e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440001', '110e8400-e29b-41d4-a716-446655440001', 5, '최고의 스마트폰!', 'iPhone 14 Pro는 정말 훌륭한 스마트폰입니다. 카메라 성능이 뛰어나고 배터리도 오래갑니다.', true, 12, '2023-02-01 10:30:00'),
('510e8400-e29b-41d4-a716-446655440002', '310e8400-e29b-41d4-a716-446655440001', '110e8400-e29b-41d4-a716-446655440002', 4, '좋지만 비싸요', '성능은 정말 좋지만 가격이 부담스럽습니다. 할인 때 구매하시는 걸 추천합니다.', true, 8, '2023-02-05 14:20:00'),
('510e8400-e29b-41d4-a716-446655440003', '310e8400-e29b-41d4-a716-446655440002', '110e8400-e29b-41d4-a716-446655440003', 5, '갤럭시 최고!', 'Samsung Galaxy S23는 정말 만족합니다. 화면도 선명하고 성능도 훌륭해요.', true, 15, '2023-02-10 09:15:00'),
('510e8400-e29b-41d4-a716-446655440004', '310e8400-e29b-41d4-a716-446655440003', '110e8400-e29b-41d4-a716-446655440004', 4, '만족하지만...', '전반적으로 만족하지만 배터리 소모가 빠른 편입니다.', true, 6, '2023-02-15 16:45:00'),
('510e8400-e29b-41d4-a716-446655440005', '310e8400-e29b-41d4-a716-446655440003', '110e8400-e29b-41d4-a716-446655440005', 3, '그냥 그래요', '기대했던 것보다는 별로였습니다. 가격에 비해 기능이 부족한 느낌.', true, 2, '2023-02-20 11:30:00'),
('510e8400-e29b-41d4-a716-446655440006', '310e8400-e29b-41d4-a716-446655440005', '110e8400-e29b-41d4-a716-446655440001', 5, '음질 최고!', 'Sony 헤드폰은 정말 대박입니다. 노이즈 캔슬링도 완벽하고 음질도 훌륭해요.', true, 20, '2023-03-01 13:20:00'),
('510e8400-e29b-41d4-a716-446655440007', '310e8400-e29b-41d4-a716-446655440008', '110e8400-e29b-41d4-a716-446655440002', 5, 'PostgreSQL 바이블', 'PostgreSQL을 배우는 데 정말 좋은 책입니다. 초보자도 쉽게 따라할 수 있어요.', true, 18, '2023-03-05 10:15:00'),
('510e8400-e29b-41d4-a716-446655440008', '310e8400-e29b-41d4-a716-446655440009', '110e8400-e29b-41d4-a716-446655440003', 4, 'SQL 필수 도서', 'SQL 개발자라면 꼭 읽어봐야 할 책입니다. 실용적인 예제가 많아요.', true, 14, '2023-03-10 15:30:00');

-- 쿠폰 데이터 삽입
INSERT INTO coupons (id, code, description, discount_type, discount_value, minimum_amount, usage_limit, used_count, valid_from, valid_until, is_active, created_at) VALUES
('610e8400-e29b-41d4-a716-446655440001', 'WELCOME10', '신규 가입 환영 10% 할인', 'percentage', 10.00, 10000, 1000, 245, '2023-01-01 00:00:00', '2023-12-31 23:59:59', true, '2023-01-01 00:00:00'),
('610e8400-e29b-41d4-a716-446655440002', 'SUMMER20', '여름 시즌 20% 할인', 'percentage', 20.00, 50000, 500, 123, '2023-06-01 00:00:00', '2023-08-31 23:59:59', true, '2023-06-01 00:00:00'),
('610e8400-e29b-41d4-a716-446655440003', 'FLAT5000', '5,000원 고정 할인', 'fixed_amount', 5000.00, 30000, 2000, 567, '2023-01-01 00:00:00', '2023-12-31 23:59:59', true, '2023-01-01 00:00:00'),
('610e8400-e29b-41d4-a716-446655440004', 'FLASH30', '플래시 세일 30% 할인', 'percentage', 30.00, 100000, 100, 45, '2023-07-01 00:00:00', '2023-07-07 23:59:59', true, '2023-07-01 00:00:00');

-- 주문 데이터 삽입
INSERT INTO orders (id, user_id, order_number, status, subtotal, tax_amount, shipping_amount, discount_amount, total_amount, currency, shipping_address, billing_address, notes, created_at, updated_at, shipped_at, delivered_at) VALUES
('710e8400-e29b-41d4-a716-446655440001', '110e8400-e29b-41d4-a716-446655440001', 'ORD-20230115-0001', 'delivered', 1200000, 120000, 3000, 120000, 1303000, 'KRW', '{"street": "123 Main St", "city": "Seoul", "state": "Seoul", "postal_code": "12345", "country": "South Korea", "recipient": "John Doe"}', '{"street": "123 Main St", "city": "Seoul", "state": "Seoul", "postal_code": "12345", "country": "South Korea"}', '빠른 배송 부탁드립니다.', '2023-01-15 10:30:00', '2023-01-17 14:20:00', '2023-01-16 10:00:00', '2023-01-17 14:20:00'),
('710e8400-e29b-41d4-a716-446655440002', '110e8400-e29b-41d4-a716-446655440002', 'ORD-20230220-0001', 'shipped', 1100000, 110000, 3000, NULL, 1213000, 'KRW', '{"street": "456 Oak Ave", "city": "Busan", "state": "Busan", "postal_code": "23456", "country": "South Korea", "recipient": "Jane Smith"}', '{"street": "456 Oak Ave", "city": "Busan", "state": "Busan", "postal_code": "23456", "country": "South Korea"}', NULL, '2023-02-20 14:15:00', '2023-02-22 09:30:00', '2023-02-21 15:00:00', NULL),
('710e8400-e29b-41d4-a716-446655440003', '110e8400-e29b-41d4-a716-446655440003', 'ORD-20230310-0001', 'processing', 2550000, 255000, 0, 255000, 2550000, 'KRW', '{"street": "789 Pine Rd", "city": "Incheon", "state": "Incheon", "postal_code": "34567", "country": "South Korea", "recipient": "Bob Wilson"}', '{"street": "789 Pine Rd", "city": "Incheon", "state": "Incheon", "postal_code": "34567", "country": "South Korea"}', '주말 배송 희망', '2023-03-10 09:45:00', '2023-03-10 10:00:00', NULL, NULL),
('710e8400-e29b-41d4-a716-446655440004', '110e8400-e29b-41d4-a716-446655440004', 'ORD-20230405-0001', 'pending', 350000, 35000, 3000, 35000, 353000, 'KRW', '{"street": "123 Main St", "city": "Seoul", "state": "Seoul", "postal_code": "12345", "country": "South Korea", "recipient": "Alice Brown"}', '{"street": "123 Main St", "city": "Seoul", "state": "Seoul", "postal_code": "12345", "country": "South Korea"}', NULL, '2023-04-05 16:20:00', '2023-04-05 16:25:00', NULL, NULL),
('710e8400-e29b-41d4-a716-446655440005', '110e8400-e29b-41d4-a716-446655440001', 'ORD-20230201-0002', 'delivered', 105000, 10500, 3000, 10500, 108000, 'KRW', '{"street": "123 Main St", "city": "Seoul", "state": "Seoul", "postal_code": "12345", "country": "South Korea", "recipient": "John Doe"}', '{"street": "123 Main St", "city": "Seoul", "state": "Seoul", "postal_code": "12345", "country": "South Korea"}', NULL, '2023-02-01 11:00:00', '2023-02-03 16:45:00', '2023-02-02 11:30:00', '2023-02-03 16:45:00'),
('710e8400-e29b-41d4-a716-446655440006', '110e8400-e29b-41d4-a716-446655440002', 'ORD-20230305-0002', 'cancelled', 80000, 8000, 3000, 8000, 83000, 'KRW', '{"street": "456 Oak Ave", "city": "Busan", "state": "Busan", "postal_code": "23456", "country": "South Korea", "recipient": "Jane Smith"}', '{"street": "456 Oak Ave", "city": "Busan", "state": "Busan", "postal_code": "23456", "country": "South Korea"}', '고객 요청으로 취소', '2023-03-05 13:30:00', '2023-03-05 14:00:00', NULL, NULL);

-- 주문 항목 데이터 삽입
INSERT INTO order_items (id, order_id, product_id, quantity, unit_price, total_price, product_snapshot, created_at) VALUES
('810e8400-e29b-41d4-a716-446655440001', '710e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440001', 1, 1200000, 1200000, '{"id": "310e8400-e29b-41d4-a716-446655440001", "name": "iPhone 14 Pro", "sku": "IPHONE14PRO001", "attributes": {"brand": "Apple", "model": "iPhone 14 Pro", "color": "Deep Purple"}}', '2023-01-15 10:30:00'),
('810e8400-e29b-41d4-a716-446655440002', '710e8400-e29b-41d4-a716-446655440002', '310e8400-e29b-41d4-a716-446655440002', 1, 1100000, 1100000, '{"id": "310e8400-e29b-41d4-a716-446655440002", "name": "Samsung Galaxy S23", "sku": "GALAXYS23001", "attributes": {"brand": "Samsung", "model": "Galaxy S23", "color": "Phantom Black"}}', '2023-02-20 14:15:00'),
('810e8400-e29b-41d4-a716-446655440003', '710e8400-e29b-41d4-a716-446655440003', '310e8400-e29b-41d4-a716-446655440003', 1, 2400000, 2400000, '{"id": "310e8400-e29b-41d4-a716-446655440003", "name": "MacBook Pro 14\"", "sku": "MACBOOKPRO14001", "attributes": {"brand": "Apple", "model": "MacBook Pro 14\"", "processor": "M2 Pro"}}', '2023-03-10 09:45:00'),
('810e8400-e29b-41d4-a716-446655440004', '710e8400-e29b-41d4-a716-446655440004', '310e8400-e29b-41d4-a716-446655440005', 1, 320000, 320000, '{"id": "310e8400-e29b-41d4-a716-446655440005", "name": "Sony WH-1000XM4", "sku": "SONYWH1000XM4", "attributes": {"brand": "Sony", "model": "WH-1000XM4", "type": "Over-ear"}}', '2023-04-05 16:20:00'),
('810e8400-e29b-41d4-a716-446655440005', '710e8400-e29b-41d4-a716-446655440005', '310e8400-e29b-41d4-a716-446655440005', 1, 350000, 350000, '{"id": "310e8400-e29b-41d4-a716-446655440005", "name": "Sony WH-1000XM4", "sku": "SONYWH1000XM4", "attributes": {"brand": "Sony", "model": "WH-1000XM4", "type": "Over-ear"}}', '2023-02-01 11:00:00'),
('810e8400-e29b-41d4-a716-446655440006', '710e8400-e29b-41d4-a716-446655440006', '310e8400-e29b-41d4-a716-446655440007', 1, 80000, 80000, '{"id": "310e8400-e29b-41d4-a716-446655440007", "name": "Women''s Jeans", "sku": "WOMENS_JEANS001", "attributes": {"brand": "Fashion Denim", "size": "M", "color": "Blue"}}', '2023-03-05 13:30:00');

-- 결제 데이터 삽입
INSERT INTO payments (id, order_id, payment_method, amount, status, transaction_id, payment_details, created_at, processed_at) VALUES
('910e8400-e29b-41d4-a716-446655440001', '710e8400-e29b-41d4-a716-446655440001', 'credit_card', 1303000, 'completed', 'TXN_20230115_001', '{"card_type": "Visa", "last4": "1234", "approval_code": "AP123456"}', '2023-01-15 10:35:00', '2023-01-15 10:36:00'),
('910e8400-e29b-41d4-a716-446655440002', '710e8400-e29b-41d4-a716-446655440002', 'bank_transfer', 1213000, 'completed', 'TXN_20230220_001', '{"bank": "KB Bank", "account": "1234567890"}', '2023-02-20 14:20:00', '2023-02-21 09:15:00'),
('910e8400-e29b-41d4-a716-446655440003', '710e8400-e29b-41d4-a716-446655440003', 'mobile_payment', 2550000, 'pending', 'TXN_20230310_001', '{"provider": "Kakao Pay", "phone": "010-3456-7890"}', '2023-03-10 09:50:00', NULL),
('910e8400-e29b-41d4-a716-446655440004', '710e8400-e29b-41d4-a716-446655440004', 'credit_card', 353000, 'pending', 'TXN_20230405_001', '{"card_type": "Mastercard", "last4": "5678"}', '2023-04-05 16:25:00', NULL),
('910e8400-e29b-41d4-a716-446655440005', '710e8400-e29b-41d4-a716-446655440005', 'cash_on_delivery', 108000, 'completed', 'TXN_20230201_002', '{"delivery_person": "김배송", "contact": "010-9999-8888"}', '2023-02-01 11:05:00', '2023-02-03 16:50:00'),
('910e8400-e29b-41d4-a716-446655440006', '710e8400-e29b-41d4-a716-446655440006', 'credit_card', 83000, 'refunded', 'TXN_20230305_002', '{"card_type": "Visa", "last4": "9876", "refund_reason": "Customer request"}', '2023-03-05 13:35:00', '2023-03-06 10:20:00');

-- 주문 쿠폰 사용 데이터 삽입
INSERT INTO order_coupons (id, order_id, coupon_id, discount_amount, created_at) VALUES
('a10e8400-e29b-41d4-a716-446655440001', '710e8400-e29b-41d4-a716-446655440001', '610e8400-e29b-41d4-a716-446655440001', 120000, '2023-01-15 10:30:00'),
('a10e8400-e29b-41d4-a716-446655440002', '710e8400-e29b-41d4-a716-446655440004', '610e8400-e29b-41d4-a716-446655440002', 105000, '2023-04-05 16:20:00'),
('a10e8400-e29b-41d4-a716-446655440003', '710e8400-e29b-41d4-a716-446655440005', '610e8400-e29b-41d4-a716-446655440003', 5000, '2023-02-01 11:00:00'),
('a10e8400-e29b-41d4-a716-446655440004', '710e8400-e29b-41d4-a716-446655440006', '610e8400-e29b-41d4-a716-446655440003', 5000, '2023-03-05 13:30:00');

-- 장바구니 데이터 삽입
INSERT INTO shopping_cart (id, user_id, product_id, quantity, created_at, updated_at) VALUES
('b10e8400-e29b-41d4-a716-446655440001', '110e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440002', 1, '2023-04-10 10:00:00', '2023-04-10 10:00:00'),
('b10e8400-e29b-41d4-a716-446655440002', '110e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440008', 2, '2023-04-10 10:05:00', '2023-04-10 10:05:00'),
('b10e8400-e29b-41d4-a716-446655440003', '110e8400-e29b-41d4-a716-446655440002', '310e8400-e29b-41d4-a716-446655440004', 1, '2023-04-11 14:30:00', '2023-04-11 14:30:00'),
('b10e8400-e29b-41d4-a716-446655440004', '110e8400-e29b-41d4-a716-446655440003', '310e8400-e29b-41d4-a716-446655440006', 3, '2023-04-12 09:15:00', '2023-04-12 09:15:00'),
('b10e8400-e29b-41d4-a716-446655440005', '110e8400-e29b-41d4-a716-446655440004', '310e8400-e29b-41d4-a716-446655440010', 1, '2023-04-13 16:45:00', '2023-04-13 16:45:00');

-- 위시리스트 데이터 삽입
INSERT INTO wishlist (id, user_id, product_id, created_at) VALUES
('c10e8400-e29b-41d4-a716-446655440001', '110e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440003', '2023-02-01 12:00:00'),
('c10e8400-e29b-41d4-a716-446655440002', '110e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440004', '2023-02-05 15:30:00'),
('c10e8400-e29b-41d4-a716-446655440003', '110e8400-e29b-41d4-a716-446655440002', '310e8400-e29b-41d4-a716-446655440001', '2023-02-10 10:15:00'),
('c10e8400-e29b-41d4-a716-446655440004', '110e8400-e29b-41d4-a716-446655440003', '310e8400-e29b-41d4-a716-446655440009', '2023-03-01 14:20:00'),
('c10e8400-e29b-41d4-a716-446655440005', '110e8400-e29b-41d4-a716-446655440004', '310e8400-e29b-41d4-a716-446655440005', '2023-03-15 11:45:00');

-- 재고 이력 데이터 삽입
INSERT INTO inventory_logs (id, product_id, change_type, quantity_change, previous_quantity, new_quantity, reason, reference_id, created_at, created_by) VALUES
('d10e8400-e29b-41d4-a716-446655440001', '310e8400-e29b-41d4-a716-446655440001', 'sale', -1, 51, 50, 'Order item INSERT', '810e8400-e29b-41d4-a716-446655440001', '2023-01-15 10:30:00', '110e8400-e29b-41d4-a716-446655440006'),
('d10e8400-e29b-41d4-a716-446655440002', '310e8400-e29b-41d4-a716-446655440002', 'sale', -1, 76, 75, 'Order item INSERT', '810e8400-e29b-41d4-a716-446655440002', '2023-02-20 14:15:00', '110e8400-e29b-41d4-a716-446655440006'),
('d10e8400-e29b-41d4-a716-446655440003', '310e8400-e29b-41d4-a716-446655440003', 'sale', -1, 31, 30, 'Order item INSERT', '810e8400-e29b-41d4-a716-446655440003', '2023-03-10 09:45:00', '110e8400-e29b-41d4-a716-446655440006'),
('d10e8400-e29b-41d4-a716-446655440004', '310e8400-e29b-41d4-a716-446655440005', 'sale', -1, 101, 100, 'Order item INSERT', '810e8400-e29b-41d4-a716-446655440005', '2023-02-01 11:00:00', '110e8400-e29b-41d4-a716-446655440006'),
('d10e8400-e29b-41d4-a716-446655440005', '310e8400-e29b-41d4-a716-446655440005', 'sale', -1, 100, 99, 'Order item INSERT', '810e8400-e29b-41d4-a716-446655440004', '2023-04-05 16:20:00', '110e8400-e29b-41d4-a716-446655440006'),
('d10e8400-e29b-41d4-a716-446655440006', '310e8400-e29b-41d4-a716-446655440007', 'sale', -1, 81, 80, 'Order item INSERT', '810e8400-e29b-41d4-a716-446655440006', '2023-03-05 13:30:00', '110e8400-e29b-41d4-a716-446655440006'),
('d10e8400-e29b-41d4-a716-446655440007', '310e8400-e29b-41d4-a716-446655440007', 'return', 1, 80, 81, 'Order cancelled', '810e8400-e29b-41d4-a716-446655440006', '2023-03-06 10:00:00', '110e8400-e29b-41d4-a716-446655440006'),
('d10e8400-e29b-41d4-a716-446655440008', '310e8400-e29b-41d4-a716-446655440008', 'purchase', 50, 150, 200, 'Initial stock', NULL, '2023-01-01 00:00:00', '110e8400-e29b-41d4-a716-446655440007'),
('d10e8400-e29b-41d4-a716-446655440009', '310e8400-e29b-41d4-a716-446655440009', 'purchase', 30, 120, 150, 'Initial stock', NULL, '2023-01-01 00:00:00', '110e8400-e29b-41d4-a716-446655440007'),
('d10e8400-e29b-41d4-a716-446655440010', '310e8400-e29b-41d4-a716-446655440010', 'purchase', 20, 40, 60, 'Initial stock', NULL, '2023-01-01 00:00:00', '110e8400-e29b-41d4-a716-446655440007');

-- 쿠폰 사용 횟수 업데이트
UPDATE coupons SET used_count = used_count + 1 WHERE id IN (
    '610e8400-e29b-41d4-a716-446655440001',
    '610e8400-e29b-41d4-a716-446655440002',
    '610e8400-e29b-41d4-a716-446655440003'
);

-- 검색 벡터 업데이트 (트리거가 자동으로 처리하지만 명시적으로 업데이트)
UPDATE products SET search_vector = to_tsvector('korean', 
    COALESCE(name, '') || ' ' || 
    COALESCE(description, '') || ' ' ||
    COALESCE(array_to_string(tags, ' '), '')
);

-- 통계 정보 업데이트
ANALYZE;

-- 데이터 확인 쿼리
SELECT 'Users' as table_name, COUNT(*) as record_count FROM users
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
UNION ALL
SELECT 'Shopping Cart', COUNT(*) FROM shopping_cart
UNION ALL
SELECT 'Wishlist', COUNT(*) FROM wishlist
ORDER BY table_name;