# 데이터베이스 설계 베스트 프랙티스

## 학습 목표

- 정규화의 원리와 중요성 이해
- 효율적인 테이블 설계 방법 학습
- 관계 설계와 외래 키 사용법 습득
- 데이터 무결성 보장 방법 이해

## 정규화(Normalization)

정규화는 데이터 중복을 최소화하고 데이터 무결성을 보장하기 위해 테이블을 구조화하는 과정입니다.

### 제1정규형 (1NF)

테이블의 모든 속성 값이 원자값(atomic value)을 가져야 합니다.

```sql
-- 나쁜 예: 비원자적 값
CREATE TABLE orders_bad (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    products TEXT  -- "노트북,마우스,키보드"
);

-- 좋은 예: 제1정규형
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_name VARCHAR(100),
    quantity INTEGER,
    price DECIMAL(10,2)
);
```

### 제2정규형 (2NF)

제1정규형을 만족하면서, 모든 비주요 속성이 기본 키에 완전히 함수적으로 종속되어야 합니다.

```sql
-- 나쁜 예: 부분 종속성
CREATE TABLE order_items_bad (
    order_id INTEGER,
    product_id INTEGER,
    product_name VARCHAR(100),  -- product_id에만 종속
    quantity INTEGER,            -- (order_id, product_id)에 종속
    price DECIMAL(10,2),         -- product_id에만 종속
    PRIMARY KEY (order_id, product_id)
);

-- 좋은 예: 제2정규형
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2)
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER
);
```

### 제3정규형 (3NF)

제2정규형을 만족하면서, 모든 비주요 속성이 기본 키에 이행적으로 종속되지 않아야 합니다.

```sql
-- 나쁜 예: 이행 종속성
CREATE TABLE employees_bad (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER,
    department_name VARCHAR(100),  -- department_id에 종속
    location VARCHAR(100)          -- department_id에 종속
);

-- 좋은 예: 제3정규형
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    location VARCHAR(100)
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER REFERENCES departments(id)
);
```

## 기본 키 설계

### 기본 키 선택 원칙

1. **고유성**: 각 행을 고유하게 식별
2. **NULL이 아님**: 항상 값을 가짐
3. **불변성**: 변경되지 않음
4. **단순성**: 가능한 짧고 단순할 것

```sql
-- 자연 키 vs 인조 키
-- 자연 키 (Natural Key)
CREATE TABLE users_natural (
    email VARCHAR(255) PRIMARY KEY,  -- 이메일은 자연 키가 될 수 있음
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인조 키 (Surrogate Key) - 권장
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- 인조 키
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 복합 기본 키

```sql
-- 연결 테이블에서의 복합 기본 키
CREATE TABLE user_roles (
    user_id INTEGER REFERENCES users(id),
    role_id INTEGER REFERENCES roles(id),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_id)
);

-- 또는 인조 키 사용
CREATE TABLE user_roles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    role_id INTEGER REFERENCES roles(id),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, role_id)
);
```

## 외래 키와 관계 설계

### 일대다 관계

```sql
-- 부서와 직원 (일대다)
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    -- 참조 무결성 옵션
    -- ON DELETE CASCADE: 부서 삭제 시 모든 직원 삭제
    -- ON DELETE SET NULL: 부서 삭제 시 직원의 department_id를 NULL로 설정
    -- ON DELETE RESTRICT: 부서 삭제 시 직원이 있으면 삭제 방지 (기본값)
    ON DELETE SET NULL
);
```

### 다대다 관계

```sql
-- 학생과 과목 (다대다)
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    credits INTEGER
);

-- 연결 테이블
CREATE TABLE enrollments (
    id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    grade VARCHAR(2),
    UNIQUE (student_id, course_id)
);
```

### 일대일 관계

```sql
-- 사용자와 프로필 (일대일)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    bio TEXT,
    avatar_url VARCHAR(255)
);
```

## 데이터 무결성 제약 조건

### NOT NULL 제약 조건

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,  -- 필수 값
    description TEXT,              -- 선택적 값
    price DECIMAL(10,2) NOT NULL CHECK (price > 0)
);
```

### CHECK 제약 조건

```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INTEGER CHECK (age >= 18),
    salary DECIMAL(10,2) CHECK (salary >= 0),
    email VARCHAR(255) CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- 복합 CHECK 제약 조건
ALTER TABLE employees 
ADD CONSTRAINT check_salary_age 
CHECK (age < 25 OR salary >= 20000);
```

### UNIQUE 제약 조건

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20)
);

-- 복합 UNIQUE 제약 조건
CREATE TABLE room_bookings (
    id SERIAL PRIMARY KEY,
    room_id INTEGER NOT NULL,
    booking_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    UNIQUE (room_id, booking_date, start_time)
);
```

## 테이블과 컬럼 명명 규칙

### 권장 명명 규칙

1. **소문자 사용**: `user_profiles` (권장) vs `UserProfiles` (비권장)
2. **밑줄로 단어 구분**: `order_items` (권장) vs `orderitems` (비권장)
3. **복수형 테이블 이름**: `users`, `products`, `orders`
4. **의미 있는 이름**: `created_at` (권장) vs `c_date` (비권장)
5. **일관된 접두사/접미사**: `is_active`, `has_permission`, `can_edit`

```sql
-- 좋은 예
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 나쁜 예
CREATE TABLE UserProfile (
    ID SERIAL PRIMARY KEY,
    UserID INTEGER NOT NULL,
    FName VARCHAR(100),
    LName VARCHAR(100),
    active BOOLEAN DEFAULT true,
    cdate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mdate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 타임스탬프와 감사 컬럼

```sql
CREATE TABLE audit_example (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    
    -- 생성 및 수정 시간
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 생성자 및 수정자 (사용자 ID)
    created_by INTEGER,
    updated_by INTEGER,
    
    -- 버전 관리 (낙관적 잠금)
    version INTEGER DEFAULT 1
);

-- updated_at 자동 업데이트를 위한 트리거
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_audit_example_updated_at 
    BEFORE UPDATE ON audit_example 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

## 실습: 데이터베이스 설계 연습

간단한 블로그 시스템의 데이터베이스를 설계해 봅시다.

```sql
-- 사용자 테이블
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 카테고리 테이블
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 블로그 글 테이블
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    excerpt TEXT,
    author_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'archived')),
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 태그 테이블
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 글-태그 연결 테이블 (다대다 관계)
CREATE TABLE post_tags (
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, tag_id)
);

-- 댓글 테이블
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    author_name VARCHAR(100),  -- 비회원 댓글을 위한 필드
    author_email VARCHAR(255),
    content TEXT NOT NULL,
    is_approved BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 요약

효과적인 데이터베이스 설계를 위한 핵심 원칙:

1. **정규화**: 데이터 중복 최소화와 무결성 보장
2. **적절한 기본 키 선택**: 인조 키 사용 권장
3. **명확한 관계 정의**: 외래 키와 참조 무결성
4. **데이터 무결성 제약 조건**: NOT NULL, CHECK, UNIQUE
5. **일관된 명명 규칙**: 가독성과 유지보수성 향상
6. **감사 컬럼 포함**: 생성/수정 시간과 사용자 정보

다음 섹션에서는 PostgreSQL의 다양한 데이터 타입과 스키마 설계 가이드라인에 대해 알아보겠습니다.

## 추가 자료

- [PostgreSQL 문서: 데이터베이스 정규화](https://www.postgresql.org/docs/current/ddl-constraints.html)
- [데이터베이스 설계 베스트 프랙티스](https://www.1keydata.com/kr/database-normalization/)