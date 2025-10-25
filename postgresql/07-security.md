# 보안 베스트 프랙티스

## 학습 목표

- PostgreSQL 인증 및 권한 관리 시스템 이해
- 데이터 암호화와 보안 설정 방법 학습
- SQL 인젝션 및 일반적인 공격 방어 기법 습득
- 감사 로그와 모니터링 방법 이해
- 데이터베이스 보안 강화 전략 파악

## 인증 및 권한 관리

### 사용자 관리

```sql
-- 사용자(롤) 생성
CREATE USER app_user WITH PASSWORD 'secure_password_123';
CREATE USER readonly_user WITH PASSWORD 'readonly_password_456';
CREATE USER admin_user WITH PASSWORD 'admin_password_789' SUPERUSER;

-- 사용자 속성 설정
ALTER USER app_user WITH CONNECTION LIMIT 10;  -- 최대 연결 수 제한
ALTER USER app_user WITH VALID UNTIL '2024-12-31';  -- 계정 만료일 설정
ALTER USER app_user WITH LOGIN;  -- 로그인 허용
ALTER USER readonly_user WITH NOLOGIN;  -- 로그인 불가

-- 사용자 정보 확인
\du  -- 사용자 목록 및 권한
SELECT usename, usecreatedb, usesuper, usecatupd, valuntil 
FROM pg_user;

-- 사용자 삭제
DROP USER IF EXISTS unused_user;
```

### 롤(Role) 기반 권한 관리

```sql
-- 롤 생성
CREATE ROLE app_developer;
CREATE ROLE app_readonly;
CREATE ROLE app_admin;

-- 롤에 권한 부여
GRANT CONNECT ON DATABASE myapp TO app_developer;
GRANT USAGE ON SCHEMA public TO app_developer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_developer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_developer;

-- 읽기 전용 롤 권한
GRANT CONNECT ON DATABASE myapp TO app_readonly;
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;

-- 관리자 롤 권한
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO app_admin;

-- 사용자에게 롤 부여
GRANT app_developer TO app_user;
GRANT app_readonly TO readonly_user;
GRANT app_admin TO admin_user;

-- 롤 계층 구조
GRANT app_readonly TO app_developer;  -- 개발자는 읽기 권한도 가짐
GRANT app_developer TO app_admin;   -- 관리자는 개발자 권한도 가짐

-- 롤 멤버십 확인
SELECT roleid, rolname, member, rolname AS member_role
FROM pg_roles
JOIN pg_auth_members ON pg_roles.oid = pg_auth_members.roleid
JOIN pg_roles AS member ON pg_auth_members.member = member.oid;
```

### 객체별 권한 관리

```sql
-- 테이블 권한
GRANT SELECT ON sensitive_data TO app_readonly;
GRANT SELECT, INSERT ON public_logs TO app_developer;
GRANT ALL ON users TO app_admin;

-- 컬럼 수준 권한
GRANT SELECT(id, name, email) ON users TO app_readonly;
GRANT UPDATE(email) ON users TO app_developer;

-- 행 수준 보안 (Row Level Security)
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;

-- 정책 생성
CREATE POLICY employee_own_policy ON employees
    FOR ALL
    TO app_user
    USING (employee_id = current_user_id());

CREATE POLICY manager_view_policy ON employees
    FOR SELECT
    TO app_manager
    USING (department_id IN (SELECT department_id FROM departments WHERE manager_id = current_user_id()));

-- 정책 확인
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual
FROM pg_policies;

-- 함수 권한
GRANT EXECUTE ON FUNCTION calculate_salary(employee_id INTEGER) TO app_developer;
REVOKE EXECUTE ON FUNCTION delete_user(user_id INTEGER) FROM app_developer;

-- 스키마 권한
GRANT USAGE ON SCHEMA app_schema TO app_user;
GRANT CREATE ON SCHEMA app_schema TO app_developer;
REVOKE ALL ON SCHEMA app_schema FROM public;
```

## 데이터 암호화

### 전송 암호화

```sql
-- SSL/TLS 설정 확인
SHOW ssl;  -- SSL 활성화 여부

-- SSL 인증서 정보 확인
SELECT ssl_is_used(), ssl_version(), ssl_cipher() FROM (SELECT 1) AS dummy;

-- 강제 SSL 연결 설정
-- postgresql.conf 파일 설정:
-- ssl = on
-- ssl_cert_file = 'server.crt'
-- ssl_key_file = 'server.key'

-- pg_hba.conf 파일 설정:
-- hostssl all all 0.0.0.0/0 md5

-- SSL 연결 테스트
psql "host=localhost dbname=mydb user=app_user sslmode=require"
```

### 저장 데이터 암호화

```sql
-- pgcrypto 확장 활성화
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 암호화 함수
-- 대칭키 암호화
SELECT encrypt('Secret data', 'encryption_key', 'aes');
SELECT decrypt(encrypt('Secret data', 'encryption_key', 'aes'), 'encryption_key', 'aes');

-- 비밀번호 해싱
SELECT crypt('my_password', gen_salt('bf'));  -- Blowfish 해시
SELECT crypt('my_password', gen_salt('md5'));  -- MD5 해시

-- 비밀번호 확인
SELECT crypt('my_password', stored_hash) = stored_hash AS password_match
FROM users 
WHERE username = 'test_user';

-- 암호화 컬럼 예제
CREATE TABLE secure_data (
    id SERIAL PRIMARY KEY,
    sensitive_info BYTEA,  -- 암호화된 데이터
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 데이터 암호화하여 삽입
INSERT INTO secure_data (sensitive_info)
VALUES (encrypt('Very sensitive information', 'secret_key_123', 'aes'));

-- 데이터 복호화하여 조회
SELECT 
    id,
    convert_from(decrypt(sensitive_info, 'secret_key_123', 'aes'), 'UTF-8') as decrypted_info
FROM secure_data;

-- 암호화 뷰
CREATE VIEW decrypted_secure_data AS
SELECT 
    id,
    convert_from(decrypt(sensitive_info, 'secret_key_123', 'aes'), 'UTF-8') as info,
    created_at
FROM secure_data;
```

### 투명 데이터 암호화 (TDE)

PostgreSQL은 기본적으로 TDE를 지원하지 않지만, 다음 방법으로 구현할 수 있습니다:

```sql
-- 파일 시스템 수준 암호화 (Linux 예제)
-- 1. 암호화된 파일 시스템 생성
-- sudo cryptsetup luksFormat /dev/sdb1
-- sudo cryptsetup open /dev/sdb1 encrypted_db
-- sudo mkfs.ext4 /dev/mapper/encrypted_db
-- sudo mount /dev/mapper/encrypted_db /var/lib/postgresql

-- 2. PostgreSQL 데이터 디렉토리 이동
-- sudo systemctl stop postgresql
-- sudo mv /var/lib/postgresql/* /encrypted_db/
-- sudo mount /dev/mapper/encrypted_db /var/lib/postgresql
-- sudo systemctl start postgresql

-- 애플리케이션 수준 암호화 함수
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(data TEXT, key TEXT)
RETURNS BYTEA AS $$
BEGIN
    RETURN encrypt(data::bytea, key, 'aes');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION decrypt_sensitive_data(encrypted_data BYTEA, key TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN convert_from(decrypt(encrypted_data, key, 'aes'), 'UTF-8');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

## SQL 인젝션 방어

### 매개변수화된 쿼리 사용

```sql
-- 나쁜 예: 문자열 연결으로 쿼리 생성 (SQL 인젝션 취약)
-- 애플리케이션 코드 예시 (Python)
# cursor.execute("SELECT * FROM users WHERE username = '" + username + "'")

-- 좋은 예: 매개변수화된 쿼리
# cursor.execute("SELECT * FROM users WHERE username = %s", (username,))

-- PostgreSQL 함수에서의 안전한 쿼리 작성
CREATE OR REPLACE FUNCTION get_user_by_username_safe(username TEXT)
RETURNS TABLE(id INTEGER, name TEXT, email TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT id, name, email
    FROM users
    WHERE users.username = get_user_by_username_safe.username;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 동적 SQL 사용 시 주의사항
CREATE OR REPLACE FUNCTION search_users_safe(search_term TEXT)
RETURNS TABLE(id INTEGER, name TEXT, email TEXT) AS $$
BEGIN
    -- format() 함수와 %I, %L 사용
    RETURN QUERY EXECUTE format('
        SELECT id, name, email 
        FROM users 
        WHERE name ILIKE %L OR email ILIKE %L
    ', '%' || search_term || '%', '%' || search_term || '%');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

### 입력 검증 및 정화

```sql
-- 입력 검증 함수
CREATE OR REPLACE FUNCTION validate_email(email TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
END;
$$ LANGUAGE plpgsql;

-- 도메인 타입으로 입력 제약
CREATE DOMAIN valid_email AS TEXT
CHECK (VALUE ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

CREATE DOMAIN positive_integer AS INTEGER
CHECK (VALUE > 0);

-- 도메인 사용
CREATE TABLE user_registrations (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email valid_email NOT NULL,
    age positive_integer
);

-- 트리거를 통한 입력 검증
CREATE OR REPLACE FUNCTION validate_user_input()
RETURNS TRIGGER AS $$
BEGIN
    -- 사용자 이름 검증
    IF NEW.username ~* '[^a-zA-Z0-9_]' THEN
        RAISE EXCEPTION '사용자 이름은 영문, 숫자, 밑줄만 포함할 수 있습니다';
    END IF;
    
    -- 이메일 검증
    IF NOT validate_email(NEW.email) THEN
        RAISE EXCEPTION '유효하지 않은 이메일 주소입니다';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_validate_user_input
    BEFORE INSERT OR UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION validate_user_input();
```

## 감사 로그 및 모니터링

### 로깅 설정

```sql
-- 로깅 설정 확인
SHOW log_destination;
SHOW logging_collector;
SHOW log_directory;
SHOW log_filename;
SHOW log_min_duration_statement;

-- 로깅 설정 변경
-- postgresql.conf 파일 설정:
-- log_destination = 'stderr'
-- logging_collector = on
-- log_directory = 'log'
-- log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
-- log_min_duration_statement = 1000  -- 1초 이상 걸리는 쿼리 로그
-- log_checkpoints = on
-- log_connections = on
-- log_disconnections = on
-- log_lock_waits = on

-- 세션별 로깅
SET log_min_duration_statement = 500;  -- 현재 세션에서만 0.5초 이상 쿼리 로그
SET log_statement = 'all';  -- 모든 쿼리 로그
```

### 감사 트리거

```sql
-- 감사 테이블 생성
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    user_name TEXT NOT NULL,
    old_values JSONB,
    new_values JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 감사 트리거 함수
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, user_name, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, current_user, row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, user_name, old_values, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, current_user, row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, user_name, old_values)
        VALUES (TG_TABLE_NAME, TG_OP, current_user, row_to_json(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 감사 트리거 적용
CREATE TRIGGER audit_users_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- 민감한 테이블에 대한 감사
CREATE TRIGGER audit_sensitive_data_trigger
    AFTER INSERT OR UPDATE OR DELETE ON sensitive_data
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
```

### 접근 모니터링

```sql
-- 접근 로그 뷰
CREATE VIEW recent_access_log AS
SELECT 
    datname as database_name,
    usename as username,
    client_addr as client_ip,
    state,
    query_start,
    state_change,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start DESC;

-- 활성 세션 모니터링
SELECT 
    pid,
    usename,
    client_addr,
    application_name,
    state,
    backend_start,
    query_start,
    state_change,
    query
FROM pg_stat_activity
WHERE state != 'idle';

-- 장기 실행 쿼리 모니터링
SELECT 
    pid,
    usename,
    client_addr,
    query_start,
    now() - query_start as duration,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 minutes'
ORDER BY duration DESC;

-- 잠금 대기 모니터링
SELECT 
    pid,
    state,
    wait_event_type,
    wait_event,
    query
FROM pg_stat_activity
WHERE wait_event IS NOT NULL;
```

## 보안 강화 전략

### 네트워크 보안

```sql
-- pg_hba.conf 설정 예제
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# 로컬 연결
local   all             postgres                                peer
local   all             all                                     md5

# IPv4 로컬 연결
host    all             all             127.0.0.1/32            md5

# 특정 IP 대역만 허용
host    myapp           app_user        192.168.1.0/24          md5
host    myapp           readonly_user   192.168.1.0/24          md5

# SSL 강제
hostssl all             all             0.0.0.0/0               md5

# 관리자 접근 제한
host    all             admin_user      10.0.0.100/32           md5

# 연결 제한
# postgresql.conf 설정:
# max_connections = 100
# superuser_reserved_connections = 3
```

### 데이터베이스 보안 설정

```sql
-- 기본 보안 설정
-- postgresql.conf 설정:

# 연결 보안
listen_addresses = 'localhost'  # 필요한 주소만 지정
port = 5432
max_connections = 100

# 인증 보안
authentication_timeout = 1min
password_encryption = scram-sha-256

# 로깅 보안
log_connections = on
log_disconnections = on
log_duration = on
log_statement = 'mod'  # 수정 작업만 로그

# 메모리 보안
shared_buffers = 256MB
temp_buffers = 32MB
work_mem = 4MB

# 자동 vacuum 설정
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
```

### 정기적인 보안 점검

```sql
-- 보안 점검 쿼리 모음

-- 1. 슈퍼유저 확인
SELECT usename, usesuper, usecreatedb, usecatupd 
FROM pg_user 
WHERE usesuper = true;

-- 2. 비밀번호 없는 사용자 확인
SELECT usename 
FROM pg_shadow 
WHERE passwd IS NULL;

-- 3. 오래된 비밀번호 확인
SELECT usename, valuntil 
FROM pg_shadow 
WHERE valuntil < CURRENT_DATE;

-- 4. 과도한 권한을 가진 사용자 확인
SELECT 
    grantee,
    table_schema,
    table_name,
    privilege_type
FROM information_schema.role_table_grants
WHERE grantee != 'postgres'
  AND grantee NOT IN (SELECT rolname FROM pg_roles WHERE rolsuper = true);

-- 5. 공개 스키마의 권한 확인
SELECT 
    grantee,
    table_schema,
    table_name,
    privilege_type
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND grantee = 'PUBLIC';

-- 6. 활성 연결 모니터링
SELECT 
    datname,
    usename,
    client_addr,
    count(*) as connection_count
FROM pg_stat_activity
GROUP BY datname, usename, client_addr
ORDER BY connection_count DESC;

-- 7. 최근 로그인 활동
SELECT 
    usename,
    client_addr,
    backend_start
FROM pg_stat_activity
WHERE backend_start > CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY backend_start DESC;
```

## 실습: 보안 설정 연습

안전한 사용자 관리 시스템을 구현하는 연습을 해봅시다.

```sql
-- 보안 설정을 위한 테이블
CREATE TABLE app_users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    failed_login_attempts INTEGER DEFAULT 0,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 역할 테이블
CREATE TABLE user_roles (
    user_id INTEGER REFERENCES app_users(id) ON DELETE CASCADE,
    role_name VARCHAR(50) NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_name)
);

-- 로그인 시도 기록
CREATE TABLE login_attempts (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    ip_address INET NOT NULL,
    success BOOLEAN NOT NULL,
    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 보안 함수
CREATE OR REPLACE FUNCTION hash_password(password TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN crypt(password, gen_salt('bf', 12));
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION verify_password(password TEXT, stored_hash TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN crypt(password, stored_hash) = stored_hash;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 로그인 함수
CREATE OR REPLACE FUNCTION authenticate_user(username TEXT, password TEXT, ip_address INET)
RETURNS TABLE(success BOOLEAN, user_id INTEGER, message TEXT) AS $$
DECLARE
    user_record RECORD;
    attempts INTEGER;
BEGIN
    -- 로그인 시도 기록
    INSERT INTO login_attempts (username, ip_address, success)
    VALUES (username, ip_address, false);
    
    -- 사용자 확인
    SELECT * INTO user_record
    FROM app_users
    WHERE username = authenticate_user.username
      AND is_active = true;
    
    IF NOT FOUND THEN
        -- 사용자 없음
        RETURN QUERY SELECT false, NULL::INTEGER, '사용자 이름 또는 비밀번호가 올바르지 않습니다'::TEXT;
        RETURN;
    END IF;
    
    -- 비밀번호 확인
    IF NOT verify_password(password, user_record.password_hash) THEN
        -- 비밀번호 불일치
        UPDATE app_users
        SET failed_login_attempts = failed_login_attempts + 1
        WHERE id = user_record.id;
        
        -- 계정 잠금 확인
        IF user_record.failed_login_attempts >= 4 THEN  -- 이번 실패로 5회
            UPDATE app_users
            SET is_active = false
            WHERE id = user_record.id;
            
            RETURN QUERY SELECT false, NULL::INTEGER, '로그인 실패 횟수 초과로 계정이 잠겼습니다'::TEXT;
        END IF;
        
        RETURN QUERY SELECT false, NULL::INTEGER, '사용자 이름 또는 비밀번호가 올바르지 않습니다'::TEXT;
    END IF;
    
    -- 로그인 성공
    UPDATE login_attempts
    SET success = true
    WHERE id = currval('login_attempts_id_seq');
    
    UPDATE app_users
    SET 
        failed_login_attempts = 0,
        last_login = CURRENT_TIMESTAMP
    WHERE id = user_record.id;
    
    RETURN QUERY SELECT true, user_record.id, '로그인 성공'::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 사용자 생성 함수
CREATE OR REPLACE FUNCTION create_app_user(
    username TEXT,
    email TEXT,
    password TEXT,
    roles TEXT[] DEFAULT ARRAY['user']
) RETURNS INTEGER AS $$
DECLARE
    user_id INTEGER;
BEGIN
    -- 사용자 생성
    INSERT INTO app_users (username, email, password_hash)
    VALUES (username, email, hash_password(password))
    RETURNING id INTO user_id;
    
    -- 역할 할당
    INSERT INTO user_roles (user_id, role_name)
    SELECT user_id, unnest(roles);
    
    RETURN user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 테스트
SELECT create_app_user('testuser', 'test@example.com', 'securepassword123', ARRAY['user', 'editor']);

-- 인증 테스트
SELECT * FROM authenticate_user('testuser', 'wrongpassword', '127.0.0.1');
SELECT * FROM authenticate_user('testuser', 'securepassword123', '127.0.0.1');

-- 결과 확인
SELECT * FROM app_users WHERE username = 'testuser';
SELECT * FROM user_roles WHERE user_id = (SELECT id FROM app_users WHERE username = 'testuser');
SELECT * FROM login_attempts WHERE username = 'testuser' ORDER BY attempt_time DESC;
```

## 요약

PostgreSQL 보안 강화를 위한 핵심 원칙:

1. **최소 권한 원칙**: 사용자에게 필요한 최소한의 권한만 부여
2. **강력한 인증**: 안전한 비밀번호 정책과 SSL/TLS 사용
3. **데이터 암호화**: 전송 및 저장 데이터 암호화
4. **입력 검증**: SQL 인젝션 방어를 위한 입력 검증과 매개변수화된 쿼리
5. **감사 및 모니터링**: 접근 로그와 활동 모니터링
6. **정기적인 보안 점검**: 취약점 식별 및 수정

다음 섹션에서는 PostgreSQL 백업 및 복구 전략에 대해 알아보겠습니다.

## 추가 자료

- [PostgreSQL 문서: 인증](https://www.postgresql.org/docs/current/auth-methods.html)
- [PostgreSQL 문서: 권한](https://www.postgresql.org/docs/current/ddl-priv.html)
- [PostgreSQL 문서: 암호화](https://www.postgresql.org/docs/current/encryption-options.html)
- [OWASP SQL 인젝션 방지 가이드](https://owasp.org/www-community/attacks/SQL_Injection)