# 트랜잭션 관리와 동시성 제어

## 학습 목표

- 트랜잭션의 ACID 특성 이해
- PostgreSQL 트랜잭션 제어 명령어 학습
- 동시성 제어 메커니즘과 잠금 이해
- 고립 수준과 그 영향 파악
- 데드락 예방 및 해결 방법 습득

## 트랜잭션 기본 개념

### ACID 특성

트랜잭션은 데이터베이스의 상태를 변경하는 논리적 작업 단위로, 다음 4가지 특성을 만족해야 합니다.

```sql
-- 원자성(Atomicity) 예제: 계좌 이체
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- 두 작업 모두 성공하거나 모두 실패

-- 일관성(Consistency) 예제: 제품 재고와 주문
BEGIN;
UPDATE products SET stock = stock - 1 WHERE id = 123 AND stock >= 1;
INSERT INTO orders (product_id, quantity) VALUES (123, 1);
COMMIT;  -- 데이터베이스 제약 조건 항상 만족

-- 고립성(Isolation) 예제: 동시 트랜잭션 간 간섭 방지
-- 격리 수준에 따라 다른 트랜잭션의 영향 제어

-- 지속성(Durability) 예제: 커밋된 데이터 영구 저장
-- 시스템 장애 후에도 커밋된 데이터는 유지
```

### 기본 트랜잭션 명령어

```sql
-- 트랜잭션 시작
BEGIN;
-- 또는
START TRANSACTION;

-- 트랜잭션 커밋 (변경사항 저장)
COMMIT;

-- 트랜잭션 롤백 (변경사항 취소)
ROLLBACK;

-- 저장점 설정
SAVEPOINT savepoint_name;

-- 저장점으로 롤백
ROLLBACK TO savepoint_name;

-- 저장점 해제
RELEASE SAVEPOINT savepoint_name;
```

### 트랜잭션 예제

```sql
-- 계좌 이체 트랜잭션
BEGIN;

-- 출금 계좌 확인
SELECT balance FROM accounts WHERE id = 1;

-- 출금
UPDATE accounts SET balance = balance - 500 WHERE id = 1;

-- 입금 계좌 확인
SELECT balance FROM accounts WHERE id = 2;

-- 입금
UPDATE accounts SET balance = balance + 500 WHERE id = 2;

-- 이체 기록
INSERT INTO transfers (from_account, to_account, amount, transfer_date)
VALUES (1, 2, 500, CURRENT_TIMESTAMP);

-- 트랜잭션 커밋
COMMIT;

-- 저장점 사용 예제
BEGIN;
UPDATE products SET price = price * 1.1 WHERE category = 'electronics';
SAVEPOINT price_update;

UPDATE products SET stock = stock - 10 WHERE id = 123;
-- 재고 부족 확인 후 롤백 결정
ROLLBACK TO price_update;

-- 가격 인상만 유지
COMMIT;
```

## 동시성 제어

### 동시성 문제 유형

```sql
-- 더티 읽기(Dirty Read): 커밋되지 않은 데이터 읽기
-- 세션 1
BEGIN;
UPDATE accounts SET balance = 1000 WHERE id = 1;
-- 아직 커밋하지 않음

-- 세션 2 (READ UNCOMMITTED 격리 수준에서만 가능)
SELECT balance FROM accounts WHERE id = 1;  -- 1000 읽음 (커밋되지 않은 데이터)

-- 반복 불가능 읽기(Non-repeatable Read): 같은 데이터를 다시 읽을 때 값이 변경
-- 세션 1
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 2000 읽음

-- 세션 2
BEGIN;
UPDATE accounts SET balance = 1500 WHERE id = 1;
COMMIT;

-- 세션 1
SELECT balance FROM accounts WHERE id = 1;  -- 1500 읽음 (값 변경)

-- 팬텀 읽기(Phantom Read): 같은 쿼리를 다시 실행할 때 새로운 행이 나타남
-- 세션 1
BEGIN;
SELECT * FROM accounts WHERE balance > 1000;  -- 5개 행

-- 세션 2
BEGIN;
INSERT INTO accounts (id, balance) VALUES (10, 2000);
COMMIT;

-- 세션 1
SELECT * FROM accounts WHERE balance > 1000;  -- 6개 행 (새로운 행)
```

### 잠금(Locking) 메커니즘

```sql
-- 명시적 잠금
-- 테이블 잠금
LOCK TABLE accounts IN SHARE MODE;           -- 공유 잠금 (읽기만 가능)
LOCK TABLE accounts IN EXCLUSIVE MODE;       -- 배타적 잠금 (읽기/쓰기 가능, 다른 트랜잭션 접근 불가)
LOCK TABLE accounts IN ACCESS EXCLUSIVE MODE; -- 접근 배타 잠금 (모든 접근 차단)

-- 행 잠금 (SELECT FOR UPDATE)
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;  -- 행 잠금
-- 다른 트랜잭션은 이 행을 수정할 수 없음
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- 행 잠금 (SELECT FOR SHARE)
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR SHARE;   -- 행 공유 잠금
-- 다른 트랜잭션은 읽을 수 있지만 수정할 수 없음
COMMIT;

-- NOWAIT 옵션 (잠금 대기하지 않음)
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;

-- SKIP LOCKED 옵션 (잠긴 행 건너뛰기)
SELECT * FROM accounts FOR UPDATE SKIP LOCKED;
```

### 잠금 충돌 확인

```sql
-- 잠금 상태 확인
SELECT 
    pid,
    state,
    query,
    wait_event_type,
    wait_event
FROM pg_stat_activity 
WHERE state != 'idle';

-- 잠금 대기 확인
SELECT 
    pid,
    mode,
    granted,
    relation::regclass as table_name
FROM pg_locks
WHERE NOT granted;

-- 잠금 충돌 상세 정보
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

## 격리 수준(Isolation Levels)

PostgreSQL은 4가지 격리 수준을 지원합니다.

### READ COMMITTED (기본값)

```sql
-- 격리 수준 설정
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- 또는 세션 전체에 설정
SET default_transaction_isolation = 'read committed';

-- READ COMMITTED 동작
-- 세션 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT balance FROM accounts WHERE id = 1;  -- 2000 읽음

-- 세션 2
BEGIN;
UPDATE accounts SET balance = 1500 WHERE id = 1;
COMMIT;

-- 세션 1
SELECT balance FROM accounts WHERE id = 1;  -- 1500 읽음 (커밋된 변경사항 보임)
COMMIT;
```

### REPEATABLE READ

```sql
-- REPEATABLE READ 설정
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- REPEATABLE READ 동작
-- 세션 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM accounts WHERE id = 1;  -- 2000 읽음 (스냅샷 생성)

-- 세션 2
BEGIN;
UPDATE accounts SET balance = 1500 WHERE id = 1;
COMMIT;

-- 세션 1
SELECT balance FROM accounts WHERE id = 1;  -- 여전히 2000 읽음 (스냅샷 유지)
COMMIT;

-- 직렬화 실패 예제
-- 세션 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

-- 세션 2
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
COMMIT;  -- 성공

-- 세션 1
COMMIT;  -- 직렬화 실패 오류 발생 가능
```

### SERIALIZABLE

```sql
-- SERIALIZABLE 설정
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- SERIALIZABLE 동작
-- 세션 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM accounts WHERE balance > 1000;  -- 5개 행

-- 세션 2
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
INSERT INTO accounts (id, balance) VALUES (10, 2000);
COMMIT;  -- 성공

-- 세션 1
SELECT COUNT(*) FROM accounts WHERE balance > 1000;  -- 여전히 5개 행
COMMIT;  -- 직렬화 실패 오류 발생 가능
```

### READ UNCOMMITTED

```sql
-- READ UNCOMMITTED 설정 (PostgreSQL에서는 READ COMMITTED로 동작)
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

-- PostgreSQL에서는 실제로 READ UNCOMMITTED를 지원하지 않음
-- 대신 READ COMMITTED로 동작함
```

## 데드락(Deadlock)

### 데드락 발생 조건

```sql
-- 데드락 예제
-- 세션 1
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- 계좌 1 잠금
UPDATE accounts SET balance = balance + 100 WHERE id = 2;  -- 계좌 2 잠금 대기

-- 세션 2
BEGIN;
UPDATE accounts SET balance = balance - 50 WHERE id = 2;   -- 계좌 2 잠금
UPDATE accounts SET balance = balance + 50 WHERE id = 1;   -- 계좌 1 잠금 대기

-- 데드락 발생! PostgreSQL이 한 트랜잭션을 롤백시킴
```

### 데드락 예방 전략

```sql
-- 1. 일관된 잠금 순서
-- 나쁜 예: 다른 순서로 잠금
-- 세션 1: 계좌 1 -> 계좌 2
-- 세션 2: 계좌 2 -> 계좌 1

-- 좋은 예: 항상 같은 순서로 잠금
-- 세션 1: 계좌 1 -> 계좌 2
-- 세션 2: 계좌 1 -> 계좌 2

-- 2. 잠금 시간 최소화
-- 나쁜 예: 긴 트랜잭션
BEGIN;
SELECT * FROM large_table WHERE condition;  -- 오래 걸리는 쿼리
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- 좋은 예: 짧은 트랜잭션
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- 3. 적절한 격리 수준 사용
-- 높은 격리 수준은 데드락 가능성 증가
-- 필요한 경우에만 높은 격리 수준 사용

-- 4. 재시도 로직 구현
-- 데드락 발생 시 자동 재시도
DO $$
BEGIN
    LOOP
        BEGIN
            -- 트랜잭션 실행
            UPDATE accounts SET balance = balance - 100 WHERE id = 1;
            UPDATE accounts SET balance = balance + 100 WHERE id = 2;
            EXIT;  -- 성공 시 루프 탈출
        EXCEPTION
            WHEN deadlock_detected THEN
                -- 데드락 발생 시 재시도
                PERFORM pg_sleep(0.1);  -- 짧은 대기
                CONTINUE;
        END;
    END LOOP;
END $$;
```

### 데드락 모니터링

```sql
-- 데드락 로그 확인
-- postgresql.conf 설정: log_lock_waits = on

-- 데드락 통계 확인
SELECT 
    datname,
    deadlock_count
FROM pg_stat_database
WHERE deadlock_count > 0;

-- 현재 잠금 대기 상황 확인
SELECT 
    pid,
    state,
    wait_event_type,
    wait_event,
    query
FROM pg_stat_activity
WHERE wait_event IS NOT NULL;
```

## 고급 트랜잭션 기법

### 저장점 활용

```sql
-- 복잡한 트랜잭션에서 부분 롤백
BEGIN;
    
    -- 첫 번째 작업
    INSERT INTO orders (customer_id, total_amount) VALUES (1, 1000);
    SAVEPOINT order_created;
    
    -- 재고 확인 및 차감
    UPDATE products SET stock = stock - 1 WHERE id = 123;
    
    -- 재고 부족 확인
    IF NOT FOUND THEN
        ROLLBACK TO order_created;  -- 주문 생성만 유지
        -- 대체 상품으로 주문 처리
        INSERT INTO orders (customer_id, total_amount) VALUES (1, 800);
    END IF;
    
    -- 결제 처리
    INSERT INTO payments (order_id, amount) VALUES (currval('orders_id_seq'), 1000);
    SAVEPOINT payment_processed;
    
    -- 결제 실패 시
    IF payment_failed THEN
        ROLLBACK TO payment_processed;  -- 주문은 유지, 결제만 취소
        INSERT INTO payment_failures (order_id, reason) VALUES (currval('orders_id_seq'), 'Insufficient funds');
    END IF;
    
COMMIT;
```

### 트랜잭션과 성능

```sql
-- 벌크 삽입 최적화
-- 나쁜 예: 개별 트랜잭션
INSERT INTO logs (message, created_at) VALUES ('Log 1', CURRENT_TIMESTAMP);
INSERT INTO logs (message, created_at) VALUES ('Log 2', CURRENT_TIMESTAMP);
INSERT INTO logs (message, created_at) VALUES ('Log 3', CURRENT_TIMESTAMP);

-- 좋은 예: 단일 트랜잭션
BEGIN;
INSERT INTO logs (message, created_at) VALUES ('Log 1', CURRENT_TIMESTAMP);
INSERT INTO logs (message, created_at) VALUES ('Log 2', CURRENT_TIMESTAMP);
INSERT INTO logs (message, created_at) VALUES ('Log 3', CURRENT_TIMESTAMP);
COMMIT;

-- 더 좋은 예: COPY 명령어 사용
COPY logs (message, created_at) FROM stdin;
Log 1	2023-01-01 10:00:00
Log 2	2023-01-01 10:00:01
Log 3	2023-01-01 10:00:02
\.

-- 트랜잭션 크기 조절
-- 너무 큰 트랜잭션은 메모리 사용량 증가 및 잠금 시간 증가
-- 적절한 크기로 트랜잭션 분할
DO $$
DECLARE
    batch_size INTEGER := 1000;
    offset_val INTEGER := 0;
    processed INTEGER := 1;
BEGIN
    WHILE processed > 0 LOOP
        BEGIN
            -- 배치 처리
            UPDATE large_table SET processed = true 
            WHERE id IN (
                SELECT id FROM large_table 
                WHERE processed = false 
                LIMIT batch_size
                OFFSET offset_val
            );
            
            GET DIAGNOSTICS processed = ROW_COUNT;
            offset_val := offset_val + batch_size;
            
            COMMIT;
        EXCEPTION
            WHEN OTHERS THEN
                ROLLBACK;
                RAISE;
        END;
    END LOOP;
END $$;
```

## 실습: 트랜잭션 관리 연습

은행 계좌 시스템의 트랜잭션을 안전하게 구현하는 연습을 해봅시다.

```sql
-- 계좌 테이블
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    account_number VARCHAR(20) UNIQUE NOT NULL,
    owner_name VARCHAR(100) NOT NULL,
    balance DECIMAL(15, 2) NOT NULL CHECK (balance >= 0),
    account_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 거래 내역 테이블
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    from_account INTEGER REFERENCES accounts(id),
    to_account INTEGER REFERENCES accounts(id),
    amount DECIMAL(15, 2) NOT NULL CHECK (amount > 0),
    transaction_type VARCHAR(20) NOT NULL,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- 안전한 이체 함수
CREATE OR REPLACE FUNCTION safe_transfer(
    from_account_num VARCHAR(20),
    to_account_num VARCHAR(20),
    amount DECIMAL(15, 2),
    description TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    from_id INTEGER;
    to_id INTEGER;
    current_balance DECIMAL(15, 2);
BEGIN
    -- 계좌 확인
    SELECT id, balance INTO from_id, current_balance
    FROM accounts 
    WHERE account_number = from_account_num;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION '출금 계좌가 존재하지 않습니다: %', from_account_num;
    END IF;
    
    SELECT id INTO to_id
    FROM accounts 
    WHERE account_number = to_account_num;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION '입금 계좌가 존재하지 않습니다: %', to_account_num;
    END IF;
    
    -- 잔액 확인
    IF current_balance < amount THEN
        RAISE EXCEPTION '잔액이 부족합니다. 현재 잔액: %, 이체 금액: %', current_balance, amount;
    END IF;
    
    -- 트랜잭션 시작
    BEGIN
        -- 출금
        UPDATE accounts 
        SET balance = balance - amount 
        WHERE id = from_id;
        
        -- 입금
        UPDATE accounts 
        SET balance = balance + amount 
        WHERE id = to_id;
        
        -- 거래 기록
        INSERT INTO transactions (from_account, to_account, amount, transaction_type, description)
        VALUES (from_id, to_id, amount, 'TRANSFER', description);
        
        RETURN TRUE;
    EXCEPTION
        WHEN OTHERS THEN
            RAISE EXCEPTION '이체 실패: %', SQLERRM;
            RETURN FALSE;
    END;
END;
$$ LANGUAGE plpgsql;

-- 동시성 제어를 위한 이체 함수 (잠금 사용)
CREATE OR REPLACE FUNCTION concurrent_safe_transfer(
    from_account_num VARCHAR(20),
    to_account_num VARCHAR(20),
    amount DECIMAL(15, 2),
    description TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    from_id INTEGER;
    to_id INTEGER;
    current_balance DECIMAL(15, 2);
BEGIN
    -- 계좌 확인 및 잠금
    SELECT id, balance INTO from_id, current_balance
    FROM accounts 
    WHERE account_number = from_account_num
    FOR UPDATE;  -- 행 잠금
    
    IF NOT FOUND THEN
        RAISE EXCEPTION '출금 계좌가 존재하지 않습니다: %', from_account_num;
    END IF;
    
    SELECT id INTO to_id
    FROM accounts 
    WHERE account_number = to_account_num
    FOR UPDATE;  -- 행 잠금
    
    IF NOT FOUND THEN
        RAISE EXCEPTION '입금 계좌가 존재하지 않습니다: %', to_account_num;
    END IF;
    
    -- 잔액 확인
    IF current_balance < amount THEN
        RAISE EXCEPTION '잔액이 부족합니다. 현재 잔액: %, 이체 금액: %', current_balance, amount;
    END IF;
    
    -- 출금
    UPDATE accounts 
    SET balance = balance - amount 
    WHERE id = from_id;
    
    -- 입금
    UPDATE accounts 
    SET balance = balance + amount 
    WHERE id = to_id;
    
    -- 거래 기록
    INSERT INTO transactions (from_account, to_account, amount, transaction_type, description)
    VALUES (from_id, to_id, amount, 'TRANSFER', description);
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION '이체 실패: %', SQLERRM;
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- 테스트 데이터
INSERT INTO accounts (account_number, owner_name, balance, account_type)
VALUES 
('123-456-789', '김철수', 1000000, 'checking'),
('987-654-321', '이영희', 500000, 'savings'),
('555-666-777', '박지성', 2000000, 'checking');

-- 이체 테스트
SELECT concurrent_safe_transfer('123-456-789', '987-654-321', 100000, '월세 이체');

-- 결과 확인
SELECT * FROM accounts ORDER BY id;
SELECT * FROM transactions ORDER BY id DESC LIMIT 1;
```

## 요약

효과적인 트랜잭션 관리를 위한 핵심 원칙:

1. **ACID 특성 이해**: 원자성, 일관성, 고립성, 지속성 보장
2. **적절한 격리 수준 선택**: 비즈니스 요구에 맞는 격리 수준 사용
3. **잠금 최소화**: 필요한 최소한의 잠금만 사용하고 시간 단축
4. **데드락 예방**: 일관된 잠금 순서와 짧은 트랜잭션 유지
5. **오류 처리**: 적절한 예외 처리와 재시도 로직 구현
6. **성능 고려**: 벌크 작업 최적화와 트랜잭션 크기 조절

다음 섹션에서는 PostgreSQL 보안 베스트 프랙티스에 대해 알아보겠습니다.

## 추가 자료

- [PostgreSQL 문서: 트랜잭션 제어](https://www.postgresql.org/docs/current/tutorial-transactions.html)
- [PostgreSQL 문서: 잠금](https://www.postgresql.org/docs/current/explicit-locking.html)
- [PostgreSQL 문서: 격리 수준](https://www.postgresql.org/docs/current/transaction-iso.html)