# 성능 튜닝과 모니터링

## 학습 목표

- PostgreSQL 성능 튜닝의 핵심 원리 이해
- 메모리, 디스크 I/O, CPU 최적화 방법 학습
- 성능 모니터링 도구와 지표 파악
- 병목 현상 식별 및 해결 방법 습득
- 장기적인 성능 관리 전략 이해

## PostgreSQL 아키텍처와 성능

### 메모리 구조

```sql
-- 메모리 설정 확인
SHOW shared_buffers;      -- 공유 버퍼 (기본값: 128MB)
SHOW effective_cache_size; -- 효과적 캐시 크기 (기본값: 4GB)
SHOW work_mem;            -- 작업 메모리 (기본값: 4MB)
SHOW maintenance_work_mem; -- 유지보수 작업 메모리 (기본값: 64MB)
SHOW wal_buffers;         -- WAL 버퍼 (기본값: 4MB)

-- 메모리 사용 통계
SELECT 
    name,
    setting,
    unit,
    short_desc
FROM pg_settings
WHERE name LIKE '%mem%' OR name LIKE '%buffer%'
ORDER BY name;
```

### 프로세스 구조

```sql
-- 활성 프로세스 확인
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    backend_start,
    query_start,
    state_change,
    query
FROM pg_stat_activity
ORDER BY backend_start;

-- 대기 중인 프로세스 확인
SELECT 
    pid,
    wait_event_type,
    wait_event,
    query
FROM pg_stat_activity
WHERE wait_event IS NOT NULL
ORDER BY wait_event_type, wait_event;
```

## 메모리 튜닝

### 공유 버퍼 최적화

```sql
-- 공유 버퍼 크기 계산 (시스템 메모리의 25% 권장)
-- 8GB RAM 시: 2GB
-- 16GB RAM 시: 4GB
-- 32GB RAM 시: 8GB

-- postgresql.conf 설정
shared_buffers = 2GB                    -- 시스템 메모리의 25%
effective_cache_size = 6GB              -- 시스템 메모리의 75%
work_mem = 16MB                         -- 정렬, 해시 조인 등에 사용
maintenance_work_mem = 128MB            -- VACUUM, CREATE INDEX 등에 사용
wal_buffers = 16MB                      -- WAL 버퍼
checkpoint_completion_target = 0.9       -- 체크포인트 완료 목표
```

### 작업 메모리 튜닝

```sql
-- 현재 작업 메모리 사용량 확인
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    temp_blks_read,
    temp_blks_written
FROM pg_stat_statements
WHERE temp_blks_read > 0 OR temp_blks_written > 0
ORDER BY temp_blks_written DESC
LIMIT 10;

-- 메모리 집약적 쿼리 식별
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    local_blks_hit,
    local_blks_read,
    temp_blks_read,
    temp_blks_written
FROM pg_stat_statements
WHERE temp_blks_read > 0 OR temp_blks_written > 0
ORDER BY (temp_blks_read + temp_blks_written) DESC
LIMIT 10;
```

## 디스크 I/O 튜닝

### WAL 설정 최적화

```sql
-- WAL 설정 확인
SHOW wal_level;              -- WAL 레벨 (minimal, replica, logical)
SHOW wal_sync_method;        -- 동기화 방법
SHOW wal_compression;        -- WAL 압축
SHOW wal_writer_delay;       -- WAL 작성자 지연 시간
SHOW commit_delay;           -- 커밋 지연

-- WAL 튜닝 설정 (postgresql.conf)
wal_level = replica
wal_sync_method = fdatasync
wal_compression = on
wal_writer_delay = 200ms
commit_delay = 0
commit_siblings = 5
```

### 체크포인트 튜닝

```sql
-- 체크포인트 설정 확인
SHOW checkpoint_timeout;        -- 체크포인트 타임아웃
SHOW checkpoint_completion_target; -- 체크포인트 완료 목표
SHOW checkpoint_warning;         -- 체크포인트 경고

-- 체크포인트 튜닝 설정
checkpoint_timeout = 15min
checkpoint_completion_target = 0.9
checkpoint_warning = 30s

-- 체크포인트 통계 확인
SELECT 
    checkpoint_timed,
    checkpoint_req,
    checkpoint_write_time,
    checkpoint_sync_time,
    buffers_checkpoint,
    buffers_clean,
    maxwritten_clean,
    buffers_backend,
    buffers_backend_fsync,
    buffers_alloc
FROM pg_stat_bgwriter;
```

### 자동 Vacuum 튜닝

```sql
-- Vacuum 설정 확인
SHOW autovacuum;               -- 자동 Vacuum 활성화
SHOW autovacuum_max_workers;   -- 최대 작업자 수
SHOW autovacuum_naptime;       -- Vacuum 간격
SHOW autovacuum_vacuum_threshold; -- Vacuum 임계값
SHOW autovacuum_analyze_threshold; -- 분석 임계값

-- Vacuum 튜닝 설정
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.2
autovacuum_analyze_scale_factor = 0.1

-- 테이블별 Vacuum 통계
SELECT 
    schemaname,
    tablename,
    last_vacuum,
    last_autovacuum,
    vacuum_count,
    autovacuum_count,
    last_analyze,
    last_autoanalyze,
    analyze_count,
    autoanalyze_count
FROM pg_stat_user_tables
ORDER BY autovacuum_count DESC;
```

## 쿼리 성능 튜닝

### 실행 계획 분석

```sql
-- 실행 계획 분석
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT u.username, COUNT(o.id) as order_count
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
GROUP BY u.id, u.username
ORDER BY order_count DESC;

-- 쿼리 성능 통계
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

-- 느린 쿼리 식별
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
WHERE mean_time > 1000  -- 1초 이상
ORDER BY mean_time DESC;
```

### 인덱스 효율성 분석

```sql
-- 인덱스 사용 통계
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 사용되지 않는 인덱스 식별
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- 인덱스 크기 분석
SELECT 
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size,
    pg_size_pretty(pg_relation_size(tablename::regclass)) as table_size,
    (pg_relation_size(indexname::regclass)::float / pg_relation_size(tablename::regclass)::float * 100) as size_ratio
FROM pg_indexes
JOIN pg_stat_user_tables ON tablename = relname
WHERE schemaname = 'public'
ORDER BY size_ratio DESC;
```

## 동시성 튜닝

### 연결 풀링

```sql
-- 연결 설정 확인
SHOW max_connections;         -- 최대 연결 수
SHOW superuser_reserved_connections; -- 슈퍼유저 예약 연결
SHOW shared_preload_libraries; -- 사전 로드 라이브러리

-- PgBouncer 설정 예제 (pgbouncer.ini)
[databases]
myapp = host=localhost port=5432 dbname=myapp

[pgbouncer]
listen_port = 6432
listen_addr = 127.0.0.1
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
logfile = /var/log/pgbouncer/pgbouncer.log
pidfile = /var/run/pgbouncer/pgbouncer.pid
admin_users = postgres
stats_users = stats, postgres

# 풀 모드 설정
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 5
max_db_connections = 50
max_user_connections = 50

# 서버 수명 주기
server_reset_query = DISCARD ALL
server_check_delay = 30
server_check_query = select 1
server_lifetime = 3600
server_idle_timeout = 600
```

### 잠금 튜닝

```sql
-- 잠금 통계 확인
SELECT 
    pid,
    locktype,
    mode,
    granted,
    query
FROM pg_locks
JOIN pg_stat_activity ON pg_locks.pid = pg_stat_activity.pid
WHERE NOT granted
ORDER BY locktype, mode;

-- 잠금 대기 시간 분석
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process,
    blocked_activity.application_name AS blocked_application
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

## 모니터링 도구

### pg_stat_statements

```sql
-- pg_stat_statements 확장 활성화
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 쿼리 성능 분석
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- 쿼리 패턴 분석
SELECT 
    LEFT(query, 50) as query_pattern,
    COUNT(*) as query_count,
    SUM(calls) as total_calls,
    SUM(total_exec_time) as total_time,
    AVG(mean_exec_time) as avg_time
FROM pg_stat_statements
GROUP BY LEFT(query, 50)
ORDER BY total_time DESC;
```

### pg_stat_activity 모니터링

```sql
-- 활성 세션 모니터링 뷰
CREATE VIEW active_sessions AS
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    backend_start,
    query_start,
    state_change,
    age(backend_start) as session_age,
    age(query_start) as query_age,
    query
FROM pg_stat_activity
WHERE state != 'idle';

-- 장기 실행 쿼리 모니터링
CREATE VIEW long_running_queries AS
SELECT 
    pid,
    usename,
    application_name,
    query_start,
    now() - query_start as duration,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 minutes'
ORDER BY duration DESC;

-- 대기 중인 세션 모니터링
CREATE VIEW waiting_sessions AS
SELECT 
    pid,
    usename,
    application_name,
    wait_event_type,
    wait_event,
    query
FROM pg_stat_activity
WHERE wait_event IS NOT NULL
ORDER BY wait_event_type, wait_event;
```

### 성능 대시보드

```sql
-- 성능 요약 대시보드
CREATE VIEW performance_dashboard AS
SELECT 
    'Active Connections' as metric,
    COUNT(*) as value
FROM pg_stat_activity
WHERE state = 'active'
UNION ALL
SELECT 
    'Idle Connections' as metric,
    COUNT(*) as value
FROM pg_stat_activity
WHERE state = 'idle'
UNION ALL
SELECT 
    'Waiting Sessions' as metric,
    COUNT(*) as value
FROM pg_stat_activity
WHERE wait_event IS NOT NULL
UNION ALL
SELECT 
    'Database Size (MB)' as metric,
    pg_size_pretty(pg_database_size(current_database())) as value
UNION ALL
SELECT 
    'Cache Hit Ratio (%)' as metric,
    ROUND(100.0 * sum(blks_hit)::numeric / (sum(blks_hit) + sum(blks_read)), 2) as value
FROM pg_stat_database
WHERE datname = current_database();

-- 테이블 성능 요약
CREATE VIEW table_performance AS
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    vacuum_count,
    autovacuum_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

## 실시간 모니터링

### 성능 모니터링 스크립트

```bash
#!/bin/bash
# pg_monitor.sh - PostgreSQL 실시간 모니터링 스크립트

DB_NAME="myapp"
PG_USER="postgres"
INTERVAL=5

while true; do
    clear
    echo "PostgreSQL Performance Monitor - $(date)"
    echo "=========================================="
    
    # 활성 연결 수
    ACTIVE_CONNECTIONS=$(psql -U $PG_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active';")
    echo "Active Connections: $ACTIVE_CONNECTIONS"
    
    # 대기 중인 세션 수
    WAITING_SESSIONS=$(psql -U $PG_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM pg_stat_activity WHERE wait_event IS NOT NULL;")
    echo "Waiting Sessions: $WAITING_SESSIONS"
    
    # 캐시 히트율
    CACHE_HIT_RATIO=$(psql -U $PG_USER -d $DB_NAME -t -c "SELECT ROUND(100.0 * sum(blks_hit)::numeric / (sum(blks_hit) + sum(blks_read)), 2) FROM pg_stat_database WHERE datname = '$DB_NAME';")
    echo "Cache Hit Ratio: $CACHE_HIT_RATIO%"
    
    # 데이터베이스 크기
    DB_SIZE=$(psql -U $PG_USER -d $DB_NAME -t -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));")
    echo "Database Size: $DB_SIZE"
    
    echo ""
    echo "Top 5 Longest Running Queries:"
    echo "------------------------------"
    psql -U $PG_USER -d $DB_NAME -c "
    SELECT 
        pid,
        usename,
        now() - query_start as duration,
        LEFT(query, 50) as query
    FROM pg_stat_activity
    WHERE state = 'active'
    ORDER BY duration DESC
    LIMIT 5;"
    
    echo ""
    echo "Waiting Events:"
    echo "--------------"
    psql -U $PG_USER -d $DB_NAME -c "
    SELECT 
        wait_event_type,
        wait_event,
        COUNT(*) as count
    FROM pg_stat_activity
    WHERE wait_event IS NOT NULL
    GROUP BY wait_event_type, wait_event
    ORDER BY count DESC;"
    
    sleep $INTERVAL
done
```

### 경고 시스템

```sql
-- 성능 경고 함수
CREATE OR REPLACE FUNCTION performance_alerts()
RETURNS TABLE(alert_type TEXT, message TEXT, severity TEXT) AS $$
DECLARE
    active_connections INTEGER;
    waiting_sessions INTEGER;
    cache_hit_ratio NUMERIC;
    long_queries INTEGER;
BEGIN
    -- 활성 연결 수 경고
    SELECT COUNT(*) INTO active_connections
    FROM pg_stat_activity
    WHERE state = 'active';
    
    IF active_connections > 80 THEN
        RETURN QUERY SELECT 'High Connections'::TEXT, 
                            'Active connections: ' || active_connections::TEXT,
                            'WARNING'::TEXT;
    END IF;
    
    -- 대기 세션 경고
    SELECT COUNT(*) INTO waiting_sessions
    FROM pg_stat_activity
    WHERE wait_event IS NOT NULL;
    
    IF waiting_sessions > 5 THEN
        RETURN QUERY SELECT 'High Waiting Sessions'::TEXT,
                            'Waiting sessions: ' || waiting_sessions::TEXT,
                            'WARNING'::TEXT;
    END IF;
    
    -- 캐시 히트율 경고
    SELECT ROUND(100.0 * sum(blks_hit)::numeric / (sum(blks_hit) + sum(blks_read)), 2)
    INTO cache_hit_ratio
    FROM pg_stat_database
    WHERE datname = current_database();
    
    IF cache_hit_ratio < 95 THEN
        RETURN QUERY SELECT 'Low Cache Hit Ratio'::TEXT,
                            'Cache hit ratio: ' || cache_hit_ratio::TEXT || '%',
                            'WARNING'::TEXT;
    END IF;
    
    -- 장기 실행 쿼리 경고
    SELECT COUNT(*) INTO long_queries
    FROM pg_stat_activity
    WHERE state = 'active'
      AND now() - query_start > interval '10 minutes';
    
    IF long_queries > 0 THEN
        RETURN QUERY SELECT 'Long Running Queries'::TEXT,
                            'Queries running > 10min: ' || long_queries::TEXT,
                            'CRITICAL'::TEXT;
    END IF;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- 경고 확인
SELECT * FROM performance_alerts();
```

## 실습: 성능 튜닝 연습

성능 문제를 식별하고 해결하는 연습을 해봅시다.

```sql
-- 성능 테스트용 테이블 생성
CREATE TABLE performance_test (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    order_date TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL
);

-- 대량 데이터 삽입
INSERT INTO performance_test (user_id, product_id, quantity, price, order_date, status)
SELECT 
    (random() * 10000 + 1)::integer,
    (random() * 1000 + 1)::integer,
    (random() * 10 + 1)::integer,
    (random() * 100000 + 1000)::decimal(10, 2),
    CURRENT_TIMESTAMP - (random() * 365 || ' days')::interval,
    ARRAY['pending', 'processing', 'shipped', 'delivered'][floor(random() * 4) + 1]
FROM generate_series(1, 1000000);

-- 인덱스 없이 쿼리 실행
EXPLAIN ANALYZE 
SELECT user_id, COUNT(*) as order_count, SUM(price) as total_spent
FROM performance_test
WHERE order_date >= '2023-01-01'
  AND status = 'delivered'
GROUP BY user_id
ORDER BY total_spent DESC
LIMIT 100;

-- 인덱스 생성
CREATE INDEX idx_performance_test_order_date ON performance_test(order_date);
CREATE INDEX idx_performance_test_status ON performance_test(status);
CREATE INDEX idx_performance_test_user_id ON performance_test(user_id);

-- 복합 인덱스 생성
CREATE INDEX idx_performance_test_date_status ON performance_test(order_date, status);

-- 인덱스 생성 후 쿼리 실행
EXPLAIN ANALYZE 
SELECT user_id, COUNT(*) as order_count, SUM(price) as total_spent
FROM performance_test
WHERE order_date >= '2023-01-01'
  AND status = 'delivered'
GROUP BY user_id
ORDER BY total_spent DESC
LIMIT 100;

-- 성능 비교
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
WHERE query LIKE '%performance_test%'
ORDER BY total_time DESC;
```

## 요약

PostgreSQL 성능 튜닝을 위한 핵심 원칙:

1. **메모리 최적화**: shared_buffers, work_mem, effective_cache_size 적절한 설정
2. **디스크 I/O 튜닝**: WAL 설정, 체크포인트, 자동 Vacuum 최적화
3. **쿼리 최적화**: 실행 계획 분석, 적절한 인덱스 사용
4. **동시성 관리**: 연결 풀링, 잠금 최적화
5. **지속적인 모니터링**: 성능 지표 추적 및 경고 시스템
6. **정기적인 유지보수**: Vacuum, 분석, 인덱스 재구성

다음 섹션에서는 예제 데이터베이스 스키마와 샘플 데이터를 생성하겠습니다.

## 추가 자료

- [PostgreSQL 문서: 성능 튜닝](https://www.postgresql.org/docs/current/performance-tips.html)
- [PostgreSQL 문서: 서버 설정](https://www.postgresql.org/docs/current/runtime-config.html)
- [pg_stat_statements 문서](https://www.postgresql.org/docs/current/pgstatstatements.html)
- [PgBouncer 문서](https://pgbouncer.github.io/)