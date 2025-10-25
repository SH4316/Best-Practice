# 백업 및 복구 전략

## 학습 목표

- PostgreSQL 백업의 종류와 특징 이해
- 논리적 백업과 물리적 백업의 차이점 학습
- 백업 자동화 및 스케줄링 방법 습득
- 복구 시나리오와 절차 파악
- 고가용성 및 재해 복구 전략 이해

## 백업 기본 개념

### 백업의 중요성

데이터베이스 백업은 다음과 같은 상황에서 필수적입니다:
- 하드웨어 장애
- 데이터 손상 또는 삭제
- 소프트웨어 버그
- 자연재해
- 보안 침해

### 백업 유형

```sql
-- 1. 논리적 백업 (pg_dump)
-- SQL 스크립트로 데이터베이스 구조와 데이터 내보내기

-- 전체 데이터베이스 백업
pg_dump -U postgres -h localhost -d myapp > myapp_backup.sql

-- 특정 테이블만 백업
pg_dump -U postgres -h localhost -d myapp -t users > users_backup.sql

-- 데이터만 백업 (구조 제외)
pg_dump -U postgres -h localhost -d myapp -a > myapp_data_only.sql

-- 구조만 백업 (데이터 제외)
pg_dump -U postgres -h localhost -d myapp -s > myapp_schema_only.sql

-- 압축 백업
pg_dump -U postgres -h localhost -d myapp | gzip > myapp_backup.sql.gz

-- 2. 물리적 백업 (기본 백업)
-- 데이터베이스 파일 시스템의 직접 복사

-- 기본 백업 시작
SELECT pg_start_backup('full_backup_20231024');

-- 파일 시스템 복사 (운영체제 명령어)
-- rsync -av /var/lib/postgresql/13/main/ /backup/postgresql/20231024/

-- 기본 백업 종료
SELECT pg_stop_backup();
```

## 논리적 백업

### pg_dump 사용법

```bash
# 기본 사용법
pg_dump [옵션] [데이터베이스명]

# 주요 옵션
# -U, --username=USERNAME: 사용자 이름
# -h, --host=HOSTNAME: 호스트 주소
# -p, --port=PORT: 포트 번호
# -d, --dbname=DBNAME: 데이터베이스 이름
# -f, --file=FILENAME: 출력 파일
# -F, --format=FORMAT: 출력 형식 (p=plain, c=custom, d=directory, t=tar)
# -a, --data-only: 데이터만 백업
# -s, --schema-only: 구조만 백업
# -t, --table=TABLE: 특정 테이블만 백업
# -n, --schema=SCHEMA: 특정 스키마만 백업
# -v, --verbose: 상세 정보 출력
# -Z, --compress=LEVEL: 압축 레벨 (0-9)
```

### 다양한 백업 형식

```bash
# 1. 플레인 SQL 형식 (기본값)
pg_dump -U postgres -d myapp -f myapp_backup.sql

# 복원 방법
psql -U postgres -d myapp_restored < myapp_backup.sql

# 2. 커스텀 형식 (압축 및 객체 선택 가능)
pg_dump -U postgres -d myapp -Fc -f myapp_backup.dump

# 복원 방법
pg_restore -U postgres -d myapp_restored myapp_backup.dump

# 3. 디렉토리 형식 (병렬 백업 가능)
pg_dump -U postgres -d myapp -Fd -f myapp_backup_dir/

# 복원 방법
pg_restore -U postgres -d myapp_restored myapp_backup_dir/

# 4. TAR 형식
pg_dump -U postgres -d myapp -Ft -f myapp_backup.tar

# 복원 방법
pg_restore -U postgres -d myapp_restored myapp_backup.tar
```

### 고급 pg_dump 옵션

```bash
# 병렬 백업 (디렉토리 형식만 가능)
pg_dump -U postgres -d myapp -Fd -j 4 -f myapp_backup_parallel/

# 특정 스키마만 백업
pg_dump -U postgres -d myapp -n public -f public_schema.sql

# 특정 테이블 제외
pg_dump -U postgres -d myapp -T logs -T temp_* -f myapp_excluding_logs.sql

# 조건부 데이터 백업
pg_dump -U postgres -d myapp \
  --where="created_at >= '2023-01-01'" \
  -t orders -f recent_orders.sql

# 롤과 권한 포함
pg_dump -U postgres -d myapp --role=postgres -f myapp_with_roles.sql

# 객체 소유자 변경
pg_dump -U postgres -d myapp --no-owner -f myapp_no_owner.sql
```

## 물리적 백업

### 기본 백업 설정

```sql
-- WAL 아카이브 활성화
-- postgresql.conf 설정:

# WAL 설정
wal_level = replica
max_wal_senders = 3
wal_keep_segments = 64
archive_mode = on
archive_command = 'cp %p /backup/wal_archive/%f'

# 체크포인트 설정
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

### 기본 백업 스크립트

```bash
#!/bin/bash
# base_backup.sh - 기본 백업 스크립트

BACKUP_DIR="/backup/postgresql/$(date +%Y%m%d_%H%M%S)"
DB_NAME="myapp"
PG_USER="postgres"

# 백업 디렉토리 생성
mkdir -p $BACKUP_DIR

# 기본 백업 시작
psql -U $PG_USER -d $DB_NAME -c "SELECT pg_start_backup('base_backup_$(date +%Y%m%d_%H%M%S)');"

# 데이터 파일 복사
rsync -av --exclude="pg_xlog" --exclude="postmaster.pid" \
  /var/lib/postgresql/13/main/ $BACKUP_DIR/

# 기본 백업 종료
psql -U $PG_USER -d $DB_NAME -c "SELECT pg_stop_backup();"

# 백업 정보 저장
echo "Base backup completed at $(date)" > $BACKUP_DIR/backup_info.txt
echo "Database: $DB_NAME" >> $BACKUP_DIR/backup_info.txt
echo "Size: $(du -sh $BACKUP_DIR | cut -f1)" >> $BACKUP_DIR/backup_info.txt

echo "Base backup completed: $BACKUP_DIR"
```

### pg_basebackup 사용

```bash
# pg_basebackup 기본 사용법
pg_basebackup -U postgres -h localhost -D /backup/base/20231024 -Ft -z -P

# 옵션 설명
# -D, --pgdata=DIRECTORY: 백업 디렉토리
# -F, --format=FORMAT: 출력 형식 (p=plain, t=tar)
# -z, --gzip: 압축
# -P, --progress: 진행률 표시
# -v, --verbose: 상세 정보
# -x, --xlog: WAL 파일 포함
# -X, --xlog-method=METHOD: WAL 백업 방법 (f=fetch, s=stream)

# 스트리밍 복제를 위한 백업
pg_basebackup -U postgres -h localhost -D /backup/base/replica -Fp -Xs -P -v

# 압축된 TAR 형식 백업
pg_basebackup -U postgres -h localhost -D /backup/base/20231024.tar -Ft -z -P

# 특정 테이블스페이스만 백업
pg_basebackup -U postgres -h localhost -D /backup/base/20231024 -t pg_default -t ts_data
```

## PITR (Point-in-Time Recovery)

### WAL 아카이빙 설정

```sql
-- postgresql.conf 설정
wal_level = replica
archive_mode = on
archive_command = 'test ! -f /backup/wal_archive/%f && cp %p /backup/wal_archive/%f'
archive_timeout = 600  -- 10분마다 WAL 아카이브

-- pg_hba.conf 설정 (복제용)
local   replication   postgres                                peer
host    replication   postgres        127.0.0.1/32            md5
```

### 복구 절차

```bash
# 1. 데이터베이스 중지
sudo systemctl stop postgresql

# 2. 데이터 디렉토리 백업
sudo mv /var/lib/postgresql/13/main /var/lib/postgresql/13/main_backup

# 3. 기본 백업 복원
sudo mkdir /var/lib/postgresql/13/main
sudo tar -xzf /backup/base/20231024.tar.gz -C /var/lib/postgresql/13/main

# 4. 복구 신호 파일 생성
sudo touch /var/lib/postgresql/13/main/recovery.signal

# 5. 복구 설정 파일 생성
sudo tee /var/lib/postgresql/13/main/postgresql.auto.conf > /dev/null <<EOF
restore_command = 'cp /backup/wal_archive/%f %p'
recovery_target_time = '2023-10-24 15:30:00'
EOF

# 6. 데이터베이스 시작
sudo systemctl start postgresql

# 7. 복구 상태 확인
sudo -u postgres psql -c "SELECT pg_is_in_recovery();"
```

### 복구 옵션

```sql
-- 복구 대상 설정 옵션

# 1. 특정 시간으로 복구
recovery_target_time = '2023-10-24 15:30:00'

# 2. 특정 트랜잭션 ID로 복구
recovery_target_xid = '12345678'

# 3. 특정 LSN(Log Sequence Number)로 복구
recovery_target_lsn = '0/12345678'

# 4. 즉시 복구 (가장 최신 상태)
# recovery_target = 'immediate'

# 5. 복구 후 동작
recovery_target_action = 'promote'  -- promote, pause, shutdown

# 6. 복구 정확도
recovery_target_inclusive = true  -- 목표 지점 포함 여부
```

## 자동화된 백업 전략

### 백업 스크립트

```bash
#!/bin/bash
# automated_backup.sh - 자동화된 백업 스크립트

# 설정 변수
BACKUP_BASE_DIR="/backup/postgresql"
DB_NAME="myapp"
PG_USER="postgres"
RETENTION_DAYS=30
LOG_FILE="/var/log/postgresql_backup.log"

# 날짜 형식
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE_DIR/$DATE"

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 백업 디렉토리 생성
mkdir -p $BACKUP_DIR
log "백업 디렉토리 생성: $BACKUP_DIR"

# 1. 논리적 백업 (전체)
log "논리적 백업 시작..."
pg_dump -U $PG_USER -d $DB_NAME -Fc -f "$BACKUP_DIR/logical_backup.dump"
if [ $? -eq 0 ]; then
    log "논리적 백업 완료"
else
    log "오류: 논리적 백업 실패"
    exit 1
fi

# 2. 기본 백업
log "기본 백업 시작..."
pg_basebackup -U $PG_USER -h localhost -D "$BACKUP_DIR/base_backup" -Ft -z -P
if [ $? -eq 0 ]; then
    log "기본 백업 완료"
else
    log "오류: 기본 백업 실패"
    exit 1
fi

# 3. 중요 테이블별 백업
IMPORTANT_TABLES=("users" "orders" "products")
for table in "${IMPORTANT_TABLES[@]}"; do
    log "테이블 백업: $table"
    pg_dump -U $PG_USER -d $DB_NAME -t $table -f "$BACKUP_DIR/table_${table}.sql"
done

# 4. 백업 정보 저장
cat > "$BACKUP_DIR/backup_info.txt" <<EOF
백업 정보
=========
백업 시간: $(date)
데이터베이스: $DB_NAME
백업 타입: 전체 백업
백업 크기: $(du -sh $BACKUP_DIR | cut -f1)
WAL 위치: $(psql -U $PG_USER -d $DB_NAME -t -c "SELECT pg_current_wal_lsn();")
EOF

# 5. 오래된 백업 정리
log "오래된 백업 정리 시작..."
find $BACKUP_BASE_DIR -maxdepth 1 -type d -name "20*" -mtime +$RETENTION_DAYS -exec rm -rf {} \;
log "오래된 백업 정리 완료"

# 6. 백업 검증
log "백업 검증 시작..."
if pg_restore -U $PG_USER --list "$BACKUP_DIR/logical_backup.dump" > /dev/null 2>&1; then
    log "백업 검증 성공"
else
    log "오류: 백업 검증 실패"
    exit 1
fi

log "백업 프로세스 완료: $BACKUP_DIR"
```

### 크론 스케줄링

```bash
# crontab -e

# 매일 새벽 2시 전체 백업
0 2 * * * /usr/local/bin/automated_backup.sh

# 매시간 WAL 아카이브 백업
0 * * * * rsync -av /backup/wal_archive/ /backup/remote/wal_archive/

# 매주 일요일 새벽 3시 전체 백업 검증
0 3 * * 0 /usr/local/bin/verify_backups.sh

# 매월 1일 백업 보고서 전송
0 4 1 * * /usr/local/bin/backup_report.sh
```

## 복구 시나리오

### 시나리오 1: 테이블 삭제 복구

```bash
# 1. 특정 시점으로 복구
sudo systemctl stop postgresql

# 데이터 디렉토리 백업
sudo mv /var/lib/postgresql/13/main /var/lib/postgresql/13/main_corrupted

# 기본 백업 복원
sudo mkdir /var/lib/postgresql/13/main
sudo tar -xzf /backup/postgresql/20231024/base_backup.tar.gz -C /var/lib/postgresql/13/main

# 복구 설정 (테이블 삭제 10분 전으로 복구)
sudo tee /var/lib/postgresql/13/main/postgresql.auto.conf > /dev/null <<EOF
restore_command = 'cp /backup/wal_archive/%f %p'
recovery_target_time = '2023-10-24 14:50:00'
recovery_target_action = 'promote'
EOF

sudo touch /var/lib/postgresql/13/main/recovery.signal
sudo systemctl start postgresql
```

### 시나리오 2: 데이터베이스 전체 복구

```bash
#!/bin/bash
# full_recovery.sh - 전체 데이터베이스 복구 스크립트

BACKUP_DIR=$1
RECOVERY_TIME=$2

if [ -z "$BACKUP_DIR" ]; then
    echo "사용법: $0 <백업_디렉토리> [복구_시간]"
    exit 1
fi

# PostgreSQL 중지
sudo systemctl stop postgresql

# 현재 데이터 디렉토리 백업
sudo mv /var/lib/postgresql/13/main /var/lib/postgresql/13/main_$(date +%Y%m%d_%H%M%S)

# 새 데이터 디렉토리 생성
sudo mkdir -p /var/lib/postgresql/13/main

# 백업 복원
if [[ $BACKUP_DIR == *.dump ]]; then
    # 논리적 백업 복원
    sudo -u postgres createdb myapp_restored
    sudo -u postgres pg_restore -d myapp_restored $BACKUP_DIR
else
    # 물리적 백업 복원
    sudo tar -xzf $BACKUP_DIR -C /var/lib/postgresql/13/main
    
    # 복구 설정
    sudo tee /var/lib/postgresql/13/main/postgresql.auto.conf > /dev/null <<EOF
restore_command = 'cp /backup/wal_archive/%f %p'
EOF

    if [ ! -z "$RECOVERY_TIME" ]; then
        echo "recovery_target_time = '$RECOVERY_TIME'" | sudo tee -a /var/lib/postgresql/13/main/postgresql.auto.conf
    fi
    
    echo "recovery_target_action = 'promote'" | sudo tee -a /var/lib/postgresql/13/main/postgresql.auto.conf
    sudo touch /var/lib/postgresql/13/main/recovery.signal
fi

# 권한 설정
sudo chown -R postgres:postgres /var/lib/postgresql/13/main

# PostgreSQL 시작
sudo systemctl start postgresql

# 복구 상태 확인
sleep 5
sudo -u postgres psql -c "SELECT pg_is_in_recovery();"
```

### 시나리오 3: 특정 테이블만 복구

```bash
#!/bin/bash
# table_recovery.sh - 특정 테이블 복구 스크립트

TABLE_NAME=$1
BACKUP_FILE=$2
DB_NAME="myapp"

if [ -z "$TABLE_NAME" ] || [ -z "$BACKUP_FILE" ]; then
    echo "사용법: $0 <테이블_이름> <백업_파일>"
    exit 1
fi

# 임시 테이블로 복원
echo "임시 테이블로 복원 시작..."
sudo -u postgres pg_restore -d $DB_NAME --table=$TABLE_NAME --use-list=/tmp/table_list.txt $BACKUP_FILE

# 임시 테이블 생성
sudo -u postgres psql -d $DB_NAME -c "ALTER TABLE $TABLE_NAME RENAME TO ${TABLE_NAME}_corrupted_$(date +%Y%m%d);"

# 백업에서 테이블 복원
echo "테이블 복원 시작..."
sudo -u postgres pg_restore -d $DB_NAME --table=$TABLE_NAME --clean --if-exists $BACKUP_FILE

# 데이터 확인
echo "복원된 데이터 확인..."
sudo -u postgres psql -d $DB_NAME -c "SELECT COUNT(*) FROM $TABLE_NAME;"

echo "테이블 복구 완료: $TABLE_NAME"
```

## 고가용성 및 복제

### 스트리밍 복제 설정

```sql
-- 마스터 서버 설정 (postgresql.conf)
wal_level = replica
max_wal_senders = 3
wal_keep_segments = 64
archive_mode = on
archive_command = 'cp %p /backup/wal_archive/%f'

-- pg_hba.conf 설정
host replication replicator 192.168.1.101/32 md5
```

```bash
# 슬레이브 서버 설정

# 1. 기본 백업으로 슬레이브 초기화
pg_basebackup -h 192.168.1.100 -D /var/lib/postgresql/13/main -U replicator -v -P -W

# 2. 복제 신호 파일 생성
touch /var/lib/postgresql/13/main/standby.signal

# 3. 복제 설정
cat >> /var/lib/postgresql/13/main/postgresql.auto.conf <<EOF
primary_conninfo = 'host=192.168.1.100 port=5432 user=replicator'
restore_command = 'cp /backup/wal_archive/%f %p'
EOF

# 4. PostgreSQL 시작
systemctl start postgresql
```

### 복제 상태 모니터링

```sql
-- 마스터 서버에서 복제 상태 확인
SELECT 
    pid,
    state,
    client_addr,
    sync_state,
    replay_lag
FROM pg_stat_replication;

-- 슬레이브 서버에서 복제 상태 확인
SELECT 
    pg_is_in_recovery(),
    pg_last_wal_receive_lsn(),
    pg_last_wal_replay_lsn(),
    pg_last_wal_replay_time();
```

## 실습: 백업 및 복구 연습

전체 백업 및 복구 프로세스를 실습해 봅시다.

```sql
-- 테스트용 데이터베이스 생성
CREATE DATABASE backup_test;

-- 테스트용 테이블 생성
CREATE TABLE backup_test.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE backup_test.orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES backup_test.users(id),
    product_name VARCHAR(100) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 테스트 데이터 삽입
INSERT INTO backup_test.users (username, email)
SELECT 
    'user_' || i,
    'user_' || i || '@example.com'
FROM generate_series(1, 1000) i;

INSERT INTO backup_test.orders (user_id, product_name, amount)
SELECT 
    (random() * 999 + 1)::integer,
    'Product ' || (random() * 100 + 1)::integer,
    (random() * 10000 + 100)::decimal(10, 2)
FROM generate_series(1, 5000) i;

-- 백업 전 데이터 확인
SELECT 'users' as table_name, COUNT(*) as record_count FROM backup_test.users
UNION ALL
SELECT 'orders' as table_name, COUNT(*) as record_count FROM backup_test.orders;
```

```bash
# 1. 논리적 백업 실행
pg_dump -U postgres -d backup_test -Fc -f /tmp/backup_test.dump

# 2. 기본 백업 실행
pg_basebackup -U postgres -h localhost -D /tmp/base_backup -Ft -z -P

# 3. 데이터 변경 시뮬레이션
psql -U postgres -d backup_test -c "DELETE FROM backup_test.orders WHERE id < 100;"
psql -U postgres -d backup_test -c "UPDATE backup_test.users SET username = 'modified' WHERE id = 1;"

# 4. 복구 시뮬레이션
dropdb backup_test
createdb backup_test

# 5. 논리적 백업에서 복원
pg_restore -U postgres -d backup_test /tmp/backup_test.dump

# 6. 복원된 데이터 확인
psql -U postgres -d backup_test -c "SELECT COUNT(*) FROM users;"
psql -U postgres -d backup_test -c "SELECT COUNT(*) FROM orders;"
psql -U postgres -d backup_test -c "SELECT * FROM users WHERE id = 1;"
```

## 요약

효과적인 백업 및 복구 전략을 위한 핵심 원칙:

1. **3-2-1 규칙**: 3개의 복사본, 2개의 다른 미디어, 1개는 오프사이트
2. **정기적인 백업**: 자동화된 스케줄링으로 일관된 백업 수행
3. **백업 검증**: 정기적인 복원 테스트로 백업 유효성 확인
4. **다양한 백업 방법**: 논리적, 물리적 백업 조합으로 유연성 확보
5. **PITR 준비**: WAL 아카이빙으로 특정 시점 복구 capability 확보
6. **문서화**: 복구 절차 문서화와 정기적인 훈련

다음 섹션에서는 PostgreSQL 성능 튜닝과 모니터링에 대해 알아보겠습니다.

## 추가 자료

- [PostgreSQL 문서: 백업 및 복구](https://www.postgresql.org/docs/current/backup.html)
- [PostgreSQL 문서: 연속 아카이빙](https://www.postgresql.org/docs/current/continuous-archiving.html)
- [PostgreSQL 문서: 스트리밍 복제](https://www.postgresql.org/docs/current/streaming-replication.html)
- [pg_dump 문서](https://www.postgresql.org/docs/current/app-pgdump.html)