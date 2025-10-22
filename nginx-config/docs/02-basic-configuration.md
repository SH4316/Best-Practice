# 기본 Nginx 설정

## Nginx 설정 파일 구조

Nginx 설정은 계층적 구조를 가지며, 주로 `/etc/nginx/nginx.conf` 파일에서 시작됩니다. 설정 파일은 다음과 같은 구조를 가집니다:

```
nginx.conf (전역 설정)
├── events 블록 (연결 처리)
├── http 블록 (웹 서버 설정)
│   ├── server 블록 (가상 호스트)
│   │   ├── location 블록 (URL 매칭)
│   │   └── ...
│   └── upstream 블록 (백엔드 서버 그룹)
└── ...
```

## 전역 설정 (Global Context)

### 문제 시나리오: 기본 설정으로 인한 성능 저하
**상황**: 새로 설치한 Nginx 서버가 예상보다 응답이 느리고 동시 접속자 수가 적습니다.

**원인 분석**: 기본 설정이 서버 사양에 맞지 않아 성능이 저하되었습니다.

**해결 과정**:

#### 1. 워커 프로세스 최적화

```nginx
# /etc/nginx/nginx.conf

# CPU 코어 수에 맞게 워커 프로세스 수 설정
worker_processes auto;  # 자동으로 CPU 코어 수에 맞춤

# 각 워커 프로세스가 처리할 수 있는 최대 연결 수
events {
    worker_connections 1024;
}

# 최대 동시 접속자 수 = worker_processes × worker_connections
# 예: 4코어 CPU × 1024 = 4,096 동시 접속 가능
```

#### 2. 파일 디스크립터 제한 설정

```nginx
# 시스템 제한 확인 (터미널에서 실행)
# ulimit -n

# 시스템 제한이 낮을 경우 설정
worker_rlimit_nofile 65535;
```

#### 3. 로그 설정 최적화

```nginx
# 로그 형식 정의
log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                '$status $body_bytes_sent "$http_referer" '
                '"$http_user_agent" "$http_x_forwarded_for"';

# 접근 로그
access_log /var/log/nginx/access.log main;

# 에러 로그 레벨 설정
error_log /var/log/nginx/error.log warn;
```

## 이벤트 블록 (Events Context)

이벤트 블록은 연결 처리 방식을 설정합니다.

### 문제 시나리오: 고부하 상황에서 연결 거부
**상황**: 트래픽이 급증할 때 일부 사용자가 연결 거부 메시지를 받습니다.

**해결 과정**:

```nginx
events {
    # 워커당 최대 연결 수
    worker_connections 2048;
    
    # 연결 처리 방식 (기본값: epoll)
    use epoll;
    
    # 한 워커가 동시에 받아들일 수 있는 새 연결 수
    multi_accept on;
}
```

## HTTP 블록 (HTTP Context)

HTTP 블록은 웹 서버 관련 모든 설정을 포함합니다.

### 기본 HTTP 설정

```nginx
http {
    # MIME 타입 설정
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    # 파일 전송 설정
    sendfile        on;
    tcp_nopush      on;
    tcp_nodelay     on;
    
    # 연결 유지 시간
    keepalive_timeout  65;
    
    # 클라이언트 요청 본문 크기 제한
    client_max_body_size 64M;
    
    # 클라이언트 헤더 버퍼 크기
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
}
```

### 문제 시나리오: 대용량 파일 업로드 실패
**상황**: 사용자가 10MB 이상의 파일을 업로드하려고 할 때 실패합니다.

**해결 과정**:

```nginx
http {
    # 클라이언트 요청 본문 크기 제한 증가
    client_max_body_size 100M;
    
    # 임시 파일 저장 디렉토리
    client_body_temp_path /var/nginx/client_temp;
    
    # 요청 타임아웃 설정
    client_body_timeout 60s;
    client_header_timeout 60s;
}
```

## 서버 블록 (Server Context)

서버 블록은 가상 호스트 설정을 정의합니다.

### 기본 서버 설정

```nginx
server {
    listen 80;
    server_name example.com www.example.com;
    
    # 루트 디렉토리
    root /var/www/html;
    
    # 기본 파일
    index index.html index.htm;
    
    # 서버 정보 숨김 (보안)
    server_tokens off;
}
```

### 문제 시나리오: 도메인 접속 시 올바른 사이트가 표시되지 않음
**상황**: 여러 도메인을 호스팅하는데 어떤 도메인으로 접속해도 동일한 웹사이트가 표시됩니다.

**원인 분석**: server_name 설정이 올바르지 않거나 기본 서버 설정이 누락되었습니다.

**해결 과정**:

```nginx
# 첫 번째 도메인
server {
    listen 80;
    server_name example.com www.example.com;
    root /var/www/example;
    index index.html;
    
    # 접근 로그
    access_log /var/log/nginx/example.access.log;
    error_log /var/log/nginx/example.error.log;
}

# 두 번째 도메인
server {
    listen 80;
    server_name another.com www.another.com;
    root /var/www/another;
    index index.html;
    
    # 접근 로그
    access_log /var/log/nginx/another.access.log;
    error_log /var/log/nginx/another.error.log;
}

# 기본 서버 (정의되지 않은 도메인 처리)
server {
    listen 80 default_server;
    server_name _;
    
    # 접근 거부 또는 기본 페이지
    return 444;
}
```

## 위치 블록 (Location Context)

위치 블록은 URL 패턴에 따라 다른 설정을 적용합니다.

### Location 매칭 규칙

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 정확히 일치
    location = / {
        # http://example.com/ 에만 일치
    }
    
    # 정규표현식 (대소문자 구분)
    location ~ \.php$ {
        # .php로 끝나는 모든 URL
    }
    
    # 정규표현식 (대소문자 무시)
    location ~* \.(jpg|jpeg|png|gif)$ {
        # 이미지 파일 확장자
    }
    
    # 접두사 일치 (가장 긴 것이 우선)
    location /images/ {
        # /images/로 시작하는 모든 URL
    }
    
    # 접두사 일치 (정규표현식 우선)
    location /documents/ {
        # /documents/로 시작하는 모든 URL
    }
}
```

### 문제 시나리오: 정적 파일과 동적 콘텐츠 처리 분리
**상황**: 정적 파일과 동적 콘텐츠가 혼재되어 있어 성능이 저하됩니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 정적 파일 캐싱
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # PHP 스크립트 처리
    location ~ \.php$ {
        fastcgi_pass 127.0.0.1:9000;
        fastcgi_index index.php;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }
    
    # 기타 요청 처리
    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }
}
```

## 설정 테스트 및 재로드

Nginx 설정을 변경한 후에는 항상 구문을 검증하고 재로드해야 합니다.

```bash
# 설정 구문 검증
sudo nginx -t

# Nginx 재시작 (다운타임 발생)
sudo systemctl restart nginx

# 설정 재로드 (무중단)
sudo systemctl reload nginx

# 워커 프로세스 상태 확인
ps aux | grep nginx
```

## 모범 사례 요약

1. **워커 프로세스**: CPU 코어 수에 맞게 `worker_processes auto` 설정
2. **연결 수**: 서버 사양에 맞게 `worker_connections` 조정
3. **보안**: `server_tokens off`로 서버 정보 숨김
4. **파일 크기**: `client_max_body_size`로 업로드 파일 크기 제한
5. **로깅**: 적절한 로그 레벨과 형식 설정
6. **테스트**: 설정 변경 후 항상 `nginx -t`로 검증

## 다음 단계

이제 Nginx의 기본 설정을 이해했습니다. 다음 장에서는 웹 서버로서의 Nginx 설정에 대해 더 자세히 알아보겠습니다.