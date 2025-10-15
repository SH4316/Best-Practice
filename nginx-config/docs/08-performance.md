# 성능 최적화

고트래픽 웹사이트에서 Nginx의 성능을 최적화하는 것은 사용자 경험과 서버 안정성에 매우 중요합니다. 이 장에서는 다양한 성능 최적화 기법과 실제 시나리오에 맞는 해결 방법을 다룹니다.

## 워커 프로세스 최적화

### 문제 시나리오: 고부하 상황에서 응답 속도 저하
**상황**: 트래픽이 증가하면 서버 응답 속도가 급격히 저하됩니다.

**원인 분석**: 워커 프로세스 수가 시스템 자원에 맞지 않게 설정되어 있습니다.

**해결 과정**:

#### 1. 워커 프로세스 최적화

```nginx
# /etc/nginx/nginx.conf

# CPU 코어 수에 맞게 워커 프로세스 수 자동 설정
worker_processes auto;

# 워커 프로세스에 CPU 코어 친화도 설정
worker_cpu_affinity auto;

# 워커 프로세스 우선순위 설정 (낮을수록 높은 우선순위)
worker_priority -10;

# 워커 프로세스가 열 수 있는 최대 파일 디스크립터 수
worker_rlimit_nofile 65535;

events {
    # 워커당 최대 연결 수
    worker_connections 4096;
    
    # 여러 연결을 동시에 수락
    multi_accept on;
    
    # 이벤트 처리 방식 (Linux에서는 epoll)
    use epoll;
}
```

#### 2. 시스템 레벨 최적화

```bash
# /etc/sysctl.conf 파일에 추가

# 네트워크 최적화
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.ipv4.tcp_congestion_control = cubic

# 파일 디스크립터 제한 증가
fs.file-max = 2097152

# 시스템 적용
sudo sysctl -p
```

## 커넥션 최적화

### 문제 시나리오: 동시 접속자 수 제한으로 인한 연결 거부
**상황**: 동시 접속자가 많아지면 "Connection refused" 오류가 발생합니다.

**해결 과정**:

```nginx
http {
    # 클라이언트 연결 유지 시간
    keepalive_timeout 65;
    
    # 하나의 연결에서 처리할 수 있는 최대 요청 수
    keepalive_requests 1000;
    
    # 클라이언트 요청 본문 크기 제한
    client_max_body_size 10M;
    
    # 클라이언트 헤더 버퍼 크기
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    # 클라이언트 요청 타임아웃
    client_body_timeout 12;
    client_header_timeout 12;
    
    # 클라이언트 연결 타임아웃
    send_timeout 10;
    
    # Reset 연결 제어
    reset_timedout_connection on;
    
    server {
        listen 80;
        server_name example.com;
        
        # 서버별 연결 제한
        limit_conn_zone $binary_remote_addr zone=addr:10m;
        limit_conn addr 100;
        
        # 요청 속도 제한
        limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;
        limit_req zone=one burst=20 nodelay;
        
        location / {
            root /var/www/html;
            index index.html;
        }
    }
}
```

## 파일 전송 최적화

### 문제 시나리오: 대용량 파일 다운로드 속도 저하
**상황**: 사용자가 대용량 파일을 다운로드할 때 속도가 매우 느립니다.

**해결 과정**:

```nginx
http {
    # 파일 전송 최적화
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    
    # 파일 전송 버퍼 크기
    sendfile_max_chunk 1m;
    
    # 압축 설정
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json
        image/svg+xml;
    
    # Brotli 압축 (ngx_brotli 모듈 필요)
    # brotli on;
    # brotli_comp_level 6;
    # brotli_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 정적 파일 최적화
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            access_log off;
            
            # 파일 다운로드 속도 제한 (필요시)
            limit_rate 1m;
        }
        
        # 대용량 파일 다운로드
        location /downloads/ {
            # 부분 요청 지원
            add_header Accept-Ranges bytes;
            
            # 다운로드 속도 제한
            limit_rate 5m;
            
            # 연결 유지 시간 증가
            keepalive_timeout 300;
        }
        
        location / {
            try_files $uri $uri/ =404;
        }
    }
}
```

## 메모리 최적화

### 문제 시나리오: 메모리 사용량 증가로 인한 시스템 불안정
**상황**: 장시간 운영 후 메모리 사용량이 계속 증가하여 시스템이 불안정해집니다.

**해결 과정**:

```nginx
http {
    # 버퍼 최적화
    client_body_buffer_size 128k;
    client_max_body_size 10m;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    # 프록시 버퍼 최적화
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 4k;
    proxy_busy_buffers_size 8k;
    
    # FastCGI 버퍼 최적화
    fastcgi_buffering on;
    fastcgi_buffer_size 4k;
    fastcgi_buffers 8 4k;
    fastcgi_busy_buffers_size 8k;
    
    # 임시 파일 크기 제한
    client_body_temp_path /var/nginx/client_temp;
    proxy_temp_path /var/nginx/proxy_temp;
    fastcgi_temp_path /var/nginx/fastcgi_temp;
    uwsgi_temp_path /var/nginx/uwsgi_temp;
    scgi_temp_path /var/nginx/scgi_temp;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        location / {
            try_files $uri $uri/ =404;
        }
        
        # 프록시 설정
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 프록시 버퍼링 최적화
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            proxy_busy_buffers_size 8k;
        }
    }
}
```

## 캐싱 최적화

### 문제 시나리오: 반복 요청으로 인한 백엔드 부하
**상황**: 동일한 요청이 반복되어 백엔드 서버에 불필요한 부하가 발생합니다.

**해결 과정**:

```nginx
http {
    # 프록시 캐시 최적화
    proxy_cache_path /var/nginx/cache levels=1:2 keys_zone=proxy_cache:100m 
                     max_size=10g inactive=60m use_temp_path=off loader_files=1000 loader_sleep=50ms;
    
    # Open File Cache
    open_file_cache max=10000 inactive=20s;
    open_file_cache_valid 30s;
    open_file_cache_min_uses 2;
    open_file_cache_errors on;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 정적 파일 캐싱
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            access_log off;
            
            # 파일 캐시 활용
            open_file_cache max=10000 inactive=20s;
            open_file_cache_valid 30s;
            open_file_cache_min_uses 2;
            open_file_cache_errors on;
        }
        
        # API 캐싱
        location /api/ {
            proxy_cache proxy_cache;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_valid 200 5m;
            proxy_cache_valid 404 1m;
            
            # 캐시 우바이 조건
            proxy_cache_bypass $http_pragma $http_authorization;
            proxy_no_cache $http_pragma $http_authorization;
            
            add_header X-Proxy-Cache $upstream_cache_status;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location / {
            try_files $uri $uri/ =404;
        }
    }
}
```

## HTTP/2 최적화

### 문제 시나리오: HTTP/2 사용 시 성능 향상 미미
**상황**: HTTP/2를 활성화했지만 예상만큼 성능 향상이 없습니다.

**해결 과정**:

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;
    
    # SSL 설정
    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # HTTP/2 최적화
    http2_max_concurrent_streams 128;
    http2_max_field_size 4k;
    http2_max_header_size 16k;
    http2_max_requests 1000;
    http2_recv_buffer_size 256k;
    
    # 서버 푸시 설정 (필요시)
    location = /index.html {
        http2_push /css/style.css;
        http2_push /js/main.js;
        http2_push /images/logo.png;
        
        try_files $uri $uri/ =404;
    }
    
    # 정적 파일 최적화
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    location / {
        root /var/www/html;
        index index.html;
        try_files $uri $uri/ =404;
    }
}
```

## 로깅 최적화

### 문제 시나리오: 과도한 로깅으로 인한 디스크 I/O 부하
**상황**: 접근 로그가 너무 많이 기록되어 디스크 I/O에 부하가 발생합니다.

**해결 과정**:

```nginx
http {
    # 간소한 로그 형식
    log_format simple '$remote_addr [$time_local] "$request" $status $body_bytes_sent';
    
    # JSON 형식 로그 (로그 분석 도구와 연동 시)
    log_format json_combined escape=json
        '{'
        '"time_local":"$time_local",'
        '"remote_addr":"$remote_addr",'
        '"request":"$request",'
        '"status": "$status",'
        '"body_bytes_sent":"$body_bytes_sent",'
        '"request_time":"$request_time",'
        '"upstream_response_time":"$upstream_response_time"'
        '}';
    
    # 접근 로그 버퍼링
    access_log /var/log/nginx/access.log simple buffer=32k flush=1m;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 정적 파일은 로깅하지 않음
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            access_log off;
        }
        
        # 헬스 체크는 로깅하지 않음
        location /health {
            access_log off;
            return 200 "OK";
        }
        
        location / {
            try_files $uri $uri/ =404;
        }
    }
}
```

## 성능 모니터링

### 문제 시나리오: 성능 병목 지점 파악 어려움
**상황**: 어떤 부분이 성능 저하의 원인인지 파악하기 어렵습니다.

**해결 과정**:

```nginx
http {
    # 요청 처리 시간 로깅
    log_format detailed '$remote_addr - $remote_user [$time_local] "$request" '
                       '$status $body_bytes_sent "$http_referer" '
                       '"$http_user_agent" "$http_x_forwarded_for" '
                       'rt=$request_time uct="$upstream_connect_time" '
                       'uht="$upstream_header_time" urt="$upstream_response_time"';
    
    # 상태 모듈 활성화
    server {
        listen 80;
        server_name localhost;
        
        location /nginx_status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            deny all;
        }
        
        # 요청 처리 시간 모니터링
        location /request_time {
            return 200 "Request time: $request_time\n";
            access_log off;
            allow 127.0.0.1;
            deny all;
        }
    }
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 응답 시간 헤더 추가
        add_header X-Response-Time $request_time always;
        
        location / {
            try_files $uri $uri/ =404;
        }
        
        # 프록시 응답 시간 모니터링
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 업스트림 응답 시간 헤더
            add_header X-Upstream-Time $upstream_response_time always;
        }
    }
}
```

## 모범 사례 요약

1. **워커 프로세스**: CPU 코어 수에 맞게 워커 프로세스 수 설정
2. **커넥션**: 적절한 연결 수와 타임아웃 값 설정
3. **파일 전송**: sendfile, tcp_nopush, tcp_nodelay 옵션 활용
4. **메모리**: 버퍼 크기 최적화로 메모리 사용량 제어
5. **캐싱**: 다양한 캐싱 레벨로 불필요한 처리 감소
6. **HTTP/2**: 최신 프로토콜 기능 최적화로 전송 효율 향상
7. **로깅**: 불필요한 로깅 제한으로 I/O 부하 감소
8. **모니터링**: 성능 지표 지속 모니터링으로 병목 지점 파악

## 다음 단계

이제 성능 최적화를 마쳤습니다. 다음 장에서는 보안 강화에 대해 알아보겠습니다.