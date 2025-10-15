# 웹 서버 설정

Nginx를 웹 서버로 사용하는 것은 가장 기본적인 사용 사례입니다. 이 장에서는 정적 콘텐츠 서빙, 가상 호스트 설정, 로깅, 그리고 성능 최적화에 대해 다룹니다.

## 정적 콘텐츠 서빙

### 문제 시나리오: 정적 웹사이트 로딩 속도 저하
**상황**: 단순한 HTML/CSS/JavaScript로 구성된 웹사이트가 예상보다 로딩이 느립니다.

**원인 분석**:
1. 브라우저 캐싱이 적절히 설정되지 않음
2. 압축이 비활성화됨
3. 적절한 MIME 타입이 설정되지 않음

**해결 과정**:

#### 1. 기본 정적 콘텐츠 서빙 설정

```nginx
server {
    listen 80;
    server_name static-site.com www.static-site.com;
    
    # 웹사이트 루트 디렉토리
    root /var/www/static-site;
    index index.html index.htm;
    
    # 정적 파일 처리
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 2. 브라우저 캐싱 최적화

```nginx
server {
    listen 80;
    server_name static-site.com;
    root /var/www/static-site;
    index index.html;
    
    # 정적 파일 캐싱 설정
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;  # 정적 파일은 접근 로그 기록 안 함
        
        # CORS 헤더 (다른 도메인에서 리소스 접근 시)
        add_header Access-Control-Allow-Origin "*";
    }
    
    # HTML 파일은 짧게 캐싱 (콘텐츠 변경 빈번)
    location ~* \.(html)$ {
        expires 1h;
        add_header Cache-Control "public, must-revalidate";
    }
    
    # 기타 파일
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 3. 압축 설정

```nginx
http {
    # Gzip 압축 활성화
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;  # 1KB 이상 파일만 압축
    gzip_proxied any;
    gzip_comp_level 6;     # 압축 레벨 (1-9)
    
    # 압축할 MIME 타입
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json
        image/svg+xml;
    
    # Brotli 압축 (더 효율적이지만 모든 브라우저에서 지원하지 않음)
    # brotli on;
    # brotli_comp_level 6;
    # brotli_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
}
```

### 문제 시나리오: 대용량 미디어 파일 전송 문제
**상황**: 사용자가 동영상이나 대용량 이미지 다운로드 시 중단되거나 속도가 매우 느립니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name media-site.com;
    root /var/www/media;
    
    # 대용량 파일 전송 최적화
    location ~* \.(mp4|mov|avi|wmv|flv|webm|mkv)$ {
        mp4;  # H.264/AAC 스트리밍 지원
        
        # 비디오 스트리밍을 위한 부분 요청 지원
        add_header Accept-Ranges bytes;
        
        # 긴 만료 시간
        expires 1y;
        add_header Cache-Control "public";
        
        # 다운로드 속도 제한 (필요시)
        limit_rate 1m;  # 1MB/s
        
        # 클라이언트 타임아웃 증가
        client_body_timeout 300s;
        client_header_timeout 300s;
    }
    
    # 이미지 최적화
    location ~* \.(jpg|jpeg|png|gif)$ {
        expires 1y;
        add_header Cache-Control "public";
        
        # WebP 지원 (브라우저가 지원할 경우)
        location ~* \.(jpg|jpeg|png)$ {
            add_header Vary Accept;
            try_files $uri$webp_suffix $uri =404;
        }
    }
}
```

## 가상 호스트 설정

### 문제 시나리오: 여러 웹사이트 호스팅 시 혼선
**상황**: 한 서버에서 여러 도메인을 호스팅하는데 특정 도메인이 잘못된 웹사이트로 연결됩니다.

**해결 과정**:

#### 1. 기본 가상 호스트 구성

```nginx
# 첫 번째 도메인
server {
    listen 80;
    server_name site1.com www.site1.com;
    root /var/www/site1;
    index index.html;
    
    # 로그 분리
    access_log /var/log/nginx/site1.access.log;
    error_log /var/log/nginx/site1.error.log;
    
    location / {
        try_files $uri $uri/ =404;
    }
}

# 두 번째 도메인
server {
    listen 80;
    server_name site2.com www.site2.com;
    root /var/www/site2;
    index index.html;
    
    # 로그 분리
    access_log /var/log/nginx/site2.access.log;
    error_log /var/log/nginx/site2.error.log;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 2. 하위 도메인 처리

```nginx
# 메인 도메인
server {
    listen 80;
    server_name example.com www.example.com;
    root /var/www/main;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}

# 블로그 하위 도메인
server {
    listen 80;
    server_name blog.example.com;
    root /var/www/blog;
    index index.php;
    
    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }
    
    location ~ \.php$ {
        fastcgi_pass 127.0.0.1:9000;
        fastcgi_index index.php;
        include fastcgi_params;
    }
}

# API 하위 도메인
server {
    listen 80;
    server_name api.example.com;
    root /var/www/api;
    index index.php;
    
    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }
    
    location ~ \.php$ {
        fastcgi_pass 127.0.0.1:9000;
        fastcgi_index index.php;
        include fastcgi_params;
    }
}
```

#### 3. 와일드카드 서브도메인

```nginx
# 와일드카드 서브도메인 처리
server {
    listen 80;
    server_name *.example.com;
    root /var/www/subdomains;
    index index.html;
    
    # 서브도메인 이름을 디렉토리로 사용
    set $subdomain "";
    if ($host ~* ^([a-z0-9-]+)\.example\.com$) {
        set $subdomain $1;
    }
    
    location / {
        root /var/www/subdomains/$subdomain;
        try_files $uri $uri/ =404;
    }
}
```

## 로깅 설정

### 문제 시나리오: 로그 파일이 너무 빨리 커지고 디스크 공간 부족
**상황**: 접근 로그 파일이 하루 만에 수 GB 크기로 커져서 디스크 공간 부족 문제가 발생합니다.

**해결 과정**:

#### 1. 로그 형식 최적화

```nginx
http {
    # 상세한 로그 형식
    log_format detailed '$remote_addr - $remote_user [$time_local] "$request" '
                       '$status $body_bytes_sent "$http_referer" '
                       '"$http_user_agent" "$http_x_forwarded_for" '
                       'rt=$request_time uct="$upstream_connect_time" '
                       'uht="$upstream_header_time" urt="$upstream_response_time"';
    
    # 간소한 로그 형식 (고트래픽 사이트용)
    log_format simple '$remote_addr [$time_local] "$request" $status $body_bytes_sent';
    
    # JSON 형식 로그 (로그 분석 도구와 연동 시)
    log_format json_combined escape=json
        '{'
        '"time_local":"$time_local",'
        '"remote_addr":"$remote_addr",'
        '"remote_user":"$remote_user",'
        '"request":"$request",'
        '"status": "$status",'
        '"body_bytes_sent":"$body_bytes_sent",'
        '"request_time":"$request_time",'
        '"http_referrer":"$http_referer",'
        '"http_user_agent":"$http_user_agent"'
        '}';
}
```

#### 2. 조건부 로깅

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/example;
    
    # 상태 파일 체크를 통한 조건부 로깅
    set $loggable 1;
    
    # 정적 파일은 로깅하지 않음
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
        set $loggable 0;
    }
    
    # 헬스 체크 요청은 로깅하지 않음
    location /health {
        access_log off;
        set $loggable 0;
        return 200 "OK";
    }
    
    # 조건부 로깅 적용
    access_log /var/log/nginx/example.access.log detailed if=$loggable;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 3. 로그 회전 설정

```bash
# /etc/logrotate.d/nginx 파일 생성
/var/log/nginx/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 nginx adm
    sharedscripts
    postrotate
        if [ -f /var/run/nginx.pid ]; then
            kill -USR1 `cat /var/run/nginx.pid`
        fi
    endscript
}
```

## 보안 설정

### 문제 시나리오: 웹사이트가 취약점 스캐너에 노출됨
**상황**: 보안 스캔 결과 서버 정보 노출, 불필요한 HTTP 메소드 허용 등의 취약점이 발견되었습니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name secure-site.com;
    root /var/www/secure-site;
    
    # 서버 정보 숨김
    server_tokens off;
    
    # 불필요한 HTTP 메소드 차단
    if ($request_method !~ ^(GET|HEAD|POST)$ ) {
        return 405;
    }
    
    # 숨길 파일 및 디렉토리
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # 백업 및 임시 파일 접근 차단
    location ~* \.(bak|backup|old|orig|save|tmp)$ {
        deny all;
    }
    
    # 보안 헤더 추가
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # 클라이언트 요청 크기 제한
    client_max_body_size 10M;
    
    # 연결 제한
    limit_conn_zone $binary_remote_addr zone=addr:10m;
    limit_conn addr 10;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## 에러 페이지 처리

### 문제 시나리오: 404 에러 시 사용자 경험 저하
**상황**: 사용자가 존재하지 않는 페이지에 접속하면 기본 404 페이지가 표시되어 이탈율이 높습니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/example;
    
    # 커스텀 에러 페이지
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
    
    # 에러 페이지 위치
    location = /404.html {
        root /var/www/error;
        internal;
    }
    
    location = /50x.html {
        root /var/www/error;
        internal;
    }
    
    # 특정 조건에서 에러 처리
    location /api {
        # API가 다운되면 502 대신 200과 메시지 반환
        error_page 502 = @api_down;
        
        proxy_pass http://backend;
    }
    
    location @api_down {
        return 200 '{"status": "error", "message": "Service temporarily unavailable"}';
        add_header Content-Type application/json;
    }
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## 모범 사례 요약

1. **정적 파일 최적화**: 적절한 캐싱 헤더와 압축 설정
2. **가상 호스트**: 각 도메인별로 분리된 설정과 로그
3. **로깅**: 조건부 로깅과 로그 회전으로 디스크 공간 관리
4. **보안**: 불필요한 정보 노출 방지와 적절한 보안 헤더
5. **에러 처리**: 사용자 친화적인 에러 페이지 제공

## 다음 단계

이제 웹 서버로서의 Nginx 설정을 마쳤습니다. 다음 장에서는 리버스 프록시로서의 Nginx 설정에 대해 알아보겠습니다.