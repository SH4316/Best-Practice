# 보안 강화

Nginx를 웹 서버나 리버스 프록시로 사용할 때 보안은 매우 중요합니다. 이 장에서는 다양한 보안 위협으로부터 Nginx와 백엔드 서버를 보호하는 방법과 실제 시나리오에 맞는 보안 설정을 다룹니다.

## 기본 보안 설정

### 문제 시나리오: 서버 정보 노출로 인한 정보 유출
**상황**: 보안 스캔 결과 Nginx 버전과 서버 정보가 노출되고 있습니다.

**원인 분석**: 기본 설정에서 서버 정보가 응답 헤더에 포함됩니다.

**해결 과정**:

#### 1. 서버 정보 숨김

```nginx
# /etc/nginx/nginx.conf

http {
    # 서버 정보 숨김
    server_tokens off;
    
    # 서버 이름 변경 (선택사항)
    server_name_in_redirect off;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        location / {
            try_files $uri $uri/ =404;
        }
    }
}
```

#### 2. 불필요한 HTTP 메소드 차단

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 불필요한 HTTP 메소드 차단
    if ($request_method !~ ^(GET|HEAD|POST)$ ) {
        return 405;
    }
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## 보안 헤더 설정

### 문제 시나리오: 클릭재킹 및 XSS 공격에 취약
**상황**: 보안 스캔 결과 클릭재킹, XSS 등 클라이언트 측 공격에 취약합니다.

**해결 과정**:

#### 1. 보안 헤더 추가

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 보안 헤더 추가
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:;" always;
    
    # HSTS (HTTPS 전용)
    # add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 2. 콘텐츠 보안 정책 (CSP) 최적화

```nginx
server {
    listen 443 ssl;
    server_name secure.example.com;
    root /var/www/html;
    
    # 엄격한 CSP 정책
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self';" always;
    
    # 개발 환경에서는 덜 엄격한 정책
    # add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:;" always;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## 접근 제어

### 문제 시나리오: 관리자 페이지에 무분별한 접근 시도
**상황**: 관리자 페이지에 대한 무차별 대입 공격이 감지됩니다.

**해결 과정**:

#### 1. IP 기반 접근 제어

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 관리자 페이지 IP 제한
    location /admin {
        allow 192.168.1.0/24;
        allow 10.0.0.0/8;
        allow 127.0.0.1;
        deny all;
        
        try_files $uri $uri/ =404;
    }
    
    # API 엔드포인트 IP 제한
    location /api/admin {
        allow 192.168.1.100;
        allow 10.0.0.50;
        deny all;
        
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
```

#### 2. 기본 인증 (Basic Authentication)

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 관리자 페이지 기본 인증
    location /admin {
        auth_basic "Restricted Area";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        try_files $uri $uri/ =404;
    }
    
    # API 엔드포인트 기본 인증
    location /api/secure {
        auth_basic "API Access";
        auth_basic_user_file /etc/nginx/.api_htpasswd;
        
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
```

#### 3. htpasswd 파일 생성

```bash
# htpasswd 파일 생성
sudo apt-get install apache2-utils

# 사용자 추가
sudo htpasswd -c /etc/nginx/.htpasswd admin
sudo htpasswd /etc/nginx/.htpasswd user1

# API 사용자 추가
sudo htpasswd -c /etc/nginx/.api_htpasswd api_user
```

## 속도 제한

### 문제 시나리오: DDoS 공격으로 인한 서비스 중단
**상황**: 대량의 요청이 서버를 공격하여 정상적인 서비스가 불가능합니다.

**해결 과정**:

#### 1. 요청 속도 제한

```nginx
http {
    # 속도 제한 영역 정의
    limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;
    limit_req_zone $binary_remote_addr zone=admin:10m rate=1r/s;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 일반 요청 속도 제한
        location / {
            limit_req zone=one burst=20 nodelay;
            
            try_files $uri $uri/ =404;
        }
        
        # API 요청 속도 제한
        location /api/ {
            limit_req zone=api burst=10 nodelay;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 관리자 페이지 속도 제한
        location /admin {
            limit_req zone=admin burst=5 nodelay;
            
            try_files $uri $uri/ =404;
        }
    }
}
```

#### 2. 연결 수 제한

```nginx
http {
    # 연결 제한 영역 정의
    limit_conn_zone $binary_remote_addr zone=addr:10m;
    limit_conn_zone $server_name zone=servers:10m;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 클라이언트당 연결 수 제한
        limit_conn addr 10;
        
        # 서버당 총 연결 수 제한
        limit_conn servers 100;
        
        location / {
            try_files $uri $uri/ =404;
        }
    }
}
```

## 파일 접근 제어

### 문제 시나리오: 민감한 파일에 대한 무단 접근
**상황**: 설정 파일, 백업 파일 등 민감한 파일에 접근할 수 있습니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 숨김 파일 및 디렉토리 접근 차단
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # 백업 및 임시 파일 접근 차단
    location ~* \.(bak|backup|old|orig|save|tmp|log|sql)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # 설정 파일 접근 차단
    location ~* \.(conf|config|ini|htaccess|htpasswd)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # 소스 코드 파일 접근 차단
    location ~* \.(inc|tpl|py|pl|sh|rb)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # 특정 디렉토리 접근 제어
    location ~ ^/(config|backup|logs|tmp)/ {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## DDoS 방어

### 문제 시나리오: 복잡한 DDoS 공격으로 인한 서비스 중단
**상황**: 다양한 IP에서 동시에 발생하는 DDoS 공격을 막아야 합니다.

**해결 과정**:

#### 1. 다단계 DDoS 방어

```nginx
http {
    # 속도 제한 영역 정의
    limit_req_zone $binary_remote_addr zone=global:10m rate=100r/s;
    limit_req_zone $binary_remote_addr zone=strict:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;
    
    # 연결 제한 영역 정의
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
    
    # 요청 크기 제한
    client_max_body_size 10M;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 전역 속도 제한
        limit_req zone=global burst=200 nodelay;
        
        # 연결 수 제한
        limit_conn conn_limit 20;
        
        # 의심스러운 요청 필터링
        if ($http_user_agent ~* (bot|crawler|spider|scraper)) {
            limit_req zone=strict burst=5 nodelay;
        }
        
        # 정적 파일 (더 높은 제한)
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf)$ {
            limit_req zone=global burst=500 nodelay;
            expires 1y;
            add_header Cache-Control "public, immutable";
            access_log off;
        }
        
        # API 엔드포인트 (더 엄격한 제한)
        location /api/ {
            limit_req zone=api burst=10 nodelay;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location / {
            limit_req zone=strict burst=20 nodelay;
            try_files $uri $uri/ =404;
        }
    }
}
```

#### 2. 동적 차단 시스템

```nginx
# 이 설정은 Lua 모듈이 필요함

http {
    # 공격자 IP 목록
    lua_shared_dict blocked_ips 10m;
    
    init_by_lua '
        -- 초기화 시 블록된 IP 목록 로드
        local blocked_ips = ngx.shared.blocked_ips
        -- 파일이나 데이터베이스에서 블록된 IP 로드 가능
    ';
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        access_by_lua '
            local blocked_ips = ngx.shared.blocked_ips
            local remote_addr = ngx.var.remote_addr
            
            -- 블록된 IP 확인
            if blocked_ips:get(remote_addr) then
                ngx.exit(403)
                return
            end
            
            -- 요청 빈도 확인
            local key = "req:" .. remote_addr
            local req_count = blocked_ips:get(key) or 0
            
            if req_count > 100 then
                -- 100개 이상의 요청이 있으면 IP 차단
                blocked_ips:set(remote_addr, true, 3600)  -- 1시간 차단
                ngx.exit(429)  -- Too Many Requests
                return
            end
            
            -- 요청 카운트 증가
            blocked_ips:incr(key, 1, 1, 60)  -- 1분간 유지
        ';
        
        location / {
            try_files $uri $uri/ =404;
        }
    }
}
```

## WAF (Web Application Firewall)

### 문제 시나리오: 웹 애플리케이션 공격 방어
**상황**: SQL 인젝션, XSS 등 웹 애플리케이션 공격을 방어해야 합니다.

**해결 과정**:

#### 1. ModSecurity 연동

```nginx
# ModSecurity 모듈이 필요함

http {
    # ModSecurity 설정
    modsecurity on;
    modsecurity_rules_file /etc/nginx/modsec/main.conf;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # ModSecurity 활성화
        modsecurity on;
        
        location / {
            try_files $uri $uri/ =404;
        }
        
        # API 엔드포인트에 더 엄격한 규칙 적용
        location /api/ {
            modsecurity_rules '
                SecRuleEngine On
                SecRule ARGS "@detectSQLi" "id:1001,phase:2,deny,status:403,msg:\'SQL Injection Attack Detected\'"
                SecRule ARGS "@detectXSS" "id:1002,phase:2,deny,status:403,msg:\'XSS Attack Detected\'"
            ';
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### 2. 간단한 WAF 규칙

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # SQL 인젝션 방어
    if ($args ~* "union.*select.*\(") {
        return 403;
    }
    
    if ($args ~* "concat.*\(") {
        return 403;
    }
    
    # XSS 방어
    if ($args ~* "<script") {
        return 403;
    }
    
    if ($args ~* "javascript:") {
        return 403;
    }
    
    # 경로 순회 공격 방어
    if ($uri ~* "\.\./") {
        return 403;
    }
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## 보안 모니터링

### 문제 시나리오: 보안 이벤트 감지 및 대응
**상황**: 보안 침해 시도를 실시간으로 감지하고 대응해야 합니다.

**해결 과정**:

```nginx
http {
    # 보안 로그 형식
    log_format security '$remote_addr - $remote_user [$time_local] "$request" '
                       '$status $body_bytes_sent "$http_referer" '
                       '"$http_user_agent" "$http_x_forwarded_for" '
                       'rt=$request_time uc=$upstream_cache_status';
    
    # 보안 로그 파일
    access_log /var/log/nginx/security.log security;
    
    server {
        listen 80;
        server_name example.com;
        root /var/www/html;
        
        # 보안 헤더
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
        
        # 의심스러운 활동 로깅
        location / {
            # 4xx 및 5xx 응답 로깅
            if ($status >= 400) {
                access_log /var/log/nginx/security.log security;
            }
            
            try_files $uri $uri/ =404;
        }
        
        # 관리자 페이지 접근 로깅
        location /admin {
            access_log /var/log/nginx/admin_access.log security;
            
            try_files $uri $uri/ =404;
        }
        
        # 보안 상태 페이지
        location /security-status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            deny all;
        }
    }
}
```

## 모범 사례 요약

1. **서버 정보 숨김**: server_tokens off로 버전 정보 노출 방지
2. **보안 헤더**: X-Frame-Options, CSP 등으로 클라이언트 측 공격 방어
3. **접근 제어**: IP 기반 제어와 기본 인증으로 민감한 리소스 보호
4. **속도 제한**: limit_req와 limit_conn으로 DDoS 공격 방어
5. **파일 접근 제어**: 민감한 파일과 디렉토리 접근 차단
6. **WAF**: ModSecurity 등으로 웹 애플리케이션 공격 방어
7. **모니터링**: 보안 이벤트 로깅과 실시간 감지

## 다음 단계

이제 보안 강화를 마쳤습니다. 다음 장에서는 마이크로서비스 및 API 게이트웨이에 대해 알아보겠습니다.