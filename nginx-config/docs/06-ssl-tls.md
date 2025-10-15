# SSL/TLS 보안 설정

SSL/TLS는 웹 통신을 암호화하여 중간자 공격을 방지하는 중요한 보안 기술입니다. 이 장에서는 Nginx에서 SSL/TLS를 안전하게 설정하는 방법과 모범 사례를 다룹니다.

## 기본 SSL/TLS 설정

### 문제 시나리오: 기본 SSL 설정으로 인한 보안 취약점
**상황**: SSL 인증서를 설치했지만 보안 스캔에서 여러 취약점이 발견되었습니다.

**원인 분석**: 기본 SSL 설정은 오래된 프로토콜과 암호화 방식을 사용하여 보안에 취약합니다.

**해결 과정**:

#### 1. 기본 SSL 설정

```nginx
server {
    listen 443 ssl;
    server_name example.com;
    
    # SSL 인증서 경로
    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;
    
    root /var/www/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 2. 보안 강화 SSL 설정

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;
    
    # SSL 인증서 경로
    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;
    
    # SSL 프로토콜 설정 (오래된 프로토콜 제외)
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # 암호화 스위트 설정 (안전한 순서로)
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # SSL 세션 설정
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;
    
    # 보안 헤더
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    root /var/www/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## HTTP에서 HTTPS로 리디렉션

### 문제 시나리오: HTTP와 HTTPS 혼용으로 인한 보안 위험
**상황**: 사용자가 HTTP로 접속하면 암호화되지 않은 통신이 발생합니다.

**해결 과정**:

```nginx
# HTTP 서버 블록 (HTTPS로 리디렉션)
server {
    listen 80;
    server_name example.com www.example.com;
    
    # HTTPS로 영구 리디렉션
    return 301 https://$host$request_uri;
}

# HTTPS 서버 블록
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;
    
    # SSL 설정
    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;
    
    # 보안 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    # HSTS 헤더
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    
    root /var/www/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## Let's Encrypt 인증서 설정

### 문제 시나리오: 무료 SSL 인증서 자동 갱신
**상황**: Let's Encrypt 인증서를 사용하지만 수동 갱신이 번거롭고 만료될 위험이 있습니다.

**해결 과정**:

#### 1. Certbot 설치 및 인증서 발급

```bash
# Certbot 설치
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# 인증서 발급
sudo certbot --nginx -d example.com -d www.example.com
```

#### 2. 자동 갱신 설정

```bash
# 갱신 테스트
sudo certbot renew --dry-run

# cron에 자동 갱신 등록
sudo crontab -e

# 다음 내용 추가 (매일 새벽 2시에 갱신 확인)
0 2 * * * /usr/bin/certbot renew --quiet --post-hook "systemctl reload nginx"
```

#### 3. Nginx 설정 (Certbot이 자동으로 생성)

```nginx
server {
    server_name example.com www.example.com;
    root /var/www/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
    
    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot
}

server {
    if ($host = www.example.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot
    
    if ($host = example.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot
    
    listen 80;
    server_name example.com www.example.com;
    return 404; # managed by Certbot
}
```

## 와일드카드 인증서 설정

### 문제 시나리오: 여러 서브도메인에 SSL 적용
**상황**: 여러 서브도메인(app.example.com, api.example.com, blog.example.com)에 SSL을 적용해야 합니다.

**해결 과정**:

#### 1. 와일드카드 인증서 발급

```bash
# DNS 인증을 통한 와일드카드 인증서 발급
sudo certbot certonly --manual --preferred-challenges dns -d "*.example.com" -d example.com
```

#### 2. 와일드카드 인증서 설정

```nginx
# 모든 서브도메인에 적용되는 설정
server {
    listen 443 ssl http2;
    server_name ~^(?<subdomain>.+)\.example\.com$;
    
    # 와일드카드 인증서
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    
    # SSL 보안 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    # 서브도메인별 루트 디렉토리 설정
    root /var/www/$subdomain;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}

# 도메인 리디렉션
server {
    listen 80;
    server_name .example.com;
    return 301 https://$host$request_uri;
}
```

## SSL/TLS 튜닝 및 최적화

### 문제 시나리오: SSL 핸드셰이크로 인한 성능 저하
**상황**: HTTPS 사이트가 HTTP 사이트보다 현저히 느립니다.

**해결 과정**:

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;
    
    # SSL 인증서
    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;
    
    # SSL 프로토콜 및 암호화
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # SSL 세션 최적화
    ssl_session_cache shared:SSL:50m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;
    
    # HTTP/2 설정
    http2_max_concurrent_streams 128;
    http2_max_field_size 4k;
    http2_max_header_size 16k;
    http2_max_requests 1000;
    
    # SSL 리디렉션 캐싱
    error_page 497 301 =307 https://$host:$server_port$request_uri;
    
    root /var/www/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## Mutual TLS (mTLS) 설정

### 문제 시나리오: 클라이언트 인증이 필요한 API
**상황**: 특정 클라이언트만 접근할 수 있는 API 서버가 필요합니다.

**해결 과정**:

#### 1. CA 인증서 및 클라이언트 인증서 생성

```bash
# CA 개인 키 생성
openssl genrsa -out ca.key 4096

# CA 인증서 생성
openssl req -new -x509 -days 365 -key ca.key -out ca.crt

# 서버 개인 키 생성
openssl genrsa -out server.key 4096

# 서버 CSR 생성
openssl req -new -key server.key -out server.csr

# 서버 인증서 서명
openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt

# 클라이언트 개인 키 생성
openssl genrsa -out client.key 4096

# 클라이언트 CSR 생성
openssl req -new -key client.key -out client.csr

# 클라이언트 인증서 서명
openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 02 -out client.crt
```

#### 2. mTLS 설정

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    # 서버 인증서
    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;
    
    # 클라이언트 인증서 검증
    ssl_client_certificate /etc/nginx/ssl/ca.crt;
    ssl_verify_client on;
    ssl_verify_depth 2;
    
    # SSL 보안 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    location / {
        # 클라이언트 인증서 정보 추출
        if ($ssl_client_verify != SUCCESS) {
            return 403;
        }
        
        # 클라이언트 정보 헤더 추가
        proxy_set_header X-SSL-Client-DN $ssl_client_s_dn;
        proxy_set_header X-SSL-Client-Verify $ssl_client_verify;
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## SSL/TLS 모니터링

### 문제 시나리오: 인증서 만료로 인한 서비스 중단
**상황**: SSL 인증서가 만료되어 서비스가 중단되었습니다.

**해결 과정**:

#### 1. 인증서 만료 모니터링 스크립트

```bash
#!/bin/bash
# ssl-check.sh

DOMAIN="example.com"
DAYS_WARNING=30

# 인증서 만료일 확인
EXPIRE_DATE=$(openssl s_client -connect $DOMAIN:443 -servername $DOMAIN 2>/dev/null | openssl x509 -noout -enddate | cut -d= -f2)
EXPIRE_EPOCH=$(date -d "$EXPIRE_DATE" +%s)
CURRENT_EPOCH=$(date +%s)
DAYS_LEFT=$(( ($EXPIRE_EPOCH - $CURRENT_EPOCH) / 86400 ))

# 경고 메시지
if [ $DAYS_LEFT -lt $DAYS_WARNING ]; then
    echo "경고: $DOMAIN의 SSL 인증서가 $DAYS_LEFT일 후에 만료됩니다."
    # 이메일 또는 슬랙 알림
    # mail -s "SSL 인증서 만료 경고" admin@example.com
else
    echo "$DOMAIN의 SSL 인증서는 $DAYSLeft일 후에 만료됩니다."
fi
```

#### 2. Nginx 상태 모니터링

```nginx
server {
    listen 80;
    server_name localhost;
    
    location /ssl-status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
    
    location /cert-info {
        default_type text/plain;
        return 200 "Domain: $ssl_server_name\nProtocol: $ssl_protocol\nCipher: $ssl_cipher\n";
        allow 127.0.0.1;
        deny all;
    }
}
```

## 모범 사례 요약

1. **프로토콜**: TLSv1.2 이상만 사용, 오래된 프로토콜 제외
2. **암호화 스위트**: 강력한 암호화 방식 사용, 서버 선호 설정
3. **세션 관리**: SSL 세션 캐싱으로 핸드셰이크 최적화
4. **OCSP Stapling**: 인증서 유효성 검사 최적화
5. **보안 헤더**: HSTS, X-Frame-Options 등 보안 헤더 추가
6. **인증서 관리**: 자동 갱신 설정으로 만료 방지
7. **모니터링**: 인증서 만료일과 SSL 상태 지속 모니터링

## 다음 단계

이제 SSL/TLS 보안 설정을 마쳤습니다. 다음 장에서는 캐싱 전략에 대해 알아보겠습니다.