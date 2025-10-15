# 캐싱 전략

캐싱은 웹 애플리케이션의 성능을 향상시키고 서버 부하를 줄이는 가장 효과적인 방법 중 하나입니다. 이 장에서는 Nginx의 다양한 캐싱 기능과 실제 시나리오에 맞는 캐싱 전략을 다룹니다.

## 브라우저 캐싱

### 문제 시나리오: 반복 방문 시에도 리소스를 계속 다운로드
**상황**: 사용자가 사이트를 재방문할 때마다 동일한 CSS, JavaScript, 이미지 파일을 다시 다운로드하여 로딩 속도가 느립니다.

**원인 분석**: 브라우저 캐싱 헤더가 적절히 설정되지 않았습니다.

**해결 과정**:

#### 1. 기본 브라우저 캐싱 설정

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 정적 파일 캐싱 (1년)
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    # HTML 파일 캐싱 (1시간)
    location ~* \.(html)$ {
        expires 1h;
        add_header Cache-Control "public, must-revalidate";
    }
    
    # API 응답 캐싱 (5분)
    location /api/ {
        expires 5m;
        add_header Cache-Control "public, must-revalidate";
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # 기타 파일
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 2. 조건부 캐싱 설정

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 쿼리 문자열이 있는 파일은 캐싱하지 않음
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg)$ {
        if ($args) {
            expires -1;
            add_header Cache-Control "no-cache, no-store, must-revalidate";
        }
        
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    # 개발 환경에서는 캐싱 비활성화
    location ~* \.(css|js)$ {
        if ($http_cookie ~* "dev_mode=on") {
            expires -1;
            add_header Cache-Control "no-cache, no-store, must-revalidate";
            break;
        }
        
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

## 프록시 캐싱

### 문제 시나리오: API 서버 부하로 인한 응답 속도 저하
**상황**: 동일한 API 요청이 반복되어 백엔드 서버에 부하가 발생하고 응답 속도가 저하됩니다.

**해결 과정**:

#### 1. 기본 프록시 캐싱 설정

```nginx
http {
    # 프록시 캐시 경로 설정
    proxy_cache_path /var/nginx/cache levels=1:2 keys_zone=api_cache:10m 
                     max_size=1g inactive=60m use_temp_path=off;
    
    server {
        listen 80;
        server_name api.example.com;
        
        location /api/data {
            # 캐시 활성화
            proxy_cache api_cache;
            
            # 캐시 키 설정
            proxy_cache_key "$scheme$request_method$host$request_uri";
            
            # 캐시 유효기간 설정
            proxy_cache_valid 200 5m;    # 성공 응답은 5분
            proxy_cache_valid 404 1m;    # 404는 1분
            proxy_cache_valid any 1m;    # 기타 응답은 1분
            
            # 캐시 관련 헤더
            add_header X-Proxy-Cache $upstream_cache_status;
            
            # 백엔드 서버 설정
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### 2. 고급 프록시 캐싱 설정

```nginx
http {
    # 다중 캐시 영역 설정
    proxy_cache_path /var/nginx/cache/api levels=1:2 keys_zone=api_cache:10m 
                     max_size=2g inactive=60m use_temp_path=off;
    proxy_cache_path /var/nginx/cache/static levels=1:2 keys_zone=static_cache:10m 
                     max_size=5g inactive=24h use_temp_path=off;
    
    # 캐시 키 맵 설정
    map $request_method $cache_key_method {
        GET $request_method;
        HEAD $request_method;
        default "";
    }
    
    server {
        listen 80;
        server_name optimized.example.com;
        
        # 정적 리소스 캐싱
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf)$ {
            proxy_cache static_cache;
            proxy_cache_key "$scheme$cache_key_method$host$request_uri";
            proxy_cache_valid 200 7d;
            proxy_cache_valid 404 1h;
            
            # 캐시 우회 조건
            proxy_cache_bypass $http_pragma $http_authorization;
            proxy_no_cache $http_pragma $http_authorization;
            
            add_header X-Proxy-Cache $upstream_cache_status;
            
            # 정적 파일은 직접 서빙
            try_files $uri =404;
        }
        
        # API 캐싱
        location /api/ {
            proxy_cache api_cache;
            proxy_cache_key "$scheme$cache_key_method$host$request_uri";
            proxy_cache_valid 200 5m;
            proxy_cache_valid 404 1m;
            
            # POST 요청도 캐싱 (읽기 전용 API의 경우)
            proxy_cache_methods GET HEAD POST;
            
            # 캐시 우바이 조건
            proxy_cache_bypass $http_pragma $http_authorization $cookie_nocache;
            proxy_no_cache $http_pragma $http_authorization $cookie_nocache;
            
            add_header X-Proxy-Cache $upstream_cache_status;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 동적 콘텐츠 (캐싱 안 함)
        location /dynamic/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## FastCGI 캐싱

### 문제 시나리오: PHP 애플리케이션 응답 속도 저하
**상황**: WordPress나 다른 PHP 기반 애플리케이션의 응답 속도가 느립니다.

**해결 과정**:

#### 1. FastCGI 캐싱 설정

```nginx
http {
    # FastCGI 캐시 경로 설정
    fastcgi_cache_path /var/nginx/fastcgi_cache levels=1:2 keys_zone=fastcgi_cache:100m 
                       max_size=2g inactive=60m use_temp_path=off;
    
    # FastCGI 캐시 키 설정
    fastcgi_cache_key "$scheme$request_method$host$request_uri";
    
    server {
        listen 80;
        server_name php-app.example.com;
        root /var/www/php-app;
        index index.php;
        
        # 동적 콘텐츠 캐싱
        location ~ \.php$ {
            # FastCGI 서버 설정
            fastcgi_pass 127.0.0.1:9000;
            fastcgi_index index.php;
            include fastcgi_params;
            fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
            
            # FastCGI 캐싱
            fastcgi_cache fastcgi_cache;
            fastcgi_cache_valid 200 10m;
            fastcgi_cache_valid 404 1m;
            
            # 캐시 우바이 조건
            fastcgi_cache_bypass $cookie_nocache $arg_nocache;
            fastcgi_no_cache $cookie_nocache $arg_nocache;
            
            # 캐시 헤더
            add_header X-FastCGI-Cache $upstream_cache_status;
        }
        
        # 정적 파일
        location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            access_log off;
        }
        
        location / {
            try_files $uri $uri/ /index.php?$query_string;
        }
    }
}
```

## 캐시 무효화 전략

### 문제 시나리오: 콘텐츠 업데이트 후에도 오래된 콘텐츠가 표시됨
**상황**: 웹사이트 콘텐츠를 업데이트했는데도 사용자에게 이전 버전이 계속 표시됩니다.

**해결 과정**:

#### 1. 파일명 기반 캐시 무효화

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/html;
    
    # 해시를 포함한 파일명 자동 리라이트
    location ~* \.(css|js)$ {
        # file.abc123.css -> file.css
        rewrite ^(.+)\.([a-f0-9]{8,})\.(css|js)$ $1.$3 last;
        
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

#### 2. API 캐시 무효화

```nginx
http {
    proxy_cache_path /var/nginx/cache/api levels=1:2 keys_zone=api_cache:10m 
                     max_size=1g inactive=60m use_temp_path=off;
    
    server {
        listen 80;
        server_name api.example.com;
        
        # GET 요청 캐싱
        location /api/data {
            proxy_cache api_cache;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_valid 200 5m;
            
            add_header X-Proxy-Cache $upstream_cache_status;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # POST/PUT/DELETE 요청 처리 (캐시 무효화)
        location /api/data {
            if ($request_method ~ ^(POST|PUT|DELETE)$) {
                # 캐시 무효화 스크립트 호출
                # 이 부분은 Lua 스크립트나 외부 프로그램으로 구현 가능
                proxy_pass http://backend;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                
                # 응답 후 캐시 정리
                add_header X-Cache-Invalidated "true";
                break;
            }
        }
        
        # 캐시 정리 API
        location /admin/cache-clear {
            allow 127.0.0.1;
            deny all;
            
            # 캐시 디렉토리 정리
            # 이 부분은 스크립트로 구현 필요
            return 200 "Cache cleared";
        }
    }
}
```

## 분산 캐싱

### 문제 시나리오: 여러 Nginx 서버 간 캐시 불일치
**상황**: 로드 밸런싱된 여러 Nginx 서버가 각각 다른 캐시를 가지고 있어 일관성이 없습니다.

**해결 과정**:

#### 1. Redis를 이용한 분산 캐싱

```nginx
# 이 설정은 ngx_http_redis_module 또는 Lua 모듈이 필요함

http {
    # Redis 서버 설정
    upstream redis {
        server 127.0.0.1:6379;
        keepalive 32;
    }
    
    server {
        listen 80;
        server_name distributed.example.com;
        
        location /api/data {
            # Redis에서 캐시 확인
            set $redis_key "$request_uri";
            redis_pass redis;
            
            # 캐시 미스 시 백엔드에서 가져오기
            error_page 404 = @backend;
        }
        
        location @backend {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 응답을 Redis에 저장
            # 이 부분은 Lua 스크립트로 구현 필요
        }
    }
}
```

## 캐시 성능 모니터링

### 문제 시나리오: 캐시 효율성 측정
**상황**: 캐시가 얼마나 효과적으로 작동하는지 알 수 없습니다.

**해결 과정**:

#### 1. 캐시 상태 모니터링

```nginx
http {
    proxy_cache_path /var/nginx/cache levels=1:2 keys_zone=api_cache:10m 
                     max_size=1g inactive=60m use_temp_path=off;
    
    server {
        listen 80;
        server_name monitored.example.com;
        
        location /api/data {
            proxy_cache api_cache;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_valid 200 5m;
            
            # 캐시 상태 헤더 추가
            add_header X-Proxy-Cache $upstream_cache_status;
            add_header X-Cache-Key "$scheme$request_method$host$request_uri";
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 캐시 통계 API
        location /cache-status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            deny all;
        }
    }
}
```

#### 2. 캐시 통계 스크립트

```bash
#!/bin/bash
# cache-stats.sh

CACHE_DIR="/var/nginx/cache"
LOG_FILE="/var/log/nginx/cache_stats.log"

# 캐시 크기 확인
CACHE_SIZE=$(du -sh $CACHE_DIR | cut -f1)
CACHE_FILES=$(find $CACHE_DIR -type f | wc -l)

# 로그 기록
echo "$(date): Cache size: $CACHE_SIZE, Files: $CACHE_FILES" >> $LOG_FILE

# 캐시 히트율 계산 (로그 분석)
HIT_COUNT=$(grep "HIT" /var/log/nginx/access.log | wc -l)
MISS_COUNT=$(grep "MISS" /var/log/nginx/access.log | wc -l)
TOTAL_COUNT=$((HIT_COUNT + MISS_COUNT))

if [ $TOTAL_COUNT -gt 0 ]; then
    HIT_RATE=$(echo "scale=2; $HIT_COUNT * 100 / $TOTAL_COUNT" | bc)
    echo "$(date): Cache hit rate: $HIT_RATE%" >> $LOG_FILE
fi
```

## 모범 사례 요약

1. **브라우저 캐싱**: 정적 리소스는 장기간 캐싱, 동적 콘텐츠는 단기간 캐싱
2. **프록시 캐싱**: API 응답과 정적 리소스를 서버 측에서 캐싱
3. **캐시 무효화**: 파일명 해시나 버전 관리로 캐시 제어
4. **분산 캐싱**: 여러 서버 간 캐시 일관성 유지
5. **모니터링**: 캐시 히트율과 성능 지표 지속 모니터링
6. **조건부 캐싱**: 쿠키, 인증 상태 등에 따른 동적 캐싱 정책

## 다음 단계

이제 캐싱 전략을 마쳤습니다. 다음 장에서는 성능 최적화에 대해 알아보겠습니다.