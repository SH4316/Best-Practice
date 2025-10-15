# 리버스 프록시 설정

리버스 프록시는 클라이언트와 백엔드 서버 사이에서 중개 역할을 하는 Nginx의 중요한 기능입니다. 이 장에서는 다양한 리버스 프록시 시나리오와 문제 해결 방법을 다룹니다.

## 기본 리버스 프록시 설정

### 문제 시나리오: 단일 백엔드 애플리케이션 프록시
**상황**: Nginx를 사용하여 내부에서 실행 중인 Node.js 애플리케이션을 외부에 노출해야 합니다.

**해결 과정**:

#### 1. 기본 프록시 설정

```nginx
server {
    listen 80;
    server_name example.com;
    
    # 모든 요청을 백엔드 서버로 전달
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 2. 프록시 헤더 최적화

```nginx
server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://127.0.0.1:3000;
        
        # 기본 헤더 설정
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 추가 헤더 설정
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        proxy_set_header X-Original-URI $request_uri;
        
        # 프록시 타임아웃 설정
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # 버퍼링 설정
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
}
```

### 문제 시나리오: 백엔드 서버 응답이 느림
**상황**: 백엔드 애플리케이션이 느리게 응답하여 사용자 경험이 저하됩니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://127.0.0.1:3000;
        
        # 타임아웃 증가
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # 버퍼링 최적화
        proxy_buffering on;
        proxy_buffer_size 8k;
        proxy_buffers 16 8k;
        proxy_busy_buffers_size 16k;
        
        # 버퍼링 임시 파일 설정
        proxy_temp_path /var/nginx/proxy_temp;
        proxy_max_temp_file_size 1024m;
        
        # 재시도 설정
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_next_upstream_tries 3;
        proxy_next_upstream_timeout 30s;
    }
}
```

## 경로 기반 라우팅

### 문제 시나리오: 여러 애플리케이션을 단일 도메인에서 제공
**상황**: API 서버와 웹 애플리케이션을 각각 다른 경로(/api, /app)로 제공해야 합니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name example.com;
    
    # 웹 애플리케이션
    location /app {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 경로 재작성 (필요시)
        rewrite ^/app/(.*)$ /$1 break;
    }
    
    # API 서버
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # API 특화 설정
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # CORS 헤더 추가 (필요시)
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        
        # OPTIONS 요청 처리
        if ($request_method = OPTIONS) {
            return 204;
        }
    }
    
    # 정적 파일
    location / {
        root /var/www/static;
        try_files $uri $uri/ =404;
    }
}
```

### 문제 시나리오: 마이그레이션 중 점진적 트래픽 전환
**상황**: 새 버전의 애플리케이션으로 점진적으로 트래픽을 전환해야 합니다.

**해결 과정**:

```nginx
http {
    # IP 해시를 사용한 트래픽 분할
    split_clients "${remote_addr}" $backend {
        20%     http://127.0.0.1:3000;  # 구 버전 (20%)
        80%     http://127.0.0.1:3001;  # 신규 버전 (80%)
    }
    
    server {
        listen 80;
        server_name example.com;
        
        location / {
            proxy_pass $backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 어떤 버전을 사용하는지 헤더 추가 (디버깅용)
            add_header X-Backend-Version $backend;
        }
    }
}
```

## WebSocket 프록시

### 문제 시나리오: WebSocket 연결이 끊김
**상황**: 실시간 채팅 애플리케이션의 WebSocket 연결이 자주 끊어집니다.

**원인 분석**: Nginx가 WebSocket을 지원하도록 설정되지 않았습니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name example.com;
    
    # WebSocket 업그레이드 지원
    location /socket.io {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 타임아웃 설정
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
    
    # 일반 HTTP 요청
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 헬스 체크 및 장애 조치

### 문제 시나리오: 백엔드 서버 다운 시 서비스 중단
**상황**: 단일 백엔드 서버가 다운되면 전체 서비스가 중단됩니다.

**해결 과정**:

```nginx
http {
    # 백엔드 서버 그룹 정의
    upstream backend {
        server 127.0.0.1:3000 max_fails=3 fail_timeout=30s;
        server 127.0.0.1:3001 max_fails=3 fail_timeout=30s backup;
        
        # 세션 유지 (필요시)
        # ip_hash;
        
        # 헬스 체크 (상용 버전에서만 지원)
        # health_check;
    }
    
    server {
        listen 80;
        server_name example.com;
        
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 실패 시 다음 서버로 전환
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
            proxy_next_upstream_tries 3;
            proxy_next_upstream_timeout 30s;
        }
    }
}
```

### 문제 시나리오: 커스텀 헬스 체크 구현
**상황**: 오픈소스 버전에서 헬스 체크 기능이 필요합니다.

**해결 과**:

```nginx
http {
    # 백엔드 서버 그룹
    upstream backend {
        server 127.0.0.1:3000;
        server 127.0.0.1:3001;
    }
    
    # 헬스 체크 서버
    server {
        listen 127.0.0.1:8080;
        
        location /health {
            proxy_pass http://backend/health;
            proxy_pass_request_body off;
            proxy_set_header Content-Length "";
            proxy_set_header X-Original-URI $request_uri;
            
            # 성공 응답 캐싱
            proxy_cache health_cache;
            proxy_cache_valid 200 10s;
            proxy_cache_key $request_uri;
        }
    }
    
    # 메인 서버
    server {
        listen 80;
        server_name example.com;
        
        # 헬스 체크 결과에 따른 동적 설정
        location / {
            # 헬스 체크 결과 확인
            resolver 127.0.0.1;
            set $backend_server "";
            
            # Lua 스크립트로 동적 서버 선택 (Lua 모듈 필요)
            # content_by_lua_block {
            #     local res = ngx.location.capture("/health")
            #     if res.status == 200 then
            #         ngx.var.backend_server = "http://127.0.0.1:3000"
            #     else
            #         ngx.var.backend_server = "http://127.0.0.1:3001"
            #     end
            # }
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## 프록시 캐싱

### 문제 시나리오: 백엔드 API 응답 속도 저하
**상황**: 동일한 API 요청이 반복되어 백엔드 서버에 부하가 발생합니다.

**해결 과정**:

```nginx
http {
    # 프록시 캐시 경로 설정
    proxy_cache_path /var/nginx/cache levels=1:2 keys_zone=api_cache:10m 
                     max_size=1g inactive=60m use_temp_path=off;
    
    server {
        listen 80;
        server_name api.example.com;
        
        # API 엔드포인트 캐싱
        location /api/data {
            proxy_cache api_cache;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_valid 200 5m;  # 성공 응답은 5분 캐싱
            proxy_cache_valid 404 1m;  # 404는 1분 캐싱
            
            # 캐시 관련 헤더
            add_header X-Proxy-Cache $upstream_cache_status;
            add_header X-Cache-Status $upstream_cache_status;
            
            # 캐시 우회 조건
            proxy_cache_bypass $http_pragma $http_authorization;
            proxy_no_cache $http_pragma $http_authorization;
            
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 캐시되지 않는 API
        location /api/secure {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## 보안 강화

### 문제 시나리오: 프록시를 통한 공격 노출
**상황**: Nginx 프록시를 통해 백엔드 서버가 직접적으로 공격에 노출됩니다.

**해결 과정**:

```nginx
server {
    listen 80;
    server_name example.com;
    
    # 요청 크기 제한
    client_max_body_size 10M;
    
    # 요청 속도 제한
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        # 보안 헤더 추가
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 민감 헤더 제거
        proxy_hide_header X-Powered-By;
        proxy_hide_header Server;
        proxy_hide_header X-AspNet-Version;
        proxy_hide_header X-AspNetMvc-Version;
        
        # 프록시 리다이렉트 처리
        proxy_redirect off;
        proxy_intercept_errors on;
        
        # 에러 페이지 처리
        error_page 502 503 504 /50x.html;
        
        proxy_pass http://127.0.0.1:3000;
    }
    
    location = /50x.html {
        root /var/www/error;
    }
}
```

## 모범 사례 요약

1. **프록시 헤더**: 항상 적절한 헤더 설정으로 클라이언트 정보 전달
2. **타임아웃**: 백엔드 특성에 맞는 타임아웃 설정
3. **버퍼링**: 적절한 버퍼 크기로 성능 최적화
4. **헬스 체크**: 서버 상태 모니터링과 자동 장애 조치
5. **캐싱**: 반복 요청에 대한 프록시 캐싱으로 백엔드 부하 감소
6. **보안**: 불필요한 정보 노출 방지와 요청 제한

## 다음 단계

이제 리버스 프록시 설정을 마쳤습니다. 다음 장에서는 로드 밸런싱에 대해 알아보겠습니다.