# 마이크로서비스 및 API 게이트웨이

마이크로서비스 아키텍처에서 Nginx는 API 게이트웨이로서 중요한 역할을 합니다. 이 장에서는 Nginx를 사용한 API 게이트웨이 구축, 서비스 디스커버리, 라우팅, 그리고 마이크로서비스 환경에서의 모범 사례를 다룹니다.

## API 게이트웨이 기본 설정

### 문제 시나리오: 여러 마이크로서비스로의 복잡한 라우팅
**상황**: 여러 마이크로서비스가 있으며, 클라이언트가 각 서비스를 개별적으로 호출해야 합니다.

**원인 분석**: 단일 진입점이 없어 클라이언트 구현이 복잡하고 서비스 관리가 어렵습니다.

**해결 과정**:

#### 1. 기본 API 게이트웨이 설정

```nginx
http {
    # 마이크로서비스 업스트림 정의
    upstream user_service {
        least_conn;
        server 192.168.1.101:8001;
        server 192.168.1.102:8001;
    }
    
    upstream product_service {
        least_conn;
        server 192.168.1.201:8002;
        server 192.168.1.202:8002;
    }
    
    upstream order_service {
        least_conn;
        server 192.168.1.301:8003;
        server 192.168.1.302:8003;
    }
    
    server {
        listen 80;
        server_name api.example.com;
        
        # 사용자 서비스
        location /api/users/ {
            proxy_pass http://user_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS 헤더
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
            add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        }
        
        # 제품 서비스
        location /api/products/ {
            proxy_pass http://product_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS 헤더
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
            add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        }
        
        # 주문 서비스
        location /api/orders/ {
            proxy_pass http://order_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS 헤더
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
            add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        }
    }
}
```

## 서비스 디스커버리

### 문제 시나리오: 동적 서비스 등록 및 발견
**상황**: 마이크로서비스가 동적으로 추가되거나 제거될 때마다 Nginx 설정을 수동으로 변경해야 합니다.

**해결 과정**:

#### 1. DNS 기반 서비스 디스커버리

```nginx
http {
    # DNS 리졸버 설정
    resolver 127.0.0.1 valid=30s;
    
    server {
        listen 80;
        server_name api.example.com;
        
        # 동적 서비스 디스커버리
        location /api/users/ {
            set $user_service "user-service.default.svc.cluster.local";
            proxy_pass http://$user_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/products/ {
            set $product_service "product-service.default.svc.cluster.local";
            proxy_pass http://$product_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/orders/ {
            set $order_service "order-service.default.svc.cluster.local";
            proxy_pass http://$order_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### 2. Consul을 이용한 서비스 디스커버리

```nginx
# 이 설정은 ngx_http_consul_backend_module 또는 Lua 스크립트가 필요함

http {
    # Consul 서버 설정
    consul 127.0.0.1:8500;
    
    server {
        listen 80;
        server_name api.example.com;
        
        # Consul을 통한 동적 서비스 디스커버리
        location /api/users/ {
            consul_service user_service;
            proxy_pass http://consul_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/products/ {
            consul_service product_service;
            proxy_pass http://consul_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/orders/ {
            consul_service order_service;
            proxy_pass http://consul_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## API 버전 관리

### 문제 시나리오: 여러 버전의 API 동시 지원
**상황**: 기존 클라이언트와 새로운 클라이언트가 다른 버전의 API를 사용해야 합니다.

**해결 과정**:

```nginx
http {
    # v1 서비스
    upstream user_service_v1 {
        server 192.168.1.101:8001;
        server 192.168.1.102:8001;
    }
    
    upstream product_service_v1 {
        server 192.168.1.201:8002;
        server 192.168.1.202:8002;
    }
    
    # v2 서비스
    upstream user_service_v2 {
        server 192.168.1.111:8011;
        server 192.168.1.112:8011;
    }
    
    upstream product_service_v2 {
        server 192.168.1.211:8012;
        server 192.168.1.212:8012;
    }
    
    server {
        listen 80;
        server_name api.example.com;
        
        # v1 API
        location /api/v1/users/ {
            proxy_pass http://user_service_v1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-API-Version "v1";
        }
        
        location /api/v1/products/ {
            proxy_pass http://product_service_v1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-API-Version "v1";
        }
        
        # v2 API
        location /api/v2/users/ {
            proxy_pass http://user_service_v2;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-API-Version "v2";
        }
        
        location /api/v2/products/ {
            proxy_pass http://product_service_v2;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-API-Version "v2";
        }
        
        # 기본 버전 (v2)
        location /api/users/ {
            return 302 /api/v2$request_uri;
        }
        
        location /api/products/ {
            return 302 /api/v2$request_uri;
        }
    }
}
```

## 인증 및 인가

### 문제 시나리오: 중앙화된 인증 및 인가 시스템
**상황**: 각 마이크로서비스가 개별적으로 인증을 처리하여 중복이 발생합니다.

**해결 과정**:

#### 1. JWT 기반 인증

```nginx
http {
    # 인증 서비스
    upstream auth_service {
        server 192.168.1.50:9000;
    }
    
    server {
        listen 80;
        server_name api.example.com;
        
        # 로그인 엔드포인트 (인증 필요 없음)
        location /api/auth/login {
            proxy_pass http://auth_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 토큰 갱신 엔드포인트
        location /api/auth/refresh {
            proxy_pass http://auth_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 인증이 필요한 API
        location /api/ {
            # JWT 검증 (Lua 스크립트 필요)
            access_by_lua_block {
                local jwt = require "resty.jwt"
                local jwt_secret = "your-secret-key"
                
                local auth_header = ngx.var.http_authorization
                if not auth_header then
                    ngx.status = 401
                    ngx.say("Missing authorization header")
                    ngx.exit(401)
                end
                
                local _, _, token = string.find(auth_header, "Bearer%s+(.+)")
                if not token then
                    ngx.status = 401
                    ngx.say("Invalid authorization header")
                    ngx.exit(401)
                end
                
                local jwt_obj = jwt:verify_jwt(jwt_secret, token)
                if not jwt_obj.valid then
                    ngx.status = 401
                    ngx.say("Invalid token: " .. jwt_obj.reason)
                    ngx.exit(401)
                end
                
                -- 사용자 정보를 헤더에 추가
                ngx.req.set_header("X-User-ID", jwt_obj.payload.sub)
                ngx.req.set_header("X-User-Role", jwt_obj.payload.role)
            }
            
            # 사용자 서비스
            location /api/users/ {
                proxy_pass http://user_service;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
            }
            
            # 제품 서비스
            location /api/products/ {
                proxy_pass http://product_service;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
            }
            
            # 주문 서비스
            location /api/orders/ {
                # 관리자만 접근 가능
                access_by_lua_block {
                    local user_role = ngx.req.get_headers()["X-User-Role"]
                    if user_role ~= "admin" then
                        ngx.status = 403
                        ngx.say("Access denied")
                        ngx.exit(403)
                    end
                }
                
                proxy_pass http://order_service;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
            }
        }
    }
}
```

## 속도 제한 및 할당량

### 문제 시나리오: API 사용량 제한 및 공정한 자원 분배
**상황**: 특정 사용자가 API를 과도하게 사용하여 다른 사용자에게 영향을 줍니다.

**해결 과정**:

```nginx
http {
    # 속도 제한 영역 정의
    limit_req_zone $binary_remote_addr zone=global:10m rate=100r/m;
    limit_req_zone $http_x_user_id zone=user:10m rate=1000r/h;
    limit_req_zone $http_x_api_key zone=api_key:10m rate=10000r/h;
    
    # API 키 할당량 맵
    geo $api_key_quota {
        default 100;
        1234567890abcdef 1000;  # 프리미엄 고객
        fedcba0987654321 500;    # 일반 고객
    }
    
    server {
        listen 80;
        server_name api.example.com;
        
        # 전역 속도 제한
        limit_req zone=global burst=20 nodelay;
        
        # API 키 기반 속도 제한
        location /api/ {
            # API 키 확인
            if ($http_x_api_key = "") {
                return 401;
            }
            
            # 동적 속도 제한
            limit_req zone=api_key burst=$api_key_quota nodelay;
            
            # 사용자 ID 기반 속도 제한
            limit_req zone=user burst=100 nodelay;
            
            # 사용자 서비스
            location /api/users/ {
                proxy_pass http://user_service;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header X-API-Key $http_x_api_key;
            }
            
            # 제품 서비스
            location /api/products/ {
                proxy_pass http://product_service;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header X-API-Key $http_x_api_key;
            }
            
            # 주문 서비스
            location /api/orders/ {
                proxy_pass http://order_service;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header X-API-Key $http_x_api_key;
            }
        }
    }
}
```

## 요청/응답 변환

### 문제 시나리오: 레거시 시스템과의 호환성
**상황**: 새로운 마이크로서비스 아키텍처와 레거시 시스템 간의 데이터 형식이 다릅니다.

**해결 과정**:

```nginx
http {
    # 사용자 서비스
    upstream user_service {
        server 192.168.1.101:8001;
    }
    
    server {
        listen 80;
        server_name api.example.com;
        
        # 레거시 형식을 새 형식으로 변환
        location /api/legacy/users/ {
            # 요청 본문 변환
            rewrite ^/api/legacy/(.*)$ /api/$1 break;
            
            # 레거시 응답 형식으로 변환
            body_filter_by_lua_block {
                local cjson = require "cjson"
                local data = cjson.decode(ngx.arg[1])
                
                -- 새 형식을 레거시 형식으로 변환
                local legacy_data = {
                    status = "success",
                    result = {
                        userId = data.id,
                        userName = data.name,
                        userEmail = data.email
                    }
                }
                
                ngx.arg[1] = cjson.encode(legacy_data)
            }
            
            proxy_pass http://user_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 새 형식의 API
        location /api/users/ {
            proxy_pass http://user_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## 모니터링 및 로깅

### 문제 시나리오: 마이크로서비스 아키텍처의 복잡한 모니터링
**상황**: 여러 서비스 간의 호출 추적과 성능 모니터링이 어렵습니다.

**해결 과정**:

```nginx
http {
    # 분산 추적을 위한 헤더
    map $http_x_request_id $request_id {
        default $http_x_request_id;
        "" $request_id;
    }
    
    # 로그 형식 정의
    log_format microservices '$remote_addr - $remote_user [$time_local] "$request" '
                            '$status $body_bytes_sent "$http_referer" '
                            '"$http_user_agent" "$http_x_forwarded_for" '
                            'rt=$request_time uct="$upstream_connect_time" '
                            'uht="$upstream_header_time" urt="$upstream_response_time" '
                            'req_id=$request_id service=$service_name';
    
    server {
        listen 80;
        server_name api.example.com;
        
        # 요청 ID 생성
        set $request_id $request_id;
        if ($request_id = "") {
            set $request_id $request_id;
        }
        
        # 요청 ID 헤더 추가
        add_header X-Request-ID $request_id;
        
        # 사용자 서비스
        location /api/users/ {
            set $service_name "user_service";
            access_log /var/log/nginx/user_service.log microservices;
            
            proxy_pass http://user_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;
            
            # 응답 시간 헤더
            add_header X-Response-Time $upstream_response_time;
        }
        
        # 제품 서비스
        location /api/products/ {
            set $service_name "product_service";
            access_log /var/log/nginx/product_service.log microservices;
            
            proxy_pass http://product_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;
            
            # 응답 시간 헤더
            add_header X-Response-Time $upstream_response_time;
        }
        
        # 주문 서비스
        location /api/orders/ {
            set $service_name "order_service";
            access_log /var/log/nginx/order_service.log microservices;
            
            proxy_pass http://order_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;
            
            # 응답 시간 헤더
            add_header X-Response-Time $upstream_response_time;
        }
        
        # API 상태 엔드포인트
        location /api/status {
            default_type application/json;
            return 200 '{"status": "healthy", "timestamp": "$time_iso8601", "request_id": "$request_id"}';
        }
    }
}
```

## 모범 사례 요약

1. **단일 진입점**: API 게이트웨이를 통한 단일 진입점 제공
2. **서비스 디스커버리**: 동적 서비스 등록 및 발견 메커니즘
3. **버전 관리**: 여러 버전의 API 동시 지원
4. **중앙화된 인증**: 중앙에서 인증 및 인가 처리
5. **속도 제한**: 공정한 자원 사용을 위한 속도 제한 및 할당량
6. **요청 변환**: 레거시 시스템과의 호환성을 위한 변환
7. **모니터링**: 분산 추적과 성능 모니터링

## 다음 단계

이제 마이크로서비스 및 API 게이트웨이를 마쳤습니다. 다음 장에서는 컨테이너화된 배포에 대해 알아보겠습니다.