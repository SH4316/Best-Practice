# 문제 해결 및 디버깅

Nginx 운영 중 발생하는 다양한 문제를 체계적으로 해결하는 방법을 다룹니다. 이 장에서는 일반적인 문제 시나리오와 해결 과정, 그리고 효과적인 디버깅 기법을 설명합니다.

## 일반적인 시작 문제

### 문제 시나리오: Nginx가 시작되지 않음
**상황**: Nginx를 시작하려고 하면 오류 메시지와 함께 실패합니다.

**원인 분석**: 설정 파일 오류, 포트 충돌, 권한 문제 등 다양한 원인이 있을 수 있습니다.

**해결 과정**:

#### 1. 설정 파일 검증

```bash
# Nginx 설정 구문 검증
sudo nginx -t

# 상세한 오류 정보 확인
sudo nginx -T

# 특정 설정 파일 검증
sudo nginx -t -c /etc/nginx/nginx.conf
```

#### 2. 로그 확인

```bash
# 에러 로그 확인
sudo tail -f /var/log/nginx/error.log

# 시스템 로그 확인
sudo journalctl -u nginx -f
```

#### 3. 포트 충돌 확인

```bash
# 사용 중인 포트 확인
sudo netstat -tulpn | grep :80
sudo ss -tulpn | grep :80

# 특정 프로세스 확인
sudo lsof -i :80
```

#### 4. 권한 문제 확인

```bash
# Nginx 실행 사용자 확인
ps aux | grep nginx

# 로그 디렉토리 권한 확인
ls -la /var/log/nginx/

# 웹 콘텐츠 디렉토리 권한 확인
ls -la /var/www/
```

## 성능 문제

### 문제 시나리오: 응답 속도가 갑자기 느려짐
**상황**: 평소에는 정상적으로 작동하다가 갑자기 응답 속도가 느려집니다.

**원인 분석**: 리소스 부족, 백엔드 서버 문제, 네트워크 문제 등이 원인일 수 있습니다.

**해결 과정**:

#### 1. 시스템 리소스 확인

```bash
# CPU 사용률 확인
top
htop

# 메모리 사용량 확인
free -h
vmstat 1 5

# 디스크 I/O 확인
iostat -x 1
iotop
```

#### 2. Nginx 상태 확인

```nginx
# status 모듈 설정
server {
    listen 80;
    server_name localhost;
    
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
}
```

```bash
# Nginx 상태 확인
curl http://localhost/nginx_status

# 결과 예시
Active connections: 291
server accepts handled requests
 16630948 16630948 31070465
Reading: 6 Writing: 179 Waiting: 106
```

#### 3. 응답 시간 분석

```bash
# curl로 응답 시간 측정
curl -o /dev/null -s -w "%{time_total}\n" http://example.com

# 상세 시간 측정
curl -o /dev/null -s -w "DNS: %{time_namelookup}s\nConnect: %{time_connect}s\nTTFB: %{time_starttransfer}s\nTotal: %{time_total}s\n" http://example.com
```

#### 4. 로그 분석

```bash
# 응답 시간이 긴 요청 찾기
awk '$NF > 1.0 {print $0}' /var/log/nginx/access.log

# 상위 10개 느린 요청
awk '{print $NF, $7}' /var/log/nginx/access.log | sort -nr | head -10

# 5xx 에러 분석
awk '$9 >= 500 {print $0}' /var/log/nginx/access.log
```

## 로드 밸런싱 문제

### 문제 시나리오: 특정 백엔드 서버로만 트래픽이 전달됨
**상황**: 여러 백엔드 서버가 있는데 특정 서버로만 트래픽이 전달됩니다.

**원인 분석**: 헬스 체크 실패, 서버 다운, 로드 밸런싱 알고리즘 문제 등이 원인일 수 있습니다.

**해결 과정**:

#### 1. 업스트림 상태 확인

```nginx
# 업스트림 상태 모니터링 추가
upstream backend {
    server 192.168.1.101:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.102:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.103:8000 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 80;
    server_name example.com;
    
    location /upstream_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 업스트림 응답 헤더 추가
        add_header X-Upstream-Addr $upstream_addr;
    }
}
```

#### 2. 백엔드 서버 직접 테스트

```bash
# 각 백엔드 서버 직접 테스트
curl -I http://192.168.1.101:8000/health
curl -I http://192.168.1.102:8000/health
curl -I http://192.168.1.103:8000/health

# 포트 연결 테스트
telnet 192.168.1.101 8000
nc -zv 192.168.1.101 8000
```

#### 3. 업스트림 로그 분석

```bash
# 업스트림 응답 시간 분석
grep "upstream_response_time" /var/log/nginx/access.log | awk '{print $NF, $7}' | sort -nr

# 업스트림 실패 분석
grep "upstream" /var/log/nginx/error.log
```

## SSL/TLS 문제

### 문제 시나리오: SSL 인증서 오류
**상황**: HTTPS 접속 시 SSL 인증서 관련 오류가 발생합니다.

**원인 분석**: 인증서 만료, 잘못된 인증서 경로, 인증서 체인 문제 등이 원인일 수 있습니다.

**해결 과정**:

#### 1. 인증서 상태 확인

```bash
# 인증서 정보 확인
openssl x509 -in /etc/nginx/ssl/example.com.crt -text -noout

# 인증서 만료일 확인
openssl x509 -in /etc/nginx/ssl/example.com.crt -noout -dates

# 인증서 유효성 확인
openssl x509 -in /etc/nginx/ssl/example.com.crt -noout -subject -issuer
```

#### 2. SSL 연결 테스트

```bash
# SSL 연결 테스트
openssl s_client -connect example.com:443

# SSL 인증서 체인 확인
openssl s_client -connect example.com:443 -showcerts

# SSL 구성 테스트
nmap --script ssl-enum-ciphers -p 443 example.com
```

#### 3. SSL 디버깅 설정

```nginx
# SSL 디버깅 로그
error_log /var/log/nginx/error.log debug;

server {
    listen 443 ssl http2;
    server_name example.com;
    
    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;
    
    # SSL 디버깅
    ssl_debug_log /var/log/nginx/ssl_debug.log;
    
    location / {
        root /var/www/html;
    }
}
```

## 캐싱 문제

### 문제 시나리오: 캐시된 콘텐츠가 업데이트되지 않음
**상황**: 웹사이트 콘텐츠를 업데이트했는데도 이전 버전이 계속 표시됩니다.

**원인 분석**: 브라우저 캐시, 프록시 캐시, Nginx 캐시 등 다양한 캐시가 원인일 수 있습니다.

**해결 과정**:

#### 1. 캐시 상태 확인

```nginx
# 캐시 상태 헤더 추가
location / {
    add_header X-Cache-Status $upstream_cache_status;
    add_header X-Cache-Key "$scheme$request_method$host$request_uri";
    
    proxy_pass http://backend;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

#### 2. 캐시 제어

```bash
# 캐시 디렉토리 확인
ls -la /var/nginx/cache/

# 특정 캐시 파일 삭제
find /var/nginx/cache/ -name "*.tmp" -delete
find /var/nginx/cache/ -name "*" -type f -delete

# 캐시 통계 확인
du -sh /var/nginx/cache/
```

#### 3. 브라우저 캐시 무시

```bash
# curl로 캐시 무시하고 요청
curl -H "Cache-Control: no-cache" -H "Pragma: no-cache" http://example.com

# wget으로 캐시 무시하고 요청
wget --no-cache --no-cookies http://example.com
```

## 메모리 누수 문제

### 문제 시나리오: 메모리 사용량이 계속 증가
**상황**: 장시간 운영 후 메모리 사용량이 계속 증가하여 시스템이 불안정해집니다.

**원인 분석**: 메모리 누수, 불필요한 버퍼링, 잘못된 설정 등이 원인일 수 있습니다.

**해결 과정**:

#### 1. 메모리 사용량 모니터링

```bash
# Nginx 프로세스 메모리 사용량 확인
ps aux | grep nginx

# 메모리 사용량 추적
watch -n 1 'ps aux | grep nginx | grep -v grep'

# 상세 메모리 정보 확인
cat /proc/$(pidof nginx)/status
```

#### 2. 메모리 최적화 설정

```nginx
# 메모리 최적화 설정
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 1024;
}

http {
    # 버퍼 크기 최적화
    client_body_buffer_size 128k;
    client_max_body_size 10m;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    # 프록시 버퍼 최적화
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 4k;
    proxy_busy_buffers_size 8k;
    
    # 임시 파일 크기 제한
    client_body_temp_path /var/nginx/client_temp;
    proxy_temp_path /var/nginx/proxy_temp;
    
    # 연결 유지 시간 최적화
    keepalive_timeout 65;
    keepalive_requests 1000;
}
```

## 디버깅 도구

### 문제 시나리오: 복잡한 설정 문제 디버깅
**상황**: 복잡한 Nginx 설정에서 문제가 발생했지만 원인을 찾기 어렵습니다.

**해결 과정**:

#### 1. 설정 분석 도구

```bash
# 전체 설정 출력
sudo nginx -T

# 특정 지시어 검색
sudo nginx -T | grep "proxy_pass"

# 설정 포함 관계 확인
sudo nginx -T | grep -E "include|map|geo"
```

#### 2. 요청 추적

```nginx
# 디버깅 로그 설정
error_log /var/log/nginx/debug.log debug;

# 요청 ID 추가
add_header X-Request-ID $request_id;

# 요청 시간 기록
log_format detailed '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for" '
                   'rt=$request_time uct="$upstream_connect_time" '
                   'uht="$upstream_header_time" urt="$upstream_response_time" '
                   'req_id=$request_id';
```

#### 3. Lua 스크립트 디버깅

```nginx
# Lua 스크립트 디버깅
location /debug {
    default_type text/plain;
    content_by_lua_block {
        ngx.say("Request URI: ", ngx.var.request_uri)
        ngx.say("Request Method: ", ngx.var.request_method)
        ngx.say("Remote Address: ", ngx.var.remote_addr)
        ngx.say("Headers:")
        for k, v in pairs(ngx.req.get_headers()) do
            ngx.say("  ", k, ": ", v)
        end
    }
}
```

## 문제 해결 체크리스트

### 시작 문제
- [ ] 설정 파일 구문 검증 (`nginx -t`)
- [ ] 에러 로그 확인
- [ ] 포트 충돌 확인
- [ ] 권한 문제 확인
- [ ] 방화벽 설정 확인

### 성능 문제
- [ ] 시스템 리소스 사용량 확인
- [ ] Nginx 상태 모니터링
- [ ] 응답 시간 분석
- [ ] 로그 분석
- [ ] 워커 프로세스 설정 확인

### 로드 밸런싱 문제
- [ ] 업스트림 상태 확인
- [ ] 백엔드 서버 직접 테스트
- [ ] 헬스 체크 설정 확인
- [ ] 로드 밸런싱 알고리즘 확인
- [ ] 업스트림 로그 분석

### SSL/TLS 문제
- [ ] 인증서 정보 확인
- [ ] 인증서 만료일 확인
- [ ] SSL 연결 테스트
- [ ] SSL 설정 확인
- [ ] 인증서 체인 확인

### 캐싱 문제
- [ ] 캐시 상태 헤더 확인
- [ ] 캐시 디렉토리 확인
- [ ] 캐시 설정 확인
- [ ] 브라우저 캐시 무시 테스트
- [ ] 캐시 무효화 확인

## 모범 사례 요약

1. **체계적인 접근**: 로그 확인 → 설정 검증 → 테스트의 순서로 문제 해결
2. **모니터링**: 지속적인 상태 모니터링으로 문제 조기 발견
3. **로깅**: 상세한 로그 설정으로 문제 원인 파악
4. **테스트**: 다양한 도구를 사용한 체계적인 테스트
5. **문서화**: 문제와 해결 과정 상세히 기록

## 다음 단계

이제 문제 해결 및 디버깅을 마쳤습니다. 다음 장에서는 전체 모범 사례 체크리스트를 제공합니다.