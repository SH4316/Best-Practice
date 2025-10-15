# Nginx 설정 예제

이 디렉토리에는 Nginx Configuration Best Practices 강의 자료에서 다룬 다양한 설정 예제가 포함되어 있습니다. 각 예제는 실제 시나리오에 맞는 문제 해결 과정을 보여줍니다.

## 디렉토리 구조

```
examples/
├── README.md                    # 이 파일
├── basic/                       # 기본 Nginx 설정
│   └── nginx.conf              # 기본 설정 예제
├── web-server/                 # 웹 서버 설정
│   └── static-site.conf       # 정적 웹사이트 서빙
├── reverse-proxy/              # 리버스 프록시 설정
│   └── basic-proxy.conf        # 기본 프록시 설정
├── load-balancing/             # 로드 밸런싱 설정
│   └── algorithms.conf        # 다양한 로드 밸런싱 알고리즘
├── ssl-tls/                    # SSL/TLS 설정
│   └── ssl-config.conf        # SSL/TLS 보안 설정
├── caching/                    # 캐싱 전략
│   └── caching-strategies.conf # 다양한 캐싱 전략
├── performance/                # 성능 최적화
│   └── performance-tuning.conf # 성능 튜닝 설정
├── security/                   # 보안 강화
│   └── security-hardening.conf # 보안 설정
├── microservices/              # 마이크로서비스 및 API 게이트웨이
│   └── api-gateway.conf        # API 게이트웨이 설정
├── containerization/           # 컨테이너화된 배포
│   └── docker-k8s.conf         # Docker 및 Kubernetes 설정
└── troubleshooting/            # 문제 해결 및 디버깅
    └── debugging-tools.conf    # 디버깅 도구 설정
```

## 사용 방법

### 1. 로컬에서 테스트

Docker를 사용하여 로컬에서 예제를 테스트할 수 있습니다:

```bash
# Docker를 사용한 Nginx 실행
docker run -d -p 80:80 -v $(pwd)/basic:/etc/nginx/conf.d nginx:latest

# 설정 테스트
docker exec <container_id> nginx -t

# 설정 재로드
docker exec <container_id> nginx -s reload
```

### 2. 설정 파일 적용

예제 설정 파일을 시스템에 적용하려면:

```bash
# 설정 파일 복사
sudo cp examples/basic/nginx.conf /etc/nginx/nginx.conf

# 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

### 3. 특정 시나리오 테스트

각 디렉토리에는 특정 시나리오에 맞는 설정이 포함되어 있습니다. 예를 들어, SSL/TLS 설정을 테스트하려면:

```bash
# SSL 설정 디렉토리로 이동
cd examples/ssl-tls

# SSL 인증서 생성 (테스트용)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/example.com.key \
  -out ssl/example.com.crt

# Docker를 사용한 HTTPS 서버 실행
docker run -d -p 443:443 -v $(pwd)/ssl-config.conf:/etc/nginx/conf.d/default.conf \
  -v $(pwd)/ssl:/etc/nginx/ssl nginx:latest
```

## 예제별 설명

### 기본 설정 (basic/)
- **nginx.conf**: Nginx의 기본 설정 구조와 핵심 개념
- 워커 프로세스, 이벤트 블록, HTTP 블록 등 기본 설정

### 웹 서버 설정 (web-server/)
- **static-site.conf**: 정적 콘텐츠 서빙, 가상 호스트, 로깅
- 다중 도메인 호스팅, 정적 파일 최적화, 보안 설정

### 리버스 프록시 (reverse-proxy/)
- **basic-proxy.conf**: 기본 프록시 설정, 경로 기반 라우팅
- 헤더 처리, WebSocket 지원, 헬스 체크

### 로드 밸런싱 (load-balancing/)
- **algorithms.conf**: 다양한 로드 밸런싱 알고리즘
- 라운드 로빈, 최소 연결, IP 해시, 해시 기반 분산

### SSL/TLS (ssl-tls/)
- **ssl-config.conf**: SSL/TLS 보안 설정
- HTTP에서 HTTPS 리디렉션, Let's Encrypt, mTLS

### 캐싱 전략 (caching/)
- **caching-strategies.conf**: 다양한 캐싱 전략
- 브라우저 캐싱, 프록시 캐싱, FastCGI 캐싱

### 성능 최적화 (performance/)
- **performance-tuning.conf**: 성능 최적화 설정
- 워커 프로세스 튜닝, 커넥션 최적화, HTTP/2 설정

### 보안 강화 (security/)
- **security-hardening.conf**: 보안 강화 설정
- 보안 헤더, 접근 제어, 속도 제한, DDoS 방어

### 마이크로서비스 (microservices/)
- **api-gateway.conf**: API 게이트웨이 설정
- 서비스 디스커버리, 버전 관리, 인증, 속도 제한

### 컨테이너화 (containerization/)
- **docker-k8s.conf**: Docker 및 Kubernetes 설정
- Dockerfile, Docker Compose, Kubernetes 배포

### 문제 해결 (troubleshooting/)
- **debugging-tools.conf**: 디버깅 도구 설정
- 로그 분석, 성능 모니터링, 상태 확인

## 주의사항

1. **테스트 환경**: 모든 예제는 먼저 테스트 환경에서 검증해야 합니다.
2. **보안**: 예제 중 일부는 보안을 위해 추가 설정이 필요할 수 있습니다.
3. **버전 호환성**: Nginx 버전에 따라 일부 지시어가 다를 수 있습니다.
4. **권한**: 일부 설정은 적절한 시스템 권한이 필요합니다.

## 추가 리소스

- [Nginx 공식 문서](https://nginx.org/en/docs/)
- [Nginx 설정 가이드](https://www.nginx.com/resources/wiki/start/topics/examples/full/)
- [Nginx 모범 사례](https://www.nginx.com/blog/nginx-best-practices/)

## 기여

이 예제들을 개선하거나 새로운 예제를 추가하고 싶으시면 Pull Request를 제출해 주세요.