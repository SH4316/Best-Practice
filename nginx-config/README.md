# Nginx Configuration Best Practices

이 문서는 Nginx 설정에 대한 모범 사례를 다루는 종합적인 강의 자료입니다. 초보자부터 고급 사용자까지 다양한 수준의 사용자를 위해 구성되었으며, 실제 시나리오와 문제 해결 과정을 포함합니다.

## 목차

1. [소개 및 개요](docs/01-introduction.md)
2. [기본 Nginx 설정](docs/02-basic-configuration.md)
3. [웹 서버 설정](docs/03-web-server.md)
4. [리버스 프록시 설정](docs/04-reverse-proxy.md)
5. [로드 밸런싱](docs/05-load-balancing.md)
6. [SSL/TLS 보안 설정](docs/06-ssl-tls.md)
7. [캐싱 전략](docs/07-caching.md)
8. [성능 최적화](docs/08-performance.md)
9. [보안 강화](docs/09-security.md)
10. [마이크로서비스 및 API 게이트웨이](docs/10-microservices.md)
11. [컨테이너화된 배포](docs/11-containerization.md)
12. [문제 해결 및 디버깅](docs/12-troubleshooting.md)
13. [요약 및 체크리스트](docs/13-summary.md)

## 예제 구성 파일

각 모듈에 대한 실제 구성 파일은 [examples](examples/) 디렉토리에서 찾을 수 있습니다.

## 학습 경로

- **초보자**: 1-3장을 먼저 학습하십시오.
- **중급 사용자**: 4-7장을重点关注하십시오.
- **고급 사용자**: 8-12장을 참조하십시오.

## 실습 환경 설정

로컬에서 예제를 테스트하려면 Docker를 사용하는 것이 좋습니다:

```bash
docker run -d -p 80:80 -v $(pwd)/examples:/etc/nginx/conf.d nginx:latest