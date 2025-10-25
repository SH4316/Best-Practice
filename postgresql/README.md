# PostgreSQL 베스트 프랙티스 강의자료

이 저장소는 PostgreSQL 데이터베이스를 효과적으로 사용하기 위한 베스트 프랙티스를 다루는 강의자료입니다. 본 자료는 데이터베이스 경험이 있지만 PostgreSQL은 처음 사용하는 개발자를 대상으로 합니다.

## 목차

1. [PostgreSQL 소개](./01-introduction.md)
2. [데이터베이스 설계 베스트 프랙티스](./02-database-design.md)
3. [데이터 타입과 스키마 설계](./03-data-types.md)
4. [인덱싱 전략과 베스트 프랙티스](./04-indexing.md)
5. [쿼리 최적화 기법](./05-query-optimization.md)
6. [트랜잭션 관리와 동시성 제어](./06-transactions.md)
7. [보안 베스트 프랙티스](./07-security.md)
8. [백업 및 복구 전략](./08-backup-recovery.md)
9. [성능 튜닝과 모니터링](./09-performance-tuning.md)
10. [요약 및 추가 자료](./10-summary.md)

## 예제 데이터베이스

강의 자료 전체에서 사용되는 예제 데이터베이스는 [examples](./examples/) 디렉토리에서 확인할 수 있습니다.

- [스키마 정의](./examples/schema.sql)
- [샘플 데이터](./examples/sample-data.sql)
- [설정 가이드](./examples/setup.md)

## 실습 문제

각 섹션의 끝에는 관련 주제에 대한 실습 문제가 포함되어 있습니다. 실습 문제와 해결책은 [exercises](./exercises/) 디렉토리에서 확인할 수 있습니다.

- [기초 실습 문제](./exercises/basic-exercises.md)
- [중급 실습 문제](./exercises/intermediate-exercises.md)
- [해결책](./exercises/solutions/)

## 시작하기

1. PostgreSQL 설치 (버전 13 이상 권장)
2. 예제 데이터베이스 설정:
   ```bash
   psql -U postgres -c "CREATE DATABASE lecture_db;"
   psql -U postgres -d lecture_db -f examples/schema.sql
   psql -U postgres -d lecture_db -f examples/sample-data.sql
   ```

## 학습 목표

본 강의자료를 통해 다음을 학습할 수 있습니다:

- PostgreSQL의 핵심 기능과 장점 이해
- 효율적인 데이터베이스 스키마 설계
- 적절한 인덱싱 전략 수립
- 쿼리 성능 최적화 기법
- 안전한 트랜잭션 관리
- 데이터베이스 보안 강화
- 효과적인 백업 및 복구 전략
- 성능 모니터링과 튜닝 방법

## 요구 사항

- PostgreSQL 13 이상
- 기본적인 SQL 지식
- 데이터베이스 기본 개념 이해

## 기여

본 강의자료는 계속 업데이트될 예정입니다. 오타, 오류, 개선 사항이 있으면 이슈를 열어주세요.

## 라이선스

본 강의자료는 MIT 라이선스 하에 제공됩니다.