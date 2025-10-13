# Thymeleaf Best Practices 강의자료

이 프로젝트는 Thymeleaf 사용 시 Best Practice와 Bad Practice를 비교하여 올바른 사용법을 학습하기 위한 예제 코드와 강의 자료를 포함하고 있습니다.

## 목차

### 📚 강의 자료 (Chapter-based Documentation)

1. [Thymeleaf 소개](docs/01-introduction.md) - Thymeleaf의 기본 개념과 특징
2. [기본 설정](docs/02-basic-setup.md) - 프로젝트 설정과 환경 구성
3. [변수 표현식](docs/03-variable-expressions.md) - 다양한 표현식과 유틸리티 객체
4. [반복문과 반복](docs/04-loops-and-iterations.md) - 효율적인 반복문 작성법
5. [조건문](docs/05-conditional-statements.md) - 조건부 렌더링과 논리 처리
6. [링크와 URL 처리](docs/06-links-and-urls.md) - 동적 URL 생성과 링크 처리
7. [폼 처리](docs/07-form-handling.md) - 폼 바인딩과 유효성 검사
8. [프래그먼트와 레이아웃](docs/08-fragments-and-layouts.md) - 코드 재사용과 레이아웃 상속
9. [국제화](docs/09-internationalization.md) - 다국어 지원과 메시지 처리
10. [보안](docs/10-security.md) - XSS 방지와 권한 기반 접근 제어
11. [성능 최적화](docs/11-performance-optimization.md) - 템플릿 성능 향상 기법
12. [실전 예제와 팁](docs/12-practical-examples.md) - 실제 프로젝트 예제와 유용한 팁

### 🚀 고급 주제 (Advanced Topics)

13. [Thymeleaf 3.x 새로운 기능](docs/13-thymeleaf-3x-new-features.md) - 최신 버전의 새로운 기능과 개선사항
14. [커스텀 다이얼렉트와 프로세서](docs/14-custom-dialects-and-processors.md) - 자체 다이얼렉트 만드는 방법
15. [Thymeleaf와 REST API 통합](docs/15-thymeleaf-rest-api-integration.md) - API 응답을 템플릿으로 렌더링하는 방법
16. [Thymeleaf 테스트 전략](docs/16-thymeleaf-testing-strategies.md) - 템플릿 테스트 방법과 도구

### 🚀 실습 예제 (Interactive Examples)

- [변수 표현식 예제](http://localhost:8080/variables) - Best Practice와 Bad Practice 비교
- [반복문 예제](http://localhost:8080/loops) - 상태 변수 활용과 빈 목록 처리
- [조건문 예제](http://localhost:8080/conditions) - switch-case 문과 삼항 연산자
- [링크 처리 예제](http://localhost:8080/links) - URL 표현식과 정적 리소스
- [폼 처리 예제](http://localhost:8080/forms) - th:object와 유효성 검사
- [보안 예제](http://localhost:8080/security) - XSS 방지와 권한 체크
- [성능 최적화 예제](http://localhost:8080/performance) - 불필요한 연산 방지와 페이징
- [Best vs Bad Practice 비교](http://localhost:8080/comparison) - 모든 주제의 비교 예제

## 시작하기

### 사전 요구사항

- JDK 17 이상
- Gradle 또는 Maven
- IDE (IntelliJ IDEA, VS Code 등)

### 설치 및 실행

1. 프로젝트 클론:
```bash
git clone <repository-url>
cd thymeleaf-best-practices
```

2. 애플리케이션 실행:
```bash
./gradlew bootRun
```

3. 브라우저에서 접속:
```
http://localhost:8080
```

## 프로젝트 구조

```
src/
├── main/
│   ├── java/
│   │   └── com/example/demo/
│   │       ├── DemoApplication.java
│   │       ├── controller/
│   │       │   └── ThymeleafController.java
│   │       └── model/
│   │           └── User.java
│   └── resources/
│       ├── application.properties
│       ├── messages.properties
│       ├── static/
│       │   ├── css/
│       │   ├── js/
│       │   └── images/
│       └── templates/
│           ├── index.html
│           ├── fragments/
│           │   ├── header.html
│           │   └── footer.html
│           └── examples/
│               ├── variables.html
│               ├── loops.html
│               ├── conditions.html
│               ├── links.html
│               ├── forms.html
│               ├── security.html
│               ├── performance.html
│               └── comparison.html
└── test/
    └── java/
        └── com/example/demo/
            └── DemoApplicationTests.java
```

## 학습 방법

1. **이론 학습**: docs/ 디렉토리의 챕터별 문서를 순서대로 읽으세요.
2. **실습 예제**: 각 주제의 예제 페이지를 방문하여 Best Practice와 Bad Practice를 비교해보세요.
3. **코드 분석**: 예제 코드를 직접 수정하고 결과를 확인하며 학습하세요.
4. **비교 예제**: comparison 페이지에서 모든 주제를 한눈에 비교해보세요.

## 핵심 개념

### Best Practice ✅

- 안전한 변수 접근: `${user?.name}`
- 유틸리티 객체 활용: `${#dates.format(date, 'yyyy-MM-dd')}`
- URL 표현식 사용: `@{/users/{id}(id=${user.id})}`
- 상태 변수 활용: `th:each="user, stat : ${users}"`
- 권한 체크: `${#authorization.expression('hasRole(''ADMIN'')')}`

### Bad Practice ❌

- null 체크 없이 변수 접근: `${user.address.city}`
- 하드코딩된 URL: `href="/users/123"`
- 상태 변수 미활용: `th:each="user : ${users}"`
- 권한 체크 없음: 민감한 기능 노출
- XSS 취약점: `th:utext="${user.comment}"`

## 참고 자료

- [Thymeleaf 공식 문서](https://www.thymeleaf.org/doc/tutorials/3.0/usingthymeleaf.html)
- [Spring Boot Thymeleaf 가이드](https://spring.io/guides/gs/serving-web-content/)
- [Thymeleaf + Spring Boot 통합](https://www.thymeleaf.org/doc/tutorials/3.0/thymeleafspring.html)

## 기여

이 프로젝트를 개선하려면 다음 단계를 따르세요:

1. 이 저장소를 포크합니다.
2. 기능 브랜치를 만듭니다 (`git checkout -b feature/AmazingFeature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`).
5. Pull Request를 엽니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 사용 가능합니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

질문이나 제안이 있으시면 [Issues](https://github.com/your-username/thymeleaf-best-practices/issues)를 통해 알려주세요.

---

⭐ 이 프로젝트가 유용하다면 스타를 눌러주세요!