# 2. 기본 설정

## 프로젝트 설정

### Spring Boot 프로젝트 생성

Spring Boot를 사용하여 Thymeleaf 프로젝트를 설정하는 것이 가장 쉽습니다. Spring Initializr에서 다음 의존성을 추가하세요:

```gradle
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-thymeleaf'
    implementation 'org.springframework.boot:spring-boot-starter-web'
    testImplementation 'org.springframework.boot:spring-boot-starter-test'
}
```

### Maven 설정

Maven을 사용하는 경우:

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-thymeleaf</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

## Thymeleaf 설정

### application.properties

```properties
# Thymeleaf 기본 설정
spring.thymeleaf.prefix=classpath:/templates/
spring.thymeleaf.suffix=.html
spring.thymeleaf.mode=HTML
spring.thymeleaf.encoding=UTF-8

# 개발 환경에서는 캐시 비활성화
spring.thymeleaf.cache=false

# 컨텐츠 타입
spring.thymeleaf.servlet.content-type=text/html

# 템플릿 체크
spring.thymeleaf.check-template=true
spring.thymeleaf.check-template-location=true
```

### application.yml

YAML 형식을 선호하는 경우:

```yaml
spring:
  thymeleaf:
    prefix: classpath:/templates/
    suffix: .html
    mode: HTML
    encoding: UTF-8
    cache: false
    servlet:
      content-type: text/html
    check-template: true
    check-template-location: true
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
│       ├── templates/
│       │   ├── index.html
│       │   ├── fragments/
│       │   │   ├── header.html
│       │   │   └── footer.html
│       │   └── examples/
│       │       ├── variables.html
│       │       ├── loops.html
│       │       └── ...
│       └── static/
│           ├── css/
│           ├── js/
│           └── images/
└── test/
    └── java/
        └── com/example/demo/
            └── DemoApplicationTests.java
```

## 기본 컨트롤러 생성

```java
@Controller
public class ThymeleafController {
    
    @GetMapping("/")
    public String index(Model model) {
        model.addAttribute("title", "Thymeleaf Best Practices");
        return "index";
    }
    
    @GetMapping("/hello")
    public String hello(Model model) {
        model.addAttribute("name", "Thymeleaf");
        return "hello";
    }
}
```

## 기본 템플릿 생성

### index.html

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title th:text="${title}">기본 제목</title>
</head>
<body>
    <h1 th:text="${title}">제목</h1>
    <p>Thymeleaf에 오신 것을 환영합니다!</p>
</body>
</html>
```

### hello.html

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Hello Thymeleaf</title>
</head>
<body>
    <h1>Hello, <span th:text="${name}">World</span>!</h1>
</body>
</html>
```

## Thymeleaf 네임스페이스

Thymeleaf를 사용하려면 HTML 요소에 네임스페이스를 추가해야 합니다:

```html
<html xmlns:th="http://www.thymeleaf.org">
```

다른 네임스페이스도 추가할 수 있습니다:

```html
<html xmlns:th="http://www.thymeleaf.org"
      xmlns:layout="http://www.ultraq.net.nz/thymeleaf/layout"
      xmlns:sec="http://www.thymeleaf.org/extras/spring-security">
```

## 애플리케이션 실행

```bash
# Gradle
./gradlew bootRun

# Maven
./mvnw spring-boot:run
```

애플리케이션이 실행되면 다음 URL에서 접근할 수 있습니다:

- 홈페이지: http://localhost:8080/
- Hello 페이지: http://localhost:8080/hello

## 개발 환경 설정

### IntelliJ IDEA

1. File → New → Project 선택
2. Spring Initializr 선택
3. 필요한 의존성 추가
4. 프로젝트 생성 후 Run 버튼 클릭

### VS Code

1. Spring Boot Extension Pack 설치
2. Command Palette (Ctrl+Shift+P) 열기
3. "Spring Initializr: Create a Maven Project" 선택
4. 필요한 의존성 추가

## 다음 장에서는

다음 장에서는 Thymeleaf의 변수 표현식과 기본 문법에 대해 알아보겠습니다.