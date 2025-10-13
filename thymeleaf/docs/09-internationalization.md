# 9. 국제화 (Internationalization)

## 국제화란?

국제화(i18n)는 애플리케이션을 다양한 언어와 지역에 맞게 적응시키는 과정입니다. Thymeleaf는 내장된 국제화 지원을 통해 다국어 애플리케이션을 쉽게 구현할 수 있습니다.

## 메시지 소스 설정

### application.properties

```properties
# 메시지 소스 설정
spring.messages.basename=messages
spring.messages.encoding=UTF-8
spring.messages.cache-duration=3600
spring.messages.fallback-to-system-locale=true
```

### Java 설정

```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    
    @Bean
    public MessageSource messageSource() {
        ResourceBundleMessageSource messageSource = new ResourceBundleMessageSource();
        messageSource.setBasename("messages");
        messageSource.setDefaultEncoding("UTF-8");
        messageSource.setCacheSeconds(3600);
        messageSource.setFallbackToSystemLocale(true);
        return messageSource;
    }
    
    @Bean
    public LocaleResolver localeResolver() {
        SessionLocaleResolver slr = new SessionLocaleResolver();
        slr.setDefaultLocale(Locale.KOREAN);
        return slr;
    }
    
    @Bean
    public LocaleChangeInterceptor localeChangeInterceptor() {
        LocaleChangeInterceptor lci = new LocaleChangeInterceptor();
        lci.setParamName("lang");
        return lci;
    }
    
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(localeChangeInterceptor());
    }
}
```

## 메시지 파일 생성

### 기본 메시지 파일 (messages.properties)

```properties
# 공통 메시지
app.title=Thymeleaf Best Practices
app.welcome=Welcome to Thymeleaf Best Practices
app.description=Learn the best practices for using Thymeleaf with Spring Boot

# 네비게이션
nav.home=Home
nav.users=Users
nav.products=Products
nav.about=About
nav.contact=Contact

# 사용자 관련
user.title=User Management
user.list=User List
user.create=Create User
user.edit=Edit User
user.delete=Delete User
user.name=Name
user.email=Email
user.role=Role
user.status=Status
user.active=Active
user.inactive=Inactive

# 역할
role.admin=Administrator
role.user=User
role.moderator=Moderator

# 폼
form.save=Save
form.cancel=Cancel
form.submit=Submit
form.reset=Reset
form.edit=Edit
form.delete=Delete

# 메시지
message.success=Operation completed successfully
message.error=An error occurred
message.confirm=Are you sure?
message.no.data=No data available

# 날짜 형식
date.format=yyyy-MM-dd
date.time.format=yyyy-MM-dd HH:mm:ss

# 숫자 형식
number.decimal.format=#,##0.##
number.currency.format=#,##0.00
```

### 한국어 메시지 파일 (messages_ko.properties)

```properties
# 공통 메시지
app.title=Thymeleaf Best Practices
app.welcome=Thymeleaf Best Practices에 오신 것을 환영합니다
app.description=Spring Boot와 함께 사용하는 Thymeleaf의 Best Practice를 배워보세요

# 네비게이션
nav.home=홈
nav.users=사용자
nav.products=제품
nav.about=소개
nav.contact=연락처

# 사용자 관련
user.title=사용자 관리
user.list=사용자 목록
user.create=사용자 생성
user.edit=사용자 수정
user.delete=사용자 삭제
user.name=이름
user.email=이메일
user.role=역할
user.status=상태
user.active=활성
user.inactive=비활성

# 역할
role.admin=관리자
role.user=사용자
role.moderator=중재자

# 폼
form.save=저장
form.cancel=취소
form.submit=제출
form.reset=초기화
form.edit=수정
form.delete=삭제

# 메시지
message.success=작업이 성공적으로 완료되었습니다
message.error=오류가 발생했습니다
message.confirm=정말로 실행하시겠습니까?
message.no.data=데이터가 없습니다

# 날짜 형식
date.format=yyyy-MM-dd
date.time.format=yyyy-MM-dd HH:mm:ss

# 숫자 형식
number.decimal.format=#,##0.##
number.currency.format=#,##0.00
```

### 영어 메시지 파일 (messages_en.properties)

```properties
# 공통 메시지
app.title=Thymeleaf Best Practices
app.welcome=Welcome to Thymeleaf Best Practices
app.description=Learn the best practices for using Thymeleaf with Spring Boot

# 네비게이션
nav.home=Home
nav.users=Users
nav.products=Products
nav.about=About
nav.contact=Contact

# 사용자 관련
user.title=User Management
user.list=User List
user.create=Create User
user.edit=Edit User
user.delete=Delete User
user.name=Name
user.email=Email
user.role=Role
user.status=Status
user.active=Active
user.inactive=Inactive

# 역할
role.admin=Administrator
role.user=User
role.moderator=Moderator

# 폼
form.save=Save
form.cancel=Cancel
form.submit=Submit
form.reset=Reset
form.edit=Edit
form.delete=Delete

# 메시지
message.success=Operation completed successfully
message.error=An error occurred
message.confirm=Are you sure?
message.no.data=No data available

# 날짜 형식
date.format=MM/dd/yyyy
date.time.format=MM/dd/yyyy HH:mm:ss

# 숫자 형식
number.decimal.format=#,##0.##
number.currency.format=#,##0.00
```

## 메시지 표현식 사용

### 기본 메시지

```html
<!-- 단순 메시지 -->
<h1 th:text="#{app.title}">Thymeleaf Best Practices</h1>
<p th:text="#{app.description}">설명</p>

<!-- 네비게이션 -->
<nav>
    <a th:href="@{/}" th:text="#{nav.home}">홈</a>
    <a th:href="@{/users}" th:text="#{nav.users}">사용자</a>
    <a th:href="@{/products}" th:text="#{nav.products}">제품</a>
</nav>
```

### 파라미터화된 메시지

```html
<!-- 메시지 파일: welcome.message=Welcome, {0}! -->
<p th:text="#{welcome.message(${user.name})}">Welcome, User!</p>

<!-- 메시지 파일: user.count=Total {0} users found -->
<p th:text="#{user.count(${#lists.size(users)})}">Total users found</p>

<!-- 여러 파라미터 -->
<!-- 메시지 파일: user.info=User {0} with role {1} -->
<p th:text="#{user.info(${user.name}, ${user.role})}">User info</p>
```

### 조건부 메시지

```html
<!-- 사용자 상태에 따른 메시지 -->
<div th:if="${user.active}" class="alert alert-success">
    <span th:text="#{user.active}">활성</span>
</div>
<div th:unless="${user.active}" class="alert alert-warning">
    <span th:text="#{user.inactive}">비활성</span>
</div>
```

## 동적 언어 전환

### 언어 선택기

```html
<!-- 언어 선택 드롭다운 -->
<div class="dropdown">
    <button class="btn btn-secondary dropdown-toggle" type="button" id="languageDropdown" data-bs-toggle="dropdown">
        <i class="bi bi-globe"></i> <span th:text="#{language.current}">언어</span>
    </button>
    <ul class="dropdown-menu">
        <li>
            <a class="dropdown-item" th:href="@{''(lang='ko')}" href="#">
                🇰🇷 한국어
            </a>
        </li>
        <li>
            <a class="dropdown-item" th:href="@{''(lang='en')}" href="#">
                🇺🇸 English
            </a>
        </li>
        <li>
            <a class="dropdown-item" th:href="@{''(lang='ja')}" href="#">
                🇯🇵 日本語
            </a>
        </li>
    </ul>
</div>
```

### 언어 링크

```html
<!-- 언어 전환 링크 -->
<div class="language-selector">
    <a th:href="@{''(lang='ko')}" class="language-link" href="#">한국어</a>
    <span>|</span>
    <a th:href="@{''(lang='en')}" class="language-link" href="#">English</a>
    <span>|</span>
    <a th:href="@{''(lang='ja')}" class="language-link" href="#">日本語</a>
</div>
```

## 날짜와 숫자 국제화

### 날짜 포맷팅

```html
<!-- 메시지 파일에 정의된 날짜 형식 사용 -->
<span th:text="${#dates.format(user.createdAt, #{date.format})}">날짜</span>

<!-- 시간 포함 -->
<span th:text="${#dates.format(user.createdAt, #{date.time.format})}">날짜 시간</span>

<!-- 로케일에 따른 날짜 포맷 -->
<span th:text="${#dates.format(user.createdAt, 'medium', #locale)}">날짜</span>
```

### 숫자 포맷팅

```html
<!-- 메시지 파일에 정의된 숫자 형식 사용 -->
<span th:text="${#numbers.formatDecimal(price, 1, #{number.decimal.format})}">가격</span>

<!-- 통화 포맷 -->
<span th:text="${#numbers.formatCurrency(price, #locale)}">통화</span>

<!-- 퍼센트 -->
<span th:text="${#numbers.formatPercent(ratio, 1, 1, #locale)}">비율</span>
```

## 복수형 처리

### 복수형 메시지

```properties
# messages.properties
user.count.one={0} user found
user.count.other={0} users found

# messages_ko.properties
user.count.other=사용자 {0}명을 찾았습니다

# messages_en.properties
user.count.one={0} user found
user.count.other={0} users found
```

### 복수형 사용

```html
<!-- 복수형 메시지 사용 -->
<p th:text="#{user.count(${#lists.size(users)})}">사용자 수</p>
```

## 템플릿에서의 국제화

### 전체 템플릿 예제

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      th:lang="${#locale.language}" 
      lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title th:text="#{app.title}">Thymeleaf Best Practices</title>
    <link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">
</head>
<body>
    <header>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" th:href="@{/}" th:text="#{app.title}">앱 제목</a>
                
                <div class="collapse navbar-collapse">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item">
                            <a class="nav-link" th:href="@{/}" th:text="#{nav.home}">홈</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" th:href="@{/users}" th:text="#{nav.users}">사용자</a>
                        </li>
                    </ul>
                    
                    <!-- 언어 선택기 -->
                    <div class="dropdown">
                        <button class="btn btn-outline-light dropdown-toggle" type="button" data-bs-toggle="dropdown">
                            <i class="bi bi-globe"></i>
                        </button>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" th:href="@{''(lang='ko')}">한국어</a></li>
                            <li><a class="dropdown-item" th:href="@{''(lang='en')}">English</a></li>
                        </ul>
                    </div>
                </div>
            </div>
        </nav>
    </header>
    
    <main class="container my-4">
        <h1 th:text="#{user.title}">사용자 관리</h1>
        
        <!-- 사용자 목록 -->
        <div th:if="${not #lists.isEmpty(users)}">
            <p th:text="#{user.count(${#lists.size(users)})}">사용자 수</p>
            
            <table class="table">
                <thead>
                    <tr>
                        <th th:text="#{user.name}">이름</th>
                        <th th:text="#{user.email}">이메일</th>
                        <th th:text="#{user.role}">역할</th>
                        <th th:text="#{user.status}">상태</th>
                        <th th:text="#{form.actions}">작업</th>
                    </tr>
                </thead>
                <tbody>
                    <tr th:each="user : ${users}">
                        <td th:text="${user.name}">이름</td>
                        <td th:text="${user.email}">이메일</td>
                        <td th:text="#{role.${user.role.toLowerCase()}}">역할</td>
                        <td>
                            <span th:class="${user.active} ? 'badge bg-success' : 'badge bg-secondary'"
                                  th:text="${user.active} ? #{user.active} : #{user.inactive}">상태</span>
                        </td>
                        <td>
                            <a th:href="@{/users/{id}/edit(id=${user.id})}" 
                               class="btn btn-sm btn-primary" th:text="#{form.edit}">수정</a>
                            <a th:href="@{/users/{id}/delete(id=${user.id})}" 
                               class="btn btn-sm btn-danger" th:text="#{form.delete}">삭제</a>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- 빈 목록 메시지 -->
        <div th:if="${#lists.isEmpty(users)}" class="alert alert-info">
            <span th:text="#{message.no.data}">데이터가 없습니다</span>
        </div>
    </main>
    
    <script th:src="@{/js/bootstrap.bundle.min.js}"></script>
</body>
</html>
```

## Best Practice

1. **메시지 키 규칙성**: 일관된 명명 규칙을 사용하여 메시지 키를 관리하세요.
2. **파라미터화된 메시지**: 동적인 값이 필요한 경우 파라미터화된 메시지를 사용하세요.
3. **로케일별 형식**: 날짜, 숫자, 통화 등은 로케일에 맞는 형식을 사용하세요.
4. **기본값 제공**: 누락된 메시지를 대비하여 기본값을 제공하세요.

## Bad Practice

1. **하드코딩된 텍스트**: 모든 텍스트는 메시지 파일에서 관리해야 합니다.
2. **일관성 없는 키**: 명명 규칙이 없으면 유지보수가 어려워집니다.
3. **복잡한 파라미터**: 너무 많은 파라미터를 사용하면 메시지 관리가 복잡해집니다.

## 다음 장에서는

다음 장에서는 보안과 XSS 방지에 대해 알아보겠습니다.