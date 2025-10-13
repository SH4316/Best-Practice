# 13. Thymeleaf 3.x 새로운 기능

## Thymeleaf 3.x 소개

Thymeleaf 3.x는 이전 버전에 비해 많은 개선과 새로운 기능을 도입했습니다. 이 장에서는 Thymeleaf 3.x의 주요 새로운 기능과 개선사항에 대해 알아보겠습니다.

## 주요 개선사항

### 1. 자연 템플릿 개선

Thymeleaf 3.x는 자연 템플릿(Natural Templates) 개념을 더욱 강화했습니다.

#### HTML5 호환성

```html
<!-- Thymeleaf 2.x -->
<table>
    <tr th:each="user : ${users}">
        <td th:text="${user.name}">Name</td>
    </tr>
</table>

<!-- Thymeleaf 3.x: 더 자연적인 HTML5 구조 -->
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Email</th>
        </tr>
    </thead>
    <tbody>
        <tr th:each="user : ${users}">
            <td th:text="${user.name}">Name</td>
            <td th:text="${user.email}">Email</td>
        </tr>
    </tbody>
</table>
```

#### 데이터 속성 지원

```html
<!-- Thymeleaf 3.x: 데이터 속성을 사용한 더 깔끔한 마크업 -->
<div th:each="user : ${users}" th:data-user-id="${user.id}" class="user-card">
    <h3 th:text="${user.name}">User Name</h3>
    <p th:text="${user.email}">user@example.com</p>
</div>
```

### 2. 개선된 표현식 구문

#### null 안전 탐색 연산자

```html
<!-- Thymeleaf 2.x -->
<span th:if="${user != null and user.address != null and user.address.city != null}" 
      th:text="${user.address.city}">City</span>

<!-- Thymeleaf 3.x: 더 간결한 null 안전 탐색 -->
<span th:text="${user?.address?.city}">City</span>
```

#### 리터럴 대체

```html
<!-- Thymeleaf 3.x: 파이프(|)를 사용한 리터럴 대체 -->
<span th:text="|Hello, ${user.name}! Today is ${#dates.format(today, 'EEEE')}|">Greeting</span>
```

### 3. 개선된 프래그먼트 시나렉스

#### 와일드카드 프래그먼트 선택

```html
<!-- Thymeleaf 3.x: 와일드카드를 사용한 프래그먼트 선택 -->
<div th:replace="~{fragments/user :: *}"></div> <!-- user.html의 모든 프래그먼트 -->
<div th:replace="~{fragments/user :: user-*}"></div> <!-- user-로 시작하는 모든 프래그먼트 -->
```

#### 매개변수화된 프래그먼트

```html
<!-- fragments/user.html -->
<div th:fragment="user-card(user, showDetails, cardClass)" th:class="${cardClass}">
    <h3 th:text="${user.name}">User Name</h3>
    <div th:if="${showDetails}">
        <p th:text="${user.email}">Email</p>
        <p th:text="${user.phone}">Phone</p>
    </div>
</div>

<!-- 사용 예제 -->
<div th:replace="~{fragments/user :: user-card(currentUser, true, 'card mb-3')}"></div>
```

## 새로운 기능

### 1. 확장된 표준 표현식

#### 이중 중괄호 표현식

```html
<!-- Thymeleaf 3.x: 이중 중괄호를 사용한 pre-processed 표현식 -->
<span th:text="${{user.${fieldName}}}">Dynamic Field</span>
```

#### 링크 URL 표현식 개선

```html
<!-- Thymeleaf 3.x: 더 유연한 URL 표현식 -->
<a th:href="@{/users/{id}/edit(id=${user.id}, section='profile', back=${#request.requestURI})}">
    Edit Profile
</a>
```

### 2. 개선된 유틸리티 객체

#### 임시 변수 지원

```html
<!-- Thymeleaf 3.x: #temporals 유틸리티 -->
<div th:with="tempVar=${#temporals.format(user.createdAt, 'yyyy-MM-dd')}">
    <span th:text="${tempVar}">Formatted Date</span>
</div>
```

#### 컨버전 서비스

```html
<!-- Thymeleaf 3.x: #conversions 유틸리티 -->
<span th:text="${#conversions.convert(user, 'com.example.dto.UserDTO').name}">Converted Name</span>
```

### 3. 새로운 처리기(Processor) 속성

#### th:block

```html
<!-- Thymeleaf 3.x: 논리적 블록을 만드는 th:block -->
<th:block th:each="user : ${users}">
    <div th:if="${user.active}" class="user-card active">
        <h3 th:text="${user.name}">User Name</h3>
    </div>
    <div th:unless="${user.active}" class="user-card inactive">
        <h3 th:text="${user.name}">User Name</h3>
    </div>
</th:block>
```

#### th:with

```html
<!-- Thymeleaf 3.x: 변수 정의를 위한 th:with -->
<div th:with="fullName=${user.firstName + ' ' + user.lastName}, 
                  userAge=${#dates.year(user.birthDate)},
                  isAdult=${userAge >= 18}">
    <h3 th:text="${fullName}">Full Name</h3>
    <p th:text="${userAge + ' years old'}">Age</p>
    <span th:class="${isAdult} ? 'badge bg-success' : 'badge bg-warning'"
          th:text="${isAdult} ? 'Adult' : 'Minor'">Status</span>
</div>
```

## 자동 이스케이프 개선

### 기본 이스케이프 정책

```html
<!-- Thymeleaf 3.x: 기본 이스케이프 정책 개선 -->
<div>
    <!-- 기본적으로 HTML 이스케이프 -->
    <p th:text="${user.comment}">User Comment</p>
    
    <!-- 명시적으로 비이스케이프 -->
    <div th:utext="${user.htmlContent}">HTML Content</div>
</div>
```

### 텍스트 노드 처리

```html
<!-- Thymeleaf 3.x: 텍스트 노드 처리 개선 -->
<div th:text="${user.description}">Description</div>

<!-- 여러 줄 텍스트 처리 -->
<pre th:text="${user.multiLineDescription}">Multi-line Description</pre>
```

## 템플릿 모드 개선

### 새로운 템플릿 모드

```html
<!-- Thymeleaf 3.x: HTML5 모드 지원 -->
<html xmlns:th="http://www.thymeleaf.org" th:mode="HTML5">

<!-- 자동 모드 감지 -->
<div th:switch="${contentType}">
    <div th:case="text" th:text="${content}">Text Content</div>
    <div th:case="html" th:utext="${content}">HTML Content</div>
    <div th:case="javascript" th:text="${content}">JavaScript Content</div>
</div>
```

### 원시 텍스트 모드

```html
<!-- Thymeleaf 3.x: 원시 텍스트 모드 -->
<textarea th:text="${user.rawContent}" th:mode="TEXT">Raw Content</textarea>

<!-- CSS/JS 내용 처리 -->
<style th:text="${customCSS}" th:mode="TEXT">Custom CSS</style>
<script th:text="${customJS}" th:mode="TEXT">Custom JS</script>
```

## 성능 개선

### 템플릿 파싱 최적화

```properties
# application.properties: Thymeleaf 3.x 성능 설정
spring.thymeleaf.cache=true  # 프로덕션 환경에서는 캐시 활성화
spring.thymeleaf.cache-ttl-ms=3600000  # 캐시 만료 시간 (1시간)
spring.thymeleaf.template-resolver-order=1  # 템플릿 리졸버 순서
```

### 메모리 사용 최적화

```java
// Thymeleaf 3.x: 커스텀 캐시 설정
@Bean
public ICacheManager cacheManager() {
    StandardCacheManager cacheManager = new StandardCacheManager();
    // 템플릿 캐시 크기 제한
    cacheManager.setTemplateCacheMaxSize(100);
    // 파싱된 템플릿 캐시 크기 제한
    cacheManager.setParsedTemplateCacheMaxSize(50);
    return cacheManager;
}
```

## 호환성 및 마이그레이션

### Thymeleaf 2.x에서 3.x로 마이그레이션

```xml
<!-- Thymeleaf 2.x 의존성 -->
<dependency>
    <groupId>org.thymeleaf</groupId>
    <artifactId>thymeleaf</artifactId>
    <version>2.1.6.RELEASE</version>
</dependency>

<!-- Thymeleaf 3.x 의존성 -->
<dependency>
    <groupId>org.thymeleaf</groupId>
    <artifactId>thymeleaf</artifactId>
    <version>3.1.2.RELEASE</version>
</dependency>
```

### 주요 호환성 변경 사항

1. **표현식 구문**: 더 엄격한 구문 검사
2. **속성 처리**: 일부 속성의 처리 방식 변경
3. **유틸리티 객체**: 일부 유틸리티 객체의 메서드 변경
4. **템플릿 모드**: 기본 템플릿 모드 변경

```html
<!-- Thymeleaf 2.x: 호환되지 않는 구문 -->
<div th:with="user=${userService.getUser()}">
    <span th:text="${user.name}">Name</span>
</div>

<!-- Thymeleaf 3.x: 호환되는 구문 -->
<div th:with="user=${userService.getUser()}">
    <span th:text="${user.name}">Name</span>
</div>
```

## Best Practice

1. **자연 템플릿 활용**: Thymeleaf 3.x의 자연 템플릿 기능을 최대한 활용하세요.
2. **null 안전 탐색 연산자**: `?.` 연산자를 사용하여 NullPointerException을 방지하세요.
3. **프래그먼트 재사용**: 개선된 프래그먼트 문법을 사용하여 코드 재사용성을 높이세요.
4. **성능 최적화**: 캐시 설정과 메모리 최적화를 통해 성능을 향상시키세요.

## Bad Practice

1. **이전 버전 호환성 무시**: 마이그레이션 시 호환성 문제를 무시하지 마세요.
2. **새로운 기능 오용**: 새로운 기능을 올바르게 이해하지 않고 사용하지 마세요.
3. **성능 고려 부족**: 새로운 기능을 사용할 때 성능 영향을 고려하세요.

## 다음 장에서는

다음 장에서는 커스텀 다이얼렉트와 프로세서 만드는 방법에 대해 알아보겠습니다.