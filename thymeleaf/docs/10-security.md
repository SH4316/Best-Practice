# 10. 보안

## 웹 보안의 중요성

현대 웹 애플리케이션에서 보안은 매우 중요합니다. Thymeleaf는 다양한 보안 기능을 내장하고 있어 일반적인 웹 보안 위협으로부터 애플리케이션을 보호할 수 있습니다.

## XSS (Cross-Site Scripting) 방지

### 기본 HTML 이스케이프

Thymeleaf는 기본적으로 모든 출력을 자동으로 이스케이프하여 XSS 공격을 방지합니다.

```html
<!-- 안전한 방법: 자동 이스케이프 -->
<div th:text="${user.comment}">사용자 댓글</div>

<!-- 위험한 방법: 이스케이프 없음 -->
<div th:utext="${user.comment}">사용자 댓글</div>
```

### XSS 공격 예시

```html
<!-- 사용자 입력에 악성 스크립트가 포함된 경우 -->
<!-- user.comment = "<script>alert('XSS 공격!');</script>안녕하세요" -->

<!-- 안전한 출력: 스크립트가 실행되지 않음 -->
<div th:text="${user.comment}">
    <script>alert('XSS 공격!');</script>안녕하세요
</div>

<!-- 위험한 출력: 스크립트가 실행됨 -->
<div th:utext="${user.comment}">
    <script>alert('XSS 공격!');</script>안녕하세요
</div>
```

### 안전한 HTML 처리

```html
<!-- 허용된 HTML 태그만 처리 -->
<div th:utext="${#strings.escapeXml(user.htmlContent)}">HTML 내용</div>

<!-- 특정 태그만 허용 -->
<div th:utext="${#strings.replace(user.htmlContent, '<script>', '<script>')}">필터링된 HTML</div>
```

## CSRF (Cross-Site Request Forgery) 방지

### Spring Security와 CSRF 토큰

```html
<!-- Spring Security가 자동으로 CSRF 토큰을 추가 -->
<form th:action="@{/users}" th:object="${user}" method="post">
    <input type="text" th:field="*{name}">
    <button type="submit">저장</button>
</form>
```

### 수동 CSRF 토큰 추가

```html
<!-- 수동으로 CSRF 토큰 추가 -->
<form th:action="@{/users}" method="post">
    <input type="hidden" th:name="${_csrf.parameterName}" th:value="${_csrf.token}">
    <input type="text" name="name">
    <button type="submit">저장</button>
</form>
```

### AJAX 요청에서의 CSRF

```html
<script>
// AJAX 요청에 CSRF 토큰 포함
const csrfToken = document.querySelector('meta[name="_csrf"]').getAttribute('content');
const csrfHeader = document.querySelector('meta[name="_csrf_header"]').getAttribute('content');

fetch('/api/users', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        [csrfHeader]: csrfToken
    },
    body: JSON.stringify(userData)
});
</script>
```

## 권한 기반 접근 제어

### Spring Security 통합

```html
<!-- 관리자 권한 확인 -->
<div th:if="${#authorization.expression('hasRole(''ADMIN'')')}">
    <button class="btn btn-danger" th:text="#{button.delete}">삭제</button>
</div>

<!-- 특정 권한 확인 -->
<div th:if="${#authorization.expression('hasAuthority(''USER_EDIT'')')}">
    <a th:href="@{/users/{id}/edit(id=${user.id})}" class="btn btn-primary">수정</a>
</div>

<!-- 인증된 사용자 확인 -->
<div th:if="${#authorization.expression('isAuthenticated()')}">
    <span th:text="${#authentication.name}">사용자 이름</span>
</div>

<!-- 특정 사용자 확인 -->
<div th:if="${#authorization.expression('hasRole(''USER'') and #authentication.name == user.name')}">
    <button class="btn btn-secondary">프로필 수정</button>
</div>
```

### 조건부 콘텐츠 표시

```html
<!-- 사용자 역할에 따른 메뉴 -->
<nav class="sidebar">
    <!-- 모든 사용자 -->
    <a th:href="@{/dashboard}" class="nav-link">대시보드</a>
    <a th:href="@{/profile}" class="nav-link">프로필</a>
    
    <!-- 일반 사용자 -->
    <div th:if="${#authorization.expression('hasRole(''USER'')')}">
        <a th:href="@{/my-posts}" class="nav-link">내 게시물</a>
    </div>
    
    <!-- 관리자 -->
    <div th:if="${#authorization.expression('hasRole(''ADMIN'')')}">
        <a th:href="@{/admin/users}" class="nav-link">사용자 관리</a>
        <a th:href="@{/admin/settings}" class="nav-link">시스템 설정</a>
    </div>
</nav>
```

## 입력 값 검증

### 서버측 검증

```html
<!-- 유효성 검사 에러 표시 -->
<form th:action="@{/users}" th:object="${user}" method="post">
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" 
               th:field="*{name}"
               th:classappend="${#fields.hasErrors('name')} ? 'is-invalid' : ''">
        
        <div class="invalid-feedback" th:if="${#fields.hasErrors('name')}" 
             th:errors="*{name}">이름 오류</div>
    </div>
    
    <div class="mb-3">
        <label for="email" class="form-label">이메일</label>
        <input type="email" class="form-control" 
               th:field="*{email}"
               th:classappend="${#fields.hasErrors('email')} ? 'is-invalid' : ''">
        
        <div class="invalid-feedback" th:if="${#fields.hasErrors('email')}" 
             th:errors="*{email}">이메일 오류</div>
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

### 클라이언트측 검증

```html
<!-- HTML5 유효성 검사 -->
<form th:action="@{/users}" method="post">
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" 
               name="name" required minlength="2" maxlength="50">
    </div>
    
    <div class="mb-3">
        <label for="email" class="form-label">이메일</label>
        <input type="email" class="form-control" 
               name="email" required>
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

## 민감 정보 처리

### 정보 마스킹

```html
<!-- 이메일 마스킹 -->
<div th:text="${#strings.substring(user.email, 0, 3) + '***' + #strings.substring(user.email, user.email.indexOf('@'))}">
    a***@example.com
</div>

<!-- 전화번호 마스킹 -->
<div th:text="${#strings.replace(user.phone, #strings.substring(user.phone, 4, 8), '****')}">
    010-****-5678
</div>

<!-- 신용카드 번호 마스킹 -->
<div th:text="'****-****-****-' + #strings.substring(user.cardNumber, 12, 16)">
    ****-****-****-1234
</div>
```

### 권한별 정보 표시

```html
<!-- 일반 사용자에게는 기본 정보만 -->
<div th:if="${#authorization.expression('hasRole(''USER'')')}">
    <p>이름: <span th:text="${user.name}">이름</span></p>
    <p>이메일: <span th:text="${user.email}">이메일</span></p>
</div>

<!-- 관리자에게는 모든 정보 -->
<div th:if="${#authorization.expression('hasRole(''ADMIN'')')}">
    <p>이름: <span th:text="${user.name}">이름</span></p>
    <p>이메일: <span th:text="${user.email}">이메일</span></p>
    <p>전화번호: <span th:text="${user.phone}">전화번호</span></p>
    <p>주소: <span th:text="${user.address}">주소</span></p>
    <p>가입일: <span th:text="${#dates.format(user.createdAt, 'yyyy-MM-dd')}">가입일</span></p>
    <p>마지막 로그인: <span th:text="${#dates.format(user.lastLogin, 'yyyy-MM-dd HH:mm')}">로그인</span></p>
</div>
```

## 콘텐츠 보안 정책 (CSP)

### CSP 헤더 설정

```html
<!-- 메타 태그로 CSP 설정 -->
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
              script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; 
              style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; 
              img-src 'self' data: https:; 
              font-src 'self' https://cdn.jsdelivr.net;">
```

### 인라인 스크립트 제한

```html
<!-- 위험한 인라인 스크립트 -->
<script>
    // 이 스크립트는 CSP에 의해 차단될 수 있음
    function showAlert() {
        alert('안녕하세요!');
    }
</script>

<!-- 안전한 외부 스크립트 -->
<script th:src="@{/js/app.js}"></script>
```

## 보안 헤더 설정

### 보안 관련 헤더

```html
<!-- 보안 헤더 설정 -->
<meta http-equiv="X-Content-Type-Options" content="nosniff">
<meta http-equiv="X-Frame-Options" content="DENY">
<meta http-equiv="X-XSS-Protection" content="1; mode=block">
<meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
```

## 파일 업로드 보안

### 파일 타입 검증

```html
<!-- 허용된 파일 타입만 업로드 -->
<div class="mb-3">
    <label for="profileImage" class="form-label">프로필 이미지</label>
    <input type="file" class="form-control" 
           id="profileImage" name="profileImage" 
           accept="image/jpeg,image/png,image/gif">
    <div class="form-text">JPG, PNG, GIF 파일만 업로드 가능합니다.</div>
</div>
```

### 파일 크기 제한

```html
<!-- 파일 크기 제한 -->
<div class="mb-3">
    <label for="document" class="form-label">문서</label>
    <input type="file" class="form-control" 
           id="document" name="document">
    <div class="form-text">최대 10MB까지 업로드 가능합니다.</div>
</div>
```

## Best Practice

1. **항상 th:text 사용**: 사용자 입력을 출력할 때는 항상 th:text를 사용하여 자동 이스케이프를 적용하세요.
2. **th:utext 신중 사용**: th:utext는 신중하게 사용하고, 반드시 입력값을 검증/필터링하세요.
3. **권한 확인**: 민감한 기능은 반드시 권한을 확인한 후 표시하세요.
4. **CSRF 토큰**: 모든 폼 제출에는 CSRF 토큰을 포함하세요.
5. **서버측 검증**: 클라이언트측 검증과 함께 서버측 검증을 항상 수행하세요.

## Bad Practice

1. **XSS 취약점**: th:utext를 무분별하게 사용하면 XSS 공격에 취약해집니다.
2. **권한 체크 없음**: 권한 확인 없이 민감한 정보나 기능을 노출하지 마세요.
3. **CSRF 토큰 없음**: CSRF 토큰 없는 폼은 CSRF 공격에 취약합니다.
4. **클라이언트측 검증만 의존**: 클라이언트측 검증은 우회할 수 있으므로 서버측 검증이 필수입니다.

## 다음 장에서는

다음 장에서는 성능 최적화 기법에 대해 알아보겠습니다.