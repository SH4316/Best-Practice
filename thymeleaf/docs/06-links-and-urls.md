# 6. 링크와 URL 처리

## URL 표현식 `@{...}`

Thymeleaf의 URL 표현식은 동적 URL을 생성하는 강력한 방법을 제공합니다. 컨텍스트 경로를 자동으로 처리하고 다양한 URL 패턴을 지원합니다.

### 기본 URL 생성

```html
<!-- 단순 URL -->
<a th:href="@{/users}">사용자 목록</a>

<!-- 결과: /context/users -->
```

### 경로 변수

```html
<!-- 단일 경로 변수 -->
<a th:href="@{/users/{id}(id=${user.id})}">사용자 정보</a>

<!-- 여러 경로 변수 -->
<a th:href="@{/users/{id}/edit(id=${user.id}, tab='profile')}">프로필 편집</a>

<!-- 결과: /context/users/123/edit?tab=profile -->
```

### 쿼리 파라미터

```html
<!-- 단일 쿼리 파라미터 -->
<a th:href="@{/search(q=${keyword})}">검색</a>

<!-- 여러 쿼리 파라미터 -->
<a th:href="@{/search(q=${keyword}, page=${currentPage}, size=${pageSize})}">검색</a>

<!-- 결과: /context/search?q=keyword&page=1&size=10 -->
```

### URL 조합

```html
<!-- 경로 변수와 쿼리 파라미터 조합 -->
<a th:href="@{/users/{id}/posts(id=${user.id}, category='tech', sort='date')}">
    사용자 게시물
</a>

<!-- 결과: /context/users/123/posts?category=tech&sort=date -->
```

## 다양한 URL 유형

### 상대 URL

```html
<!-- 현재 경로 기준 상대 URL -->
<a th:href="@{../users}">상위 경로</a>
<a th:href="@{./profile}">현재 경로</a>
```

### 컨텍스트 경로

```html
<!-- 컨텍스트 경로를 포함한 URL -->
<a th:href="@{/~/users}">전체 애플리케이션 사용자 목록</a>

<!-- 결과: /users (컨텍스트 경로 무시) -->
```

### 서버 루트 URL

```html
<!-- 서버 루트 기준 URL -->
<a th:href="@{//static.example.com/images/logo.png}">외부 이미지</a>

<!-- 프로토콜 기준 URL -->
<a th:href="@{/users}" th:with="protocol=${#request.isSecure()} ? 'https' : 'http'">
    <span th:text="${protocol} + '://example.com' + @{/users}">링크</span>
</a>
```

### 절대 URL

```html
<!-- 절대 URL 생성 -->
<a th:href="@{https://example.com/users}" class="external-link">외부 사이트</a>

<!-- 동적 프로토콜 -->
<a th:href="@{#{server.protocol}://#{server.host}/users}" class="dynamic-link">동적 링크</a>
```

## 정적 리소스 처리

### CSS, JavaScript, 이미지

```html
<!-- CSS 파일 -->
<link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">

<!-- JavaScript 파일 -->
<script th:src="@{/js/jquery.min.js}"></script>

<!-- 이미지 -->
<img th:src="@{/images/logo.png}" alt="로고">

<!-- favicon -->
<link rel="icon" th:href="@{/favicon.ico}">
```

### 버전 관리

```html
<!-- 버전 파라미터 추가 -->
<link rel="stylesheet" th:href="@{/css/style.css(v=${app.version})}">

<!-- 결과: /context/css/style.css?v=1.0.0 -->
```

### CDN과 로컬 리소스 결합

```html
<!-- 로컬 리소스가 있으면 사용하고 없으면 CDN 사용 -->
<link th:href="@{/css/bootstrap.min.css}" 
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" 
      rel="stylesheet">
```

## 동적 링크 생성

### 페이징 링크

```html
<!-- 페이징 네비게이션 -->
<nav>
    <ul class="pagination">
        <li class="page-item" th:if="${currentPage > 1}">
            <a class="page-link" th:href="@{/users(page=${currentPage - 1})}">이전</a>
        </li>
        
        <li class="page-item" th:each="i : ${#numbers.sequence(1, totalPages)}">
            <a class="page-link" 
               th:href="@{/users(page=${i})}" 
               th:text="${i}">페이지</a>
        </li>
        
        <li class="page-item" th:if="${currentPage < totalPages}">
            <a class="page-link" th:href="@{/users(page=${currentPage + 1})}">다음</a>
        </li>
    </ul>
</nav>
```

### 정렬 링크

```html
<!-- 정렬 가능한 테이블 헤더 -->
<table class="table">
    <thead>
        <tr>
            <th>
                <a th:href="@{/users(sort='name', order=${sort == 'name' and order == 'asc' ? 'desc' : 'asc'})}">
                    이름
                    <span th:if="${sort == 'name'}">
                        <i th:class="${order == 'asc'} ? 'bi bi-arrow-up' : 'bi bi-arrow-down'"></i>
                    </span>
                </a>
            </th>
            <th>
                <a th:href="@{/users(sort='email', order=${sort == 'email' and order == 'asc' ? 'desc' : 'asc'})}">
                    이메일
                    <span th:if="${sort == 'email'}">
                        <i th:class="${order == 'asc'} ? 'bi bi-arrow-up' : 'bi bi-arrow-down'"></i>
                    </span>
                </a>
            </th>
        </tr>
    </thead>
    <tbody>
        <!-- 테이블 내용 -->
    </tbody>
</table>
```

### 검색 링크

```html
<!-- 검색 폼과 결과 링크 -->
<form th:action="@{/search}" method="get">
    <div class="input-group">
        <input type="text" class="form-control" name="q" th:value="${param.q}" placeholder="검색어">
        <button class="btn btn-primary" type="submit">검색</button>
    </div>
</form>

<!-- 검색 결과 -->
<div th:if="${param.q != null}">
    <h3>'<span th:text="${param.q}">검색어</span>' 검색 결과</h3>
    
    <div class="list-group">
        <a th:each="item : ${searchResults}" 
           th:href="@{/items/{id}(id=${item.id}, q=${param.q})}" 
           class="list-group-item">
            <h5 th:text="${item.title}">제목</h5>
            <p th:text="${item.description}">설명</p>
        </a>
    </div>
</div>
```

## 링크 상태 처리

### 활성 링크

```html
<!-- 현재 페이지에 따른 활성 링크 -->
<nav class="nav">
    <a class="nav-link" 
       th:href="@{/}" 
       th:classappend="${currentPage == 'home'} ? 'active' : ''">홈</a>
    <a class="nav-link" 
       th:href="@{/users}" 
       th:classappend="${currentPage == 'users'} ? 'active' : ''">사용자</a>
    <a class="nav-link" 
       th:href="@{/products}" 
       th:classappend="${currentPage == 'products'} ? 'active' : ''">제품</a>
</nav>
```

### 동적 메뉴

```html
<!-- 사용자 권한에 따른 메뉴 -->
<div class="sidebar">
    <div class="menu-item">
        <a th:href="@{/dashboard}" class="menu-link">
            <i class="bi bi-speedometer2"></i> 대시보드
        </a>
    </div>
    
    <!-- 관리자 메뉴 -->
    <div th:if="${user.admin}" class="menu-item">
        <a th:href="@{/admin/users}" class="menu-link">
            <i class="bi bi-people"></i> 사용자 관리
        </a>
    </div>
    
    <div th:if="${user.admin}" class="menu-item">
        <a th:href="@{/admin/settings}" class="menu-link">
            <i class="bi bi-gear"></i> 시스템 설정
        </a>
    </div>
    
    <!-- 중재자 메뉴 -->
    <div th:if="${user.moderator}" class="menu-item">
        <a th:href="@{/moderate/posts}" class="menu-link">
            <i class="bi bi-chat-dots"></i> 게시물 관리
        </a>
    </div>
</div>
```

## AJAX 링크

### JavaScript와 통합

```html
<!-- AJAX 링크 -->
<a th:href="@{/api/users/{id}(id=${user.id})}" 
   class="ajax-link" 
   data-user-id="${user.id}">
    사용자 정보
</a>

<script>
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.ajax-link').forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const url = this.getAttribute('href');
            const userId = this.getAttribute('data-user-id');
            
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    console.log('User data:', data);
                    // 데이터 처리
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        });
    });
});
</script>
```

### REST API 링크

```html
<!-- RESTful API 링크 -->
<div class="api-actions">
    <button th:attr="data-url=@{/api/users/{id}(id=${user.id})}" 
            class="btn btn-sm btn-primary get-user">
        조회
    </button>
    <button th:attr="data-url=@{/api/users/{id}(id=${user.id})}" 
            class="btn btn-sm btn-warning update-user">
        수정
    </button>
    <button th:attr="data-url=@{/api/users/{id}(id=${user.id})}" 
            class="btn btn-sm btn-danger delete-user">
        삭제
    </button>
</div>
```

## 보안과 링크

### CSRF 토큰 포함

```html
<!-- POST 요청 링크 -->
<form th:action="@{/users/{id}/delete(id=${user.id})}" method="post">
    <input type="hidden" th:name="${_csrf.parameterName}" th:value="${_csrf.token}">
    <button type="submit" class="btn btn-danger">삭제</button>
</form>
```

### 보안 검사

```html
<!-- 권한에 따른 링크 표시 -->
<div th:if="${#authorization.expression('hasRole(''ADMIN'')')}">
    <a th:href="@{/admin/users}" class="btn btn-primary">사용자 관리</a>
</div>

<div th:if="${#authorization.expression('hasAuthority(''USER_EDIT'')')}">
    <a th:href="@{/users/{id}/edit(id=${user.id})}" class="btn btn-secondary">
        <span th:text="${user.name}">이름</span> 수정
    </a>
</div>
```

## URL 인코딩

```html
<!-- 자동 URL 인코딩 -->
<a th:href="@{/search(q=${keyword})}">검색</a>

<!-- 수동 인코딩 -->
<a th:with="encodedKeyword=${#urllib.escape(keyword, 'UTF-8')}" 
   th:href="@{/search(q=${encodedKeyword})}">검색</a>
```

## Best Practice

1. **항상 URL 표현식 사용**: 하드코딩된 URL 대신 `@{...}` 표현식을 사용하세요.
2. **정적 리소스도 URL 표현식 사용**: 일관성을 위해 모든 리소스는 URL 표현식을 사용하세요.
3. **링크 상태 처리**: 현재 페이지를 시각적으로 표시하여 사용자 경험을 향상시키세요.
4. **보안 검사**: 권한에 따라 링크를 표시/숨기세요.

## Bad Practice

1. **하드코딩된 URL**: 컨텍스트 경로 변경 시 문제가 발생합니다.
2. **상대 경로 사용**: 파일 위치 변경 시 링크가 깨질 수 있습니다.
3. **보안 검사 없음**: 인가되지 않은 사용자에게 민감한 링크를 노출하지 마세요.

## 다음 장에서는

다음 장에서는 폼 처리와 데이터 바인딩에 대해 알아보겠습니다.