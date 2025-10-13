# 8. 프래그먼트와 레이아웃

## 프래그먼트란?

프래그먼트는 Thymeleaf의 강력한 재사용 기능으로, 템플릿의 일부를 정의하고 다른 템플릿에서 가져와 사용할 수 있게 해줍니다. 이를 통해 코드 중복을 줄이고 유지보수성을 향상시킬 수 있습니다.

## 프래그먼트 정의

### 기본 프래그먼트

```html
<!-- fragments/header.html -->
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head th:fragment="header-head">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title th:text="${title}">기본 제목</title>
    <link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">
</head>

<header th:fragment="header" class="site-header">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" th:href="@{/}">애플리케이션</a>
            
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" th:href="@{/}">홈</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" th:href="@{/users}">사용자</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
</header>
</html>
```

### 매개변수가 있는 프래그먼트

```html
<!-- fragments/pagination.html -->
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<nav th:fragment="pagination(currentPage, totalPages, url)" class="d-flex justify-content-center">
    <ul class="pagination">
        <!-- 이전 페이지 -->
        <li class="page-item" th:if="${currentPage > 1}">
            <a class="page-link" th:href="@{${url}(page=${currentPage - 1})}">이전</a>
        </li>
        
        <!-- 페이지 번호 -->
        <li class="page-item" th:each="i : ${#numbers.sequence(1, totalPages)}">
            <a class="page-link" 
               th:href="@{${url}(page=${i})}" 
               th:class="${i == currentPage} ? 'active' : ''"
               th:text="${i}">페이지</a>
        </li>
        
        <!-- 다음 페이지 -->
        <li class="page-item" th:if="${currentPage < totalPages}">
            <a class="page-link" th:href="@{${url}(page=${currentPage + 1})}">다음</a>
        </li>
    </ul>
</nav>
</html>
```

## 프래그먼트 사용

### th:insert

```html
<!-- 프래그먼트를 현재 태그 안에 삽입 -->
<div th:insert="~{fragments/header :: header}"></div>

<!-- 결과 -->
<div>
    <header class="site-header">
        <!-- 헤더 내용 -->
    </header>
</div>
```

### th:replace

```html
<!-- 프래그먼트로 현재 태그를 완전히 교체 -->
<div th:replace="~{fragments/header :: header}"></div>

<!-- 결과 -->
<header class="site-header">
    <!-- 헤더 내용 -->
</header>
```

### th:include

```html
<!-- 프래그먼트의 내용만 삽입 (태그 제외) -->
<div th:include="~{fragments/header :: header}"></div>

<!-- 결과 -->
<div>
    <!-- 헤더 내용 (header 태그 제외) -->
</div>
```

## 매개변수 전달

### 명시적 매개변수

```html
<!-- 매개변수 전달 -->
<div th:replace="~{fragments/pagination :: pagination(${currentPage}, ${totalPages}, '/users')}"></div>

<!-- 또는 명명된 매개변수 -->
<div th:replace="~{fragments/pagination :: pagination(currentPage=${currentPage}, totalPages=${totalPages}, url='/users')}"></div>
```

### th:with를 사용한 매개변수

```html
<!-- th:with로 변수 설정 후 전달 -->
<div th:with="currentPage=1, totalPages=10, url='/users'">
    <div th:replace="~{fragments/pagination :: pagination(currentPage, totalPages, url)}"></div>
</div>
```

## 레이아웃 다이얼렉트

Thymeleaf Layout Dialect를 사용하면 레이아웃 상속을 쉽게 구현할 수 있습니다.

### 의존성 추가

```gradle
implementation 'nz.net.ultraq.thymeleaf:thymeleaf-layout-dialect'
```

### 기본 레이아웃

```html
<!-- layouts/default.html -->
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:layout="http://www.ultraq.net.nz/thymeleaf/layout">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title th:text="${title}">기본 제목</title>
    <link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">
    
    <!-- 페이지별 CSS 추가 -->
    <th:block layout:fragment="styles"></th:block>
</head>
<body>
    <!-- 헤더 -->
    <header th:replace="~{fragments/header :: header}"></header>
    
    <!-- 메인 콘텐츠 -->
    <main class="container my-4">
        <div layout:fragment="content">
            <!-- 페이지 콘텐츠가 여기에 삽입됨 -->
        </div>
    </main>
    
    <!-- 푸터 -->
    <footer th:replace="~{fragments/footer :: footer}"></footer>
    
    <!-- 기본 JavaScript -->
    <script th:src="@{/js/bootstrap.bundle.min.js}"></script>
    
    <!-- 페이지별 JavaScript 추가 -->
    <th:block layout:fragment="scripts"></th:block>
</body>
</html>
```

### 레이아웃 사용

```html
<!-- users/list.html -->
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:layout="http://www.ultraq.net.nz/thymeleaf/layout"
      layout:decorate="~{layouts/default}">

<head>
    <title>사용자 목록</title>
</head>

<th:block layout:fragment="styles">
    <style>
        .user-card {
            transition: transform 0.2s;
        }
        .user-card:hover {
            transform: translateY(-5px);
        }
    </style>
</th:block>

<div layout:fragment="content">
    <h1>사용자 목록</h1>
    
    <table class="table">
        <thead>
            <tr>
                <th>ID</th>
                <th>이름</th>
                <th>이메일</th>
                <th>역할</th>
            </tr>
        </thead>
        <tbody>
            <tr th:each="user : ${users}">
                <td th:text="${user.id}">ID</td>
                <td th:text="${user.name}">이름</td>
                <td th:text="${user.email}">이메일</td>
                <td th:text="${user.role}">역할</td>
            </tr>
        </tbody>
    </table>
    
    <!-- 페이지네이션 -->
    <div th:replace="~{fragments/pagination :: pagination(${currentPage}, ${totalPages}, '/users')}"></div>
</div>

<th:block layout:fragment="scripts">
    <script>
        // 페이지별 JavaScript
        document.addEventListener('DOMContentLoaded', function() {
            console.log('사용자 목록 페이지 로드됨');
        });
    </script>
</th:block>

</html>
```

## 고급 프래그먼트 기법

### 조건부 프래그먼트

```html
<!-- fragments/user-card.html -->
<div th:fragment="userCard(user, showDetails)" class="card mb-3">
    <div class="card-header">
        <h5 th:text="${user.name}">이름</h5>
        <span class="badge bg-primary" th:text="${user.role}">역할</span>
    </div>
    <div class="card-body">
        <p class="card-text" th:text="${user.email}">이메일</p>
        
        <!-- 조건부 상세 정보 -->
        <div th:if="${showDetails}">
            <p><strong>가입일:</strong> <span th:text="${#dates.format(user.createdAt, 'yyyy-MM-dd')}">날짜</span></p>
            <p><strong>상태:</strong> 
                <span th:text="${user.active} ? '활성' : '비활성'">상태</span>
            </p>
        </div>
        
        <div class="d-flex justify-content-between">
            <a th:href="@{/users/{id}(id=${user.id})}" class="btn btn-primary">상세</a>
            <a th:href="@{/users/{id}/edit(id=${user.id})}" class="btn btn-secondary">수정</a>
        </div>
    </div>
</div>
```

### 프래그먼트 재귀

```html
<!-- fragments/category-tree.html -->
<ul th:fragment="categoryTree(categories)" class="list-group">
    <li th:each="category : ${categories}" class="list-group-item">
        <div class="d-flex justify-content-between align-items-center">
            <span th:text="${category.name}">카테고리 이름</span>
            <span class="badge bg-primary" th:text="${category.itemCount}">0</span>
        </div>
        
        <!-- 재귀적으로 하위 카테고리 표시 -->
        <ul th:if="${not #lists.isEmpty(category.children)}" class="list-group mt-2">
            <li th:replace="~{::categoryTree(category.children)}" 
                th:each="category : ${category.children}"></li>
        </ul>
    </li>
</ul>
```

### 프래그먼트 확장

```html
<!-- fragments/alerts.html -->
<div th:fragment="alert(type, message)" 
     th:class="'alert alert-' + ${type} + ' alert-dismissible fade show'" 
     role="alert">
    <span th:text="${message}">메시지</span>
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
</div>

<!-- 확장된 프래그먼트 -->
<div th:fragment="successAlert(message)" 
     th:replace="~{::alert('success', ${message})}"></div>

<div th:fragment="errorAlert(message)" 
     th:replace="~{::alert('danger', ${message})}"></div>

<div th:fragment="warningAlert(message)" 
     th:replace="~{::alert('warning', ${message})}"></div>
```

## 프래그먼트 캐싱

```html
<!-- 캐시 가능한 프래그먼트 -->
<div th:fragment="userStats" th:with="stats=${userService.getStats()}">
    <div class="row">
        <div class="col-md-4">
            <div class="card text-center">
                <div class="card-body">
                    <h5 class="card-title">총 사용자</h5>
                    <p class="card-text display-4" th:text="${stats.totalUsers}">0</p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card text-center">
                <div class="card-body">
                    <h5 class="card-title">활성 사용자</h5>
                    <p class="card-text display-4" th:text="${stats.activeUsers}">0</p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card text-center">
                <div class="card-body">
                    <h5 class="card-title">신규 사용자</h5>
                    <p class="card-text display-4" th:text="${stats.newUsers}">0</p>
                </div>
            </div>
        </div>
    </div>
</div>
```

## Best Practice

1. **프래그먼트로 재사용 가능한 부분 분리**: 헤더, 푸터, 메뉴 등 공통 부분을 프래그먼트로 분리하세요.
2. **레이아웃 다이얼렉트 활용**: 페이지 구조가 유사할 때 레이아웃 상속을 사용하세요.
3. **명명된 매개변수 사용**: 프래그먼트에 매개변수를 전달할 때 명명된 매개변수를 사용하여 가독성을 높이세요.
4. **적절한 프래그먼트 선택**: th:insert, th:replace, th:include를 상황에 맞게 사용하세요.

## Bad Practice

1. **코드 중복**: 동일한 코드를 여러 템플릿에 복사-붙여넣기 하지 마세요.
2. **과도한 프래그먼트 분할**: 너무 작은 단위로 프래그먼트를 분할하면 관리가 어려워질 수 있습니다.
3. **프래그먼트 의존성 복잡성**: 프래그먼트 간의 의존성이 복잡해지지 않도록 주의하세요.

## 다음 장에서는

다음 장에서는 국제화(i18n)와 메시지 처리에 대해 알아보겠습니다.