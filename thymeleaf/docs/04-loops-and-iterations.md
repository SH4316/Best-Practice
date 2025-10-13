# 4. 반복문과 반복

## th:each 속성

Thymeleaf에서 반복문은 `th:each` 속성을 사용하여 구현합니다.

### 기본 문법

```html
<tr th:each="user : ${users}">
    <td th:text="${user.id}">ID</td>
    <td th:text="${user.name}">이름</td>
    <td th:text="${user.email}">이메일</td>
</tr>
```

### 상태 변수

`th:each`는 두 번째 변수로 상태 정보를 제공합니다.

```html
<tr th:each="user, stat : ${users}">
    <td th:text="${stat.count}">번호</td>
    <td th:text="${user.name}">이름</td>
    <td th:text="${user.email}">이메일</td>
    <td th:text="${stat.even} ? '짝수' : '홀수'">상태</td>
</tr>
```

### 상태 변수 속성

| 속성 | 설명 | 예시 |
|------|------|------|
| `index` | 0부터 시작하는 인덱스 | `${stat.index}` |
| `count` | 1부터 시작하는 카운트 | `${stat.count}` |
| `size` | 반복 요소의 총 개수 | `${stat.size}` |
| `current` | 현재 반복 요소 | `${stat.current}` |
| `even` | 현재 인덱스가 짝수인지 여부 | `${stat.even}` |
| `odd` | 현재 인덱스가 홀수인지 여부 | `${stat.odd}` |
| `first` | 첫 번째 요소인지 여부 | `${stat.first}` |
| `last` | 마지막 요소인지 여부 | `${stat.last}` |

## 다양한 반복문 예제

### 기본 테이블

```html
<table class="table">
    <thead>
        <tr>
            <th>번호</th>
            <th>이름</th>
            <th>이메일</th>
            <th>역할</th>
            <th>상태</th>
        </tr>
    </thead>
    <tbody>
        <tr th:each="user, stat : ${users}">
            <td th:text="${stat.count}">1</td>
            <td th:text="${user.name}">홍길동</td>
            <td th:text="${user.email}">user@example.com</td>
            <td>
                <span class="badge bg-primary" th:text="${user.role}">USER</span>
            </td>
            <td>
                <span class="badge" 
                      th:class="${user.active} ? 'bg-success' : 'bg-secondary'"
                      th:text="${user.active} ? '활성' : '비활성'">상태</span>
            </td>
        </tr>
    </tbody>
</table>
```

### 짝수/홀수 스타일링

```html
<table class="table">
    <tbody>
        <tr th:each="user, stat : ${users}" 
            th:class="${stat.even} ? 'table-light' : 'table-white'">
            <td th:text="${stat.count}">번호</td>
            <td th:text="${user.name}">이름</td>
            <td th:text="${user.email}">이메일</td>
        </tr>
    </tbody>
</table>
```

### 첫 번째와 마지막 요소 특별 처리

```html
<div th:each="user, stat : ${users}">
    <div th:class="${stat.first} ? 'alert alert-danger' : 'alert alert-primary'">
        <span th:if="${stat.first}" class="badge bg-danger">첫 번째</span>
        <span th:if="${stat.last}" class="badge bg-warning">마지막</span>
        <span th:text="${user.name}">이름</span>
    </div>
</div>
```

### 인덱스 기반 처리

```html
<div class="list-group">
    <a href="#" class="list-group-item list-group-item-action" 
       th:each="user, stat : ${users}"
       th:href="@{/users/{id}(id=${user.id})}">
        <div class="d-flex w-100 justify-content-between">
            <h5 class="mb-1" th:text="${stat.index} + '. ' + ${user.name}">이름</h5>
            <small th:text="${user.createdAt}">날짜</small>
        </div>
        <p class="mb-1" th:text="${user.email}">이메일</p>
    </a>
</div>
```

## 중첩 반복문

```html
<div th:each="department : ${departments}">
    <h3 th:text="${department.name}">부서명</h3>
    <ul class="list-group">
        <li class="list-group-item" 
            th:each="user, stat2 : ${department.users}"
            th:text="${stat2.count} + '. ' + ${user.name}">
            사용자 이름
        </li>
    </ul>
</div>
```

## 반복문과 조건문 결합

```html
<table class="table">
    <tbody>
        <tr th:each="user, stat : ${users}">
            <td th:text="${stat.count}">번호</td>
            <td th:text="${user.name}">이름</td>
            <td th:text="${user.email}">이메일</td>
            <td>
                <!-- 활성 사용자만 배지 표시 -->
                <span th:if="${user.active}" 
                      class="badge bg-success" 
                      th:text="${user.role}">역할</span>
                <!-- 비활성 사용자는 회색 텍스트 -->
                <span th:unless="${user.active}" 
                      class="text-muted" 
                      th:text="${user.role}">역할</span>
            </td>
        </tr>
    </tbody>
</table>
```

## 빈 목록 처리

```html
<!-- 목록이 비어있을 때 메시지 표시 -->
<div th:if="${#lists.isEmpty(users)}" class="alert alert-warning">
    <h4>사용자가 없습니다</h4>
    <p>등록된 사용자가 없습니다. 새 사용자를 추가해주세요.</p>
</div>

<!-- 목록이 있을 때만 테이블 표시 -->
<table th:unless="${#lists.isEmpty(users)}" class="table">
    <thead>
        <tr>
            <th>번호</th>
            <th>이름</th>
            <th>이메일</th>
        </tr>
    </thead>
    <tbody>
        <tr th:each="user, stat : ${users}">
            <td th:text="${stat.count}">번호</td>
            <td th:text="${user.name}">이름</td>
            <td th:text="${user.email}">이메일</td>
        </tr>
    </tbody>
</table>
```

## 반복문 성능 최적화

### 페이징 처리

```html
<!-- 컨트롤러에서 페이징된 데이터 전달 -->
<table class="table">
    <tbody>
        <tr th:each="user, stat : ${users.content}">
            <td th:text="${stat.count}">번호</td>
            <td th:text="${user.name}">이름</td>
            <td th:text="${user.email}">이메일</td>
        </tr>
    </tbody>
</table>

<!-- 페이지네이션 -->
<nav>
    <ul class="pagination">
        <li class="page-item" th:each="i : ${#numbers.sequence(0, users.totalPages - 1)}">
            <a class="page-link" th:href="@{/users(page=${i})}" th:text="${i + 1}">페이지</a>
        </li>
    </ul>
</nav>
```

### 지연 로딩

```html
<!-- 필요한 시점에 데이터 로드 -->
<div th:with="activeUsers=${userService.getActiveUsers()}">
    <table class="table">
        <tbody>
            <tr th:each="user, stat : ${activeUsers}">
                <td th:text="${stat.count}">번호</td>
                <td th:text="${user.name}">이름</td>
                <td th:text="${user.email}">이메일</td>
            </tr>
        </tbody>
    </table>
</div>
```

## 데이터 그리드 예제

```html
<div class="row">
    <div class="col-md-4" th:each="user, stat : ${users}">
        <div class="card mb-3">
            <div class="card-header">
                <span class="badge bg-primary" th:text="${stat.count}">번호</span>
                <span class="badge bg-secondary" th:text="${user.role}">역할</span>
            </div>
            <div class="card-body">
                <h5 class="card-title" th:text="${user.name}">이름</h5>
                <p class="card-text" th:text="${user.email}">이메일</p>
                <div class="d-flex justify-content-between">
                    <span th:class="${user.active} ? 'text-success' : 'text-muted'"
                          th:text="${user.active} ? '활성' : '비활성'">상태</span>
                    <div>
                        <a th:href="@{/users/{id}/edit(id=${user.id})}" class="btn btn-sm btn-primary">수정</a>
                        <a th:href="@{/users/{id}/delete(id=${user.id})}" class="btn btn-sm btn-danger">삭제</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
```

## Best Practice

1. **상태 변수 활용**: 인덱스, 카운트, 홀짝 정보를 활용하여 풍부한 템플릿을 만드세요.
2. **빈 목록 처리**: 빈 목록일 때 사용자에게 적절한 메시지를 표시하세요.
3. **페이징 구현**: 대용량 데이터는 페이징 처리를 구현하세요.
4. **지연 로딩**: 필요한 시점에 데이터를 로드하여 성능을 향상시키세요.

## Bad Practice

1. **상태 변수 미활용**: 반복문의 상태 정보를 활용하지 않으면 템플릿이 제한적입니다.
2. **빈 목록 처리 없음**: 빈 목록일 때 빈 테이블만 표시되면 사용자 경험이 저하됩니다.
3. **대용량 데이터 전체 로딩**: 메모리 문제와 성능 저하를 유발할 수 있습니다.

## 다음 장에서는

다음 장에서는 조건문과 논리 처리에 대해 알아보겠습니다.