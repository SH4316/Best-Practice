# 5. 조건문

## 조건문 속성

Thymeleaf는 다양한 조건문 속성을 제공합니다:

| 속성 | 설명 | 예시 |
|------|------|------|
| `th:if` | 조건이 참일 때 요소를 렌더링 | `<div th:if="${user.active}">...</div>` |
| `th:unless` | 조건이 거짓일 때 요소를 렌더링 | `<div th:unless="${user.active}">...</div>` |
| `th:switch` | 다중 조건 처리 | `<div th:switch="${user.role}">...</div>` |
| `th:case` | switch의 case | `<span th:case="'ADMIN'">...</span>` |

## th:if와 th:unless

### 기본 사용법

```html
<!-- th:if: 조건이 참일 때만 렌더링 -->
<div th:if="${user.active}" class="alert alert-success">
    <span th:text="${user.name}">이름</span>님은 활성 상태입니다.
</div>

<!-- th:unless: 조건이 거짓일 때만 렌더링 -->
<div th:unless="${user.active}" class="alert alert-warning">
    <span th:text="${user.name}">이름</span>님은 비활성 상태입니다.
</div>
```

### 다중 조건

```html
<!-- AND 조건 -->
<div th:if="${user.active and user.age >= 18}" class="alert alert-info">
    활성 성인 사용자입니다.
</div>

<!-- OR 조건 -->
<div th:if="${user.admin or user.moderator}" class="alert alert-danger">
    관리자 권한이 있습니다.
</div>

<!-- NOT 조건 -->
<div th:if="${not user.banned}" class="alert alert-success">
    정상 사용자입니다.
</div>
```

### null 체크

```html
<!-- null이 아닐 때만 표시 -->
<div th:if="${user.name != null}" class="user-name">
    <span th:text="${user.name}">이름</span>
</div>

<!-- null이거나 비어있을 때 표시 -->
<div th:if="${user.name == null or #strings.isEmpty(user.name)}" class="alert alert-warning">
    이름이 없습니다.
</div>
```

## th:switch와 th:case

### 기본 switch 문

```html
<div th:switch="${user.role}">
    <p th:case="'ADMIN'" class="alert alert-danger">
        <span th:text="${user.name}">이름</span>님은 관리자입니다.
    </p>
    <p th:case="'USER'" class="alert alert-primary">
        <span th:text="${user.name}">이름</span>님은 일반 사용자입니다.
    </p>
    <p th:case="'MODERATOR'" class="alert alert-warning">
        <span th:text="${user.name}">이름</span>님은 중재자입니다.
    </p>
    <p th:case="*" class="alert alert-secondary">
        <span th:text="${user.name}">이름</span>님은 알 수 없는 역할입니다.
    </p>
</div>
```

### switch와 스타일링

```html
<div th:switch="${user.status}">
    <span th:case="'ACTIVE'" class="badge bg-success" th:text="${user.status}">상태</span>
    <span th:case="'INACTIVE'" class="badge bg-secondary" th:text="${user.status}">상태</span>
    <span th:case="'BANNED'" class="badge bg-danger" th:text="${user.status}">상태</span>
    <span th:case="*" class="badge bg-warning" th:text="${user.status}">상태</span>
</div>
```

### 복잡한 switch 문

```html
<div th:switch="${user.role}">
    <!-- 관리자 -->
    <div th:case="'ADMIN'" class="card border-danger">
        <div class="card-header bg-danger text-white">
            <i class="bi bi-shield-check"></i> 관리자 권한
        </div>
        <div class="card-body">
            <p><strong>이름:</strong> <span th:text="${user.name}">이름</span></p>
            <p><strong>권한:</strong> 모든 기능 접근 가능</p>
            <button class="btn btn-danger">관리자 패널</button>
        </div>
    </div>
    
    <!-- 중재자 -->
    <div th:case="'MODERATOR'" class="card border-warning">
        <div class="card-header bg-warning text-dark">
            <i class="bi bi-shield"></i> 중재자 권한
        </div>
        <div class="card-body">
            <p><strong>이름:</strong> <span th:text="${user.name}">이름</span></p>
            <p><strong>권한:</strong> 콘텐츠 관리 가능</p>
            <button class="btn btn-warning">중재자 패널</button>
        </div>
    </div>
    
    <!-- 일반 사용자 -->
    <div th:case="'USER'" class="card border-primary">
        <div class="card-header bg-primary text-white">
            <i class="bi bi-person"></i> 사용자 권한
        </div>
        <div class="card-body">
            <p><strong>이름:</strong> <span th:text="${user.name}">이름</span></p>
            <p><strong>권한:</strong> 기본 기능만 사용 가능</p>
            <button class="btn btn-primary">마이페이지</button>
        </div>
    </div>
    
    <!-- 기본 -->
    <div th:case="*" class="card border-secondary">
        <div class="card-header bg-secondary text-white">
            <i class="bi bi-question-circle"></i> 알 수 없는 권한
        </div>
        <div class="card-body">
            <p><strong>이름:</strong> <span th:text="${user.name}">이름</span></p>
            <p><strong>권한:</strong> 권한 정보 없음</p>
            <button class="btn btn-secondary" disabled>기능 제한</button>
        </div>
    </div>
</div>
```

## 삼항 연산자와 Elvis 연산자

### 삼항 연산자

```html
<!-- 기본 삼항 연산자 -->
<span th:text="${user.active} ? '활성' : '비활성'">상태</span>

<!-- 복잡한 삼항 연산자 -->
<span th:class="${user.active} ? 'badge bg-success' : 'badge bg-secondary'" 
      th:text="${user.active} ? '활성' : '비활성'">상태</span>

<!-- 중첩 삼항 연산자 -->
<span th:text="${user.age >= 18} ? 
               (${user.admin} ? '성인 관리자' : '성인 사용자') : 
               '미성년자'">사용자 타입</span>
```

### Elvis 연산자

```html
<!-- 기본 Elvis 연산자 -->
<span th:text="${user.name ?: '이름 없음'}">이름</span>

<!-- 중첩 Elvis 연산자 -->
<span th:text="${user.profile?.nickname ?: user.name ?: '게스트'}">닉네임</span>

<!-- 복잡한 Elvis 연산자 -->
<div th:with="displayName=${user.profile?.displayName ?: user.name ?: '알 수 없는 사용자'}">
    <h3 th:text="${displayName}">표시 이름</h3>
</div>
```

## 조건문과 반복문 결합

```html
<table class="table">
    <thead>
        <tr>
            <th>번호</th>
            <th>이름</th>
            <th>역할</th>
            <th>상태</th>
            <th>작업</th>
        </tr>
    </thead>
    <tbody>
        <tr th:each="user, stat : ${users}">
            <td th:text="${stat.count}">1</td>
            <td th:text="${user.name}">이름</td>
            
            <!-- 역할에 따른 스타일링 -->
            <td>
                <span th:switch="${user.role}">
                    <span th:case="'ADMIN'" class="badge bg-danger">관리자</span>
                    <span th:case="'MODERATOR'" class="badge bg-warning">중재자</span>
                    <span th:case="'USER'" class="badge bg-primary">사용자</span>
                    <span th:case="*" class="badge bg-secondary">알 수 없음</span>
                </span>
            </td>
            
            <!-- 상태에 따른 표시 -->
            <td>
                <span th:if="${user.active}" class="badge bg-success">활성</span>
                <span th:unless="${user.active}" class="badge bg-secondary">비활성</span>
            </td>
            
            <!-- 권한에 따른 작업 버튼 -->
            <td>
                <button th:if="${user.admin}" class="btn btn-sm btn-danger">관리</button>
                <button th:if="${user.moderator}" class="btn btn-sm btn-warning">중재</button>
                <button th:if="${user.active}" class="btn btn-sm btn-primary">수정</button>
                <button th:unless="${user.active}" class="btn btn-sm btn-secondary" disabled>비활성</button>
            </td>
        </tr>
    </tbody>
</table>
```

## 복잡한 조건 처리

### 사용자 권한에 따른 메뉴 표시

```html
<nav class="navbar">
    <div class="container">
        <a class="navbar-brand" href="#">애플리케이션</a>
        
        <div class="navbar-nav">
            <!-- 모든 사용자에게 표시 -->
            <a class="nav-link" href="/">홈</a>
            <a class="nav-link" href="/profile">프로필</a>
            
            <!-- 로그인한 사용자에게만 표시 -->
            <a th:if="${user != null}" class="nav-link" href="/dashboard">대시보드</a>
            
            <!-- 관리자에게만 표시 -->
            <a th:if="${user != null and user.admin}" class="nav-link" href="/admin">관리자</a>
            
            <!-- 중재자에게만 표시 -->
            <a th:if="${user != null and user.moderator}" class="nav-link" href="/moderate">중재</a>
            
            <!-- 로그인하지 않은 사용자에게만 표시 -->
            <a th:if="${user == null}" class="nav-link" href="/login">로그인</a>
            <a th:if="${user == null}" class="nav-link" href="/register">회원가입</a>
            
            <!-- 로그인한 사용자에게만 표시 -->
            <a th:if="${user != null}" class="nav-link" href="/logout">로그아웃</a>
        </div>
    </div>
</nav>
```

### 동적 폼 필드

```html
<form th:object="${user}" method="post">
    <!-- 기본 필드 -->
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" th:field="*{name}">
    </div>
    
    <div class="mb-3">
        <label for="email" class="form-label">이메일</label>
        <input type="email" class="form-control" th:field="*{email}">
    </div>
    
    <!-- 관리자인 경우에만 표시 -->
    <div th:if="${user.admin}" class="mb-3">
        <label for="role" class="form-label">역할</label>
        <select class="form-select" th:field="*{role}">
            <option value="USER">사용자</option>
            <option value="MODERATOR">중재자</option>
            <option value="ADMIN">관리자</option>
        </select>
    </div>
    
    <!-- 비활성 사용자인 경우에만 표시 -->
    <div th:unless="${user.active}" class="mb-3">
        <label for="reason" class="form-label">비활성 사유</label>
        <textarea class="form-control" th:field="*{inactiveReason}"></textarea>
    </div>
    
    <!-- 신규 사용자인 경우에만 표시 -->
    <div th:if="${user.id == null}" class="mb-3">
        <label for="password" class="form-label">비밀번호</label>
        <input type="password" class="form-control" th:field="*{password}">
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

## Best Practice

1. **switch-case 문 활용**: 복잡한 조건은 switch-case 문을 사용하여 간결하게 표현하세요.
2. **삼항 연산자**: 간단한 조건은 삼항 연산자를 사용하여 한 줄로 표현하세요.
3. **Elvis 연산자**: 기본값 설정에는 Elvis 연산자를 사용하세요.
4. **조건문 중첩 피하기**: 복잡한 조건은 논리 연산자를 사용하여 중첩을 피하세요.

## Bad Practice

1. **중복된 if 문**: 동일한 조건을 여러 번 확인하는 것은 비효율적입니다.
2. **복잡한 중첩 조건**: 가독성이 저하되고 유지보수가 어려워집니다.
3. **불필요한 조건**: 항상 참이거나 거짓인 조건은 제거하세요.

## 다음 장에서는

다음 장에서는 링크 처리와 URL 생성에 대해 알아보겠습니다.