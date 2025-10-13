# 7. 폼 처리

## 폼 기본 설정

Thymeleaf는 Spring MVC와 완벽하게 통합되어 폼 처리를 간단하게 만들어 줍니다.

### th:object와 th:field

```html
<form th:action="@{/users}" th:object="${user}" method="post">
    <input type="hidden" th:field="*{id}">
    
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" 
               id="name" th:field="*{name}" 
               placeholder="이름을 입력하세요">
    </div>
    
    <div class="mb-3">
        <label for="email" class="form-label">이메일</label>
        <input type="email" class="form-control" 
               id="email" th:field="*{email}" 
               placeholder="이메일을 입력하세요">
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

### th:field의 장점

1. **자동 ID 생성**: `name` 필드에서 자동으로 `id="name"`을 생성합니다.
2. **값 바인딩**: 객체의 값을 자동으로 입력 필드에 설정합니다.
3. **에러 처리**: 유효성 검사 에러를 자동으로 표시합니다.
4. **CSRF 보호**: Spring Security와 통합되어 자동으로 CSRF 토큰을 추가합니다.

## 입력 필드 유형

### 텍스트 필드

```html
<!-- 일반 텍스트 -->
<input type="text" th:field="*{name}" class="form-control">

<!-- 비밀번호 -->
<input type="password" th:field="*{password}" class="form-control">

<!-- 숫자 -->
<input type="number" th:field="*{age}" class="form-control">

<!-- 이메일 -->
<input type="email" th:field="*{email}" class="form-control">

<!-- 전화번호 -->
<input type="tel" th:field="*{phone}" class="form-control">

<!-- URL -->
<input type="url" th:field="*{website}" class="form-control">
```

### 선택 필드

```html
<!-- 라디오 버튼 -->
<div class="form-check">
    <input class="form-check-input" type="radio" th:field="*{gender}" value="M" id="male">
    <label class="form-check-label" for="male">남성</label>
</div>
<div class="form-check">
    <input class="form-check-input" type="radio" th:field="*{gender}" value="F" id="female">
    <label class="form-check-label" for="female">여성</label>
</div>

<!-- 체크박스 -->
<div class="form-check">
    <input class="form-check-input" type="checkbox" th:field="*{active}" id="active">
    <label class="form-check-label" for="active">활성 상태</label>
</div>

<!-- 셀렉트 박스 -->
<select class="form-select" th:field="*{role}">
    <option value="">역할을 선택하세요</option>
    <option value="USER">사용자</option>
    <option value="ADMIN">관리자</option>
    <option value="MODERATOR">중재자</option>
</select>

<!-- 다중 셀렉트 -->
<select class="form-select" th:field="*{skills}" multiple>
    <option value="JAVA">Java</option>
    <option value="PYTHON">Python</option>
    <option value="JAVASCRIPT">JavaScript</option>
</select>
```

### 텍스트 영역

```html
<textarea class="form-control" th:field="*{description}" rows="3" 
          placeholder="설명을 입력하세요"></textarea>
```

### 파일 업로드

```html
<div class="mb-3">
    <label for="profileImage" class="form-label">프로필 이미지</label>
    <input type="file" class="form-control" id="profileImage" name="profileImage">
</div>
```

## 유효성 검사

### 에러 표시

```html
<form th:action="@{/users}" th:object="${user}" method="post">
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" 
               id="name" th:field="*{name}"
               th:classappend="${#fields.hasErrors('name')} ? 'is-invalid' : ''">
        
        <!-- 필드별 에러 메시지 -->
        <div class="invalid-feedback" th:if="${#fields.hasErrors('name')}" 
             th:errors="*{name}">이름 오류</div>
    </div>
    
    <div class="mb-3">
        <label for="email" class="form-label">이메일</label>
        <input type="email" class="form-control" 
               id="email" th:field="*{email}"
               th:classappend="${#fields.hasErrors('email')} ? 'is-invalid' : ''">
        
        <div class="invalid-feedback" th:if="${#fields.hasErrors('email')}" 
             th:errors="*{email}">이메일 오류</div>
    </div>
    
    <!-- 전역 에러 메시지 -->
    <div class="alert alert-danger" th:if="${#fields.hasErrors('*')}">
        <h5>폼 에러:</h5>
        <ul>
            <li th:each="err : ${#fields.errors('*')}" th:text="${err}">에러 메시지</li>
        </ul>
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

### 에러 스타일링

```html
<!-- 부트스트랩 유효성 검사 스타일 -->
<div class="mb-3">
    <label for="name" class="form-label">이름</label>
    <input type="text" class="form-control" 
           id="name" th:field="*{name}"
           th:classappend="${#fields.hasErrors('name')} ? 'is-invalid' : (${#fields.hasErrors('name')} ? '' : 'is-valid')">
    
    <div class="valid-feedback">
        올바른 이름입니다.
    </div>
    <div class="invalid-feedback" th:if="${#fields.hasErrors('name')}" 
         th:errors="*{name}">이름 오류</div>
</div>
```

## 동적 폼

### 동적 필드 추가

```html
<form th:action="@{/users}" th:object="${user}" method="post">
    <!-- 기본 필드 -->
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" id="name" th:field="*{name}">
    </div>
    
    <!-- 동적 전화번호 필드 -->
    <div class="mb-3">
        <label class="form-label">전화번호</label>
        <div id="phoneFields">
            <div th:each="phone, stat : *{phones}" class="input-group mb-2">
                <input type="tel" class="form-control" 
                       th:field="*{phones[__${stat.index}__]}" 
                       th:id="'phone-' + ${stat.index}">
                <button type="button" class="btn btn-outline-danger remove-phone" 
                        th:if="${stat.count > 1}">삭제</button>
            </div>
        </div>
        <button type="button" class="btn btn-outline-primary" id="addPhone">전화번호 추가</button>
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>

<script>
document.getElementById('addPhone').addEventListener('click', function() {
    const phoneFields = document.getElementById('phoneFields');
    const newField = document.createElement('div');
    newField.className = 'input-group mb-2';
    newField.innerHTML = `
        <input type="tel" class="form-control" name="phones[]" placeholder="전화번호">
        <button type="button" class="btn btn-outline-danger remove-phone">삭제</button>
    `;
    phoneFields.appendChild(newField);
});

document.addEventListener('click', function(e) {
    if (e.target.classList.contains('remove-phone')) {
        e.target.parentElement.remove();
    }
});
</script>
```

### 조건부 필드

```html
<form th:action="@{/users}" th:object="${user}" method="post">
    <!-- 기본 필드 -->
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" id="name" th:field="*{name}">
    </div>
    
    <!-- 역할 선택 -->
    <div class="mb-3">
        <label for="role" class="form-label">역할</label>
        <select class="form-select" id="role" th:field="*{role}">
            <option value="">역할을 선택하세요</option>
            <option value="USER">사용자</option>
            <option value="ADMIN">관리자</option>
            <option value="MODERATOR">중재자</option>
        </select>
    </div>
    
    <!-- 관리자 전용 필드 -->
    <div th:if="*{role == 'ADMIN'}" class="mb-3">
        <label for="adminLevel" class="form-label">관리자 레벨</label>
        <select class="form-select" id="adminLevel" th:field="*{adminLevel}">
            <option value="1">레벨 1</option>
            <option value="2">레벨 2</option>
            <option value="3">레벨 3</option>
        </select>
    </div>
    
    <!-- 비활성 사용자용 필드 -->
    <div th:unless="*{active}" class="mb-3">
        <label for="inactiveReason" class="form-label">비활성 사유</label>
        <textarea class="form-control" id="inactiveReason" th:field="*{inactiveReason}"></textarea>
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

## 폼 데이터 전처리

### 데이터 변환

```html
<!-- 날짜 포맷팅 -->
<div class="mb-3">
    <label for="birthDate" class="form-label">생일</label>
    <input type="date" class="form-control" 
           id="birthDate" th:field="*{birthDate}"
           th:value="${#dates.format(user.birthDate, 'yyyy-MM-dd')}">
</div>

<!-- 숫자 포맷팅 -->
<div class="mb-3">
    <label for="salary" class="form-label">급여</label>
    <input type="number" class="form-control" 
           id="salary" th:field="*{salary}"
           th:value="${#numbers.formatDecimal(user.salary, 1, 2)}">
</div>
```

### 기본값 설정

```html
<!-- 새 사용자용 기본값 -->
<div class="mb-3">
    <label for="status" class="form-label">상태</label>
    <select class="form-select" id="status" th:field="*{status}">
        <option value="">상태를 선택하세요</option>
        <option value="ACTIVE" th:selected="${user.status == null or user.status == 'ACTIVE'}">활성</option>
        <option value="INACTIVE" th:selected="${user.status == 'INACTIVE'}">비활성</option>
    </select>
</div>
```

## 폼 보안

### CSRF 보호

```html
<!-- Spring Security가 자동으로 CSRF 토큰 추가 -->
<form th:action="@{/users}" th:object="${user}" method="post">
    <!-- 폼 필드 -->
    <button type="submit" class="btn btn-primary">저장</button>
</form>

<!-- 수동 CSRF 토큰 추가 -->
<form th:action="@{/users}" th:object="${user}" method="post">
    <input type="hidden" th:name="${_csrf.parameterName}" th:value="${_csrf.token}">
    <!-- 폼 필드 -->
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

### 권한 기반 폼

```html
<!-- 관리자만 접근 가능한 폼 -->
<div th:if="${#authorization.expression('hasRole(''ADMIN'')')}">
    <form th:action="@{/admin/users}" th:object="${user}" method="post">
        <!-- 관리자 전용 필드 -->
        <button type="submit" class="btn btn-danger">관리자 저장</button>
    </form>
</div>

<!-- 일반 사용자용 폼 -->
<div th:unless="${#authorization.expression('hasRole(''ADMIN'')')}">
    <form th:action="@{/users}" th:object="${user}" method="post">
        <!-- 일반 필드 -->
        <button type="submit" class="btn btn-primary">저장</button>
    </form>
</div>
```

## 폼 제출 처리

### 다중 버튼

```html
<form th:action="@{/users}" th:object="${user}" method="post">
    <!-- 폼 필드 -->
    
    <div class="btn-group">
        <button type="submit" name="action" value="save" class="btn btn-primary">저장</button>
        <button type="submit" name="action" value="saveAndContinue" class="btn btn-success">저장하고 계속</button>
        <button type="submit" name="action" value="delete" class="btn btn-danger">삭제</button>
    </div>
</form>
```

### AJAX 폼 제출

```html
<form th:action="@{/users}" th:object="${user}" method="post" id="userForm">
    <div class="mb-3">
        <label for="name" class="form-label">이름</label>
        <input type="text" class="form-control" id="name" th:field="*{name}">
    </div>
    
    <div class="mb-3">
        <label for="email" class="form-label">이메일</label>
        <input type="email" class="form-control" id="email" th:field="*{email}">
    </div>
    
    <button type="submit" class="btn btn-primary">저장</button>
</form>

<div id="formResult" class="mt-3"></div>

<script>
document.getElementById('userForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    
    fetch(this.action, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('formResult');
        if (data.success) {
            resultDiv.innerHTML = '<div class="alert alert-success">저장되었습니다.</div>';
        } else {
            resultDiv.innerHTML = '<div class="alert alert-danger">저장 실패: ' + data.message + '</div>';
        }
    })
    .catch(error => {
        document.getElementById('formResult').innerHTML = 
            '<div class="alert alert-danger">오류 발생: ' + error.message + '</div>';
    });
});
</script>
```

## Best Practice

1. **th:object와 th:field 사용**: 폼 바인딩을 자동화하여 코드를 단순화하세요.
2. **유효성 검사 구현**: 서버측 유효성 검사를 통해 데이터 무결성을 보장하세요.
3. **CSRF 보호**: Spring Security를 사용하여 자동으로 CSRF 토큰을 추가하세요.
4. **에러 메시지 제공**: 사용자에게 명확한 에러 메시지를 표시하세요.

## Bad Practice

1. **수동 폼 처리**: th:object와 th:field를 사용하지 않으면 코드가 길어지고 오류가 발생하기 쉽습니다.
2. **클라이언트측 검증만**: 서버측 검증 없이 클라이언트측 검증만 의존하면 보안에 취약합니다.
3. **CSRF 토큰 없음**: CSRF 공격에 취약해집니다.

## 다음 장에서는

다음 장에서는 프래그먼트 재사용과 레이아웃에 대해 알아보겠습니다.