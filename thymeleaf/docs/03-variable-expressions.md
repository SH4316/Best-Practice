# 3. 변수 표현식

## 표현식 종류

Thymeleaf는 다양한 종류의 표현식을 제공합니다:

| 표현식 | 설명 | 예시 |
|--------|------|------|
| `${...}` | 변수 표현식 | `${user.name}` |
| `*{...}` | 선택 변수 표현식 | `*{name}` |
| `#{...}` | 메시지 표현식 | `#{message.key}` |
| `@{...}` | 링크 URL 표현식 | `@{/users/{id}(id=${user.id})}` |
| `~{...}` | 프래그먼트 표현식 | `~{fragments/header :: header}` |
| `@{/...}` | 리소스 URL 표현식 | `@{/css/style.css}` |

## 변수 표현식 `${...}`

### 기본 사용법

```html
<!-- 단일 변수 -->
<span th:text="${user.name}">홍길동</span>

<!-- 중첩 속성 -->
<span th:text="${user.address.city}">서울</span>
```

### 안전한 내비게이션 연산자

```html
<!-- null 체크 없이 사용 -->
<span th:text="${user.address.city}">서울</span>

<!-- 안전한 내비게이션 연산자 사용 -->
<span th:text="${user?.address?.city}">서울</span>
```

### 기본값 설정

```html
<!-- 기본값 제공 -->
<span th:text="${user.name ?: 'guest'}">게스트</span>
```

## 선택 변수 표현식 `*{...}`

선택 변수 표현식은 th:object로 지정된 객체의 속성에 간단하게 접근할 수 있게 해줍니다.

```html
<form th:object="${user}">
    <input type="text" th:field="*{name}" />
    <input type="email" th:field="*{email}" />
    <input type="text" th:field="*{address.city}" />
</form>
```

## 메시지 표현식 `#{...}`

메시지 표현식은 국제화에 사용됩니다.

### messages.properties

```properties
welcome.message=Welcome, {0}!
page.title=Thymeleaf Best Practices
user.name=User Name
```

### 템플릿에서 사용

```html
<!-- 단순 메시지 -->
<h1 th:text="#{page.title}">Default Title</h1>

<!-- 파라미터화된 메시지 -->
<p th:text="#{welcome.message(${user.name})}">Welcome!</p>
```

## 링크 URL 표현식 `@{...}`

### 기본 URL 생성

```html
<!-- 단순 URL -->
<a th:href="@{/users}">사용자 목록</a>

<!-- 경로 변수 -->
<a th:href="@{/users/{id}(id=${user.id})">사용자 정보</a>

<!-- 여러 경로 변수 -->
<a th:href="@{/users/{id}/edit(id=${user.id}, tab='profile')}">프로필 편집</a>
```

### 쿼리 파라미터

```html
<!-- 단일 쿼리 파라미터 -->
<a th:href="@{/search(q=${keyword})}">검색</a>

<!-- 여러 쿼리 파라미터 -->
<a th:href="@{/search(q=${keyword}, page=${currentPage}, size=${pageSize})}">검색</a>
```

### 절대 URL

```html
<!-- 상대 URL -->
<a th:href="@{/users}">사용자 목록</a>

<!-- 컨텍스트 경로를 포함한 URL -->
<a th:href="@{/~/users}">전체 애플리케이션 사용자 목록</a>

<!-- 서버 루트 URL -->
<a th:href="@{//static.example.com/images/logo.png}">외부 이미지</a>
```

## 프래그먼트 표현식 `~{...}`

```html
<!-- 프래그먼트 포함 -->
<div th:replace="~{fragments/header :: header}"></div>

<!-- 조건부 프래그먼트 -->
<div th:replace="~{fragments/header :: header(${pageTitle})}"></div>

<!-- 변수로 프래그먼트 지정 -->
<div th:replace="~{${content}}"></div>
```

## 리터럴 및 연산

### 문자열 리터럴

```html
<!-- 작은따옴표 사용 -->
<span th:text="'Hello, ' + ${user.name}">인사</span>

<!-- 큰따옴표 사용 -->
<span th:text="|Hello, ${user.name}|">인사</span>
```

### 숫자 리터럴

```html
<span th:text="${user.age + 1}">나이</span>
<span th:text="${user.age > 18} ? '성인' : '미성년자'">상태</span>
```

### 불리언 리터럴

```html
<div th:if="${user.active}" class="active">활성 사용자</div>
<div th:unless="${user.active}" class="inactive">비활성 사용자</div>
```

### null 리터럴

```html
<span th:text="${user.name != null} ? ${user.name} : '이름 없음'">이름</span>
```

## 연산자

### 산술 연산자

```html
<span th:text="${user.age + 1}">나이+1</span>
<span th:text="${user.age - 1}">나이-1</span>
<span th:text="${user.age * 2}">나이*2</span>
<span th:text="${user.age / 2}">나이/2</span>
<span th:text="${user.age % 2}">나이%2</span>
```

### 비교 연산자

```html
<span th:text="${user.age > 18}">성인</span>
<span th:text="${user.age >= 18}">성인 이상</span>
<span th:text="${user.age < 18}">미성년자</span>
<span th:text="${user.age <= 18}">성인 이하</span>
<span th:text="${user.age == 18}">18세</span>
<span th:text="${user.age != 18}">18세 아님</span>
```

### 논리 연산자

```html
<div th:if="${user.active and user.age >= 18}">
    활성 성인 사용자
</div>

<div th:if="${user.active or user.admin}">
    활성 또는 관리자
</div>

<div th:if="${not user.active}">
    비활성 사용자
</div>
```

### 조건 연산자

```html
<!-- 삼항 연산자 -->
<span th:text="${user.active} ? '활성' : '비활성'">상태</span>

<!-- Elvis 연산자 -->
<span th:text="${user.name ?: '이름 없음'}">이름</span>
```

## 유틸리티 객체

Thymeleaf는 다양한 유틸리티 객체를 제공합니다:

### #dates

```html
<!-- 날짜 포맷팅 -->
<span th:text="${#dates.format(user.createdAt, 'yyyy-MM-dd')}">날짜</span>

<!-- 날짜 생성 -->
<span th:text="${#dates.createNow()}">현재 날짜</span>

<!-- 날짜 연산 -->
<span th:text="${#dates.plusDays(user.createdAt, 7)}">7일 후</span>
```

### #strings

```html
<!-- 문자열 길이 -->
<span th:text="${#strings.length(user.name)}">이름 길이</span>

<!-- 문자열 변환 -->
<span th:text="${#strings.toUpperCase(user.name)}">대문자</span>

<!-- 문자열 포함 여부 -->
<span th:text="${#strings.contains(user.email, '@')}">이메일 확인</span>

<!-- 빈 문자열 확인 -->
<span th:text="${#strings.isEmpty(user.name)} ? '이름 없음' : ${user.name}">이름</span>
```

### #numbers

```html
<!-- 숫자 포맷팅 -->
<span th:text="${#numbers.formatDecimal(price, 1, 2)}">가격</span>

<!-- 정수 포맷팅 -->
<span th:text="${#numbers.formatInteger(count, 3)}">카운트</span>

<!-- 퍼센트 -->
<span th:text="${#numbers.formatPercent(ratio, 2, 2)}">비율</span>
```

### #lists

```html
<!-- 리스트 크기 -->
<span th:text="${#lists.size(users)}">사용자 수</span>

<!-- 빈 리스트 확인 -->
<div th:if="${#lists.isEmpty(users)}">사용자가 없습니다</div>

<!-- 리스트 포함 여부 -->
<span th:text="${#lists.contains(roles, 'ADMIN')} ? '관리자' : '일반 사용자'">역할</span>
```

## Best Practice

1. **안전한 내비게이션 연산자 사용**: null 값으로 인한 오류를 방지하기 위해 `?.`를 사용하세요.
2. **유틸리티 객체 활용**: 날짜, 문자열, 숫자 처리는 유틸리티 객체를 사용하세요.
3. **Elvis 연산자**: 기본값 설정에는 `?:` 연산자를 사용하세요.
4. **URL 표현식**: 모든 URL은 `@{...}` 표현식을 사용하여 생성하세요.

## Bad Practice

1. **null 체크 없이 변수 접근**: NullPointerException을 유발할 수 있습니다.
2. **하드코딩된 URL**: 컨텍스트 경로 변경 시 문제가 발생합니다.
3. **복잡한 연산**: 템플릿에서 복잡한 연산은 피하고 컨트롤러에서 처리하세요.

## 다음 장에서는

다음 장에서는 반복문과 상태 변수 활용법에 대해 알아보겠습니다.