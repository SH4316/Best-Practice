# 11. 성능 최적화

## 성능 최적화의 중요성

웹 애플리케이션의 성능은 사용자 경험에 직접적인 영향을 미칩니다. Thymeleaf 템플릿을 최적화하면 페이지 로딩 속도를 향상시키고 서버 리소스 사용을 줄일 수 있습니다.

## 템플릿 캐싱

### 개발 환경 vs 운영 환경

```properties
# application.properties (개발 환경)
spring.thymeleaf.cache=false
spring.thymeleaf.cache-ttl=0

# application.properties (운영 환경)
spring.thymeleaf.cache=true
spring.thymeleaf.cache-ttl=3600
```

### 캐시 설정

```java
@Configuration
public class ThymeleafConfig {
    
    @Bean
    public SpringTemplateEngine templateEngine() {
        SpringTemplateEngine templateEngine = new SpringTemplateEngine();
        templateEngine.setTemplateResolver(templateResolver());
        
        // 캐시 설정
        StandardCacheManager cacheManager = new StandardCacheManager();
        cacheManager.setTemplateCacheMaxSize(100);
        templateEngine.setCacheManager(cacheManager);
        
        return templateEngine;
    }
}
```

## 불필요한 연산 방지

### th:with를 사용한 변수 캐싱

```html
<!-- Bad Practice: 반복적인 연산 -->
<div>
    <span th:text="${#dates.format(user.createdAt, 'yyyy-MM-dd')}">날짜</span>
    <span th:text="${#dates.format(user.createdAt, 'yyyy-MM-dd')}">날짜</span>
    <span th:text="${#dates.format(user.createdAt, 'yyyy-MM-dd')}">날짜</span>
</div>

<!-- Good Practice: th:with 사용 -->
<div th:with="formattedDate=${#dates.format(user.createdAt, 'yyyy-MM-dd')}">
    <span th:text="${formattedDate}">날짜</span>
    <span th:text="${formattedDate}">날짜</span>
    <span th:text="${formattedDate}">날짜</span>
</div>
```

### 복잡한 연산 최적화

```html
<!-- Bad Practice: 복잡한 연산 반복 -->
<div th:each="user : ${users}">
    <span th:text="${#strings.toUpperCase(user.name) + ' - ' + #strings.substring(user.email, 0, user.email.indexOf('@'))}">정보</span>
</div>

<!-- Good Practice: 연산 결과 캐싱 -->
<div th:each="user : ${users}">
    <div th:with="displayName=${#strings.toUpperCase(user.name)}, 
                  emailDomain=${#strings.substring(user.email, user.email.indexOf('@') + 1)}">
        <span th:text="${displayName + ' - ' + emailDomain}">정보</span>
    </div>
</div>
```

## 지연 로딩

### 필요한 시점에 데이터 로드

```html
<!-- Bad Practice: 모든 데이터를 미리 로드 -->
<table>
    <tr th:each="user : ${allUsers}">
        <td th:text="${user.name}">이름</td>
        <td th:text="${user.email}">이메일</td>
        <td th:text="${user.profile.bio}">소개</td>
    </tr>
</table>

<!-- Good Practice: 필요한 데이터만 로드 -->
<div th:with="activeUsers=${userService.getActiveUsers()}">
    <table>
        <tr th:each="user : ${activeUsers}">
            <td th:text="${user.name}">이름</td>
            <td th:text="${user.email}">이메일</td>
            <!-- 상세 정보는 필요할 때만 로드 -->
            <td>
                <button class="btn btn-sm" 
                        th:attr="data-user-id=${user.id}"
                        onclick="loadUserProfile(this.getAttribute('data-user-id'))">
                    프로필 보기
                </button>
            </td>
        </tr>
    </table>
</div>
```

## 대용량 데이터 처리

### 페이징 구현

```html
<!-- Bad Practice: 전체 데이터 로드 -->
<table>
    <tr th:each="user : ${allUsers}">
        <td th:text="${user.name}">이름</td>
        <td th:text="${user.email}">이메일</td>
    </tr>
</table>

<!-- Good Practice: 페이징된 데이터 로드 -->
<table>
    <tr th:each="user : ${users.content}">
        <td th:text="${user.name}">이름</td>
        <td th:text="${user.email}">이메일</td>
    </tr>
</table>

<!-- 페이지네이션 -->
<nav>
    <ul class="pagination">
        <li class="page-item" th:if="${users.hasPrevious()}">
            <a class="page-link" th:href="@{/users(page=${users.number - 1})}">이전</a>
        </li>
        
        <li class="page-item" th:each="i : ${#numbers.sequence(0, users.totalPages - 1)}">
            <a class="page-link" 
               th:href="@{/users(page=${i})}" 
               th:text="${i + 1}">페이지</a>
        </li>
        
        <li class="page-item" th:if="${users.hasNext()}">
            <a class="page-link" th:href="@{/users(page=${users.number + 1})}">다음</a>
        </li>
    </ul>
</nav>
```

### 지연 로딩과 AJAX

```html
<!-- 초기에는 기본 정보만 표시 -->
<div class="user-list">
    <div th:each="user : ${users}" class="user-item" th:id="'user-' + ${user.id}">
        <h4 th:text="${user.name}">이름</h4>
        <p th:text="${user.email}">이메일</p>
        <button class="btn btn-sm btn-primary" 
                th:onclick="loadUserDetails([[${user.id}]])">
            상세 정보
        </button>
    </div>
</div>

<script>
function loadUserDetails(userId) {
    fetch(`/api/users/${userId}`)
        .then(response => response.json())
        .then(user => {
            const userElement = document.getElementById(`user-${userId}`);
            const detailsHtml = `
                <div class="user-details">
                    <p><strong>전화번호:</strong> ${user.phone}</p>
                    <p><strong>주소:</strong> ${user.address}</p>
                    <p><strong>가입일:</strong> ${user.createdAt}</p>
                </div>
            `;
            userElement.insertAdjacentHTML('beforeend', detailsHtml);
        });
}
</script>
```

## 조건문 최적화

### 서버측 필터링

```html
<!-- Bad Practice: 템플릿에서 필터링 -->
<table>
    <tr th:each="user : ${allUsers}">
        <div th:if="${user.active}">
            <td th:text="${user.name}">이름</td>
            <td th:text="${user.email}">이메일</td>
        </div>
    </tr>
</table>

<!-- Good Practice: 서버측에서 필터링 -->
<table>
    <tr th:each="user : ${activeUsers}">
        <td th:text="${user.name}">이름</td>
        <td th:text="${user.email}">이메일</td>
    </tr>
</table>
```

### 간단한 조건문

```html
<!-- Bad Practice: 복잡한 중첩 조건 -->
<div th:if="${user.active}">
    <div th:if="${user.age >= 18}">
        <div th:if="${user.role == 'ADMIN'}">
            <span>성인 관리자</span>
        </div>
        <div th:unless="${user.role == 'ADMIN'}">
            <span>성인 사용자</span>
        </div>
    </div>
    <div th:unless="${user.age >= 18}">
        <span>미성년자</span>
    </div>
</div>

<!-- Good Practice: 간단한 조건문 -->
<div th:switch="${user.role}">
    <span th:case="'ADMIN'" th:text="${user.age >= 18} ? '성인 관리자' : '미성년자 관리자'">역할</span>
    <span th:case="*" th:text="${user.age >= 18} ? '성인 사용자' : '미성년자'">역할</span>
</div>
```

## 프래그먼트 최적화

### 재사용 가능한 프래그먼트

```html
<!-- fragments/user-card.html -->
<div th:fragment="userCard(user, showDetails)" class="card">
    <div class="card-header">
        <h5 th:text="${user.name}">이름</h5>
    </div>
    <div class="card-body">
        <p th:text="${user.email}">이메일</p>
        
        <!-- 조건부 상세 정보 -->
        <div th:if="${showDetails}">
            <p th:text="${user.phone}">전화번호</p>
            <p th:text="${user.address}">주소</p>
        </div>
    </div>
</div>
```

### 프래그먼트 캐싱

```html
<!-- 캐시 가능한 프래그먼트 -->
<div th:fragment="userStats" th:with="stats=${userService.getStats()}">
    <div class="stats-container">
        <div class="stat-item">
            <span th:text="${stats.totalUsers}">총 사용자</span>
        </div>
        <div class="stat-item">
            <span th:text="${stats.activeUsers}">활성 사용자</span>
        </div>
    </div>
</div>
```

## 정적 리소스 최적화

### 리소스 버전 관리

```html
<!-- 버전 관리된 CSS -->
<link rel="stylesheet" th:href="@{/css/style.css(v=${app.version})}">

<!-- 버전 관리된 JavaScript -->
<script th:src="@{/js/app.js(v=${app.version})}"></script>
```

### CDN과 로컬 리소스 결합

```html
<!-- 로컬 리소스가 있으면 사용하고 없으면 CDN 사용 -->
<link rel="stylesheet" th:href="@{/css/bootstrap.min.css}" 
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">

<script th:src="@{/js/jquery.min.js}" 
        src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
```

## 성능 모니터링

### 렌더링 시간 측정

```html
<!-- 템플릿 렌더링 시작 시간 기록 -->
<div th:with="startTime=${System.currentTimeMillis()}">
    <!-- 템플릿 내용 -->
    
    <!-- 렌더링 시간 표시 (개발 환경에서만) -->
    <div th:if="${@environment.getProperty('spring.profiles.active') == 'dev'}" 
         class="debug-info">
        렌더링 시간: <span th:text="${System.currentTimeMillis() - startTime}">0</span>ms
    </div>
</div>
```

## Best Practice

1. **th:with 활용**: 반복적인 연산은 th:with를 사용하여 한 번만 수행하세요.
2. **페이징 구현**: 대용량 데이터는 페이징 처리를 구현하세요.
3. **지연 로딩**: 필요한 시점에 데이터를 로드하여 초기 로딩 속도를 향상시키세요.
4. **템플릿 캐싱**: 운영 환경에서는 템플릿 캐싱을 활성화하세요.
5. **서버측 필터링**: 템플릿에서 필터링하지 말고 서버측에서 데이터를 필터링하세요.

## Bad Practice

1. **대용량 데이터 전체 로딩**: 메모리 문제와 성능 저하를 유발합니다.
2. **반복적인 연산**: 불필요한 연산 반복은 성능을 저하시킵니다.
3. **복잡한 조건문**: 중첩된 조건문은 처리 속도를 느리게 합니다.
4. **캐시 비활성화**: 운영 환경에서 캐시를 비활성화하면 성능이 크게 저하됩니다.

## 다음 장에서는

다음 장에서는 실전 예제와 팁에 대해 알아보겠습니다.