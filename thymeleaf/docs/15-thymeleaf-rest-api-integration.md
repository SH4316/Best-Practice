# 15. Thymeleaf와 REST API 통합

## REST API와 Thymeleaf 통합 소개

현대 웹 애플리케이션에서는 프론트엔드와 백엔드를 분리하고 REST API를 통해 통신하는 구조가 일반적입니다. 그러나 Thymeleaf와 REST API를 효과적으로 통합하면 서버 측 렌더링의 장점과 클라이언트 측 동적 업데이트의 장점을 모두 활용할 수 있습니다.

## 통합 방식

### 1. 하이브리드 접근 방식

하이브리드 접근 방식은 초기 페이지 로드는 서버 측에서 처리하고, 이후에는 AJAX를 통해 REST API를 호출하여 동적으로 콘텐츠를 업데이트하는 방식입니다.

```java
// 컨트롤러 예제
@Controller
@RequestMapping("/api")
public class ApiController {

    @Autowired
    private UserService userService;

    // 전체 페이지 렌더링 (SSR)
    @GetMapping("/users")
    public String usersPage(Model model) {
        model.addAttribute("users", userService.findAll());
        return "users/list";
    }

    // 사용자 데이터 API (REST)
    @GetMapping("/users/data")
    @ResponseBody
    public List<User> getUsersData() {
        return userService.findAll();
    }

    // 특정 사용자 데이터 API (REST)
    @GetMapping("/users/{id}/data")
    @ResponseBody
    public ResponseEntity<User> getUserData(@PathVariable Long id) {
        Optional<User> user = userService.findById(id);
        return user.map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
}
```

### 2. HTMX와 Thymeleaf 통합

HTMX는 서버 측 템플릿을 클라이언트 측에서 동적으로 로드할 수 있게 해주는 라이브러리입니다.

```html
<!-- HTMX와 Thymeleaf 통합 예제 -->
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>HTMX + Thymeleaf</title>
    <script src="https://unpkg.com/htmx.org@1.9.6"></script>
    <link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">
</head>
<body>
    <div class="container my-4">
        <h1>HTMX + Thymeleaf Examples</h1>
        
        <!-- HTMX를 사용한 동적 콘텐츠 로드 -->
        <button hx-get="/api/users/table" 
                hx-target="#userTable" 
                class="btn btn-primary">
            사용자 목록 로드
        </button>
        
        <!-- 결과를 표시할 컨테이너 -->
        <div id="userTable" class="mt-3">
            <p>사용자 목록을 로드하려면 버튼을 클릭하세요.</p>
        </div>
        
        <!-- 폼 제출 후 동적 업데이트 -->
        <form hx-post="/api/users/create" 
              hx-target="#userFormResult" 
              hx-swap="innerHTML">
            <div class="mb-3">
                <label for="name" class="form-label">이름</label>
                <input type="text" class="form-control" id="name" name="name" required>
            </div>
            <div class="mb-3">
                <label for="email" class="form-label">이메일</label>
                <input type="email" class="form-control" id="email" name="email" required>
            </div>
            <button type="submit" class="btn btn-success">사용자 생성</button>
        </form>
        
        <div id="userFormResult" class="mt-3"></div>
    </div>
</body>
</html>
```

### 3. Spring MVC와 REST API 결합

```java
// 통합 컨트롤러 예제
@Controller
@RequestMapping("/users")
public class UserApiController {

    @Autowired
    private UserService userService;

    // 페이지 렌더링 엔드포인트
    @GetMapping
    public String index(Model model) {
        model.addAttribute("users", userService.findAll());
        return "users/index";
    }

    // REST API 엔드포인트
    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public List<User> getAllUsers() {
        return userService.findAll();
    }

    @GetMapping(value = "/{id}", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public ResponseEntity<User> getUserById(@PathVariable Long id) {
        Optional<User> user = userService.findById(id);
        return user.map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    // 부분 템플릿 렌더링 엔드포인트
    @GetMapping(value = "/table", produces = MediaType.TEXT_HTML_VALUE)
    public String getUserTable(Model model) {
        model.addAttribute("users", userService.findAll());
        return "users/_table :: table";
    }

    @PostMapping(value = "/create", produces = MediaType.TEXT_HTML_VALUE)
    public String createUser(@ModelAttribute User user, Model model) {
        User savedUser = userService.save(user);
        model.addAttribute("user", savedUser);
        model.addAttribute("message", "사용자가 성공적으로 생성되었습니다.");
        return "users/_form-result :: result";
    }
}
```

## 프래그먼트 기반 AJAX

### 부분 템플릿 프래그먼트

```html
<!-- users/_table.html -->
<html xmlns:th="http://www.thymeleaf.org" th:fragment="table">
<table class="table table-striped">
    <thead>
        <tr>
            <th>ID</th>
            <th>이름</th>
            <th>이메일</th>
            <th>역할</th>
            <th>작업</th>
        </tr>
    </thead>
    <tbody>
        <tr th:each="user : ${users}">
            <td th:text="${user.id}">ID</td>
            <td th:text="${user.name}">이름</td>
            <td th:text="${user.email}">이메일</td>
            <td th:text="${user.role}">역할</td>
            <td>
                <button class="btn btn-sm btn-primary" 
                        th:attr="onclick='loadUserDetails(' + ${user.id} + ')'">
                    상세
                </button>
            </td>
        </tr>
    </tbody>
</table>
</html>
```

```html
<!-- users/_form-result.html -->
<html xmlns:th="http://www.thymeleaf.org" th:fragment="result">
<div th:if="${user}" class="alert alert-success">
    <strong th:text="${user.name}">사용자</strong>님이 성공적으로 생성되었습니다.
</div>
<div th:if="${message}" class="alert alert-info" th:text="${message}">메시지</div>
</html>
```

### JavaScript와의 통합

```javascript
// 사용자 상세 정보 로드
function loadUserDetails(userId) {
    fetch(`/api/users/${userId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(user => {
            // 사용자 정보를 모달에 표시
            document.getElementById('modalUserId').textContent = user.id;
            document.getElementById('modalUserName').textContent = user.name;
            document.getElementById('modalUserEmail').textContent = user.email;
            document.getElementById('modalUserRole').textContent = user.role;
            
            // 모달 표시
            const userModal = new bootstrap.Modal(document.getElementById('userModal'));
            userModal.show();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('사용자 정보를 가져오는 데 실패했습니다.');
        });
}

// 사용자 목록 새로고침
function refreshUserTable() {
    fetch('/api/users/table')
        .then(response => response.text())
        .then(html => {
            document.getElementById('userTable').innerHTML = html;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('사용자 목록을 새로고치는 데 실패했습니다.');
        });
}

// 사용자 생성 폼 제출
function createUserForm(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const userData = Object.fromEntries(formData.entries());
    
    fetch('/api/users/create', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams(userData)
    })
    .then(response => response.text())
    .then(html => {
        document.getElementById('userFormResult').innerHTML = html;
        // 폼 초기화
        event.target.reset();
        // 사용자 목록 새로고침
        refreshUserTable();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('사용자 생성에 실패했습니다.');
    });
}
```

## 실시간 데이터 업데이트

### WebSocket과 Thymeleaf 통합

```java
// WebSocket 설정
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(new UserUpdatesHandler(), "/ws/user-updates")
                .setAllowedOrigins("*");
    }
}

// WebSocket 핸들러
public class UserUpdatesHandler extends TextWebSocketHandler {

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        // 연결 설정 후 처리
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        // 메시지 수신 후 처리
        String payload = message.getPayload();
        // 모든 클라이언트에게 메시지 브로드캐스트
        broadcastUpdate(payload);
    }

    private void broadcastUpdate(String message) {
        // 모든 연결된 클라이언트에게 메시지 전송
    }
}
```

```html
<!-- WebSocket과 JavaScript 통합 -->
<script>
// WebSocket 연결
const socket = new WebSocket('ws://localhost:8080/ws/user-updates');

socket.onopen = function(event) {
    console.log('WebSocket 연결이 열렸습니다.');
};

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'USER_UPDATE') {
        // 사용자 정보 업데이트
        updateUserInTable(data.user);
    } else if (data.type === 'USER_LIST_REFRESH') {
        // 사용자 목록 새로고침
        refreshUserTable();
    }
};

socket.onclose = function(event) {
    console.log('WebSocket 연결이 닫혔습니다.');
};

socket.onerror = function(error) {
    console.error('WebSocket 오류:', error);
};

// 테이블에서 사용자 정보 업데이트
function updateUserInTable(user) {
    const userRow = document.getElementById(`user-row-${user.id}`);
    if (userRow) {
        userRow.querySelector('.user-name').textContent = user.name;
        userRow.querySelector('.user-email').textContent = user.email;
        userRow.querySelector('.user-role').textContent = user.role;
        
        // 업데이트 효과
        userRow.classList.add('table-success');
        setTimeout(() => {
            userRow.classList.remove('table-success');
        }, 2000);
    }
}
</script>
```

## Server-Sent Events (SSE)와 Thymeleaf

### SSE 컨트롤러

```java
// SSE 컨트롤러
@RestController
@RequestMapping("/sse")
public class SSEController {

    @GetMapping("/user-updates")
    public SseEmitter handleUserUpdates() {
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);
        
        // 비동기 처리를 위한 스레드 풀
        ExecutorService executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            try {
                // 초기 데이터 전송
                emitter.send(SseEmitter.event().name("init").data("Connection established"));
                
                // 주기적으로 데이터 전송 (예: 5초마다)
                for (int i = 0; i < 10; i++) {
                    // 사용자 데이터 가져오기
                    List<User> users = userService.findAll();
                    
                    // 데이터 전송
                    emitter.send(SseEmitter.event()
                            .name("user-update")
                            .data(users));
                    
                    // 5초 대기
                    Thread.sleep(5000);
                }
                
                // 완료 신호 전송
                emitter.send(SseEmitter.event().name("complete").data("Stream complete"));
                emitter.complete();
            } catch (Exception e) {
                emitter.completeWithError(e);
            } finally {
                executor.shutdown();
            }
        });
        
        return emitter;
    }
}
```

### SSE와 JavaScript 통합

```html
<!-- SSE와 JavaScript 통합 -->
<script>
// EventSource 연결
const eventSource = new EventSource('/sse/user-updates');

eventSource.onopen = function(event) {
    console.log('SSE 연결이 열렸습니다.');
};

eventSource.addEventListener('init', function(event) {
    console.log('초기화:', event.data);
});

eventSource.addEventListener('user-update', function(event) {
    const users = JSON.parse(event.data);
    updateUserTable(users);
});

eventSource.addEventListener('complete', function(event) {
    console.log('스트림 완료:', event.data);
    eventSource.close();
});

eventSource.onerror = function(error) {
    console.error('SSE 오류:', error);
    eventSource.close();
});

// 사용자 테이블 업데이트
function updateUserTable(users) {
    const tbody = document.querySelector('#userTable tbody');
    tbody.innerHTML = '';
    
    users.forEach(user => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${user.id}</td>
            <td>${user.name}</td>
            <td>${user.email}</td>
            <td>${user.role}</td>
            <td>
                <button class="btn btn-sm btn-info" onclick="showUserDetails(${user.id})">상세</button>
            </td>
        `;
        tbody.appendChild(row);
    });
    
    // 업데이트 시간 표시
    const updateTime = document.getElementById('updateTime');
    if (updateTime) {
        updateTime.textContent = new Date().toLocaleTimeString();
    }
}
</script>
```

## 성능 최적화

### 캐싱 전략

```java
// 캐싱 설정
@Configuration
@EnableCaching
public class CacheConfig {

    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager();
        cacheManager.setCaffeine(Caffeine.newBuilder()
                .expireAfterWrite(10, TimeUnit.MINUTES)
                .maximumSize(100));
        return cacheManager;
    }
}

// 캐시된 API 엔드포인트
@RestController
@RequestMapping("/api/cache")
public class CachedApiController {

    @Autowired
    private CacheManager cacheManager;

    @GetMapping("/users")
    @Cacheable(value = "users", key = "#root.methodName")
    public List<User> getCachedUsers() {
        return userService.findAll();
    }

    @GetMapping("/users/{id}")
    @Cacheable(value = "user", key = "#id")
    public ResponseEntity<User> getCachedUser(@PathVariable Long id) {
        Optional<User> user = userService.findById(id);
        return user.map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/users/{id}/invalidate")
    public String invalidateUserCache(@PathVariable Long id) {
        Cache userCache = cacheManager.getCache("user");
        if (userCache != null) {
            userCache.evict(id);
        }
        return "Cache invalidated for user " + id;
    }
}
```

### 데이터 프리페칭

```html
<!-- 데이터 프리페칭 -->
<script>
// 페이지 로드 시 미리 데이터 로드
document.addEventListener('DOMContentLoaded', function() {
    // 사용자 데이터 미리 로드
    prefetchUserData();
    
    // 다른 필요한 데이터 미리 로드
    prefetchCommonData();
});

function prefetchUserData() {
    fetch('/api/users')
        .then(response => response.json())
        .then(users => {
            // 전역 변수에 데이터 저장
            window.userData = users;
            console.log('사용자 데이터를 미리 로드했습니다:', users.length, '명');
        })
        .catch(error => {
            console.error('사용자 데이터 미리 로드 실패:', error);
        });
}

function prefetchCommonData() {
    // 공통 데이터 미리 로드
    const commonDataUrls = [
        '/api/roles',
        '/api/departments',
        '/api/statuses'
    ];
    
    Promise.all(commonDataUrls.map(url => 
        fetch(url).then(response => response.json())
    ))
    .then(data => {
        // 전역 변수에 데이터 저장
        window.commonData = {
            roles: data[0],
            departments: data[1],
            statuses: data[2]
        };
        console.log('공통 데이터를 미리 로드했습니다.');
    })
    .catch(error => {
        console.error('공통 데이터 미리 로드 실패:', error);
    });
}

// 미리 로드된 데이터 사용
function getUserFromCache(userId) {
    if (window.userData) {
        return window.userData.find(user => user.id === userId);
    }
    return null;
}
</script>
```

## Best Practice

1. **하이브리드 접근**: 초기 로드는 SSR, 이후 업데이트는 AJAX를 사용하는 하이브리드 접근 방식을 고려하세요.
2. **프래그먼트 활용**: 부분 템플릿 프래그먼트를 활용하여 불필요한 전체 페이지 로드를 피하세요.
3. **에러 처리**: API 호출 시 적절한 에러 처리를 구현하세요.
4. **성능 최적화**: 캐싱과 데이터 프리페칭을 통해 성능을 최적화하세요.
5. **보안 고려**: API 엔드포인트에 적절한 보안 조치를 적용하세요.

## Bad Practice

1. **전체 페이지 로드**: 작은 변경에도 전체 페이지를 로드하는 것은 비효율적입니다.
2. **에러 처리 부재**: API 호출 시 에러 처리를 하지 않으면 사용자 경험이 저하됩니다.
3. **보안 무시**: API 엔드포인트를 보호하지 않으면 보안 위험이 발생할 수 있습니다.
4. **성능 고려 부족**: 캐싱이나 최적화 없이 많은 API 호출을 하면 성능이 저하됩니다.

## 다음 장에서는

다음 장에서는 Thymeleaf 테스트 전략에 대해 알아보겠습니다.