# 12. 실전 예제와 팁

## 실전 예제

이 장에서는 실제 프로젝트에서 자주 사용되는 Thymeleaf 패턴과 예제를 살펴보겠습니다.

### 사용자 관리 시스템

#### 사용자 목록 페이지

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:layout="http://www.ultraq.net.nz/thymeleaf/layout"
      layout:decorate="~{layouts/default}">

<head>
    <title th:text="#{user.list.title}">사용자 목록</title>
</head>

<th:block layout:fragment="styles">
    <style>
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
        }
        .status-badge {
            font-size: 0.8rem;
        }
        .action-buttons .btn {
            padding: 0.25rem 0.5rem;
            font-size: 0.8rem;
        }
    </style>
</th:block>

<div layout:fragment="content">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h1 th:text="#{user.list.title}">사용자 목록</h1>
        <a th:href="@{/users/create}" class="btn btn-primary">
            <i class="bi bi-plus-circle"></i> <span th:text="#{user.create}">사용자 생성</span>
        </a>
    </div>
    
    <!-- 검색 필터 -->
    <div class="card mb-4">
        <div class="card-body">
            <form th:action="@{/users}" method="get" class="row g-3">
                <div class="col-md-4">
                    <input type="text" class="form-control" name="search" 
                           th:value="${param.search}" 
                           th:placeholder="#{user.search.placeholder}">
                </div>
                <div class="col-md-3">
                    <select class="form-select" name="role">
                        <option value="" th:text="#{user.role.all}">모든 역할</option>
                        <option value="USER" th:selected="${param.role == 'USER'}" th:text="#{role.user}">사용자</option>
                        <option value="ADMIN" th:selected="${param.role == 'ADMIN'}" th:text="#{role.admin}">관리자</option>
                        <option value="MODERATOR" th:selected="${param.role == 'MODERATOR'}" th:text="#{role.moderator}">중재자</option>
                    </select>
                </div>
                <div class="col-md-3">
                    <select class="form-select" name="status">
                        <option value="" th:text="#{user.status.all}">모든 상태</option>
                        <option value="ACTIVE" th:selected="${param.status == 'ACTIVE'}" th:text="#{user.active}">활성</option>
                        <option value="INACTIVE" th:selected="${param.status == 'INACTIVE'}" th:text="#{user.inactive}">비활성</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <button type="submit" class="btn btn-outline-primary w-100">
                        <i class="bi bi-search"></i> <span th:text="#{button.search}">검색</span>
                    </button>
                </div>
            </form>
        </div>
    </div>
    
    <!-- 사용자 목록 -->
    <div class="card">
        <div class="card-body">
            <div th:if="${#lists.isEmpty(users.content)}" class="text-center py-4">
                <i class="bi bi-people fs-1 text-muted"></i>
                <p class="mt-2" th:text="#{message.no.data}">데이터가 없습니다</p>
            </div>
            
            <div th:unless="${#lists.isEmpty(users.content)}">
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th></th>
                                <th>
                                    <a th:href="@{/users(sort='name', order=${sort == 'name' and order == 'asc' ? 'desc' : 'asc'})}" 
                                       class="text-decoration-none">
                                        <span th:text="#{user.name}">이름</span>
                                        <i th:if="${sort == 'name'}" 
                                           th:class="${order == 'asc'} ? 'bi bi-arrow-up' : 'bi bi-arrow-down'"></i>
                                    </a>
                                </th>
                                <th th:text="#{user.email}">이메일</th>
                                <th th:text="#{user.role}">역할</th>
                                <th th:text="#{user.status}">상태</th>
                                <th th:text="#{user.createdAt}">가입일</th>
                                <th th:text="#{table.actions}">작업</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr th:each="user : ${users.content}">
                                <td>
                                    <img th:src="${user.avatar ?: @{/images/default-avatar.png}}" 
                                         alt="Avatar" class="user-avatar">
                                </td>
                                <td>
                                    <div class="d-flex align-items-center">
                                        <div>
                                            <div th:text="${user.name}" class="fw-bold">이름</div>
                                            <small class="text-muted" th:text="'@' + ${user.username}">@username</small>
                                        </div>
                                    </div>
                                </td>
                                <td th:text="${user.email}">이메일</td>
                                <td>
                                    <span th:switch="${user.role}" class="badge status-badge">
                                        <span th:case="'ADMIN'" class="bg-danger" th:text="#{role.admin}">관리자</span>
                                        <span th:case="'MODERATOR'" class="bg-warning" th:text="#{role.moderator}">중재자</span>
                                        <span th:case="'USER'" class="bg-primary" th:text="#{role.user}">사용자</span>
                                        <span th:case="*" class="bg-secondary" th:text="${user.role}">기타</span>
                                    </span>
                                </td>
                                <td>
                                    <span th:class="${user.active} ? 'badge bg-success' : 'badge bg-secondary'"
                                          th:text="${user.active} ? #{user.active} : #{user.inactive}">상태</span>
                                </td>
                                <td th:text="${#dates.format(user.createdAt, #{date.format})}">가입일</td>
                                <td>
                                    <div class="action-buttons">
                                        <a th:href="@{/users/{id}(id=${user.id})}" 
                                           class="btn btn-sm btn-outline-primary" 
                                           th:title="#{button.view}">
                                            <i class="bi bi-eye"></i>
                                        </a>
                                        <a th:href="@{/users/{id}/edit(id=${user.id})}" 
                                           class="btn btn-sm btn-outline-secondary" 
                                           th:title="#{button.edit}">
                                            <i class="bi bi-pencil"></i>
                                        </a>
                                        <button th:if="${#authorization.expression('hasRole(''ADMIN'')')}" 
                                                class="btn btn-sm btn-outline-danger" 
                                                th:attr="data-bs-toggle='modal', data-bs-target='#deleteModal', data-user-id=${user.id}, data-user-name=${user.name}"
                                                th:title="#{button.delete}">
                                            <i class="bi bi-trash"></i>
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <!-- 페이지네이션 -->
                <div th:replace="~{fragments/pagination :: pagination(${users.number + 1}, ${users.totalPages}, '/users')}"></div>
            </div>
        </div>
    </div>
</div>

<!-- 삭제 확인 모달 -->
<div class="modal fade" id="deleteModal" tabindex="-1">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" th:text="#{user.delete.confirm}">사용자 삭제 확인</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <p th:text="#{user.delete.message}">정말로 삭제하시겠습니까?</p>
                <p><strong id="deleteUserName"></strong></p>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal" th:text="#{button.cancel}">취소</button>
                <form id="deleteForm" method="post" style="display: inline;">
                    <button type="submit" class="btn btn-danger" th:text="#{button.delete}">삭제</button>
                </form>
            </div>
        </div>
    </div>
</div>

</th:block>

<th:block layout:fragment="scripts">
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const deleteModal = document.getElementById('deleteModal');
            deleteModal.addEventListener('show.bs.modal', function(event) {
                const button = event.relatedTarget;
                const userId = button.getAttribute('data-user-id');
                const userName = button.getAttribute('data-user-name');
                
                document.getElementById('deleteUserName').textContent = userName;
                document.getElementById('deleteForm').action = '/users/' + userId + '/delete';
            });
        });
    </script>
</th:block>

</html>
```

### 대시보드 페이지

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:layout="http://www.ultraq.net.nz/thymeleaf/layout"
      layout:decorate="~{layouts/default}">

<head>
    <title th:text="#{dashboard.title}">대시보드</title>
</head>

<th:block layout:fragment="styles">
    <style>
        .stat-card {
            transition: transform 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .chart-container {
            height: 300px;
        }
    </style>
</th:block>

<div layout:fragment="content">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h1 th:text="#{dashboard.title}">대시보드</h1>
        <div class="btn-group">
            <button class="btn btn-outline-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown">
                <i class="bi bi-calendar"></i> <span th:text="${period}">기간</span>
            </button>
            <ul class="dropdown-menu">
                <li><a class="dropdown-item" th:href="@{/dashboard(period='today')}" th:text="#{period.today}">오늘</a></li>
                <li><a class="dropdown-item" th:href="@{/dashboard(period='week')}" th:text="#{period.week}">이번 주</a></li>
                <li><a class="dropdown-item" th:href="@{/dashboard(period='month')}" th:text="#{period.month}">이번 달</a></li>
                <li><a class="dropdown-item" th:href="@{/dashboard(period='year')}" th:text="#{period.year}">올해</a></li>
            </ul>
        </div>
    </div>
    
    <!-- 통계 카드 -->
    <div th:with="stats=${dashboardService.getStats(period)}" class="row mb-4">
        <div class="col-md-3 mb-3">
            <div class="card stat-card h-100">
                <div class="card-body text-center">
                    <i class="bi bi-people fs-1 text-primary"></i>
                    <h5 class="card-title mt-2" th:text="#{stats.users.total}">총 사용자</h5>
                    <p class="card-text display-4" th:text="${#numbers.formatInteger(stats.totalUsers, 3)}">0</p>
                    <small class="text-success" th:if="${stats.userGrowth > 0}">
                        <i class="bi bi-arrow-up"></i> <span th:text="${stats.userGrowth + '%'}">0%</span>
                    </small>
                    <small class="text-danger" th:if="${stats.userGrowth < 0}">
                        <i class="bi bi-arrow-down"></i> <span th:text="${stats.userGrowth + '%'}">0%</span>
                    </small>
                </div>
            </div>
        </div>
        
        <div class="col-md-3 mb-3">
            <div class="card stat-card h-100">
                <div class="card-body text-center">
                    <i class="bi bi-person-check fs-1 text-success"></i>
                    <h5 class="card-title mt-2" th:text="#{stats.users.active}">활성 사용자</h5>
                    <p class="card-text display-4" th:text="${#numbers.formatInteger(stats.activeUsers, 3)}">0</p>
                    <small class="text-muted" th:text="#{stats.users.active.ratio} + '%'">0%</small>
                </div>
            </div>
        </div>
        
        <div class="col-md-3 mb-3">
            <div class="card stat-card h-100">
                <div class="card-body text-center">
                    <i class="bi bi-file-text fs-1 text-info"></i>
                    <h5 class="card-title mt-2" th:text="#{stats.posts.total}">총 게시물</h5>
                    <p class="card-text display-4" th:text="${#numbers.formatInteger(stats.totalPosts, 3)}">0</p>
                    <small class="text-success" th:if="${stats.postGrowth > 0}">
                        <i class="bi bi-arrow-up"></i> <span th:text="${stats.postGrowth + '%'}">0%</span>
                    </small>
                </div>
            </div>
        </div>
        
        <div class="col-md-3 mb-3">
            <div class="card stat-card h-100">
                <div class="card-body text-center">
                    <i class="bi bi-eye fs-1 text-warning"></i>
                    <h5 class="card-title mt-2" th:text="#{stats.views.total}">조회수</h5>
                    <p class="card-text display-4" th:text="${#numbers.formatDecimal(stats.totalViews, 1, 'COMMA')}">0</p>
                    <small class="text-muted" th:text="#{stats.views.average} + ' ' + #{stats.views.per.day}">0/일</small>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 차트 -->
    <div class="row">
        <div class="col-md-8 mb-4">
            <div class="card">
                <div class="card-header">
                    <h5 th:text="#{chart.users.growth}">사용자 증가 추이</h5>
                </div>
                <div class="card-body">
                    <div class="chart-container">
                        <canvas id="userGrowthChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-4 mb-4">
            <div class="card">
                <div class="card-header">
                    <h5 th:text="#{chart.users.by.role}">역할별 사용자</h5>
                </div>
                <div class="card-body">
                    <div class="chart-container">
                        <canvas id="userRoleChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 최근 활동 -->
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5 th:text="#{activity.recent}">최근 활동</h5>
                    <a th:href="@{/activities}" class="btn btn-sm btn-outline-primary" th:text="#{button.view.all}">전체 보기</a>
                </div>
                <div class="card-body">
                    <div th:if="${#lists.isEmpty(recentActivities)}" class="text-center py-4">
                        <i class="bi bi-activity fs-1 text-muted"></i>
                        <p class="mt-2" th:text="#{activity.no.recent}">최근 활동이 없습니다</p>
                    </div>
                    
                    <div th:unless="${#lists.isEmpty(recentActivities)}" class="timeline">
                        <div th:each="activity, stat : ${recentActivities}" class="timeline-item">
                            <div class="timeline-marker" th:class="${activity.type}">
                                <i th:class="'bi bi-' + ${activity.icon}"></i>
                            </div>
                            <div class="timeline-content">
                                <div class="d-flex justify-content-between">
                                    <h6 th:text="${activity.title}">활동 제목</h6>
                                    <small class="text-muted" th:text="${#dates.format(activity.createdAt, 'HH:mm')}">시간</small>
                                </div>
                                <p th:text="${activity.description}">활동 설명</p>
                                <small class="text-muted">
                                    <span th:text="${activity.user.name}">사용자</span> • 
                                    <span th:text="${#dates.format(activity.createdAt, #{date.time.format})}">날짜</span>
                                </small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

</th:block>

<th:block layout:fragment="scripts">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script th:inline="javascript">
        document.addEventListener('DOMContentLoaded', function() {
            // 사용자 증가 추이 차트
            const userGrowthCtx = document.getElementById('userGrowthChart').getContext('2d');
            new Chart(userGrowthCtx, {
                type: 'line',
                data: {
                    labels: /*[[${userGrowthLabels}]]*/ [],
                    datasets: [{
                        label: '사용자 수',
                        data: /*[[${userGrowthData}]]*/ [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
            
            // 역할별 사용자 차트
            const userRoleCtx = document.getElementById('userRoleChart').getContext('2d');
            new Chart(userRoleCtx, {
                type: 'doughnut',
                data: {
                    labels: /*[[${userRoleLabels}]]*/ [],
                    datasets: [{
                        data: /*[[${userRoleData}]]*/ [],
                        backgroundColor: [
                            'rgb(255, 99, 132)',
                            'rgb(54, 162, 235)',
                            'rgb(255, 205, 86)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        });
    </script>
</th:block>

</html>
```

## 실용 팁

### 1. 조건부 CSS 클래스

```html
<!-- 여러 조건에 따른 클래스 적용 -->
<div th:class="${user.active} ? 'card border-success' : 'card border-secondary'">
    <!-- 내용 -->
</div>

<!-- th:classappend를 사용한 클래스 추가 -->
<div class="card" th:classappend="${user.active} ? 'border-success' : 'border-secondary'">
    <!-- 내용 -->
</div>
```

### 2. 동적 속성 생성

```html
<!-- 여러 속성 동적 생성 -->
<button th:attr="data-id=${user.id}, data-name=${user.name}, data-role=${user.role}">
    버튼
</button>
```

### 3. 데이터 형식 변환

```html
<!-- 날짜 형식 변환 -->
<span th:text="${#dates.format(user.createdAt, 'yyyy년 MM월 dd일')}">날짜</span>

<!-- 숫자 형식 변환 -->
<span th:text="${#numbers.formatDecimal(user.salary, 1, 2)}">급여</span>

<!-- 파일 크기 형식 -->
<span th:text="${#numbers.formatDecimal(fileSize / 1024 / 1024, 1, 2)} + ' MB'">파일 크기</span>
```

### 4. 복잡한 로직 처리

```html
<!-- th:with를 사용한 복잡한 로직 -->
<div th:with="fullName=${user.lastName + ' ' + user.firstName}, 
                  displayName=${user.nickname ?: fullName},
                  userAge=${#dates.calculateAge(user.birthDate)},
                  isAdult=${userAge >= 18}">
    <h4 th:text="${displayName}">표시 이름</h4>
    <p th:text="${userAge + '세'}">나이</p>
    <span th:class="${isAdult} ? 'badge bg-success' : 'badge bg-warning'"
          th:text="${isAdult} ? '성인' : '미성년자'">상태</span>
</div>
```

### 5. 유틸리티 객체 활용

```html
<!-- 리스트 유틸리티 -->
<span th:if="${#lists.isEmpty(users)}">사용자가 없습니다</span>
<span th:if="${not #lists.isEmpty(users)}" th:text="${#lists.size(users)} + '명의 사용자'"></span>

<!-- 문자열 유틸리티 -->
<span th:text="${#strings.abbreviate(user.description, 50)}">설명</span>
<span th:text="${#strings.defaultString(user.phone, '번호 없음')}">전화번호</span>

<!-- 배열 유틸리티 -->
<span th:if="${#arrays.contains(user.roles, 'ADMIN')}">관리자</span>
```

## Best Practice 요약

1. **의미 있는 변수명**: th:with를 사용할 때 의미 있는 변수명을 사용하세요.
2. **재사용 가능한 프래그먼트**: 공통 부분은 프래그먼트로 분리하세요.
3. **조건부 렌더링**: 불필요한 HTML 생성을 피하기 위해 조건부 렌더링을 사용하세요.
4. **유틸리티 객체 활용**: Thymeleaf 유틸리티 객체를 적극 활용하세요.
5. **보안**: 항상 보안을 고려하여 템플릿을 작성하세요.
6. **성능**: 불필요한 연산을 피하고 페이징을 구현하세요.

## 마무리

이제 Thymeleaf의 기본 기능부터 고급 기법까지 모두 학습했습니다. 실제 프로젝트에서 이 예제와 팁을 활용하여 효율적이고 안전한 템플릿을 작성하세요.

Thymeleaf는 매우 강력하고 유연한 템플릿 엔진입니다. 계속해서 공식 문서를 참고하고 새로운 기법을 학습하여 실력을 향상시키세요.

## 추가 학습 자료

- [Thymeleaf 공식 문서](https://www.thymeleaf.org/doc/tutorials/3.0/usingthymeleaf.html)
- [Spring Boot Thymeleaf 가이드](https://spring.io/guides/gs/serving-web-content/)
- [Thymeleaf + Spring Security](https://www.thymeleaf.org/doc/tutorials/3.0/thymeleafspring.html#security)

감사합니다!