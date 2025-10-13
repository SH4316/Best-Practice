# 16. Thymeleaf 테스트 전략

## 테스트의 중요성

Thymeleaf 템플릿은 서버 측에서 HTML을 생성하는 중요한 구성 요소입니다. 따라서 템플릿을 철저히 테스트하는 것은 애플리케이션의 안정성과 품질을 보장하는 데 매우 중요합니다. 이 장에서는 Thymeleaf 템플릿을 효과적으로 테스트하는 다양한 전략과 도구에 대해 알아보겠습니다.

## 테스트 유형

### 1. 단위 테스트 (Unit Testing)

단위 테스트는 개별 템플릿이나 컴포넌트를 독립적으로 테스트하는 방식입니다.

#### Thymeleaf 단위 테스트 설정

```java
// 테스트 설정 클래스
@SpringBootTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@TestPropertySource(properties = {
    "spring.jpa.hibernate.ddl-auto=none",
    "spring.thymeleaf.cache=false"
})
public class ThymeleafTestConfig {

    @Autowired
    private TemplateEngine templateEngine;

    @Bean
    @Primary
    public TemplateEngine templateEngine() {
        SpringTemplateEngine templateEngine = new SpringTemplateEngine();
        templateEngine.setEnableSpringELCompiler(true);
        templateEngine.setTemplateResolver(templateResolver());
        return templateEngine;
    }

    @Bean
    public ITemplateResolver templateResolver() {
        SpringResourceTemplateResolver templateResolver = new SpringResourceTemplateResolver();
        templateResolver.setApplicationContext(applicationContext);
        templateResolver.setPrefix("classpath:/templates/");
        templateResolver.setSuffix(".html");
        templateResolver.setTemplateMode(TemplateMode.HTML);
        templateResolver.setCacheable(false);
        return templateResolver;
    }
}
```

#### 템플릿 렌더링 단위 테스트

```java
// 템플릿 렌더링 테스트
@ExtendWith(MockitoExtension.class)
public class TemplateRenderingTest {

    @Autowired
    private TemplateEngine templateEngine;

    @Test
    public void testUserListTemplate() {
        // given: 테스트 데이터 준비
        List<User> users = Arrays.asList(
            new User(1L, "홍길동", "hong@example.com", "USER"),
            new User(2L, "김철수", "kim@example.com", "ADMIN")
        );

        Context context = new Context();
        context.setVariable("users", users);
        context.setVariable("title", "사용자 목록");

        // when: 템플릿 렌더링
        String result = templateEngine.process("users/list", context);

        // then: 결과 검증
        assertThat(result).contains("사용자 목록");
        assertThat(result).contains("홍길동");
        assertThat(result).contains("김철수");
        assertThat(result).contains("hong@example.com");
        assertThat(result).contains("kim@example.com");
    }

    @Test
    public void testUserDetailTemplate() {
        // given: 테스트 데이터 준비
        User user = new User(1L, "홍길동", "hong@example.com", "USER");
        user.setCreatedAt(LocalDate.of(2023, 1, 1));
        user.setActive(true);

        Context context = new Context();
        context.setVariable("user", user);

        // when: 템플릿 렌더링
        String result = templateEngine.process("users/detail", context);

        // then: 결과 검증
        assertThat(result).contains("홍길동");
        assertThat(result).contains("hong@example.com");
        assertThat(result).contains("2023-01-01");
        assertThat(result).contains("활성");
    }

    @Test
    public void testEmptyUserListTemplate() {
        // given: 빈 사용자 목록
        List<User> emptyUsers = Collections.emptyList();

        Context context = new Context();
        context.setVariable("users", emptyUsers);

        // when: 템플릿 렌더링
        String result = templateEngine.process("users/list", context);

        // then: 결과 검증
        assertThat(result).contains("사용자가 없습니다");
        assertThat(result).doesNotContain("<table>");
    }
}
```

### 2. 통합 테스트 (Integration Testing)

통합 테스트는 컨트롤러와 템플릿이 함께 동작하는 방식을 테스트합니다.

#### MockMvc를 사용한 통합 테스트

```java
// MockMvc를 사용한 통합 테스트
@AutoConfigureMockMvc
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class UserControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private UserService userService;

    @Test
    public void testUsersListPage() throws Exception {
        // given: 테스트 데이터 준비
        List<User> users = Arrays.asList(
            new User(1L, "홍길동", "hong@example.com", "USER"),
            new User(2L, "김철수", "kim@example.com", "ADMIN")
        );

        given(userService.findAll()).willReturn(users);

        // when/then: 페이지 요청 및 결과 검증
        mockMvc.perform(get("/users"))
                .andExpect(status().isOk())
                .andExpect(view().name("users/list"))
                .andExpect(model().attributeExists("users"))
                .andExpect(model().attribute("users", hasSize(2)))
                .andExpect(content().string(containsString("홍길동")))
                .andExpect(content().string(containsString("김철수")));
    }

    @Test
    public void testUserDetailPage() throws Exception {
        // given: 테스트 데이터 준비
        User user = new User(1L, "홍길동", "hong@example.com", "USER");
        given(userService.findById(1L)).willReturn(Optional.of(user));

        // when/then: 페이지 요청 및 결과 검증
        mockMvc.perform(get("/users/{id}", 1L))
                .andExpect(status().isOk())
                .andExpect(view().name("users/detail"))
                .andExpect(model().attributeExists("user"))
                .andExpect(model().attribute("user", equalTo(user)))
                .andExpect(content().string(containsString("홍길동")))
                .andExpect(content().string(containsString("hong@example.com")));
    }

    @Test
    public void testNonExistentUser() throws Exception {
        // given: 존재하지 않는 사용자
        given(userService.findById(999L)).willReturn(Optional.empty());

        // when/then: 404 에러 검증
        mockMvc.perform(get("/users/{id}", 999L))
                .andExpect(status().isNotFound());
    }
}
```

### 3. E2E 테스트 (End-to-End Testing)

E2E 테스트는 실제 브라우저를 사용하여 전체 사용자 시나리오를 테스트합니다.

#### Selenium을 사용한 E2E 테스트

```java
// Selenium을 사용한 E2E 테스트
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class UserE2ETest {

    @Autowired
    private TestRestTemplate restTemplate;

    private static WebDriver driver;
    private static WebDriverWait wait;

    @BeforeAll
    static void setup() {
        // WebDriver 설정
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver");
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--headless"); // 헤드리스 모드
        options.addArguments("--no-sandbox");
        options.addArguments("--disable-dev-shm-usage");
        
        driver = new ChromeDriver(options);
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    @AfterAll
    static void teardown() {
        if (driver != null) {
            driver.quit();
        }
    }

    @Test
    void testUserListPage() {
        // given: 사용자 목록 페이지로 이동
        driver.get("http://localhost:" + port + "/users");

        // when: 페이지 로드 대기
        wait.until(ExpectedConditions.presenceOfElementLocated(By.cssSelector(".user-table")));

        // then: 페이지 요소 검증
        assertThat(driver.getTitle()).contains("사용자 목록");
        
        // 사용자 목록이 표시되는지 확인
        List<WebElement> userRows = driver.findElements(By.cssSelector(".user-table tbody tr"));
        assertThat(userRows).isNotEmpty();
        
        // 첫 번째 사용자 정보 확인
        WebElement firstUserRow = userRows.get(0);
        assertThat(firstUserRow.getText()).contains("홍길동");
    }

    @Test
    void testUserCreationFlow() {
        // given: 사용자 생성 페이지로 이동
        driver.get("http://localhost:" + port + "/users/create");

        // when: 폼 작성 및 제출
        driver.findElement(By.id("name")).sendKeys("테스트 사용자");
        driver.findElement(By.id("email")).sendKeys("test@example.com");
        driver.findElement(By.id("role")).sendKeys("USER");
        
        driver.findElement(By.cssSelector("button[type='submit']")).click();

        // then: 결과 페이지 검증
        wait.until(ExpectedConditions.urlContains("/users/"));
        
        // 성공 메시지 확인
        WebElement alert = driver.findElement(By.cssSelector(".alert-success"));
        assertThat(alert.getText()).contains("사용자가 성공적으로 생성되었습니다");
        
        // 생성된 사용자 정보 확인
        assertThat(driver.getPageSource()).contains("테스트 사용자");
        assertThat(driver.getPageSource()).contains("test@example.com");
    }

    @Test
    void testUserSearch() {
        // given: 사용자 목록 페이지로 이동
        driver.get("http://localhost:" + port + "/users");

        // when: 검색어 입력 및 검색 실행
        WebElement searchInput = driver.findElement(By.id("searchInput"));
        searchInput.sendKeys("홍길동");
        
        driver.findElement(By.id("searchButton")).click();

        // then: 검색 결과 확인
        wait.until(ExpectedConditions.presenceOfElementLocated(By.cssSelector(".search-results")));
        
        List<WebElement> resultRows = driver.findElements(By.cssSelector(".search-results tbody tr"));
        assertThat(resultRows).hasSize(1);
        
        WebElement resultRow = resultRows.get(0);
        assertThat(resultRow.getText()).contains("홍길동");
    }
}
```

## 테스트 도구와 프레임워크

### 1. Thymeleaf 테스트 유틸리티

```java
// Thymeleaf 테스트 유틸리티 클래스
public class ThymeleafTestUtils {

    private final TemplateEngine templateEngine;

    public ThymeleafTestUtils(TemplateEngine templateEngine) {
        this.templateEngine = templateEngine;
    }

    public String processTemplate(String templateName, Map<String, Object> variables) {
        Context context = new Context();
        variables.forEach(context::setVariable);
        return templateEngine.process(templateName, context);
    }

    public void assertTemplateContains(String templateName, Map<String, Object> variables, String expectedContent) {
        String result = processTemplate(templateName, variables);
        assertThat(result).contains(expectedContent);
    }

    public void assertTemplateDoesNotContain(String templateName, Map<String, Object> variables, String unexpectedContent) {
        String result = processTemplate(templateName, variables);
        assertThat(result).doesNotContain(unexpectedContent);
    }

    public void assertTemplateCount(String templateName, Map<String, Object> variables, String selector, int expectedCount) {
        String result = processTemplate(templateName, variables);
        Document doc = Jsoup.parse(result);
        Elements elements = doc.select(selector);
        assertThat(elements).hasSize(expectedCount);
    }
}
```

### 2. 테스트 데이터 빌더

```java
// 사용자 테스트 데이터 빌더
public class UserTestDataBuilder {

    public static User createDefaultUser() {
        return User.builder()
                .id(1L)
                .name("홍길동")
                .email("hong@example.com")
                .role("USER")
                .active(true)
                .createdAt(LocalDate.of(2023, 1, 1))
                .build();
    }

    public static User createAdminUser() {
        return User.builder()
                .id(2L)
                .name("관리자")
                .email("admin@example.com")
                .role("ADMIN")
                .active(true)
                .createdAt(LocalDate.of(2023, 1, 1))
                .build();
    }

    public static List<User> createUserList(int count) {
        List<User> users = new ArrayList<>();
        for (int i = 1; i <= count; i++) {
            users.add(User.builder()
                    .id((long) i)
                    .name("사용자" + i)
                    .email("user" + i + "@example.com")
                    .role(i % 2 == 0 ? "ADMIN" : "USER")
                    .active(i % 3 != 0)
                    .createdAt(LocalDate.of(2023, 1, 1).plusDays(i))
                    .build());
        }
        return users;
    }
}
```

## 테스트 커버리지

### 1. 테스트 커버리지 측정

```gradle
// build.gradle: JaCoCo 플러그인 추가
test {
    useJUnitPlatform()
}

jacoco {
    toolVersion = "0.8.7"
}

jacocoTestReport {
    reports {
        xml.enabled(true)
        html.enabled(true)
    }
}
```

### 2. 테스트 커버리지 목표 설정

```java
// 테스트 커버리지 검증
@Test
public void testAllTemplateScenarios() {
    // 모든 사용자 시나리오 테스트
    testUserListWithData();
    testUserListWithEmptyData();
    testUserListWithSingleUser();
    testUserListWithMultipleUsers();
    
    // 모든 사용자 상태 시나리오 테스트
    testActiveUser();
    testInactiveUser();
    testUserWithNullFields();
    
    // 모든 폼 시나리오 테스트
    testUserCreationForm();
    testUserEditForm();
    testUserFormValidation();
}
```

## 테스트 자동화

### 1. CI/CD 파이프라인 통합

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    
    - name: Set up JDK 17
      uses: actions/setup-java@v2
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Cache Gradle packages
      uses: actions/cache@v2
      with:
        path: |
          ~/.gradlele/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
        restore-keys: |
          ${{ runner.os }}-gradle-
    
    - name: Grant execute permission for gradlew
      run: chmod +x gradlew
    
    - name: Run tests
      run: ./gradlew test
    
    - name: Generate test report
      run: ./gradlew jacocoTestReport
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
```

### 2. 테스트 리포트 생성

```java
// 테스트 리포트 생성기
@Component
public class TestReportGenerator {

    public void generateTemplateTestReport() {
        StringBuilder report = new StringBuilder();
        report.append("# Thymeleaf 템플릿 테스트 리포트\n\n");
        
        // 템플릿 목록
        report.append("## 테스트된 템플릿\n\n");
        report.append("| 템플릿 | 테스트 케이스 | 결과 |\n");
        report.append("|--------|-------------|------|\n");
        
        // 테스트 결과 추가
        report.append("| users/list.html | 5 | 통과 |\n");
        report.append("| users/detail.html | 3 | 통과 |\n");
        report.append("| users/create.html | 4 | 통과 |\n");
        
        // 커버리지 정보
        report.append("\n## 테스트 커버리지\n\n");
        report.append("- 전체 템플릿: 15개\n");
        report.append("- 테스트된 템플릿: 12개\n");
        report.append("- 커버리지: 80%\n");
        
        // 파일로 저장
        try (FileWriter writer = new FileWriter("template-test-report.md")) {
            writer.write(report.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

## Best Practice

1. **다층적 테스트**: 단위 테스트, 통합 테스트, E2E 테스트를 모두 수행하세요.
2. **테스트 데이터 격리**: 각 테스트는 독립적인 테스트 데이터를 사용하세요.
3. **테스트 자동화**: CI/CD 파이프라인에 테스트를 자동화하세요.
4. **테스트 커버리지 모니터링**: 테스트 커버리지를 정기적으로 확인하고 개선하세요.
5. **명확한 테스트 이름**: 테스트 이름은 무엇을 테스트하는지 명확하게 나타내야 합니다.

## Bad Practice

1. **테스트 부족**: 충분한 테스트 없이 배포하면 예기치 않은 오류가 발생할 수 있습니다.
2. ** brittle 테스트**: 구현 상세에 너무 의존적인 테스트는 유지보수가 어렵습니다.
3. **테스트 데이터 격리 부족**: 테스트 간 데이터 의존성은 테스트를 불안정하게 만듭니다.
4. **테스트 무시**: 실패한 테스트를 무시하고 넘어가면 코드 품질이 저하됩니다.

## 마무리

Thymeleaf 템플릿 테스트는 애플리케이션의 안정성과 품질을 보장하는 데 매우 중요합니다. 단위 테스트, 통합 테스트, E2E 테스트를 적절히 조합하여 포괄적인 테스트 전략을 수립하세요.

테스트를 통해 템플릿의 오류를 조기에 발견하고 수정할 수 있으며, 리팩토링 시에도 기능이 올바르게 동작하는지 확인할 수 있습니다.

## 추가 학습 자료

- [Spring Boot Test](https://spring.io/guides/gs/testing-web/)
- [MockMvc Documentation](https://docs.spring.io/spring-framework/docs/current/reference/html/testing.html#spring-mvc-test-framework)
- [Selenium WebDriver](https://www.selenium.dev/documentation/)
- [JUnit 5](https://junit.org/junit5/docs/current/user-guide/)

이제 Thymeleaf Best Practices 강의자료가 완료되었습니다. 모든 챕터를 순서대로 학습하고 실습 예제를 통해 실력을 향상시키세요. 감사합니다!