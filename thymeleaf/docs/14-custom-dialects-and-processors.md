# 14. 커스텀 다이얼렉트와 프로세서

## 커스텀 다이얼렉트란?

Thymeleaf 다이얼렉트는 특정 마크업 언어나 도메인에 맞게 사용자 정의 속성과 프로세서를 추가할 수 있는 확장 메커니즘입니다. 커스텀 다이얼렉트를 만들면 프로젝트에 특화된 기능을 구현하고 코드 재사용성을 높일 수 있습니다.

## 다이얼렉트 기본 구조

### IDialect 인터페이스 구현

```java
package com.example.demo.dialect;

import org.thymeleaf.dialect.AbstractProcessorDialect;
import org.thymeleaf.processor.IProcessor;
import org.thymeleaf.standard.StandardDialect;

import java.util.HashSet;
import java.util.Set;

public class CustomDialect extends AbstractProcessorDialect {

    private static final String PREFIX = "custom";
    private static final int PRECEDENCE = StandardStandardDialect.PROCESSOR_PRECEDENCE;

    public CustomDialect() {
        super(PREFIX, PRECEDENCE);
    }

    @Override
    public Set<IProcessor> getProcessors(String dialectPrefix) {
        final Set<IProcessor> processors = new HashSet<>();
        
        // 커스텀 프로세서 추가
        processors.add(new CustomAttributeProcessor(dialectPrefix, "format", PRECEDENCE + 100));
        processors.add(new CustomTagProcessor(dialectPrefix, "card", PRECEDENCE + 100));
        processors.add(new CustomElementProcessor(dialectPrefix, "badge", PRECEDENCE + 100));
        
        return processors;
    }
}
```

### 다이얼렉트 등록

```java
package com.example.demo.config;

import com.example.demo.dialect.CustomDialect;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ThymeleafConfig {

    @Bean
    public CustomDialect customDialect() {
        return new CustomDialect();
    }
}
```

## 커스텀 속성 프로세서

### 기본 속성 프로세서 구현

```java
package com.example.demo.dialect.processor;

import org.thymeleaf.context.ITemplateContext;
import org.thymeleaf.engine.AttributeName;
import org.thymeleaf.model.IProcessableElementTag;
import org.thymeleaf.processor.IProcessor;
import org.thymeleaf.processor.attr.AbstractAttributeProcessor;
import org.thymeleaf.templatemode.TemplateMode;

public class CustomAttributeProcessor extends AbstractAttributeProcessor {

    public CustomAttributeProcessor(String dialectPrefix, String attrName, int precedence) {
        super(TemplateMode.HTML, dialectPrefix, attrName, precedence, true);
    }

    @Override
    protected void doProcess(ITemplateContext context, IProcessableElementTag tag,
                            AttributeName attributeName, String attributeValue,
                            IElementTagStructureHandler structureHandler) {
        
        // 속성 값 처리 로직
        String processedValue = processValue(attributeValue, context);
        
        // 처리된 값을 속성으로 설정
        tag.setAttribute(attributeName.getCompleteAttributeName(), processedValue);
        
        // 구조 처리기에 변경 사항 알림
        structureHandler.setBody(processedValue, false);
    }
    
    private String processValue(String value, ITemplateContext context) {
        // 값 처리 로직 구현
        if (value == null || value.trim().isEmpty()) {
            return "";
        }
        
        // 예시: 날짜 포맷팅
        if (value.contains("date:")) {
            String dateValue = value.substring(5);
            return formatDate(dateValue, context);
        }
        
        // 예시: 숫자 포맷팅
        if (value.contains("number:")) {
            String numberValue = value.substring(7);
            return formatNumber(numberValue, context);
        }
        
        return value;
    }
    
    private String formatDate(String dateValue, ITemplateContext context) {
        // 날짜 포맷팅 로직
        return "[Formatted: " + dateValue + "]";
    }
    
    private String formatNumber(String numberValue, ITemplateContext context) {
        // 숫자 포맷팅 로직
        return "[Formatted: " + numberValue + "]";
    }
}
```

### 커스텀 속성 사용

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:custom="http://www.thymeleaf.org/extras/custom">
<head>
    <title>Custom Attribute Example</title>
</head>
<body>
    <!-- 커스텀 속성 사용 -->
    <div custom:format="date:2023-01-01">Formatted Date</div>
    <div custom:format="number:12345.67">Formatted Number</div>
</body>
</html>
```

## 커스텀 태그 프로세서

### 기본 태그 프로세서 구현

```java
package com.example.demo.dialect.processor;

import org.thymeleaf.context.ITemplateContext;
import org.thymeleaf.engine.AttributeName;
import org.thymeleaf.model.IProcessableElementTag;
import org.thymeleaf.processor.IProcessor;
import org.thymeleaf.processor.element.AbstractElementTagProcessor;
import org.thymeleaf.processor.element.IElementTagStructureHandler;
import org.thymeleaf.templatemode.TemplateMode;

public class CustomTagProcessor extends AbstractElementTagProcessor {

    private static final String TAG_NAME = "card";
    private static final int PRECEDENCE = 1000;

    public CustomTagProcessor(String dialectPrefix, String tagName, int precedence) {
        super(TemplateMode.HTML, dialectPrefix, tagName, precedence, true);
    }

    @Override
    protected void doProcess(ITemplateContext context, IProcessableElementTag tag,
                           IElementTagStructureHandler structureHandler) {
        
        // 속성 값 추출
        String title = tag.getAttributeValue("title");
        String content = tag.getAttributeValue("content");
        String type = tag.getAttributeValue("type");
        
        // 기본값 설정
        if (type == null) {
            type = "default";
        }
        
        // 카드 HTML 생성
        String cardHtml = generateCardHtml(title, content, type);
        
        // 태그를 생성된 HTML로 교체
        structureHandler.replaceWith(cardHtml, false);
    }
    
    private String generateCardHtml(String title, String content, String type) {
        StringBuilder html = new StringBuilder();
        
        html.append("<div class=\"card card-").append(type).append("\">");
        
        if (title != null && !title.trim().isEmpty()) {
            html.append("<div class=\"card-header\">");
            html.append("<h5 class=\"card-title\">").append(title).append("</h5>");
            html.append("</div>");
        }
        
        if (content != null && !content.trim().isEmpty()) {
            html.append("<div class=\"card-body\">");
            html.append("<p class=\"card-text\">").append(content).append("</p>");
            html.append("</div>");
        }
        
        html.append("</div>");
        
        return html.toString();
    }
}
```

### 커스텀 태그 사용

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:custom="http://www.thymeleaf.org/extras/custom">
<head>
    <title>Custom Tag Example</title>
    <link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">
</head>
<body>
    <div class="container my-4">
        <h1>Custom Tag Examples</h1>
        
        <!-- 기본 카드 -->
        <custom:card title="Card Title" content="This is a card content."></custom:card>
        
        <!-- 타입별 카드 -->
        <custom:card title="Primary Card" content="Primary card content." type="primary"></custom:card>
        <custom:card title="Success Card" content="Success card content." type="success"></custom:card>
        <custom:card title="Warning Card" content="Warning card content." type="warning"></custom:card>
        
        <!-- 동적 내용 카드 -->
        <custom:card th:title="${user.name}" th:content="${user.description}" type="info"></custom:card>
    </div>
</body>
</html>
```

## 커스텀 요소 프로세서

### 기본 요소 프로세서 구현

```java
package com.example.demo.dialect.processor;

import org.thymeleaf.context.ITemplateContext;
import org.thymeleaf.engine.AttributeName;
import org.thymeleaf.model.IProcessableElementTag;
import org.thymeleaf.processor.IProcessor;
import org.thymeleaf.processor.element.AbstractElementTagProcessor;
import org.thymeleaf.processor.element.IElementTagStructureHandler;
import org.thymeleaf.templatemode.TemplateMode;

public class CustomElementProcessor extends AbstractElementTagProcessor {

    private static final String ELEMENT_NAME = "badge";
    private static final int PRECEDENCE = 1000;

    public CustomElementProcessor(String dialectPrefix, String elementName, int precedence) {
        super(TemplateMode.HTML, dialectPrefix, elementName, precedence, true);
    }

    @Override
    protected void doProcess(ITemplateContext context, IProcessableElementTag tag,
                           IElementTagStructureHandler structureHandler) {
        
        // 속성 값 추출
        String text = tag.getAttributeValue("text");
        String type = tag.getAttributeValue("type");
        String size = tag.getAttributeValue("size");
        
        // 기본값 설정
        if (type == null) {
            type = "primary";
        }
        if (size == null) {
            size = "md";
        }
        
        // 배지 HTML 생성
        String badgeHtml = generateBadgeHtml(text, type, size);
        
        // 요소를 생성된 HTML로 교체
        structureHandler.replaceWith(badgeHtml, false);
    }
    
    private String generateBadgeHtml(String text, String type, String size) {
        StringBuilder html = new StringBuilder();
        
        html.append("<span class=\"badge badge-").append(type);
        
        // 크기 클래스 추가
        if (!"md".equals(size)) {
            html.append(" badge-").append(size);
        }
        
        html.append("\">");
        
        // 텍스트 추가
        if (text != null && !text.trim().isEmpty()) {
            html.append(text);
        } else {
            html.append("Badge");
        }
        
        html.append("</span>");
        
        return html.toString();
    }
}
```

### 커스텀 요소 사용

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:custom="http://www.thymeleaf.org/extras/custom">
<head>
    <title>Custom Element Example</title>
    <link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">
</head>
<body>
    <div class="container my-4">
        <h1>Custom Element Examples</h1>
        
        <!-- 기본 배지 -->
        <custom:badge text="Default Badge"></custom:badge>
        
        <!-- 타입별 배지 -->
        <custom:badge text="Primary" type="primary"></custom:badge>
        <custom:badge text="Success" type="success"></custom:badge>
        <custom:badge text="Warning" type="warning"></custom:badge>
        <custom:badge text="Danger" type="danger"></custom:badge>
        
        <!-- 크기별 배지 -->
        <custom:badge text="Small Badge" type="info" size="sm"></custom:badge>
        <custom:badge text="Large Badge" type="secondary" size="lg"></custom:badge>
        
        <!-- 동적 배지 -->
        <custom:badge th:text="${user.status}" th:type="${user.statusColor}" th:size="${user.statusSize}"></custom:badge>
    </div>
</body>
</html>
```

## 고급 커스텀 다이얼렉트

### 데이터 소프트웨어 다이얼렉트

```java
package com.example.demo.dialect;

import org.thymeleaf.dialect.AbstractProcessorDialect;
import org.thymeleaf.processor.IProcessor;
import org.thymeleaf.standard.StandardDialect;

import java.util.HashSet;
import java.util.Set;

public class DataSoftDialect extends AbstractProcessorDialect {

    private static final String PREFIX = "ds";
    private static final int PRECEDENCE = StandardStandardDialect.PROCESSOR_PRECEDENCE;

    public DataSoftDialect() {
        super(PREFIX, PRECEDENCE);
    }

    @Override
    public Set<IProcessor> getProcessors(String dialectPrefix) {
        final Set<IProcessor> processors = new HashSet<>();
        
        // 데이터 소프트웨어 관련 프로세서 추가
        processors.add(new DataTableProcessor(dialectPrefix, "table", PRECEDENCE + 100));
        processors.add(new DataChartProcessor(dialectPrefix, "chart", PRECEDENCE + 100));
        processors.add(new DataFormProcessor(dialectPrefix, "form", PRECEDENCE + 100));
        
        return processors;
    }
}
```

### 데이터 테이블 프로세서

```java
package com.example.demo.dialect.processor;

import org.thymeleaf.context.ITemplateContext;
import org.thymeleaf.engine.AttributeName;
import org.thymeleaf.model.IProcessableElementTag;
import org.thymeleaf.processor.element.AbstractElementTagProcessor;
import org.thymeleaf.processor.element.IElementTagStructureHandler;
import org.thymeleaf.templatemode.TemplateMode;

public class DataTableProcessor extends AbstractElementTagProcessor {

    private static final String TAG_NAME = "table";
    private static final int PRECEDENCE = 1000;

    public DataTableProcessor(String dialectPrefix, String tagName, int precedence) {
        super(TemplateMode.HTML, dialectPrefix, tagName, precedence, true);
    }

    @Override
    protected void doProcess(ITemplateContext context, IProcessableElementTag tag,
                           IElementTagStructureHandler structureHandler) {
        
        // 속성 값 추출
        String data = tag.getAttributeValue("data");
        String columns = tag.getAttributeValue("columns");
        String pageable = tag.getAttributeValue("pageable");
        
        // 데이터 테이블 HTML 생성
        String tableHtml = generateTableHtml(data, columns, "true".equals(pageable));
        
        // 태그를 생성된 HTML로 교체
        structureHandler.replaceWith(tableHtml, false);
    }
    
    private String generateTableHtml(String data, String columns, boolean pageable) {
        StringBuilder html = new StringBuilder();
        
        html.append("<div class=\"data-table-container\">");
        html.append("<table class=\"table table-striped table-hover\">");
        
        // 테이블 헤더 생성
        html.append("<thead>");
        html.append("<tr>");
        
        if (columns != null && !columns.trim().isEmpty()) {
            String[] columnArray = columns.split(",");
            for (String column : columnArray) {
                html.append("<th>").append(column.trim()).append("</th>");
            }
        }
        
        html.append("</tr>");
        html.append("</thead>");
        
        // 테이블 본문 생성
        html.append("<tbody>");
        html.append("<tr>");
        html.append("<td colspan=\"").append(columns != null ? columns.split(",").length : 1).append("\">");
        html.append("Data will be populated here: ").append(data);
        html.append("</td>");
        html.append("</tr>");
        html.append("</tbody>");
        
        html.append("</table>");
        
        // 페이지네이션 추가
        if (pageable) {
            html.append("<div class=\"data-table-pagination\">");
            html.append("<nav>");
            html.append("<ul class=\"pagination\">");
            html.append("<li class=\"page-item disabled\"><a class=\"page-link\" href=\"#\">Previous</a></li>");
            html.append("<li class=\"page-item active\"><a class=\"page-link\" href=\"#\">1</a></li>");
            html.append("<li class=\"page-item\"><a class=\"page-link\" href=\"#\">2</a></li>");
            html.append("<li class=\"page-item\"><a class="page-link" href=\"#\">Next</a></li>");
            html.append("</ul>");
            html.append("</nav>");
            html.append("</div>");
        }
        
        html.append("</div>");
        
        return html.toString();
    }
}
```

### 데이터 테이블 사용

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org" 
      xmlns:ds="http://www.thymeleaf.org/extras/datasoft">
<head>
    <title>DataSoft Dialect Example</title>
    <link rel="stylesheet" th:href="@{/css/bootstrap.min.css}">
</head>
<body>
    <div class="container my-4">
        <h1>DataSoft Dialect Examples</h1>
        
        <!-- 기본 데이터 테이블 -->
        <ds:table data="users" columns="Name,Email,Role,Status"></ds:table>
        
        <!-- 페이지 가능한 데이터 테이블 -->
        <ds:table data="products" columns="Name,Price,Category,Stock" pageable="true"></ds:table>
    </div>
</body>
</html>
```

## 커스텀 유틸리티 객체

### 유틸리티 객체 구현

```java
package com.example.demo.dialect.util;

import org.thymeleaf.context.IExpressionContext;
import org.thymeleaf.expression.IExpressionObjectFactory;

import java.util.LinkedHashSet;
import java.util.Set;

public class CustomExpressionObjectFactory implements IExpressionObjectFactory {

    @Override
    public Set<Object> getAllExpressionObjects(IExpressionContext context) {
        Set<Object> objects = new LinkedHashSet<>();
        
        // 커스텀 유틸리티 객체 추가
        objects.put("customUtils", new CustomUtils());
        objects.put("stringUtils", new StringUtils());
        objects.put("dateUtils", new DateUtils());
        
        return objects;
    }

    @Override
    public Object buildObject(IExpressionContext context, String expressionObjectName) {
        // 동적 유틸리티 객체 생성
        if ("dynamicUtils".equals(expressionObjectName)) {
            return new DynamicUtils(context);
        }
        
        return null;
    }
}
```

### 커스텀 유틸리티 클래스

```java
package com.example.demo.dialect.util;

public class CustomUtils {
    
    public String formatCurrency(double amount) {
        return String.format("$%,.2f", amount);
    }
    
    public String truncate(String text, int length) {
        if (text == null || text.length() <= length) {
            return text;
        }
        return text.substring(0, length) + "...";
    }
    
    public String highlightKeywords(String text, String keyword) {
        if (text == null || keyword == null || keyword.trim().isEmpty()) {
            return text;
        }
        
        return text.replaceAll("(?i)(" + keyword.trim() + ")", "<mark>$1</mark>");
    }
    
    public String generateSlug(String text) {
        if (text == null || text.trim().isEmpty()) {
            return "";
        }
        
        return text.toLowerCase()
                .replaceAll("[^a-z0-9\\s-]", "")
                .replaceAll("\\s+", "-")
                .replaceAll("-+", "-")
                .replaceAll("^-|-$", "");
    }
}
```

### 커스텀 유틸리티 사용

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Custom Utils Example</title>
</head>
<body>
    <div class="container my-4">
        <h1>Custom Utils Examples</h1>
        
        <!-- 통화 포맷 -->
        <p>Price: <span th:text="${customUtils.formatCurrency(1234.56)}">$1,234.56</span></p>
        
        <!-- 텍스트 자르기 -->
        <p>Truncated: <span th:text="${customUtils.truncate('This is a long text that needs to be truncated', 20)}">Truncated Text</span></p>
        
        <!-- 키워드 하이라이트 -->
        <p>Highlighted: <span th:utext="${customUtils.highlightKeywords('This is a sample text with sample keyword', 'sample')}">Highlighted Text</span></p>
        
        <!-- 슬러그 생성 -->
        <p>Slug: <span th:text="${customUtils.generateSlug('This is a Sample Title')}">this-is-a-sample-title</span></p>
    </div>
</body>
</html>
```

## Best Practice

1. **명확한 네이밍**: 커스텀 다이얼렉트와 프로세서에는 명확하고 의미 있는 이름을 사용하세요.
2. **일관된 구조**: 프로젝트 전체에서 일관된 구조와 패턴을 유지하세요.
3. **문서화**: 커스텀 기능에 대한 문서를 작성하고 예제를 제공하세요.
4. **테스트**: 커스텀 다이얼렉트와 프로세서를 철저히 테스트하세요.
5. **성능 고려**: 복잡한 로직은 템플릿이 아닌 서버 측에서 처리하세요.

## Bad Practice

1. **과도한 커스터마이징**: 필요 이상의 커스텀 기능은 유지보수를 어렵게 만듭니다.
2. **명명 충돌**: 기존 속성이나 태그와 충돌하는 이름을 사용하지 마세요.
3. **문서화 부족**: 커스텀 기능에 대한 문서화가 부족하면 다른 개발자가 사용하기 어렵습니다.
4. **테스트 부족**: 충분한 테스트 없이 커스텀 기능을 배포하면 오류가 발생할 수 있습니다.

## 다음 장에서는

다음 장에서는 Thymeleaf와 REST API 통합에 대해 알아보겠습니다.
