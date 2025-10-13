package com.example.demo.controller;

import com.example.demo.model.User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Controller
public class ThymeleafController {

    @GetMapping("/")
    public String index(Model model) {
        model.addAttribute("title", "Thymeleaf Best Practices");
        return "index";
    }

    @GetMapping("/variables")
    public String variables(Model model) {
        User user = new User(1L, "홍길동", "hong@example.com", "USER", true, LocalDateTime.now());
        User userWithNull = new User(2L, null, "null@example.com", "USER", false, LocalDateTime.now());
        
        model.addAttribute("user", user);
        model.addAttribute("userWithNull", userWithNull);
        model.addAttribute("title", "변수 표현식 예제");
        
        return "examples/variables";
    }

    @GetMapping("/loops")
    public String loops(Model model) {
        List<User> users = new ArrayList<>();
        users.add(new User(1L, "홍길동", "hong@example.com", "USER", true, LocalDateTime.now()));
        users.add(new User(2L, "김철수", "kim@example.com", "ADMIN", true, LocalDateTime.now()));
        users.add(new User(3L, "이영희", "lee@example.com", "USER", false, LocalDateTime.now()));
        users.add(new User(4L, "박민준", "park@example.com", "MANAGER", true, LocalDateTime.now()));
        users.add(new User(5L, "최지민", "choi@example.com", "USER", true, LocalDateTime.now()));
        
        model.addAttribute("users", users);
        model.addAttribute("title", "반복문 예제");
        
        return "examples/loops";
    }

    @GetMapping("/loops/empty")
    public String loopsEmpty(Model model) {
        model.addAttribute("users", new ArrayList<User>());
        model.addAttribute("title", "빈 목록 예제");
        
        return "examples/loops";
    }

    @GetMapping("/conditions")
    public String conditions(Model model) {
        User adminUser = new User(1L, "관리자", "admin@example.com", "ADMIN", true, LocalDateTime.now());
        User normalUser = new User(2L, "사용자", "user@example.com", "USER", true, LocalDateTime.now());
        User inactiveUser = new User(3L, "비활성", "inactive@example.com", "USER", false, LocalDateTime.now());
        
        model.addAttribute("adminUser", adminUser);
        model.addAttribute("normalUser", normalUser);
        model.addAttribute("inactiveUser", inactiveUser);
        model.addAttribute("title", "조건문 예제");
        
        return "examples/conditions";
    }

    @GetMapping("/links")
    public String links(Model model) {
        User user = new User(1L, "홍길동", "hong@example.com", "USER", true, LocalDateTime.now());
        
        model.addAttribute("user", user);
        model.addAttribute("keyword", "검색어");
        model.addAttribute("currentPage", 1);
        model.addAttribute("title", "링크 처리 예제");
        
        return "examples/links";
    }

    @GetMapping("/forms")
    public String form(Model model) {
        User user = new User(1L, "홍길동", "hong@example.com", "USER", true, LocalDateTime.now());
        
        model.addAttribute("user", user);
        model.addAttribute("title", "폼 처리 예제");
        
        return "examples/forms";
    }

    @PostMapping("/forms")
    public String submitForm(User user) {
        // 폼 처리 로직
        return "redirect:/forms/success";
    }

    @GetMapping("/forms/success")
    public String formSuccess(Model model) {
        model.addAttribute("title", "폼 제출 성공");
        return "examples/form-success";
    }

    @GetMapping("/fragments")
    public String fragments(Model model) {
        model.addAttribute("title", "프래그먼트 예제");
        model.addAttribute("content", "examples/content-example");
        
        return "examples/layout";
    }

    @GetMapping("/pagination")
    public String pagination(@RequestParam(defaultValue = "1") int page, Model model) {
        int totalPages = 10;
        
        model.addAttribute("page", page);
        model.addAttribute("totalPages", totalPages);
        model.addAttribute("title", "페이지네이션 예제");
        
        return "examples/pagination";
    }

    @GetMapping("/i18n")
    public String internationalization(Model model) {
        User user = new User(1L, "홍길동", "hong@example.com", "USER", true, LocalDateTime.now());
        
        model.addAttribute("user", user);
        model.addAttribute("date", LocalDateTime.now());
        model.addAttribute("price", 1234.56);
        model.addAttribute("title", "국제화 예제");
        
        return "examples/i18n";
    }

    @GetMapping("/security")
    public String security(Model model) {
        User user = new User(1L, "홍길동", "hong@example.com", "USER", true, LocalDateTime.now());
        user.setComment("<script>alert('XSS 공격!');</script>이것은 댓글입니다.");
        user.setHtmlContent("<strong>강조 텍스트</strong>");
        
        model.addAttribute("user", user);
        model.addAttribute("title", "보안 예제");
        
        return "examples/security";
    }

    @GetMapping("/performance")
    public String performance(Model model) {
        List<User> users = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            users.add(new User(
                (long) i, 
                "사용자" + i, 
                "user" + i + "@example.com", 
                i % 3 == 0 ? "ADMIN" : "USER", 
                i % 2 == 0, 
                LocalDateTime.now()
            ));
        }
        
        model.addAttribute("users", users);
        model.addAttribute("title", "성능 최적화 예제");
        
        return "examples/performance";
    }

    @GetMapping("/comparison/{topic}")
    public String comparison(@PathVariable String topic, Model model) {
        switch (topic.toLowerCase()) {
            case "variables":
                return variables(model);
            case "loops":
                return loops(model);
            case "conditions":
                return conditions(model);
            case "links":
                return links(model);
            case "forms":
                return form(model);
            case "security":
                return security(model);
            case "performance":
                return performance(model);
            default:
                return "redirect:/";
        }
    }

    @GetMapping("/comparison")
    public String comparisonMain(Model model) {
        model.addAttribute("title", "Best vs Bad Practice 비교");
        return "examples/comparison";
    }
}