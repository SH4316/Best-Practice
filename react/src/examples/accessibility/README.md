# 접근성 (Accessibility)

이 디렉토리는 React 애플리케이션에서 접근성을 구현하는 다양한 방법을 보여줍니다. 효과적인 접근성 구현은 모든 사용자가 애플리케이션을 사용할 수 있도록 보장합니다.

## 디렉토리 구조

```
accessibility/
├── bad/                    # 나쁜 예시
│   └── NavigationMenu.tsx
├── good/                   # 좋은 예시
│   ├── AccessibleApp.tsx
│   ├── AccessibleApp.css
│   ├── components/
│   │   ├── NavigationMenu.tsx
│   │   ├── Modal.tsx
│   │   └── FormField.tsx
│   ├── hooks/
│   │   └── useKeyboardNavigation.ts
│   ├── types/
│   │   └── index.ts
│   └── utils/
│       ├── focusTrap.ts
│       ├── screenReaderAnnouncer.ts
│       └── index.ts
└── README.md
```

## 나쁜 예시: 기본적인 접근성

`bad/NavigationMenu.tsx`는 기본적인 접근성을 보여줍니다. 이 접근 방식의 문제점은 다음과 같습니다:

- 키보드 접근성 부족
- 스크린 리더 지원 부족
- ARIA 속성 부족
- 포커스 관리 부족

## 좋은 예시: 포괄적인 접근성

`good/` 디렉토리는 포괄적인 접근성을 구현하는 방법을 보여줍니다. 이 접근 방식의 장점은 다음과 같습니다:

- 완전한 키보드 접근성
- 스크린 리더 지원
- 적절한 ARIA 속성
- 효과적인 포커스 관리
- 재사용 가능한 접근성 유틸리티

## 접근성 전략

### 1. 키보드 네비게이션

키보드만으로 모든 기능을 사용할 수 있도록 합니다:
- `useKeyboardNavigation`: 키보드 이벤트 처리 Hook

### 2. 포커스 관리

포커스를 효과적으로 관리하여 사용자 경험을 향상시킵니다:
- `focusTrap`: 모달과 같은 컴포넌트에서 포커스 트랩

### 3. 스크린 리더 지원

스크린 리더 사용자에게 적절한 정보를 제공합니다:
- `screenReaderAnnouncer`: 동적 콘텐츠 변경 알림

### 4. ARIA 속성

적절한 ARIA 속성을 사용하여 접근성을 향상시킵니다:
- `NavigationMenu`: 내비게이션 메뉴의 ARIA 속성
- `Modal`: 모달의 ARIA 속성
- `FormField`: 폼 필드의 ARIA 속성

## 사용 방법

```typescript
import AccessibleApp from './good/AccessibleApp';

function App() {
  return <AccessibleApp />;
}
```

## 결론

포괄적인 접근성 구현은 모든 사용자가 애플리케이션을 사용할 수 있도록 보장합니다. 이는 법적 요구사항을 충족할 뿐만 아니라, 더 넓은 사용자층에게 서비스를 제공하고 전반적인 사용자 경험을 향상시킵니다.