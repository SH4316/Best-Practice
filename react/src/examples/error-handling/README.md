# 에러 처리

이 디렉토리는 React 애플리케이션에서 에러를 처리하는 다양한 방법을 보여줍니다. 효과적인 에러 처리는 애플리케이션의 안정성과 사용자 경험을 향상시킵니다.

## 디렉토리 구조

```
error-handling/
├── bad/                    # 나쁜 예시
│   └── DataComponent.tsx
├── good/                   # 좋은 예시
│   ├── DataComponent.tsx
│   ├── DataComponent.css
│   ├── components/
│   │   ├── ErrorBoundary.tsx
│   │   ├── ErrorMessage.tsx
│   │   └── LoadingSpinner.tsx
│   ├── hooks/
│   │   └── useApi.ts
│   ├── types/
│   │   └── index.ts
│   └── utils/
│       ├── apiService.ts
│       ├── errorLogger.ts
│       └── index.ts
└── README.md
```

## 나쁜 예시: 기본적인 에러 처리

`bad/DataComponent.tsx`는 기본적인 에러 처리를 보여줍니다. 이 접근 방식의 문제점은 다음과 같습니다:

- 일관되지 않은 에러 처리
- 사용자에게 친화적이지 않은 에러 메시지
- 에러 로깅 부족
- 에러 복구 기능 부족

## 좋은 예시: 포괄적인 에러 처리

`good/` 디렉토리는 포괄적인 에러 처리를 구현하는 방법을 보여줍니다. 이 접근 방식의 장점은 다음과 같습니다:

- 일관된 에러 처리
- 사용자 친화적인 에러 메시지
- 체계적인 에러 로깅
- 에러 복구 기능
- 재사용 가능한 에러 처리 컴포넌트

## 에러 처리 전략

### 1. 에러 경계 (Error Boundary)

React 에러 경계를 사용하여 컴포넌트 트리에서 발생하는 에러를 처리합니다:
- `ErrorBoundary`: 에러를 포착하고 사용자에게 친화적인 UI 표시

### 2. 커스텀 Hook

데이터 페칭 및 에러 처리 로직을 커스텀 Hook으로 분리합니다:
- `useApi`: API 호출 및 에러 처리

### 3. 에러 로깅

에러를 체계적으로 로깅하여 디버깅과 모니터링을 용이하게 합니다:
- `errorLogger`: 에러 로깅 유틸리티

### 4. API 에러 처리

API 호출에서 발생하는 에러를 일관되게 처리합니다:
- `apiService`: API 호출 및 에러 처리

### 5. 에러 표시 컴포넌트

에러를 사용자에게 표시하는 재사용 가능한 컴포넌트를 만듭니다:
- `ErrorMessage`: 에러 메시지 표시
- `LoadingSpinner`: 로딩 상태 표시

## 사용 방법

```typescript
import DataComponent from './good/DataComponent';

function App() {
  return (
    <div>
      <DataComponent userId="user-123" />
    </div>
  );
}
```

## 결론

포괄적인 에러 처리 전략을 사용하면 애플리케이션의 안정성을 높이고 사용자 경험을 향상시킬 수 있습니다. 이는 특히 대규모 애플리케이션에서 중요합니다.