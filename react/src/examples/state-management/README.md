# 상태 관리

이 디렉토리는 React 애플리케이션에서 상태를 관리하는 다양한 방법을 보여줍니다. 효과적인 상태 관리는 애플리케이션의 예측 가능성, 유지보수성 및 확장성을 향상시킵니다.

## 디렉토리 구조

```
state-management/
├── bad/                    # 나쁜 예시
│   └── TodoApp.tsx
├── good/                   # 좋은 예시
│   ├── TodoApp.tsx
│   ├── types/
│   │   └── index.ts
│   └── store/
│       └── todoStore.ts
└── README.md
```

## 나쁜 예시: Props Drilling

`bad/TodoApp.tsx`는 props drilling 문제를 보여줍니다. 이 접근 방식의 문제점은 다음과 같습니다:

- 깊은 중첩 구조에서 데이터 전달의 복잡성
- 컴포넌트 재사용성 저하
- 상태 변경 시 불필요한 리렌더링
- 유지보수 어려움

## 좋은 예시: Zustand 사용

`good/` 디렉토리는 Zustand를 사용하여 상태를 효과적으로 관리하는 방법을 보여줍니다. 이 접근 방식의 장점은 다음과 같습니다:

- 중앙 집중식 상태 관리
- 불필요한 리렌더링 방지
- 간단한 API
- TypeScript와의 좋은 통합
- 개발 도구 지원

## 상태 관리 전략

### 1. 상태 분리

상태를 도메인별로 분리합니다:
- `todoStore`: 할 일 목록 관리

### 2. 타입 정의

명확한 타입 정의를 통해 타입 안전성을 보장합니다:
- `Todo`: 할 일 항목의 타입
- `TodoFilter`: 필터링 옵션의 타입

### 3. 액션 정의

상태를 변경하는 함수를 정의합니다:
- `addTodo`: 할 일 추가
- `toggleTodo`: 할 일 완료 상태 전환
- `deleteTodo`: 할 일 삭제
- `setFilter`: 필터링 설정

## 사용 방법

```typescript
import { useTodoStore } from './store/todoStore';

function TodoApp() {
  const { todos, filter, addTodo, toggleTodo, deleteTodo, setFilter } = useTodoStore();
  
  return (
    <div>
      {/* 컴포넌트 내용 */}
    </div>
  );
}
```

## 결론

Zustand와 같은 현대적인 상태 관리 라이브러리를 사용하면 복잡한 상태 로직을 효과적으로 관리할 수 있습니다. 이는 특히 대규모 애플리케이션에서 중요합니다.