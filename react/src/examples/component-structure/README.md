# 컴포넌트 구조

이 디렉토리는 React 컴포넌트를 구성하는 다양한 방법을 보여줍니다. 올바른 컴포넌트 구조는 코드의 재사용성, 유지보수성 및 테스트 용이성을 향상시킵니다.

## 디렉토리 구조

```
component-structure/
├── bad/                    # 나쁜 예시
│   ├── UserProfile.tsx
├── good/                   # 좋은 예시
│   ├── UserProfile.tsx
│   ├── components/
│   │   ├── Avatar.tsx
│   │   ├── UserInfo.tsx
│   │   ├── UserStats.tsx
│   │   ├── UserForm.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── ErrorMessage.tsx
│   │   ├── EmptyState.tsx
│   │   └── Button.tsx
│   ├── hooks/
│   │   └── useUser.ts
│   └── UserProfile.styles.css
└── README.md
```

## 나쁜 예시: 단일 컴포넌트

`bad/UserProfile.tsx`는 하나의 거대한 컴포넌트를 보여줍니다. 이 접근 방식의 문제점은 다음과 같습니다:

- 단일 책임 원칙 위반
- 재사용성 부족
- 테스트 어려움
- 유지보수 복잡성 증가

## 좋은 예시: 분리된 컴포넌트

`good/` 디렉토리는 컴포넌트를 더 작고 재사용 가능한 부분으로 분리하는 방법을 보여줍니다. 이 접근 방식의 장점은 다음과 같습니다:

- 단일 책임 원칙 준수
- 높은 재사용성
- 쉬운 테스트
- 향상된 유지보수성

## 컴포넌트 분리 전략

### 1. UI 컴포넌트 분리

UI를 논리적 부분으로 분리합니다:
- `Avatar`: 사용자 프로필 이미지
- `UserInfo`: 사용자 기본 정보
- `UserStats`: 사용자 통계
- `UserForm`: 사용자 정보 편집 폼

### 2. 상태 비저장 컴포넌트

재사용 가능한 상태 비저장 컴포넌트를 만듭니다:
- `LoadingSpinner`: 로딩 상태 표시
- `ErrorMessage`: 에러 메시지 표시
- `EmptyState`: 데이터가 없을 때 표시
- `Button`: 재사용 가능한 버튼

### 3. 커스텀 Hook 분리

상태 로직과 사이드 이펙트를 커스텀 Hook으로 분리합니다:
- `useUser`: 사용자 데이터 페칭 및 관리

## 사용 방법

```typescript
import UserProfile from './good/UserProfile';

function App() {
  return (
    <div>
      <UserProfile userId="user-123" />
    </div>
  );
}
```

## 결론

컴포넌트를 더 작고 재사용 가능한 부분으로 분리하면 코드가 더 명확해지고 유지보수가 쉬워집니다. 이는 특히 대규모 애플리케이션에서 중요합니다.