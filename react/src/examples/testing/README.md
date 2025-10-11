# 테스트

이 디렉토리는 React 애플리케이션을 테스트하는 다양한 방법을 보여줍니다. 효과적인 테스트는 코드의 안정성과 신뢰성을 보장합니다.

## 디렉토리 구조

```
testing/
├── bad/                    # 나쁜 예시
│   ├── UserProfile.tsx
│   └── UserProfile.test.tsx
├── good/                   # 좋은 예시
│   ├── UserProfile.tsx
│   ├── UserProfile.test.tsx
│   ├── useUser.test.ts
│   ├── components/
│   │   └── UserProfile.tsx
│   ├── hooks/
│   │   └── useUser.ts
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   ├── apiService.ts
│   │   ├── testDataFactory.ts
│   │   └── index.ts
│   └── __mocks__/
│       └── apiService.ts
└── README.md
```

## 나쁜 예시: 기본적인 테스트

`bad/` 디렉토리는 기본적인 테스트를 보여줍니다. 이 접근 방식의 문제점은 다음과 같습니다:

- 구현 세부 사항에 의존하는 테스트
- 사용자 관점에서 테스트되지 않음
- 불안정한 테스트
- 재사용 불가능한 테스트 유틸리티

## 좋은 예시: 포괄적인 테스트

`good/` 디렉토리는 포괄적인 테스트를 구현하는 방법을 보여줍니다. 이 접근 방식의 장점은 다음과 같습니다:

- 사용자 관점에서 테스트
- 안정적인 테스트
- 재사용 가능한 테스트 유틸리티
- 효과적인 모킹 전략
- 컴포넌트 및 Hook 테스트

## 테스트 전략

### 1. 컴포넌트 테스트

사용자가 컴포넌트와 상호작용하는 방식을 테스트합니다:
- `UserProfile.test.tsx`: 컴포넌트 테스트

### 2. Hook 테스트

커스텀 Hook의 동작을 테스트합니다:
- `useUser.test.ts`: Hook 테스트

### 3. 모킹

외부 의존성을 모킹하여 테스트를 격리하고 안정성을 높입니다:
- `apiService.ts`: API 서비스 모킹
- `__mocks__/apiService.ts`: 모킹된 API 서비스

### 4. 테스트 데이터 팩토리

테스트 데이터를 생성하는 재사용 가능한 팩토리를 만듭니다:
- `testDataFactory.ts`: 테스트 데이터 팩토리

### 5. 의존성 주입

의존성 주입을 통해 테스트 가능성을 높입니다:
- `UserProfile.tsx`: 의존성 주입이 가능한 컴포넌트
- `useUser.ts`: 의존성 주입이 가능한 Hook

## 사용 방법

```typescript
import UserProfile from './good/components/UserProfile';

function App() {
  return (
    <div>
      <UserProfile userId="user-123" />
    </div>
  );
}
```

## 테스트 실행

```bash
# 컴포넌트 테스트 실행
npm test UserProfile.test.tsx

# Hook 테스트 실행
npm test useUser.test.ts

# 모든 테스트 실행
npm test
```

## 결론

효과적인 테스트 전략을 사용하면 코드의 안정성과 신뢰성을 높일 수 있습니다. 이는 특히 대규모 애플리케이션에서 중요합니다.