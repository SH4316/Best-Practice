# React 프로젝트 구조

React 애플리케이션의 효과적인 프로젝트 구조는 유지보수성, 확장성, 협업 효율성에 큰 영향을 미칩니다.

## 권장되는 프로젝트 구조

```
my-react-app/
├── public/                 # 정적 파일
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/         # 재사용 가능한 컴포넌트
│   │   ├── common/         # 공통 컴포넌트
│   │   │   ├── Button/
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Button.test.tsx
│   │   │   │   ├── Button.styles.ts
│   │   │   │   └── index.ts
│   │   │   └── Input/
│   │   ├── layout/         # 레이아웃 컴포넌트
│   │   │   ├── Header/
│   │   │   ├── Footer/
│   │   │   └── Sidebar/
│   │   └── ui/             # UI 관련 컴포넌트
│   ├── pages/              # 페이지 컴포넌트
│   │   ├── Home/
│   │   │   ├── Home.tsx
│   │   │   ├── Home.styles.ts
│   │   │   └── index.ts
│   │   ├── About/
│   │   └── Contact/
│   ├── hooks/              # 커스텀 Hooks
│   │   ├── useAuth.ts
│   │   ├── useApi.ts
│   │   └── useLocalStorage.ts
│   ├── services/           # API 및 외부 서비스
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   └── storage.ts
│   ├── utils/              # 유틸리티 함수
│   │   ├── helpers.ts
│   │   ├── constants.ts
│   │   └── validators.ts
│   ├── types/              # TypeScript 타입 정의
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   └── common.ts
│   ├── store/              # 상태 관리
│   │   ├── slices/
│   │   └── index.ts
│   ├── styles/             # 전역 스타일
│   │   ├── globals.css
│   │   ├── variables.css
│   │   └── mixins.css
│   ├── assets/             # 이미지, 폰트 등
│   │   ├── images/
│   │   └── fonts/
│   ├── App.tsx             # 루트 컴포넌트
│   ├── main.tsx            # 애플리케이션 진입점
│   └── vite-env.d.ts       # Vite 타입 선언
├── docs/                   # 문서
├── tests/                  # 테스트 파일
├── .env.example            # 환경 변수 예시
├── .gitignore
├── package.json
├── tsconfig.json
└── README.md
```

## 디렉토리별 설명

### `/src/components`

재사용 가능한 UI 컴포넌트를 위치시킵니다. 각 컴포넌트는 별도의 폴더에 구성하고, 관련 파일(컴포넌트, 스타일, 테스트, 인덱스)을 함께 포함합니다.

**장점:**
- 컴포넌트를 쉽게 재사용할 수 있습니다
- 관련 파일이 함께 위치하여 관리가 용이합니다
- 테스트 파일이 컴포넌트 근처에 있어 접근성이 좋습니다

### `/src/pages`

애플리케이션의 페이지 레벨 컴포넌트를 위치시킵니다. 각 페이지는 해당 페이지와 관련된 모든 컴포넌트, 스타일, 로직을 포함할 수 있습니다.

### `/src/hooks`

재사용 가능한 커스텀 Hooks를 위치시킵니다. 상태 관리 로직, API 호출 로직 등을 컴포넌트에서 분리하여 재사용성을 높입니다.

### `/src/services`

API 호출, 인증, 외부 서비스와의 통신 등을 담당합니다. 비즈니스 로직을 컴포넌트에서 분리하여 테스트와 유지보수를 용이하게 합니다.

### `/src/utils`

순수 함수, 유틸리티 함수, 상수 등을 위치시킵니다. 애플리케이션 전역에서 사용되는 헬퍼 함수들을 모아둡니다.

### `/src/types`

TypeScript 타입 정의를 위치시킵니다. API 응답 타입, 컴포넌트 Props 타입 등을 중앙에서 관리합니다.

## 파일 명명 규칙

### 컴포넌트 파일

- **PascalCase**: `Button.tsx`, `UserProfile.tsx`
- **테스트 파일**: `Button.test.tsx`, `UserProfile.test.tsx`
- **스타일 파일**: `Button.styles.ts`, `UserProfile.styles.ts`

### 유틸리티 및 Hook 파일

- **camelCase**: `useAuth.ts`, `formatDate.ts`, `apiClient.ts`

### 폴더 구조

- **소문자**: `components/`, `pages/`, `utils/`
- **복수형**: `components/`, `hooks/`, `services/`

## 인덱스 파일 활용

각 컴포넌트 폴더에 `index.ts` 파일을 만들어 import 경로를 간결하게 유지합니다:

```typescript
// components/Button/index.ts
export { default } from './Button';
export type { ButtonProps } from './Button';

// 사용 시
import Button from '@/components/common/Button'; // ✅ 좋음
// import Button from '@/components/common/Button/Button'; // ❌ 피하기
```

## 절대 경로 설정

Vite나 Webpack에서 절대 경로를 설정하여 상대 경로의 복잡성을 줄입니다:

```typescript
// vite.config.ts
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },
});
```

## 기능별 구조 (Feature-based Structure)

대규모 애플리케이션에서는 기능별로 폴더를 구성하는 것도 좋은 방법입니다:

```
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   └── index.ts
│   ├── products/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   └── index.ts
│   └── users/
├── shared/
│   ├── components/
│   ├── hooks/
│   ├── utils/
│   └── types/
```

## 프로젝트 구조 선택 가이드

### 소규모 프로젝트
- 타입별 구조(Type-based structure)를 권장
- 간단한 구조로 시작하여 필요에 따라 확장

### 중규모 프로젝트
- 타입별 구조와 기능별 구조의 혼합을 권장
- 공통 컴포넌트와 기능별 컴포넌트를 분리

### 대규모 프로젝트
- 기능별 구조(Feature-based structure)를 권장
- 마이크로 프론트엔드 아키텍처 고려

## 결론

효과적인 프로젝트 구조는 다음을 보장합니다:
- 코드의 재사용성
- 유지보수의 용이성
- 팀원 간의 협업 효율성
- 애플리케이션의 확장성

프로젝트의 규모와 복잡성에 맞는 구조를 선택하고, 일관성을 유지하는 것이 중요합니다.