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

절대 경로(Absolute Path)는 프로젝트 루트에서 시작하는 경로를 사용하여 파일을 import하는 방식입니다. 이를 통해 상대 경로의 복잡성을 줄이고 코드의 가독성을 향상시킬 수 있습니다.

### 절대 경로 사용의 장점

1. **가독성 향상**: `../../../components/Button`과 같은 복잡한 상대 경로 대신 `@components/Button` 사용
2. **리팩토링 용이성**: 파일 위치가 변경되어도 import 경로 수정 필요 없음
3. **일관성 유지**: 프로젝트 전체에서 일관된 import 패턴 사용
4. **IDE 지원**: 대부분의 IDE에서 자동 완성 및 빠른 탐색 지원

### Vite에서 절대 경로 설정

Vite 프로젝트에서 절대 경로를 설정하려면 `vite.config.ts` 파일을 수정해야 합니다:

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },
})
```

### TypeScript 설정

절대 경로를 TypeScript에서 인식하려면 `tsconfig.app.json` 파일에 경로 매핑을 추가해야 합니다:

```json
{
  "compilerOptions": {
    // ... 기존 설정
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@hooks/*": ["src/hooks/*"],
      "@utils/*": ["src/utils/*"]
    }
    // ... 나머지 설정
  },
  "include": ["src"]
}
```

### 사용 예시

절대 경로 설정 후 다음과 같이 import를 사용할 수 있습니다:

```typescript
// 상대 경로 사용 (이전)
import Button from '../../../components/common/Button/Button';
import { useAuth } from '../../../hooks/useAuth';
import { formatDate } from '../../../utils/helpers';

// 절대 경로 사용 (개선 후)
import Button from '@components/common/Button';
import { useAuth } from '@hooks/useAuth';
import { formatDate } from '@utils/helpers';
```

### 절대 경로 마이그레이션 방법

기존 프로젝트를 절대 경로로 마이그레이션하는 방법:

1. **설정 추가**: 위에서 설명한 대로 Vite와 TypeScript 설정 추가
2. **점진적 변환**: 한 파일씩 또는 한 디렉토리씩 절대 경로로 변환
3. **자동화 도구 사용**: VSCode의 "바꾸기 기능"으로 일괄 변환 가능
4. **테스트**: 각 변환 후 애플리케이션이 정상적으로 작동하는지 확인

### 추가 경로 별칭 설정

프로젝트가 커질 경우 필요에 따라 추가 경로 별칭을 설정할 수 있습니다:

```typescript
// vite.config.ts
export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@utils': path.resolve(__dirname, './src/utils'),
      // 필요에 따라 추가
      '@pages': path.resolve(__dirname, './src/pages'),
      '@services': path.resolve(__dirname, './src/services'),
      '@types': path.resolve(__dirname, './src/types'),
    },
  },
});
```

이 경우 `tsconfig.app.json`에도 동일한 경로를 추가해야 합니다.

### 문제 해결

절대 경로 설정 시 발생할 수 있는 문제들과 해결 방법은 [별도의 문서](./path-configuration-troubleshooting.md)에서 자세히 설명합니다.

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