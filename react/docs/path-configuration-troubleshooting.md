# 절대 경로 설정 문제 해결

이 문서는 절대 경로 설정 시 발생할 수 있는 일반적인 문제들과 해결 방법을 설명합니다.

## 1. 모듈을 찾을 수 없는 오류

```
Cannot find module '@components/Button' or its corresponding type declarations.
```

### 원인
- Vite 설정과 TypeScript 설정이 일치하지 않음
- 경로 별칭이 올바르게 설정되지 않음
- 개발 서버가 재시작되지 않음

### 해결 방법
1. `vite.config.ts`와 `tsconfig.app.json`에 동일한 경로 설정이 있는지 확인
2. 설정 후 개발 서버 재시작
3. TypeScript 서버 재시작 (VSCode: Ctrl+Shift+P → "TypeScript: Restart TS Server")

## 2. 경로 자동 완성이 작동하지 않음

### 원인
- TypeScript 설정 문제
- IDE 설정 문제

### 해결 방법
1. `tsconfig.app.json`에 `baseUrl`과 `paths`가 올바르게 설정되었는지 확인
2. VSCode에서 TypeScript 버전이 워크스페이스 버전으로 설정되어 있는지 확인
3. VSCode 설정에서 TypeScript 자동 완성이 활성화되어 있는지 확인

## 3. 빌드 시 모듈 해결 오류

### 원인
- 운영체제별 경로 구분자 차이
- 잘못된 경로 해결 방식 사용

### 해결 방법
1. Windows에서는 `path.resolve(__dirname, './src')` 대신 `path.resolve(process.cwd(), 'src')` 사용
2. 경로 구분자 문제를 피하기 위해 `path.join()` 대신 `path.resolve()` 사용
3. 절대 경로를 사용할 때는 항상 `path.resolve()` 사용

## 4. 테스트 환경에서 절대 경로가 작동하지 않음

### 원인
- 테스트 프레임워크가 별도의 설정을 사용함

### 해결 방법
1. Jest 설정 파일에 moduleNameMapper 추가:
```json
{
  "moduleNameMapper": {
    "^@components/(.*)$": "<rootDir>/src/components/$1",
    "^@hooks/(.*)$": "<rootDir>/src/hooks/$1",
    "^@utils/(.*)$": "<rootDir>/src/utils/$1"
  }
}
```

2. Vitest 설정 파일에 alias 추가:
```typescript
import { defineConfig } from 'vitest/config'
import path from 'path'

export default defineConfig({
  test: {
    // 테스트 설정
  },
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

## 5. ESLint에서 모듈 해결 오류

### 원인
- ESLint가 절대 경로를 인식하지 못함

### 해결 방법
1. `eslint-import-resolver-alias` 플러그인 설치:
```bash
npm install -D eslint-import-resolver-alias
```

2. ESLint 설정 파일에 설정 추가:
```json
{
  "settings": {
    "import/resolver": {
      "alias": {
        "map": [
          ["@", "./src"],
          ["@components", "./src/components"],
          ["@hooks", "./src/hooks"],
          ["@utils", "./src/utils"]
        ],
        "extensions": [".ts", ".tsx", ".js", ".jsx"]
      }
    }
  }
}
```

## 6. 동적 import에서 절대 경로가 작동하지 않음

### 원인
- 동적 import는 번들러의 경로 해석을 따르지 않을 수 있음

### 해결 방법
1. Vite에서는 별도 설정 없이 동작해야 함
2. 문제가 발생하면 전체 경로를 사용하거나 별도의 유틸리티 함수 생성:
```typescript
// src/utils/pathResolver.ts
export const resolvePath = (path: string) => {
  switch (path) {
    case '@components': return '/src/components';
    case '@hooks': return '/src/hooks';
    case '@utils': return '/src/utils';
    default: return path;
  }
};

// 사용 예시
const module = await import(resolvePath('@components/Button'));
```

## 7. 라이브러리 개발 시 절대 경로 문제

### 원인
- 라이브러리 빌드 시 절대 경로가 상대 경로로 변환되지 않음

### 해결 방법
1. 빌드 도구 설정에서 경로 변환 확인
2. 라이브러리 배포 시에는 상대 경로 사용 권장
3. TypeScript의 `paths` 설정은 런타임이 아닌 컴파일 타임에만 적용됨을 인지

## 일반적인 문제 해결 순서

1. **설정 확인**: Vite와 TypeScript 설정이 일치하는지 확인
2. **서버 재시작**: 개발 서버와 TypeScript 서버 재시작
3. **캐시 삭제**: `node_modules/.vite` 디렉토리 삭제 후 재설치
4. **IDE 확인**: IDE 설정과 TypeScript 버전 확인
5. **빌드 테스트**: 실제 빌드 환경에서 테스트