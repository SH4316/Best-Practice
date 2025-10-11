# React Best Practices

이 저장소는 React 개발 시 따라야 할 모범 사례(Best Practices)를 정리하고 실제 코드 예제를 통해 설명하는 자료입니다.

## 목차

1. [프로젝트 구조](./docs/project-structure.md)
2. [컴포넌트 구조](./docs/component-structure.md)
3. [상태 관리](./docs/state-management.md)
4. [Hooks 사용법](./docs/hooks-usage.md)
5. [성능 최적화](./docs/performance-optimization.md)
6. [코드 조직 및 파일 구조](./docs/code-organization.md)
7. [에러 처리](./docs/error-handling.md)
8. [접근성 (Accessibility)](./docs/accessibility.md)
9. [테스트](./docs/testing.md)
10. [보안](./docs/security.md)
11. [스타일링](./docs/styling.md)

## 시작하기

이 프로젝트는 Vite + React + TypeScript로 설정되었습니다.

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 빌드
npm run build

# 린트 확인
npm run lint
```

## 실습 예제

각 베스트 프랙티스 카테고리별로 실제 코드 예제가 `src/examples` 디렉토리에 포함되어 있습니다. 각 예제는 다음과 같은 구조를 따릅니다:

```
src/examples/
├── component-structure/
│   ├── good/
│   └── bad/
├── state-management/
│   ├── good/
│   └── bad/
└── ...
```

## 학습 목표

이 자료를 통해 다음을 학습할 수 있습니다:

- React 컴포넌트를 효과적으로 구조화하는 방법
- 상태 관리의 모범 사례
- Hooks를 올바르게 사용하는 방법
- React 애플리케이션의 성능을 최적화하는 방법
- 유지보수 가능한 코드를 작성하는 방법
- 접근성을 고려한 컴포넌트를 만드는 방법
- 효과적인 테스트 전략
- 보안 취약점을 방지하는 방법

## 기여

이 자료는 계속 업데이트될 예정입니다. 추가할 베스트 프랙티스가 있다면 Pull Request를 제출해주세요.

## 라이선스

MIT License