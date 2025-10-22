# React Hooks 사용법

React Hooks는 함수형 컴포넌트에서 상태와 생명주기 기능을 사용할 수 있게 해주는 함수입니다. 올바르게 사용하면 코드의 재사용성과 가독성이 크게 향상됩니다.

## 목차

이 문서는 사용 케이스별로 분리된 React Hooks 가이드입니다:

- [상태 관리](./state-management.md) - `useState`와 `useReducer`를 사용한 컴포넌트 상태 관리
- [사이드 이펙트 처리](./side-effects.md) - `useEffect`를 사용한 사이드 이펙트 및 생명주기 관리
- [컨텍스트 관리](./context-management.md) - `useContext`를 사용한 전역 상태 공유
- [성능 최적화](./performance-optimization.md) - `useMemo`와 `useCallback`을 사용한 렌더링 최적화
- [DOM 상호작용](./dom-interaction.md) - `useRef`를 사용한 DOM 접근 및 값 저장
- [커스텀 Hooks](./custom-hooks.md) - 재사용 가능한 상태 로직 만들기
- [Hooks 사용 규칙](./hooks-rules.md) - Hooks 사용을 위한 규칙과 모범 사례

## 시작하기

React Hooks를 처음 사용하는 경우, [상태 관리](./state-management.md)부터 시작하여 점진적으로 다른 주제를 탐색하는 것을 권장합니다.

## 왜 Hooks를 사용해야 할까요?

React Hooks를 올바르게 사용하면 다음과 같은 이점이 있습니다:
- 컴포넌트 로직의 재사용성 증가
- 코드의 가독성 향상
- 클래스 컴포넌트의 복잡성 감소
- 상태 관리의 용이성

이러한 원칙과 패턴을 따르면 더 효율적이고 유지보수하기 쉬운 React 애플리케이션을 만들 수 있습니다.