# React Hooks 사용법

> **중요**: 이 문서는 사용 케이스별로 분리되었습니다. 아래 링크를 통해 더 체계적으로 구성된 문서를 확인하세요.

## 새로운 문서 구조

React Hooks 문서가 사용 케이스별로 재구성되었습니다. 각 주제에 대한 자세한 내용은 아래 링크를 참조하세요:

### 📚 [React Hooks 가이드](./hooks/)

React Hooks에 대한 포괄적인 가이드입니다. 아래 주제별 문서를 통해 필요한 정보를 찾아보세요.

#### 주요 주제

- **[상태 관리](./hooks/state-management.md)** - `useState`와 `useReducer`를 사용한 컴포넌트 상태 관리
- **[사이드 이펙트 처리](./hooks/side-effects.md)** - `useEffect`를 사용한 사이드 이펙트 및 생명주기 관리
- **[컨텍스트 관리](./hooks/context-management.md)** - `useContext`를 사용한 전역 상태 공유
- **[성능 최적화](./hooks/performance-optimization.md)** - `useMemo`와 `useCallback`을 사용한 렌더링 최적화
- **[DOM 상호작용](./hooks/dom-interaction.md)** - `useRef`를 사용한 DOM 접근 및 값 저장
- **[커스텀 Hooks](./hooks/custom-hooks.md)** - 재사용 가능한 상태 로직 만들기
- **[Hooks 사용 규칙](./hooks/hooks-rules.md)** - Hooks 사용을 위한 규칙과 모범 사례

## React Hooks란?

React Hooks는 함수형 컴포넌트에서 상태와 생명주기 기능을 사용할 수 있게 해주는 함수입니다. 올바르게 사용하면 코드의 재사용성과 가독성이 크게 향상됩니다.

### 왜 Hooks를 사용해야 할까요?

React Hooks를 올바르게 사용하면 다음과 같은 이점이 있습니다:
- 컴포넌트 로직의 재사용성 증가
- 코드의 가독성 향상
- 클래스 컴포넌트의 복잡성 감소
- 상태 관리의 용이성

## 시작하기

React Hooks를 처음 사용하는 경우, [상태 관리](./hooks/state-management.md)부터 시작하여 점진적으로 다른 주제를 탐색하는 것을 권장합니다.

## 예제 코드

다양한 Hooks 사용 예제는 각 주제별 문서에서 확인할 수 있습니다:

- [useState 예제](./hooks/state-management.md#기본-사용법)
- [useEffect 예제](./hooks/side-effects.md#기본-사용법)
- [커스텀 Hook 예제](./hooks/custom-hooks.md#데이터-페칭-hook)

## 기여하기

문서 개선에 대한 제안이나 수정 사항이 있으시면 언제든지 기여해주세요. 각 문서는 해당 사용 케이스에 특화된 내용을 포함하고 있어 필요한 정보를 더 쉽게 찾을 수 있습니다.

---

**이 문서는 [hooks/index.md](./hooks/index.md)로 이동되었습니다.** 위 링크를 통해 더 체계적으로 구성된 문서를 확인하세요.