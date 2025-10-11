# React 접근성 (Accessibility)

접근성은 모든 사용자가 웹 애플리케이션을 사용할 수 있도록 보장하는 중요한 측면입니다. React 애플리케이션에서 접근성을 구현하는 것은 법적 요구사항을 충족할 뿐만 아니라, 더 넓은 사용자층에게 서비스를 제공하고 전반적인 사용자 경험을 향상시킵니다. 이 문서에서는 React 애플리케이션의 접근성을 구현하는 다양한 기법과 모범 사례를 설명합니다.

## 시맨틱 HTML

### 1. 올바른 HTML 요소 사용

의미에 맞는 HTML 요소를 사용하는 것은 접근성의 기본입니다.

```typescript
// ❌ 나쁜 예시: div를 버튼으로 사용
const BadButton = ({ onClick, children }) => {
  return (
    <div 
      onClick={onClick}
      className="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          onClick();
          e.preventDefault();
        }
      }}
    >
      {children}
    </div>
  );
};

// ✅ 좋은 예시: button 요소 사용
const GoodButton = ({ onClick, children }) => {
  return (
    <button 
      onClick={onClick}
      className="button"
    >
      {children}
    </button>
  );
};
```

### 2. 제목 구조

논리적인 제목 구조를 사용하여 콘텐츠의 계층을 명확하게 합니다.

```typescript
// ❌ 나쁜 예시: 제목 구조가 논리적이지 않음
const BadPageStructure = () => {
  return (
    <div>
      <h1>Main Title</h1>
      <h3>Subtitle</h3>
      <h2>Another Section</h2>
      <h4>Subsection</h4>
    </div>
  );
};

// ✅ 좋은 예시: 논리적인 제목 구조
const GoodPageStructure = () => {
  return (
    <div>
      <h1>Main Title</h1>
      <h2>First Section</h2>
      <h3>Subsection</h3>
      <h2>Second Section</h2>
      <h3>Subsection</h3>
    </div>
  );
};
```

## ARIA 속성

### 1. ARIA 레이블

요소에 명확한 레이블을 제공합니다.

```typescript
// ❌ 나쁜 예시: 아이콘 버튼에 레이블이 없음
const BadIconButton = () => {
  return (
    <button onClick={() => console.log('clicked')}>
      <svg>...</svg>
    </button>
  );
};

// ✅ 좋은 예시: aria-label로 레이블 제공
const GoodIconButton = () => {
  return (
    <button 
      onClick={() => console.log('clicked')}
      aria-label="Close dialog"
    >
      <svg>...</svg>
    </button>
  );
};

// ✅ 좋은 예시: aria-labelledby로 레이블 제공
const GoodInputWithLabel = () => {
  const inputId = useId();
  
  return (
    <div>
      <label id={inputId}>Username</label>
      <input 
        aria-labelledby={inputId}
        type="text"
      />
    </div>
  );
};
```

### 2. ARIA 역할 및 상태

요소의 역할과 상태를 명확하게 표시합니다.

```typescript
// ❌ 나쁜 예시: 커스텀 체크박스에 접근성 정보 없음
const BadCustomCheckbox = ({ checked, onChange }) => {
  return (
    <div 
      className={`checkbox ${checked ? 'checked' : ''}`}
      onClick={() => onChange(!checked)}
    >
      {checked ? '✓' : ''}
    </div>
  );
};

// ✅ 좋은 예시: ARIA 속성으로 역할과 상태 제공
const GoodCustomCheckbox = ({ checked, onChange }) => {
  return (
    <div 
      className={`checkbox ${checked ? 'checked' : ''}`}
      role="checkbox"
      aria-checked={checked}
      tabIndex={0}
      onClick={() => onChange(!checked)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          onChange(!checked);
          e.preventDefault();
        }
      }}
    >
      {checked ? '✓' : ''}
    </div>
  );
};
```

## 키보드 접근성

### 1. 포커스 관리

모든 상호작용 가능한 요소는 키보드로 접근할 수 있어야 합니다.

```typescript
// ❌ 나쁜 예시: 포커스가 불가능한 커스텀 컴포넌트
const BadCustomComponent = () => {
  return (
    <div 
      onClick={() => console.log('clicked')}
      style={{ cursor: 'pointer' }}
    >
      Click me
    </div>
  );
};

// ✅ 좋은 예시: 키보드 접근 가능한 컴포넌트
const GoodCustomComponent = () => {
  const [isFocused, setIsFocused] = useState(false);
  
  return (
    <div 
      onClick={() => console.log('clicked')}
      onFocus={() => setIsFocused(true)}
      onBlur={() => setIsFocused(false)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          console.log('clicked');
          e.preventDefault();
        }
      }}
      tabIndex={0}
      role="button"
      style={{ 
        cursor: 'pointer',
        outline: isFocused ? '2px solid blue' : 'none'
      }}
    >
      Click me
    </div>
  );
};
```

### 2. 포커스 트랩

모달과 같은 컴포넌트에서 포커스를 트랩합니다.

```typescript
// ✅ 좋은 예시: 모달에서 포커스 트랩
const Modal = ({ isOpen, onClose, children }) => {
  const modalRef = useRef(null);
  const previousFocusRef = useRef(null);
  
  // 모달이 열릴 때 이전 포커스 저장
  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement;
      
      // 모달 내의 첫 번째 포커스 가능 요소로 포커스 이동
      const focusableElements = modalRef.current?.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      
      if (focusableElements && focusableElements.length > 0) {
        (focusableElements[0] as HTMLElement).focus();
      }
    } else {
      // 모달이 닫힐 때 이전 포커스 복원
      previousFocusRef.current?.focus();
    }
  }, [isOpen]);
  
  // Tab 키로 포커스 이동 시 모달 내에서 순환
  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      const focusableElements = modalRef.current?.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      
      if (focusableElements && focusableElements.length > 0) {
        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;
        
        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            lastElement.focus();
            e.preventDefault();
          }
        } else {
          if (document.activeElement === lastElement) {
            firstElement.focus();
            e.preventDefault();
          }
        }
      }
    }
    
    // ESC 키로 모달 닫기
    if (e.key === 'Escape') {
      onClose();
    }
  };
  
  if (!isOpen) return null;
  
  return (
    <div 
      className="modal-overlay"
      ref={modalRef}
      onKeyDown={handleKeyDown}
      role="dialog"
      aria-modal="true"
    >
      {children}
    </div>
  );
};
```

## 스크린 리더 지원

### 1. 라이브 영역

동적으로 변경되는 콘텐츠를 스크린 리더 사용자에게 알립니다.

```typescript
// ❌ 나쁜 예시: 상태 변경을 알리지 않음
const BadStatusMessage = () => {
  const [status, setStatus] = useState('Loading');
  
  useEffect(() => {
    setTimeout(() => setStatus('Complete'), 2000);
  }, []);
  
  return <div>{status}</div>;
};

// ✅ 좋은 예시: aria-live로 상태 변경 알림
const GoodStatusMessage = () => {
  const [status, setStatus] = useState('Loading');
  
  useEffect(() => {
    setTimeout(() => setStatus('Complete'), 2000);
  }, []);
  
  return (
    <div aria-live="polite" aria-atomic="true">
      {status}
    </div>
  );
};
```

### 2. 설명 텍스트

복잡한 컴포넌트에 추가 설명을 제공합니다.

```typescript
// ✅ 좋은 예시: aria-describedby로 설명 제공
const ComplexInput = () => {
  const inputId = useId();
  const descriptionId = useId();
  
  return (
    <div>
      <label htmlFor={inputId}>Password</label>
      <input 
        id={inputId}
        type="password"
        aria-describedby={descriptionId}
      />
      <div id={descriptionId}>
        Password must be at least 8 characters long and contain both letters and numbers.
      </div>
    </div>
  );
};
```

## 색상 및 대비

### 1. 색상 대비

텍스트와 배경 색상 사이에 충분한 대비를 제공합니다.

```css
/* ❌ 나쁜 예시: 대비가 낮은 색상 */
.low-contrast {
  color: #cccccc;
  background-color: #ffffff;
}

/* ✅ 좋은 예시: 대비가 높은 색상 */
.high-contrast {
  color: #333333;
  background-color: #ffffff;
}
```

### 2. 색상만으로 정보 전달 금지

색상 외에 다른 방법으로도 정보를 전달합니다.

```typescript
// ❌ 나쁜 예시: 색상만으로 오류 상태 표시
const BadErrorField = ({ hasError, children }) => {
  return (
    <input 
      className={hasError ? 'error' : ''}
      style={{ borderColor: hasError ? 'red' : 'gray' }}
    />
  );
};

// ✅ 좋은 예시: 아이콘과 텍스트로 오류 상태 표시
const GoodErrorField = ({ hasError, children }) => {
  return (
    <div>
      <input 
        className={hasError ? 'error' : ''}
        style={{ borderColor: hasError ? 'red' : 'gray' }}
        aria-invalid={hasError}
        aria-describedby={hasError ? 'error-message' : undefined}
      />
      {hasError && (
        <div id="error-message" className="error-message">
          <span aria-hidden="true">⚠️</span>
          This field is required
        </div>
      )}
    </div>
  );
};
```

## 폼 접근성

### 1. 레이블 연결

모든 폼 컨트롤에 레이블을 연결합니다.

```typescript
// ❌ 나쁜 예시: 레이블이 없는 입력 필드
const BadInputField = () => {
  return <input type="text" placeholder="Enter your name" />;
};

// ✅ 좋은 예시: 명시적 레이블 연결
const GoodInputField = () => {
  const inputId = useId();
  
  return (
    <div>
      <label htmlFor={inputId}>Name</label>
      <input id={inputId} type="text" />
    </div>
  );
};

// ✅ 좋은 예시: 암시적 레이블 연결
const GoodInputFieldImplicit = () => {
  return (
    <div>
      <label>
        Name:
        <input type="text" />
      </label>
    </div>
  );
};
```

### 2. 폼 유효성 검사

폼 유효성 검사 결과를 접근 가능하게 표시합니다.

```typescript
// ✅ 좋은 예시: 접근 가능한 폼 유효성 검사
const AccessibleForm = () => {
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');
  const emailId = useId();
  const emailErrorId = useId();
  
  const validateEmail = (value) => {
    if (!value) {
      setEmailError('Email is required');
      return false;
    }
    
    if (!/\S+@\S+\.\S+/.test(value)) {
      setEmailError('Please enter a valid email address');
      return false;
    }
    
    setEmailError('');
    return true;
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (validateEmail(email)) {
      // 폼 제출 로직
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor={emailId}>Email</label>
        <input
          id={emailId}
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          onBlur={() => validateEmail(email)}
          aria-invalid={!!emailError}
          aria-describedby={emailError ? emailErrorId : undefined}
        />
        {emailError && (
          <div id={emailErrorId} className="error-message">
            {emailError}
          </div>
        )}
      </div>
      
      <button type="submit">Submit</button>
    </form>
  );
};
```

## 접근성 테스트 도구

### 1. 자동화된 테스트

접근성 테스트 도구를 사용하여 문제를 식별합니다.

```json
// package.json
{
  "scripts": {
    "test:a11y": "jest --testPathPattern=a11y",
    "lint:a11y": "eslint --ext .js,.jsx,.ts,.tsx src/ --rule @eslint-js/a11y"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^5.16.5",
    "jest-axe": "^6.1.1",
    "@eslint/js/a11y": "^8.7.0"
  }
}
```

```typescript
// ✅ 좋은 예시: jest-axe로 접근성 테스트
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import MyComponent from './MyComponent';

expect.extend(toHaveNoViolations);

test('should not have accessibility violations', async () => {
  const { container } = render(<MyComponent />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

### 2. 수동 테스트

자동화된 테스트와 함께 수동 테스트도 수행합니다.

- 키보드만으로 모든 기능 사용 가능한지 확인
- 스크린 리더로 콘텐츠 탐색 가능한지 확인
- 색상 대비 및 확대/축소 기능 확인
- 다양한 보조 기술과의 호환성 확인

## 결론

효과적인 접근성 구현은 다음 원칙을 따라야 합니다:
- 시맨틱 HTML 사용
- 적절한 ARIA 속성 추가
- 키보드 접근성 보장
- 스크린 리더 지원
- 충분한 색상 대비 제공
- 폼 접근성 향상
- 자동화 및 수동 테스트 수행

이러한 원칙을 따르면 모든 사용자가 접근할 수 있는 포용적인 React 애플리케이션을 만들 수 있습니다.