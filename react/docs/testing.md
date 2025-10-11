# React 테스트

테스트는 안정적이고 신뢰할 수 있는 React 애플리케이션을 개발하는 데 필수적인 부분입니다. 적절한 테스트 전략을 통해 코드 변경으로 인한 회귀를 방지하고, 리팩토링을 안전하게 수행하며, 문서로서의 역할도 할 수 있습니다. 이 문서에서는 React 애플리케이션을 테스트하는 다양한 기법과 모범 사례를 설명합니다.

## 테스트 유형

### 1. 단위 테스트 (Unit Testing)

단위 테스트는 개별 함수, 컴포넌트 또는 모듈을 격리된 환경에서 테스트합니다.

```typescript
// ❌ 나쁜 예시: 구현 세부 사항에 의존하는 테스트
import { render, screen } from '@testing-library/react';
import UserProfile from './UserProfile';

// 컴포넌트 내부 상태를 직접 테스트
test('should set user name correctly', () => {
  render(<UserProfile userId="123" />);
  
  // 내부 상태를 직접 확인하는 안티패턴
  expect(screen.getByTestId('user-name')).toHaveTextContent('John Doe');
});

// ✅ 좋은 예시: 사용자 관점에서 테스트
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UserProfile from './UserProfile';

// 사용자가 보는 것을 테스트
test('should display user information', async () => {
  render(<UserProfile userId="123" />);
  
  // 로딩 상태 확인
  expect(screen.getByText('Loading...')).toBeInTheDocument();
  
  // 데이터 로딩 후 사용자 정보 확인
  await waitFor(() => {
    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('john@example.com')).toBeInTheDocument();
  });
});
```

### 2. 통합 테스트 (Integration Testing)

통합 테스트는 여러 컴포넌트나 모듈이 함께 작동하는 방식을 테스트합니다.

```typescript
// ✅ 좋은 예시: 여러 컴포넌트의 상호작용 테스트
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TodoApp from './TodoApp';

test('should add and complete a todo', async () => {
  render(<TodoApp />);
  
  // 할 일 추가
  const input = screen.getByPlaceholderText('What needs to be done?');
  const addButton = screen.getByRole('button', { name: 'Add' });
  
  await userEvent.type(input, 'Learn React testing');
  await userEvent.click(addButton);
  
  // 할 일이 추가되었는지 확인
  await waitFor(() => {
    expect(screen.getByText('Learn React testing')).toBeInTheDocument();
  });
  
  // 할 일 완료
  const checkbox = screen.getByRole('checkbox');
  await userEvent.click(checkbox);
  
  // 할 일이 완료되었는지 확인
  await waitFor(() => {
    expect(screen.getByText('Learn React testing')).toHaveClass('completed');
  });
});
```

### 3. E2E 테스트 (End-to-End Testing)

E2E 테스트는 실제 사용자 시나리오를 시뮬레이션하여 전체 애플리케이션 흐름을 테스트합니다.

```typescript
// ✅ 좋은 예시: Cypress를 사용한 E2E 테스트
describe('User Authentication Flow', () => {
  it('should allow a user to sign up, log in, and log out', () => {
    // 회원가입
    cy.visit('/signup');
    cy.get('[data-testid="username-input"]').type('testuser');
    cy.get('[data-testid="email-input"]').type('test@example.com');
    cy.get('[data-testid="password-input"]').type('password123');
    cy.get('[data-testid="signup-button"]').click();
    
    // 회원가입 성공 확인
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="welcome-message"]').should('contain', 'Welcome, testuser');
    
    // 로그아웃
    cy.get('[data-testid="logout-button"]').click();
    
    // 로그아웃 성공 확인
    cy.url().should('include', '/login');
    
    // 로그인
    cy.get('[data-testid="email-input"]').type('test@example.com');
    cy.get('[data-testid="password-input"]').type('password123');
    cy.get('[data-testid="login-button"]').click();
    
    // 로그인 성공 확인
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="welcome-message"]').should('contain', 'Welcome, testuser');
  });
});
```

## 테스트 도구

### 1. React Testing Library

React Testing Library는 컴포넌트를 사용자가 사용하는 방식으로 테스트하는 데 중점을 둡니다.

```typescript
// ✅ 좋은 예시: React Testing Library 사용
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Counter from './Counter';

describe('Counter Component', () => {
  test('should increment count when increment button is clicked', () => {
    render(<Counter />);
    
    // 초기 상태 확인
    expect(screen.getByText('Count: 0')).toBeInTheDocument();
    
    // 버튼 클릭
    const incrementButton = screen.getByRole('button', { name: /increment/i });
    fireEvent.click(incrementButton);
    
    // 상태 변경 확인
    expect(screen.getByText('Count: 1')).toBeInTheDocument();
  });
  
  test('should show error message when count exceeds 10', async () => {
    render(<Counter maxCount={10} />);
    
    // 10번 클릭
    const incrementButton = screen.getByRole('button', { name: /increment/i });
    for (let i = 0; i < 10; i++) {
      fireEvent.click(incrementButton);
    }
    
    // 11번째 클릭
    fireEvent.click(incrementButton);
    
    // 에러 메시지 확인
    await waitFor(() => {
      expect(screen.getByText('Maximum count reached')).toBeInTheDocument();
    });
  });
});
```

### 2. Jest Mock

Jest Mock을 사용하여 외부 의존성을 격리하고 테스트를 더 예측 가능하게 만듭니다.

```typescript
// ✅ 좋은 예시: API 모킹
import { render, screen, waitFor } from '@testing-library/react';
import UserProfile from './UserProfile';

// API 모듈 모킹
jest.mock('../api/userService', () => ({
  getUser: jest.fn(),
}));

import { getUser } from '../api/userService';

describe('UserProfile with mocked API', () => {
  beforeEach(() => {
    // 모든 모킹 초기화
    jest.clearAllMocks();
  });
  
  test('should display user data from API', async () => {
    // 모킹 데이터 설정
    const mockUser = {
      id: '123',
      name: 'John Doe',
      email: 'john@example.com',
    };
    
    getUser.mockResolvedValue(mockUser);
    
    // 컴포넌트 렌더링
    render(<UserProfile userId="123" />);
    
    // API 호출 확인
    expect(getUser).toHaveBeenCalledWith('123');
    
    // 데이터 표시 확인
    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('john@example.com')).toBeInTheDocument();
    });
  });
  
  test('should display error message when API fails', async () => {
    // API 에러 모킹
    getUser.mockRejectedValue(new Error('Failed to fetch user'));
    
    // 컴포넌트 렌더링
    render(<UserProfile userId="123" />);
    
    // 에러 메시지 확인
    await waitFor(() => {
      expect(screen.getByText('Failed to load user profile')).toBeInTheDocument();
    });
  });
});
```

## 컴포넌트 테스트

### 1. 사용자 상호작용 테스트

사용자가 컴포넌트와 상호작용하는 방식을 테스트합니다.

```typescript
// ✅ 좋은 예시: 사용자 상호작용 테스트
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LoginForm from './LoginForm';

describe('LoginForm', () => {
  test('should show validation errors for empty fields', async () => {
    render(<LoginForm />);
    
    // 제출 버튼 클릭
    const submitButton = screen.getByRole('button', { name: 'Log In' });
    await userEvent.click(submitButton);
    
    // 유효성 검사 에러 확인
    await waitFor(() => {
      expect(screen.getByText('Email is required')).toBeInTheDocument();
      expect(screen.getByText('Password is required')).toBeInTheDocument();
    });
  });
  
  test('should call onSubmit with form data when valid form is submitted', async () => {
    const mockOnSubmit = jest.fn();
    render(<LoginForm onSubmit={mockOnSubmit} />);
    
    // 폼 입력
    const emailInput = screen.getByLabelText('Email');
    const passwordInput = screen.getByLabelText('Password');
    
    await userEvent.type(emailInput, 'test@example.com');
    await userEvent.type(passwordInput, 'password123');
    
    // 폼 제출
    const submitButton = screen.getByRole('button', { name: 'Log In' });
    await userEvent.click(submitButton);
    
    // onSubmit 호출 확인
    expect(mockOnSubmit).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password123',
    });
  });
});
```

### 2. 비동기 컴포넌트 테스트

데이터 페칭과 같은 비동기 작업을 처리하는 컴포넌트를 테스트합니다.

```typescript
// ✅ 좋은 예시: 비동기 컴포넌트 테스트
import { render, screen, waitFor } from '@testing-library/react';
import UserList from './UserList';

// API 모듈 모킹
jest.mock('../api/userService', () => ({
  getUsers: jest.fn(),
}));

import { getUsers } from '../api/userService';

describe('UserList', () => {
  test('should display loading state initially', () => {
    // API 호출이 해결되지 않도록 모킹
    getUsers.mockImplementation(() => new Promise(() => {}));
    
    render(<UserList />);
    
    // 로딩 상태 확인
    expect(screen.getByText('Loading users...')).toBeInTheDocument();
  });
  
  test('should display user list when data is loaded', async () => {
    // 모킹 데이터 설정
    const mockUsers = [
      { id: '1', name: 'John Doe', email: 'john@example.com' },
      { id: '2', name: 'Jane Smith', email: 'jane@example.com' },
    ];
    
    getUsers.mockResolvedValue(mockUsers);
    
    render(<UserList />);
    
    // 로딩 상태 확인
    expect(screen.getByText('Loading users...')).toBeInTheDocument();
    
    // 사용자 목록 확인
    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('jane@example.com')).toBeInTheDocument();
    });
  });
  
  test('should display error message when API fails', async () => {
    // API 에러 모킹
    getUsers.mockRejectedValue(new Error('Failed to fetch users'));
    
    render(<UserList />);
    
    // 에러 메시지 확인
    await waitFor(() => {
      expect(screen.getByText('Failed to load users')).toBeInTheDocument();
    });
  });
});
```

## Hook 테스트

### 1. 커스텀 Hook 테스트

커스텀 Hook을 테스트하기 위해 `renderHook` 함수를 사용합니다.

```typescript
// ✅ 좋은 예시: 커스텀 Hook 테스트
import { renderHook, act } from '@testing-library/react';
import { useCounter } from './useCounter';

describe('useCounter', () => {
  test('should initialize with default value', () => {
    const { result } = renderHook(() => useCounter());
    
    expect(result.current.count).toBe(0);
  });
  
  test('should initialize with provided value', () => {
    const { result } = renderHook(() => useCounter(5));
    
    expect(result.current.count).toBe(5);
  });
  
  test('should increment count', () => {
    const { result } = renderHook(() => useCounter());
    
    act(() => {
      result.current.increment();
    });
    
    expect(result.current.count).toBe(1);
  });
  
  test('should decrement count', () => {
    const { result } = renderHook(() => useCounter(5));
    
    act(() => {
      result.current.decrement();
    });
    
    expect(result.current.count).toBe(4);
  });
  
  test('should not go below min value', () => {
    const { result } = renderHook(() => useCounter(0, { min: 0 }));
    
    act(() => {
      result.current.decrement();
    });
    
    expect(result.current.count).toBe(0);
  });
});
```

## 테스트 전략

### 1. 테스트 피라미드

테스트 피라미드는 다양한 유형의 테스트 간의 균형을 나타냅니다.

```
    /\
   /  \  E2E 테스트 (소수)
  /____\
 /      \ 통합 테스트 (더 많음)
/________\
단위 테스트 (가장 많음)
```

- **단위 테스트**: 빠르고, 격리되어 있으며, 개별 기능을 테스트
- **통합 테스트**: 여러 컴포넌트가 함께 작동하는 방식을 테스트
- **E2E 테스트**: 전체 사용자 시나리오를 테스트

### 2. 테스트 커버리지

코드 커버리지는 테스트가 코드의 얼마나 많은 부분을 실행하는지 측정합니다.

```json
// package.json
{
  "scripts": {
    "test": "jest",
    "test:coverage": "jest --coverage",
    "test:watch": "jest --watch"
  }
}
```

```javascript
// jest.config.js
module.exports = {
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.{js,jsx,ts,tsx}',
    '!src/serviceWorker.ts',
    '!src/setupTests.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};
```

## 테스트 모범 사례

### 1. 테스트 작성 원칙

- **사용자 관점에서 테스트**: 사용자가 애플리케이션을 어떻게 사용하는지에 중점을 둡니다.
- **구현 세부 사항 테스트 금지**: 컴포넌트 내부 상태나 구현 대신 사용자가 보는 것을 테스트합니다.
- **설명적인 테스트 이름**: 테스트가 무엇을 확인하는지 명확하게 설명합니다.
- **한 가지 사항만 테스트**: 각 테스트는 하나의 동작이나 상태만 확인해야 합니다.

### 2. 테스트 구조

```typescript
// ✅ 좋은 예시: 테스트 구조
describe('ComponentName', () => {
  // 설정
  beforeEach(() => {
    // 각 테스트 전에 실행할 코드
  });
  
  // 테스트 케이스
  describe('when user interacts with component', () => {
    test('should do something when user does something', () => {
      // Arrange (준비)
      // 테스트에 필요한 데이터와 환경 설정
      
      // Act (실행)
      // 테스트할 동작 실행
      
      // Assert (단언)
      // 예상 결과 확인
    });
  });
});
```

### 3. 테스트 데이터 관리

```typescript
// ✅ 좋은 예시: 테스트 데이터 팩토리
const createUser = (overrides = {}) => ({
  id: '1',
  name: 'John Doe',
  email: 'john@example.com',
  ...overrides,
});

test('should display user information', () => {
  const user = createUser({ name: 'Jane Smith' });
  
  render(<UserProfile user={user} />);
  
  expect(screen.getByText('Jane Smith')).toBeInTheDocument();
});
```

## 결론

효과적인 테스트 전략은 다음 원칙을 따라야 합니다:
- 사용자 관점에서 테스트
- 구현 세부 사항이 아닌 동작 테스트
- 다양한 유형의 테스트 균형 유지
- 테스트 커버리지 모니터링
- 명확하고 유지보수 가능한 테스트 작성

이러한 원칙을 따르면 안정적이고 신뢰할 수 있는 React 애플리케이션을 만들 수 있습니다.