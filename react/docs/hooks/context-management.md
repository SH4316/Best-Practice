# 컨텍스트 관리

`useContext`는 React Context를 통해 컴포넌트 트리 전체에 데이터를 전달할 때 사용하는 Hook입니다. Props drilling(여러 계층에 걸쳐 props를 전달)을 피하고 전역 상태를 관리하는 데 유용합니다.

## 기본 사용법

### Context 생성 및 사용

```typescript
// ✅ 좋은 예시: Theme Context 생성
import React, { createContext, useContext, useState, ReactNode } from 'react';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'light',
  toggleTheme: () => {},
});

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('light');
  
  const toggleTheme = (): void => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// 사용 예시
const Header: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <header className={`header header--${theme}`}>
      <h1>My App</h1>
      <button onClick={toggleTheme}>
        Toggle to {theme === 'light' ? 'dark' : 'light'} mode
      </button>
    </header>
  );
};
```

### 애플리케이션에 Provider 적용

```typescript
// App.tsx
import React from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { Header } from './components/Header';

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <div className="app">
        <Header />
        {/* 다른 컴포넌트들 */}
      </div>
    </ThemeProvider>
  );
};

export default App;
```

## 고급 사용 패턴

### 여러 Context 결합

```typescript
// AuthContext.tsx
interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  login: async () => {},
  logout: () => {},
});

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  
  const login = async (email: string, password: string): Promise<void> => {
    // 로그인 로직
    const userData = await api.login(email, password);
    setUser(userData);
  };
  
  const logout = (): void => {
    setUser(null);
  };
  
  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
```

### Context와 useReducer 결합

```typescript
// TodoContext.tsx
interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

type TodoAction =
  | { type: 'ADD_TODO'; text: string }
  | { type: 'TOGGLE_TODO'; id: number }
  | { type: 'DELETE_TODO'; id: number };

interface TodoContextType {
  todos: Todo[];
  dispatch: React.Dispatch<TodoAction>;
}

const TodoContext = createContext<TodoContextType>({
  todos: [],
  dispatch: () => {},
});

const todoReducer = (todos: Todo[], action: TodoAction): Todo[] => {
  switch (action.type) {
    case 'ADD_TODO':
      return [...todos, { id: Date.now(), text: action.text, completed: false }];
    case 'TOGGLE_TODO':
      return todos.map(todo =>
        todo.id === action.id ? { ...todo, completed: !todo.completed } : todo
      );
    case 'DELETE_TODO':
      return todos.filter(todo => todo.id !== action.id);
    default:
      return todos;
  }
};

export const TodoProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [todos, dispatch] = useReducer(todoReducer, []);
  
  return (
    <TodoContext.Provider value={{ todos, dispatch }}>
      {children}
    </TodoContext.Provider>
  );
};

export const useTodos = (): TodoContextType => {
  const context = useContext(TodoContext);
  if (!context) {
    throw new Error('useTodos must be used within a TodoProvider');
  }
  return context;
};
```

### 여러 Provider 결합

```typescript
// AppProviders.tsx
import React from 'react';
import { ThemeProvider } from './ThemeContext';
import { AuthProvider } from './AuthContext';
import { TodoProvider } from './TodoContext';

export const AppProviders: React.FC<{ children: ReactNode }> = ({ children }) => {
  return (
    <AuthProvider>
      <ThemeProvider>
        <TodoProvider>
          {children}
        </TodoProvider>
      </ThemeProvider>
    </AuthProvider>
  );
};

// App.tsx
import React from 'react';
import { AppProviders } from './providers/AppProviders';
import { AppContent } from './components/AppContent';

const App: React.FC = () => {
  return (
    <AppProviders>
      <AppContent />
    </AppProviders>
  );
};
```

## Context 최적화

### 불필요한 리렌더링 방지

```typescript
// ❌ 나쁜 예시: 전체 context 객체가 변경될 때마다 리렌더링
const UserProfile: React.FC = () => {
  const { user, theme, toggleTheme, login, logout } = useContext(AppContext);
  
  return <div>{user?.name}</div>; // user만 사용하지만 theme 변경 시에도 리렌더링됨
};

// ✅ 좋은 예시: Context 분리
const UserContext = createContext<{ user: User | null }>({ user: null });
const ThemeContext = createContext<{ theme: Theme; toggleTheme: () => void }>({ 
  theme: 'light', 
  toggleTheme: () => {} 
});

const UserProfile: React.FC = () => {
  const { user } = useContext(UserContext); // user 변경 시에만 리렌더링
  
  return <div>{user?.name}</div>;
};
```

### 값 메모이제이션

```typescript
// ✅ 좋은 예시: useMemo로 값 메모이제이션
export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('light');
  
  const toggleTheme = (): void => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };
  
  const value = useMemo(() => ({ theme, toggleTheme }), [theme]);
  
  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
```

## Context 사용 시 주의사항

### 1. Context 값 변경 빈도

자주 변경되는 값은 Context에 저장하지 않는 것이 좋습니다. 예를 들어, 마우스 위치와 같이 빈번하게 업데이트되는 값은 Context에 적합하지 않습니다.

```typescript
// ❌ 나쁜 예시: 빈번하게 변경되는 값을 Context에 저장
const MouseContext = createContext<{ x: number; y: number }>({ x: 0, y: 0 });

// ✅ 좋은 예시: 커스텀 Hook으로 로컬 상태 관리
const useMousePosition = () => {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);
  
  return position;
};
```

### 2. Context 구조화

너무 많은 데이터를 단일 Context에 넣지 마세요. 관련된 데이터끼리 그룹화하여 여러 Context로 나누는 것이 좋습니다.

```typescript
// ❌ 나쁜 예시: 모든 상태를 하나의 Context에 저장
const AppContext = createContext<{
  user: User | null;
  theme: Theme;
  todos: Todo[];
  notifications: Notification[];
  // ... 다른 많은 상태들
}>({
  user: null,
  theme: 'light',
  todos: [],
  notifications: [],
});

// ✅ 좋은 예시: 관련된 데이터끼리 그룹화
const UserContext = createContext<{ user: User | null }>({ user: null });
const ThemeContext = createContext<{ theme: Theme; toggleTheme: () => void }>({ 
  theme: 'light', 
  toggleTheme: () => {} 
});
const TodoContext = createContext<{ todos: Todo[] }>({ todos: [] });
```

## 다음 단계

상태 관리에 대한 더 자세한 내용은 [상태 관리](./state-management.md) 문서를 참조하세요.

커스텀 Hook 만들기에 대해서는 [커스텀 Hooks](./custom-hooks.md) 문서를 확인하세요.

성능 최적화에 대해서는 [성능 최적화](./performance-optimization.md) 문서를 참조하세요.