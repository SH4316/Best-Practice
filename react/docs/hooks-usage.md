# React Hooks 사용법

React Hooks는 함수형 컴포넌트에서 상태와 생명주기 기능을 사용할 수 있게 해주는 함수입니다. 올바르게 사용하면 코드의 재사용성과 가독성이 크게 향상됩니다.

## 기본 Hooks

### 1. useState

컴포넌트에 상태를 추가할 때 사용합니다.

```typescript
// ✅ 좋은 예시: 기본 useState 사용
const Counter = () => {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
};

// ✅ 좋은 예시: 함수형 업데이트 사용
const Counter = () => {
  const [count, setCount] = useState(0);
  
  const increment = () => {
    setCount(prevCount => prevCount + 1);
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>Increment</button>
    </div>
  );
};

// ✅ 좋은 예시: 객체 상태 관리
const UserProfile = () => {
  const [user, setUser] = useState({
    name: '',
    email: '',
    age: 0,
  });
  
  const updateName = (name: string) => {
    setUser(prevUser => ({ ...prevUser, name }));
  };
  
  return (
    <div>
      <input 
        value={user.name} 
        onChange={(e) => updateName(e.target.value)} 
      />
      <p>Name: {user.name}</p>
    </div>
  );
};
```

### 2. useEffect

사이드 이펙트를 처리할 때 사용합니다.

```typescript
// ❌ 나쁜 예시: 의존성 배열을 잘못 사용
const UserProfile = ({ userId }) => {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(data => setUser(data));
  }, []); // userId가 변경되어도 다시 fetch하지 않음
  
  return <div>{user?.name}</div>;
};

// ✅ 좋은 예시: 올바른 의존성 배열 사용
const UserProfile = ({ userId }) => {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    const fetchUser = async () => {
      const response = await fetch(`/api/users/${userId}`);
      const data = await response.json();
      setUser(data);
    };
    
    fetchUser();
  }, [userId]); // userId가 변경될 때마다 재실행
  
  return <div>{user?.name}</div>;
};

// ✅ 좋은 예시: 클린업 함수 사용
const Timer = () => {
  const [seconds, setSeconds] = useState(0);
  
  useEffect(() => {
    const intervalId = setInterval(() => {
      setSeconds(prevSeconds => prevSeconds + 1);
    }, 1000);
    
    // 컴포넌트 언마운트 시 정리
    return () => clearInterval(intervalId);
  }, []); // 빈 배열: 컴포넌트 마운트 시 한 번만 실행
  
  return <div>Seconds: {seconds}</div>;
};

// ✅ 좋은 예시: 조건부 사이드 이펙트
const DataFetcher = ({ shouldFetch, userId }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    if (!shouldFetch) return;
    
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/users/${userId}`);
        const userData = await response.json();
        setData(userData);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [shouldFetch, userId]);
  
  if (loading) return <div>Loading...</div>;
  return <div>{data?.name}</div>;
};
```

### 3. useContext

Context를 통해 컴포넌트 트리 전체에 데이터를 전달할 때 사용합니다.

```typescript
// ✅ 좋은 예시: Theme Context 생성
const ThemeContext = createContext({
  theme: 'light',
  toggleTheme: () => {},
});

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// 사용 예시
const Header = () => {
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

### 4. useReducer

복잡한 상태 로직을 관리할 때 useState보다 유용합니다.

```typescript
// ✅ 좋은 예시: useReducer를 사용한 TODO 상태 관리
interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

type TodoAction =
  | { type: 'ADD_TODO'; text: string }
  | { type: 'TOGGLE_TODO'; id: number }
  | { type: 'DELETE_TODO'; id: number };

const todoReducer = (todos: Todo[], action: TodoAction): Todo[] => {
  switch (action.type) {
    case 'ADD_TODO':
      return [
        ...todos,
        { id: Date.now(), text: action.text, completed: false },
      ];
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

const TodoApp = () => {
  const [todos, dispatch] = useReducer(todoReducer, []);
  const [inputValue, setInputValue] = useState('');
  
  const addTodo = () => {
    if (inputValue.trim()) {
      dispatch({ type: 'ADD_TODO', text: inputValue });
      setInputValue('');
    }
  };
  
  return (
    <div>
      <input
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Add a todo"
      />
      <button onClick={addTodo}>Add</button>
      
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => dispatch({ type: 'TOGGLE_TODO', id: todo.id })}
            />
            {todo.text}
            <button onClick={() => dispatch({ type: 'DELETE_TODO', id: todo.id })}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};
```

## 추가 Hooks

### 1. useMemo

비용이 많이 드는 계산을 메모이제이션할 때 사용합니다.

```typescript
// ❌ 나쁜 예시: 불필요한 재계산
const ExpensiveComponent = ({ items, filter }) => {
  const filteredItems = items.filter(item => item.type === filter);
  const sortedItems = filteredItems.sort((a, b) => a.name.localeCompare(b.name));
  
  return (
    <ul>
      {sortedItems.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
};

// ✅ 좋은 예시: useMemo로 재계산 방지
const ExpensiveComponent = ({ items, filter }) => {
  const filteredAndSortedItems = useMemo(() => {
    return items
      .filter(item => item.type === filter)
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [items, filter]);
  
  return (
    <ul>
      {filteredAndSortedItems.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
};

// ✅ 좋은 예시: 복잡한 객체 참조 안정화
const UserProfile = ({ user }) => {
  const userSettings = useMemo(() => ({
    theme: user.preferences.theme,
    language: user.preferences.language,
    notifications: user.preferences.notifications,
  }), [user.preferences]);
  
  return <SettingsPanel settings={userSettings} />;
};
```

### 2. useCallback

함수 참조를 안정화시켜 불필요한 리렌더링을 방지할 때 사용합니다.

```typescript
// ❌ 나쁜 예시: 매 렌더링마다 새 함수 생성
const ParentComponent = () => {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    console.log('Button clicked');
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <ChildComponent onClick={handleClick} />
    </div>
  );
};

// ✅ 좋은 예시: useCallback으로 함수 참조 안정화
const ParentComponent = () => {
  const [count, setCount] = useState(0);
  
  const handleClick = useCallback(() => {
    console.log('Button clicked');
  }, []); // 빈 의존성 배열: 함수가 한 번만 생성됨
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <ChildComponent onClick={handleClick} />
    </div>
  );
};

// ✅ 좋은 예시: 의존성이 있는 함수
const TodoItem = ({ todo, onUpdate }) => {
  const handleToggle = useCallback(() => {
    onUpdate(todo.id, { completed: !todo.completed });
  }, [todo.id, todo.completed, onUpdate]);
  
  const handleDelete = useCallback(() => {
    onUpdate(todo.id, { deleted: true });
  }, [todo.id, onUpdate]);
  
  return (
    <li>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={handleToggle}
      />
      {todo.text}
      <button onClick={handleDelete}>Delete</button>
    </li>
  );
};
```

### 3. useRef

DOM 요소에 직접 접근하거나 렌더링을 유발하지 않는 값을 저장할 때 사용합니다.

```typescript
// ✅ 좋은 예시: DOM 요소 참조
const TextInputWithFocusButton = () => {
  const inputRef = useRef<HTMLInputElement>(null);
  
  const onButtonClick = () => {
    inputRef.current?.focus();
  };
  
  return (
    <>
      <input ref={inputRef} type="text" />
      <button onClick={onButtonClick}>Focus the input</button>
    </>
  );
};

// ✅ 좋은 예시: 렌더링을 유발하지 않는 값 저장
const Timer = () => {
  const [count, setCount] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const startTimer = () => {
    if (intervalRef.current) return; // 이미 실행 중이면 시작하지 않음
    
    intervalRef.current = setInterval(() => {
      setCount(prevCount => prevCount + 1);
    }, 1000);
  };
  
  const stopTimer = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={startTimer}>Start</button>
      <button onClick={stopTimer}>Stop</button>
    </div>
  );
};

// ✅ 좋은 예시: 이전 값 저장
const PreviousValue = ({ value }) => {
  const prevValueRef = useRef();
  
  useEffect(() => {
    prevValueRef.current = value;
  }); // 렌더링 후 실행되어 이전 값을 저장
  
  return (
    <div>
      <p>Current: {value}</p>
      <p>Previous: {prevValueRef.current}</p>
    </div>
  );
};
```

## 커스텀 Hooks

커스텀 Hook은 상태 로직을 재사용 가능하게 만드는 강력한 방법입니다.

```typescript
// ✅ 좋은 예시: 데이터 페칭 Hook
interface UseFetchResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

const useFetch = <T,>(url: string): UseFetchResult<T> => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [url]);
  
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  
  return { data, loading, error, refetch: fetchData };
};

// 사용 예시
const UserProfile = ({ userId }) => {
  const { data: user, loading, error, refetch } = useFetch(`/api/users/${userId}`);
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!user) return <div>No user found</div>;
  
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      <button onClick={refetch}>Refresh</button>
    </div>
  );
};

// ✅ 좋은 예시: 로컬 스토리지 Hook
const useLocalStorage = <T,>(key: string, initialValue: T) => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });
  
  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);
  
  return [storedValue, setValue] as const;
};

// 사용 예시
const Settings = () => {
  const [theme, setTheme] = useLocalStorage('theme', 'light');
  const [language, setLanguage] = useLocalStorage('language', 'en');
  
  return (
    <div>
      <select value={theme} onChange={(e) => setTheme(e.target.value)}>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
      
      <select value={language} onChange={(e) => setLanguage(e.target.value)}>
        <option value="en">English</option>
        <option value="ko">Korean</option>
      </select>
    </div>
  );
};

// ✅ 좋은 예시: 디바운스 Hook
const useDebounce = <T,>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);
  
  return debouncedValue;
};

// 사용 예시
const SearchComponent = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const debouncedSearchTerm = useDebounce(searchTerm, 500);
  
  useEffect(() => {
    if (debouncedSearchTerm) {
      // API 호출 등 검색 로직
      console.log('Searching for:', debouncedSearchTerm);
    }
  }, [debouncedSearchTerm]);
  
  return (
    <input
      type="text"
      value={searchTerm}
      onChange={(e) => setSearchTerm(e.target.value)}
      placeholder="Search..."
    />
  );
};
```

## Hooks 사용 규칙

1. **최상위 레벨에서만 Hook 호출**: Hook은 반복문, 조건문, 중첩된 함수 내에서 호출할 수 없습니다.
2. **React 함수 컴포넌트에서만 Hook 호출**: 일반 JavaScript 함수에서 Hook을 호출할 수 없습니다.

```typescript
// ❌ 나쁜 예시: 조건부 Hook 호출
const UserProfile = ({ user }) => {
  if (!user) {
    const [loading, setLoading] = useState(false); // 조건부 Hook 호출
    return <div>Loading...</div>;
  }
  
  return <div>{user.name}</div>;
};

// ✅ 좋은 예시: 항상 Hook 호출
const UserProfile = ({ user }) => {
  const [loading, setLoading] = useState(false);
  
  if (!user) {
    return <div>Loading...</div>;
  }
  
  return <div>{user.name}</div>;
};
```

## 결론

React Hooks를 올바르게 사용하면 다음과 같은 이점이 있습니다:
- 컴포넌트 로직의 재사용성 증가
- 코드의 가독성 향상
- 클래스 컴포넌트의 복잡성 감소
- 상태 관리의 용이성

이러한 원칙과 패턴을 따르면 더 효율적이고 유지보수하기 쉬운 React 애플리케이션을 만들 수 있습니다.