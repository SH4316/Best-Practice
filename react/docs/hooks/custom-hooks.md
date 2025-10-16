# 커스텀 Hooks

커스텀 Hook은 상태 로직을 재사용 가능하게 만드는 강력한 방법입니다. 커스텀 Hook은 이름이 `use`로 시작하고, 다른 Hook을 호출할 수 있는 JavaScript 함수입니다. 이를 통해 컴포넌트 간에 상태 로직을 공유할 수 있습니다.

## 기본 구조

```typescript
// 커스텀 Hook 기본 구조
const useCustomHook = (parameter: Type): ReturnType => {
  // 내부에서 다른 Hook 사용 가능
  const [state, setState] = useState<Type>(initialValue);
  
  // 로직 처리
  
  return { state, setState };
};
```

## 데이터 페칭 Hook

### 기본 데이터 페칭

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
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  const fetchData = useCallback(async (): Promise<void> => {
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
interface UserProfileProps {
  userId: string | number;
}

interface UserData {
  name: string;
  email: string;
}

const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const { data: user, loading, error, refetch } = useFetch<UserData>(`/api/users/${userId}`);
  
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
```

### 조건부 데이터 페칭

```typescript
const useConditionalFetch = <T,>(
  url: string | null,
  dependencies: any[] = []
): UseFetchResult<T> => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  const fetchData = useCallback(async (): Promise<void> => {
    if (!url) return;
    
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
  }, [url, ...dependencies]);
  
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  
  return { data, loading, error, refetch: fetchData };
};
```

## 로컬 스토리지 Hook

```typescript
// ✅ 좋은 예시: 로컬 스토리지 Hook
const useLocalStorage = <T,>(
  key: string, 
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void] => {
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
const Settings: React.FC = () => {
  const [theme, setTheme] = useLocalStorage<string>('theme', 'light');
  const [language, setLanguage] = useLocalStorage<string>('language', 'en');
  
  return (
    <div>
      <select value={theme} onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setTheme(e.target.value)}>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
      
      <select value={language} onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setLanguage(e.target.value)}>
        <option value="en">English</option>
        <option value="ko">Korean</option>
      </select>
    </div>
  );
};
```

## 디바운스 Hook

```typescript
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
const SearchComponent: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const debouncedSearchTerm = useDebounce<string>(searchTerm, 500);
  
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
      onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
      placeholder="Search..."
    />
  );
};
```

## 윈도우 크기 Hook

```typescript
const useWindowSize = (): { width: number; height: number } => {
  const [windowSize, setWindowSize] = useState<{ width: number; height: number }>({
    width: window.innerWidth,
    height: window.innerHeight,
  });
  
  useEffect(() => {
    const handleResize = (): void => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  return windowSize;
};

// 사용 예시
const ResponsiveComponent: React.FC = () => {
  const { width, height } = useWindowSize();
  
  return (
    <div>
      <p>Window size: {width} x {height}</p>
      {width < 768 ? <p>Mobile view</p> : <p>Desktop view</p>}
    </div>
  );
};
```

## 이전 값 Hook

```typescript
const usePrevious = <T,>(value: T): T | undefined => {
  const ref = useRef<T>();
  
  useEffect(() => {
    ref.current = value;
  });
  
  return ref.current;
};

// 사용 예시
const CounterWithPrevious: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  const previousCount = usePrevious(count);
  
  return (
    <div>
      <p>Current: {count}</p>
      <p>Previous: {previousCount}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
};
```

## 토글 Hook

```typescript
const useToggle = (initialValue: boolean = false): [boolean, () => void] => {
  const [value, setValue] = useState<boolean>(initialValue);
  
  const toggle = useCallback((): void => {
    setValue(prev => !prev);
  }, []);
  
  return [value, toggle];
};

// 사용 예시
const ToggleComponent: React.FC = () => {
  const [isVisible, toggleVisible] = useToggle(false);
  
  return (
    <div>
      <button onClick={toggleVisible}>
        {isVisible ? 'Hide' : 'Show'}
      </button>
      {isVisible && <p>Now you can see me!</p>}
    </div>
  );
};
```

## 배열 상태 관리 Hook

```typescript
interface UseArrayActions<T> {
  add: (item: T) => void;
  remove: (index: number) => void;
  update: (index: number, item: T) => void;
  clear: () => void;
}

const useArray = <T,>(initialValue: T[] = []): { array: T[]; actions: UseArrayActions<T> } => {
  const [array, setArray] = useState<T[]>(initialValue);
  
  const actions: UseArrayActions<T> = {
    add: useCallback((item: T) => {
      setArray(prev => [...prev, item]);
    }, []),
    
    remove: useCallback((index: number) => {
      setArray(prev => prev.filter((_, i) => i !== index));
    }, []),
    
    update: useCallback((index: number, item: T) => {
      setArray(prev => prev.map((element, i) => i === index ? item : element));
    }, []),
    
    clear: useCallback(() => {
      setArray([]);
    }, []),
  };
  
  return { array, actions };
};

// 사용 예시
const TodoList: React.FC = () => {
  const { array: todos, actions } = useArray<string>(['Learn React', 'Learn TypeScript']);
  
  return (
    <div>
      <ul>
        {todos.map((todo, index) => (
          <li key={index}>
            {todo}
            <button onClick={() => actions.remove(index)}>Remove</button>
          </li>
        ))}
      </ul>
      <button onClick={() => actions.add('New Task')}>Add Task</button>
    </div>
  );
};
```

## 커스텀 Hook 만들 때 고려사항

### 1. 이름 규칙

커스텀 Hook은 항상 `use`로 시작해야 합니다. 이는 React의 규칙이며, linter가 Hook 사용 규칙을 확인하는 데 도움이 됩니다.

```typescript
// ✅ 좋은 예시
const useUserData = () => { /* ... */ };
const useFetchData = () => { /* ... */ };

// ❌ 나쁜 예시
const getUserData = () => { /* ... */ };
const fetchData = () => { /* ... */ };
```

### 2. 의존성 배열 관리

커스텀 Hook 내부에서 `useEffect`, `useMemo`, `useCallback` 등을 사용할 때 의존성 배열을 올바르게 관리해야 합니다.

```typescript
// ✅ 좋은 예시: 의존성 배열 올바르게 사용
const useApi = (url: string) => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(setData);
  }, [url]); // url이 변경될 때만 재실행
  
  return data;
};
```

### 3. 타입 안정성

TypeScript를 사용할 때는 명확한 타입 정의가 중요합니다.

```typescript
// ✅ 좋은 예시: 명확한 타입 정의
interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

const useApi = <T,>(url: string): UseApiResult<T> => {
  // 구현
};
```

### 4. 에러 처리

커스텀 Hook에서는 에러 처리를 고려해야 합니다.

```typescript
// ✅ 좋은 예시: 에러 처리 포함
const useApi = <T,>(url: string) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [url]);
  
  return { data, loading, error };
};
```

## 다음 단계

Hooks 사용 규칙에 대해서는 [Hooks 사용 규칙](./hooks-rules.md) 문서를 참조하세요.

성능 최적화에 대해서는 [성능 최적화](./performance-optimization.md) 문서를 확인하세요.