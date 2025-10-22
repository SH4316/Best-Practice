# 사이드 이펙트 처리

`useEffect`는 함수형 컴포넌트에서 사이드 이펙트(side effects)를 처리할 때 사용하는 Hook입니다. 데이터 페칭, 구독, 수동 DOM 조작 등 컴포넌트 렌더링 외부에서 발생하는 작업을 처리할 수 있습니다.

## 기본 사용법

### 올바른 의존성 배열 사용

의존성 배열은 `useEffect`가 언제 다시 실행되어야 하는지를 결정합니다. 의존성 배열을 올바르게 사용하는 것이 중요합니다.

```typescript
// ❌ 나쁜 예시: 의존성 배열을 잘못 사용
interface UserProfileProps {
  userId: string | number;
}

const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const [user, setUser] = useState<any>(null);
  
  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(data => setUser(data));
  }, []); // userId가 변경되어도 다시 fetch하지 않음
  
  return <div>{user?.name}</div>;
};

// ✅ 좋은 예시: 올바른 의존성 배열 사용
const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const [user, setUser] = useState<any>(null);
  
  useEffect(() => {
    const fetchUser = async (): Promise<void> => {
      const response = await fetch(`/api/users/${userId}`);
      const data = await response.json();
      setUser(data);
    };
    
    fetchUser();
  }, [userId]); // userId가 변경될 때마다 재실행
  
  return <div>{user?.name}</div>;
};
```

### 클린업 함수 사용

`useEffect`에서 리턴하는 함수는 컴포넌트가 언마운트되거나 의존성이 변경되어 effect가 다시 실행되기 전에 호출됩니다. 이를 통해 리소스 정리를 할 수 있습니다.

```typescript
// ✅ 좋은 예시: 클린업 함수 사용
const Timer: React.FC = () => {
  const [seconds, setSeconds] = useState<number>(0);
  
  useEffect(() => {
    const intervalId = setInterval(() => {
      setSeconds(prevSeconds => prevSeconds + 1);
    }, 1000);
    
    // 컴포넌트 언마운트 시 정리
    return (): void => clearInterval(intervalId);
  }, []); // 빈 배열: 컴포넌트 마운트 시 한 번만 실행
  
  return <div>Seconds: {seconds}</div>;
};
```

### 조건부 사이드 이펙트

때로는 특정 조건이 충족될 때만 사이드 이펙트를 실행하고 싶을 수 있습니다.

```typescript
// ✅ 좋은 예시: 조건부 사이드 이펙트
interface DataFetcherProps {
  shouldFetch: boolean;
  userId: string | number;
}

const DataFetcher: React.FC<DataFetcherProps> = ({ shouldFetch, userId }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  
  useEffect(() => {
    if (!shouldFetch) return;
    
    const fetchData = async (): Promise<void> => {
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

## 일반적인 사용 사례

### API 호출

```typescript
const useFetch = <T,>(url: string): { data: T | null; loading: boolean; error: string | null } => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchData = async (): Promise<void> => {
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
    };
    
    fetchData();
  }, [url]);
  
  return { data, loading, error };
};
```

### 이벤트 리스너

```typescript
const useKeyPress = (targetKey: string): boolean => {
  const [keyPressed, setKeyPressed] = useState<boolean>(false);
  
  useEffect(() => {
    const downHandler = (event: KeyboardEvent): void => {
      if (event.key === targetKey) {
        setKeyPressed(true);
      }
    };
    
    const upHandler = (event: KeyboardEvent): void => {
      if (event.key === targetKey) {
        setKeyPressed(false);
      }
    };
    
    window.addEventListener('keydown', downHandler);
    window.addEventListener('keyup', upHandler);
    
    // 클린업
    return (): void => {
      window.removeEventListener('keydown', downHandler);
      window.removeEventListener('keyup', upHandler);
    };
  }, [targetKey]);
  
  return keyPressed;
};
```

## useEffect 사용 규칙

### 1. 의존성 배열 규칙

- `useEffect` 내부에서 사용하는 모든 상태와 props는 의존성 배열에 포함해야 합니다
- ESLint의 `react-hooks/exhaustive-deps` 규칙을 사용하면 누락된 의존성을 쉽게 찾을 수 있습니다

### 2. 객체와 함수 의존성

```typescript
// ❌ 나쁜 예시: 객체 참조가 매번 변경되어 effect가 계속 실행됨
const MyComponent: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  const options = { enabled: true }; // 매 렌더링마다 새 객체 생성
  
  useEffect(() => {
    // options가 변경될 때마다 실행됨
    console.log('Effect ran');
  }, [options]);
  
  return <div>Count: {count}</div>;
};

// ✅ 좋은 예시: useMemo로 객체 참조 안정화
const MyComponent: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  const options = useMemo(() => ({ enabled: true }), []); // 참조 안정화
  
  useEffect(() => {
    // options가 변경될 때만 실행됨
    console.log('Effect ran');
  }, [options]);
  
  return <div>Count: {count}</div>;
};
```

### 3. 비동기 함수 사용

```typescript
// ❌ 나쁜 예시: useEffect에 비동기 함수를 직접 전달
useEffect(async () => {
  const data = await fetchData();
  setData(data);
}, [url]);

// ✅ 좋은 예시: 내부에 비동기 함수 정의
useEffect(() => {
  const fetchDataAsync = async (): Promise<void> => {
    const data = await fetchData();
    setData(data);
  };
  
  fetchDataAsync();
}, [url]);
```

## 성능 최적화

### 불필요한 effect 실행 방지

```typescript
// ✅ 좋은 예시: 조건부 실행으로 불필요한 API 호출 방지
const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const [user, setUser] = useState<any>(null);
  
  useEffect(() => {
    if (!userId) return; // userId가 없으면 실행하지 않음
    
    const fetchUser = async (): Promise<void> => {
      const response = await fetch(`/api/users/${userId}`);
      const data = await response.json();
      setUser(data);
    };
    
    fetchUser();
  }, [userId]);
  
  return <div>{user?.name}</div>;
};
```

## 다음 단계

성능 최적화에 대해서는 [성능 최적화](./performance-optimization.md) 문서를 확인하세요.

커스텀 Hook 만들기에 대해서는 [커스텀 Hooks](./custom-hooks.md) 문서를 참조하세요.