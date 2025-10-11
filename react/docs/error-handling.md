# React 에러 처리

에러 처리는 모든 애플리케이션에서 중요한 부분입니다. React 애플리케이션에서 에러를 효과적으로 처리하면 사용자 경험을 향상시키고 디버깅을 용이하게 만들 수 있습니다. 이 문서에서는 React 애플리케이션의 에러를 처리하는 다양한 기법과 모범 사례를 설명합니다.

## 에러 경계 (Error Boundaries)

### 1. 에러 경계 기본

에러 경계는 하위 컴포넌트 트리에서 발생한 JavaScript 에러를 catch하고, 에러를 기록하며, 대체 UI를 표시하는 React 컴포넌트입니다.

```typescript
// ❌ 나쁜 예시: 에러 경계가 없는 경우
const UserProfile = ({ userId }) => {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(response => response.json())
      .then(data => setUser(data))
      .catch(error => {
        // 에러가 처리되지 않으면 애플리케이션이 충돌할 수 있음
        console.error('Error fetching user:', error);
      });
  }, [userId]);
  
  if (!user) return <div>Loading...</div>;
  
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      <img src={user.avatar} alt={user.name} />
    </div>
  );
};

// ✅ 좋은 예시: 에러 경계 사용
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    
    // 에러 로깅 서비스에 에러 보내기
    logErrorToService(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="error-fallback">
            <h2>Something went wrong.</h2>
            <details>
              {this.state.error && this.state.error.toString()}
            </details>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

// 사용 예시
const App = () => {
  return (
    <div>
      <ErrorBoundary>
        <UserProfile userId={1} />
      </ErrorBoundary>
      
      <ErrorBoundary>
        <ProductList />
      </ErrorBoundary>
    </div>
  );
};
```

### 2. 함수형 컴포넌트에서의 에러 경계

함수형 컴포넌트에서는 클래스 컴포넌트로 에러 경계를 만들어 사용합니다.

```typescript
// ✅ 좋은 예시: 함수형 컴포넌트에서 에러 경 boundary 사용
const withErrorBoundary = (Component: React.ComponentType, fallback?: React.ReactNode) => {
  const WrappedComponent = (props: any) => (
    <ErrorBoundary fallback={fallback}>
      <Component {...props} />
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
};

// 사용 예시
const UserProfileWithErrorBoundary = withErrorBoundary(UserProfile, 
  <div>Failed to load user profile</div>
);
```

## 비동기 에러 처리

### 1. Promise 및 async/await 에러 처리

```typescript
// ❌ 나쁜 예시: 비동기 에러 처리가 부족한 경우
const DataFetcher = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const fetchData = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('/api/data');
      const result = await response.json();
      setData(result);
      setLoading(false);
    } catch (error) {
      // 에러가 처리되지 않음
      setLoading(false);
    }
  };
  
  return (
    <div>
      <button onClick={fetchData}>Fetch Data</button>
      {loading ? <p>Loading...</p> : <div>{JSON.stringify(data)}</div>}
    </div>
  );
};

// ✅ 좋은 예시: 적절한 비동기 에러 처리
const DataFetcher = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/data');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setData(result);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Unknown error');
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <button onClick={fetchData}>Fetch Data</button>
      {loading && <p>Loading...</p>}
      {error && <div className="error">Error: {error}</div>}
      {data && <div>{JSON.stringify(data)}</div>}
    </div>
  );
};
```

### 2. 커스텀 Hook을 사용한 에러 처리

```typescript
// ✅ 좋은 예시: 에러 처리 로직을 커스텀 Hook으로 분리
interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

const useApi = <T,>(url: string): UseApiResult<T> => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fetchData = useCallback(async () => {
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
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      console.error('API Error:', err);
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
  const { data: user, loading, error } = useApi(`/api/users/${userId}`);
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div className="error">Error: {error}</div>;
  if (!user) return <div>No user found</div>;
  
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
};
```

## 에러 상태 관리

### 1. 글로벌 에러 상태

```typescript
// ✅ 좋은 예시: 글로벌 에러 상태 관리
interface ErrorState {
  errors: Array<{
    id: string;
    message: string;
    timestamp: Date;
    dismissed: boolean;
  }>;
}

const ErrorContext = createContext<{
  errors: ErrorState['errors'];
  addError: (message: string) => void;
  dismissError: (id: string) => void;
  clearErrors: () => void;
}>({
  errors: [],
  addError: () => {},
  dismissError: () => {},
  clearErrors: () => {},
});

export const ErrorProvider = ({ children }) => {
  const [errors, setErrors] = useState<ErrorState['errors']>([]);
  
  const addError = useCallback((message: string) => {
    const newError = {
      id: Date.now().toString(),
      message,
      timestamp: new Date(),
      dismissed: false,
    };
    
    setErrors(prevErrors => [...prevErrors, newError]);
  }, []);
  
  const dismissError = useCallback((id: string) => {
    setErrors(prevErrors =>
      prevErrors.map(error =>
        error.id === id ? { ...error, dismissed: true } : error
      )
    );
  }, []);
  
  const clearErrors = useCallback(() => {
    setErrors([]);
  }, []);
  
  return (
    <ErrorContext.Provider value={{ errors, addError, dismissError, clearErrors }}>
      {children}
      <ErrorDisplay errors={errors.filter(error => !error.dismissed)} onDismiss={dismissError} />
    </ErrorContext.Provider>
  );
};

export const useError = () => {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error('useError must be used within an ErrorProvider');
  }
  return context;
};

// 에러 표시 컴포넌트
const ErrorDisplay = ({ errors, onDismiss }) => {
  if (errors.length === 0) return null;
  
  return (
    <div className="error-container">
      {errors.map(error => (
        <div key={error.id} className="error-message">
          <span>{error.message}</span>
          <button onClick={() => onDismiss(error.id)}>×</button>
        </div>
      ))}
    </div>
  );
};
```

### 2. 에러 발생 Hook

```typescript
// ✅ 좋은 예시: 에러 발생을 위한 Hook
const useThrowError = () => {
  return (error: Error | string) => {
    throw error instanceof Error ? error : new Error(error);
  };
};

// 사용 예시
const FormComponent = () => {
  const throwOnError = useThrowError();
  const [value, setValue] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!value.trim()) {
      throwOnError('Value cannot be empty');
    }
    
    // 폼 제출 로직
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button type="submit">Submit</button>
    </form>
  );
};
```

## 사용자 피드백

### 1. 에러 메시지 표시

```typescript
// ❌ 나쁜 예시: 사용자 친화적이지 않은 에러 메시지
const BadErrorDisplay = ({ error }) => {
  return (
    <div className="error">
      {error.toString()}
    </div>
  );
};

// ✅ 좋은 예시: 사용자 친화적인 에러 메시지
const GoodErrorDisplay = ({ error, onRetry }) => {
  const getErrorMessage = (error) => {
    if (error.response) {
      // 서버 응답 에러
      switch (error.response.status) {
        case 400:
          return 'Invalid request. Please check your input and try again.';
        case 401:
          return 'You are not authorized. Please log in and try again.';
        case 403:
          return 'You do not have permission to perform this action.';
        case 404:
          return 'The requested resource was not found.';
        case 500:
          return 'Server error. Please try again later.';
        default:
          return 'An unexpected error occurred. Please try again.';
      }
    } else if (error.request) {
      // 네트워크 에러
      return 'Network error. Please check your connection and try again.';
    } else {
      // 기타 에러
      return error.message || 'An unexpected error occurred.';
    }
  };
  
  return (
    <div className="error-display">
      <div className="error-icon">⚠️</div>
      <div className="error-message">
        {getErrorMessage(error)}
      </div>
      {onRetry && (
        <button className="retry-button" onClick={onRetry}>
          Try Again
        </button>
      )}
    </div>
  );
};
```

### 2. 로딩 및 에러 상태 결합

```typescript
// ✅ 좋은 예시: 로딩 및 에러 상태를 결합한 컴포넌트
const DataComponent = ({ url }) => {
  const { data, loading, error, refetch } = useApi(url);
  
  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading data...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="error-container">
        <GoodErrorDisplay 
          error={error} 
          onRetry={refetch}
        />
      </div>
    );
  }
  
  if (!data) {
    return (
      <div className="empty-state">
        <p>No data available</p>
      </div>
    );
  }
  
  return (
    <div className="data-container">
      {/* 데이터 표시 */}
    </div>
  );
};
```

## 에러 로깅

### 1. 클라이언트 에러 로깅

```typescript
// ✅ 좋은 예시: 에러 로깅 서비스
class ErrorLogger {
  static log(error: Error, errorInfo?: React.ErrorInfo, context?: any) {
    const errorData = {
      message: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      errorInfo,
      context,
    };
    
    // 개발 환경에서는 콘솔에 출력
    if (process.env.NODE_ENV === 'development') {
      console.error('Error logged:', errorData);
      return;
    }
    
    // 프로덕션 환경에서는 로깅 서비스로 전송
    this.sendToLoggingService(errorData);
  }
  
  private static async sendToLoggingService(errorData: any) {
    try {
      await fetch('/api/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(errorData),
      });
    } catch (e) {
      console.error('Failed to log error:', e);
    }
  }
}

// 에러 경계에서 사용
class LoggingErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode; context?: any },
  { hasError: boolean; error?: Error }
> {
  // ... 다른 메서드
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    ErrorLogger.log(error, errorInfo, this.props.context);
  }
  
  // ... 렌더 메서드
}
```

### 2. React DevTools Profiler와 에러 로깅

```typescript
// ✅ 좋은 예시: Profiler와 에러 로깅 결합
const withProfiler = (Component: React.ComponentType, id: string) => {
  return (props: any) => (
    <Profiler
      id={id}
      onRender={(phase, actualDuration, baseDuration, startTime, commitTime) => {
        // 렌더링 성능 로깅
        if (process.env.NODE_ENV === 'development') {
          console.log(`${id} ${phase} took ${actualDuration}ms`);
        }
      }}
    >
      <Component {...props} />
    </Profiler>
  );
};

// 사용 예시
const ProfiledComponent = withProfiler(UserProfile, 'UserProfile');
```

## 결론

효과적인 에러 처리는 다음 원칙을 따라야 합니다:
- 에러 경계를 사용하여 컴포넌트 트리의 에러를 catch
- 비동기 작업에서 적절한 에러 처리
- 사용자 친화적인 에러 메시지 제공
- 에러 상태를 명확하게 관리
- 에러 로깅을 통해 디버깅 용이성 향상
- 재시도 옵션을 제공하여 사용자 경험 개선

이러한 원칙을 따르면 안정적이고 사용자 친화적인 React 애플리케이션을 만들 수 있습니다.