# Hooks 사용 규칙

React Hooks를 사용할 때는 반드시 지켜야 할 두 가지 중요한 규칙이 있습니다. 이 규칙을 따르면 React가 컴포넌트의 상태를 올바르게 유지하고 Hook 호출 순서를 보장할 수 있습니다.

## 기본 규칙

### 1. 최상위 레벨에서만 Hook 호출

Hook은 반복문, 조건문, 중첩된 함수 내에서 호출할 수 없습니다. Hook은 항상 React 함수의 최상위 레벨에서 호출되어야 합니다. 이렇게 하면 컴포넌트가 렌더링될 때마다 항상 같은 순서로 Hook이 호출되도록 보장할 수 있습니다.

```typescript
// ❌ 나쁜 예시: 조건부 Hook 호출
interface UserProfileRuleProps {
  user: {
    name: string;
  } | null;
}

const UserProfile: React.FC<UserProfileRuleProps> = ({ user }) => {
  if (!user) {
    const [loading, setLoading] = useState<boolean>(false); // 조건부 Hook 호출
    return <div>Loading...</div>;
  }
  
  return <div>{user.name}</div>;
};

// ❌ 나쁜 예시: 반복문 내 Hook 호출
const TodoList: React.FC<{ items: string[] }> = ({ items }) => {
  const [todos, setTodos] = useState<string[]>([]);
  
  items.forEach((item, index) => {
    const [itemState, setItemState] = useState<string>(item); // 반복문 내 Hook 호출
  });
  
  return <div>{/* ... */}</div>;
};

// ❌ 나쁜 예시: 중첩 함수 내 Hook 호출
const Component: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  
  const handleClick = () => {
    const [innerState, setInnerState] = useState<number>(0); // 중첩 함수 내 Hook 호출
  };
  
  return <button onClick={handleClick}>Click</button>;
};

// ✅ 좋은 예시: 항상 Hook 호출
const UserProfile: React.FC<UserProfileRuleProps> = ({ user }) => {
  const [loading, setLoading] = useState<boolean>(false);
  
  if (!user) {
    return <div>Loading...</div>;
  }
  
  return <div>{user.name}</div>;
};
```

### 2. React 함수 컴포넌트에서만 Hook 호출

Hook은 일반 JavaScript 함수에서 호출할 수 없습니다. Hook은 React 함수 컴포넌트나 커스텀 Hook 내에서만 호출해야 합니다.

```typescript
// ❌ 나쁜 예시: 일반 JavaScript 함수에서 Hook 호출
const processUserData = (user: User) => {
  const [processedData, setProcessedData] = useState<any>(null); // 일반 함수에서 Hook 호출
  
  // 데이터 처리 로직
  return processedData;
};

// ✅ 좋은 예시: React 함수 컴포넌트에서 Hook 호출
const UserProcessor: React.FC<{ user: User }> = ({ user }) => {
  const [processedData, setProcessedData] = useState<any>(null);
  
  // 데이터 처리 로직
  useEffect(() => {
    // 처리 로직
  }, [user]);
  
  return <div>{/* ... */}</div>;
};

// ✅ 좋은 예시: 커스텀 Hook에서 Hook 호출
const useUserProcessor = (user: User) => {
  const [processedData, setProcessedData] = useState<any>(null);
  
  useEffect(() => {
    // 처리 로직
  }, [user]);
  
  return processedData;
};
```

## 왜 이 규칙이 중요한가요?

React는 Hook 호출 순서에 의존하여 상태를 올바르게 관리합니다. 각 Hook 호출은 컴포넌트의 상태 배열에서 특정 인덱스와 연결됩니다.

```typescript
function Counter() {
  // 1. useState는 첫 번째 상태와 연결됨
  const [count, setCount] = useState(0);
  
  // 2. useEffect는 두 번째 상태와 연결됨
  useEffect(() => {
    document.title = `Count: ${count}`;
  });
  
  // 3. useState는 세 번째 상태와 연결됨
  const [name, setName] = useState('');
  
  // ...
}
```

Hook 호출 순서가 렌더링마다 동일하다면 React는 각 Hook을 올바른 상태와 연결할 수 있습니다. 하지만 조건부로 Hook을 호출하면 순서가 바뀌어 버그가 발생할 수 있습니다.

## ESLint 규칙 사용

React Hooks 규칙을 자동으로 검사하기 위해 ESLint 규칙을 사용하는 것이 좋습니다.

```json
{
  "plugins": ["react-hooks"],
  "rules": {
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn"
  }
}
```

## 일반적인 실수와 해결 방법

### 1. 조건부 Hook 사용

```typescript
// ❌ 나쁜 예시
const Component: React.FC = () => {
  if (someCondition) {
    const [state, setState] = useState(initialValue);
  }
  
  return <div>{/* ... */}</div>;
};

// ✅ 좋은 예시: 조건부로 초기값 설정
const Component: React.FC = () => {
  const [state, setState] = useState(someCondition ? initialValue : defaultValue);
  
  return <div>{/* ... */}</div>;
};

// ✅ 좋은 예시: 조건부로 effect 실행
const Component: React.FC = () => {
  const [state, setState] = useState(initialValue);
  
  useEffect(() => {
    if (someCondition) {
      // 조건부 로직
    }
  }, [someCondition]);
  
  return <div>{/* ... */}</div>;
};
```

### 2. 반복문 내 Hook 사용

```typescript
// ❌ 나쁜 예시
const Component: React.FC = () => {
  const items = [1, 2, 3];
  
  items.forEach(item => {
    const [state, setState] = useState(item);
  });
  
  return <div>{/* ... */}</div>;
};

// ✅ 좋은 예시: 커스텀 Hook 사용
const useItemState = (initialValue: any) => {
  const [state, setState] = useState(initialValue);
  return [state, setState];
};

const Component: React.FC = () => {
  const items = [1, 2, 3];
  const [states, setStates] = useState(items.map(item => item));
  
  return <div>{/* ... */}</div>;
};
```

### 3. 이벤트 핸들러 내 Hook 사용

```typescript
// ❌ 나쁜 예시
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  
  const handleClick = () => {
    const [localState, setLocalState] = useState(0); // 이벤트 핸들러 내 Hook 호출
  };
  
  return <button onClick={handleClick}>Click</button>;
};

// ✅ 좋은 예시: 상태를 컴포넌트 최상위 레벨로 이동
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  const [localState, setLocalState] = useState(0);
  
  const handleClick = () => {
    setLocalState(prev => prev + 1);
  };
  
  return <button onClick={handleClick}>Click</button>;
};
```

## 모범 사례

### 1. 커스텀 Hook으로 로직 추출

복잡한 로직은 커스텀 Hook으로 추출하여 재사용성과 가독성을 높이세요.

```typescript
// ✅ 좋은 예시: 커스텀 Hook으로 로직 추출
const useUserProfile = (userId: string) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchUser = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch user');
        }
        const userData = await response.json();
        setUser(userData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };
    
    fetchUser();
  }, [userId]);
  
  return { user, loading, error };
};

const UserProfile: React.FC<{ userId: string }> = ({ userId }) => {
  const { user, loading, error } = useUserProfile(userId);
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!user) return <div>No user found</div>;
  
  return <div>{user.name}</div>;
};
```

### 2. 의존성 배열 올바르게 사용

```typescript
// ❌ 나쁜 예시: 의존성 배열 누락
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');
  
  useEffect(() => {
    console.log(`Count: ${count}, Name: ${name}`);
  }, [count]); // name을 의존성 배열에 포함하지 않음
  
  return <div>{/* ... */}</div>;
};

// ✅ 좋은 예시: 모든 의존성 포함
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');
  
  useEffect(() => {
    console.log(`Count: ${count}, Name: ${name}`);
  }, [count, name]); // 모든 의존성 포함
  
  return <div>{/* ... */}</div>;
};
```

### 3. 불필요한 의존성 제거

```typescript
// ❌ 나쁜 예시: 불필요한 의존성 포함
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    const intervalId = setInterval(() => {
      setCount(prevCount => prevCount + 1);
    }, 1000);
    
    return () => clearInterval(intervalId);
  }, [count]); // count가 변경될 때마다 새 인터벌 생성
  
  return <div>Count: {count}</div>;
};

// ✅ 좋은 예시: 함수형 업데이트 사용
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    const intervalId = setInterval(() => {
      setCount(prevCount => prevCount + 1); // 함수형 업데이트
    }, 1000);
    
    return () => clearInterval(intervalId);
  }, []); // 빈 의존성 배열
  
  return <div>Count: {count}</div>;
};
```

## 다음 단계

이제 React Hooks의 기본 규칙과 모범 사례를 이해했습니다. 다음 문서들을 통해 더 자세한 내용을 학습하세요:

- [상태 관리](./state-management.md) - useState와 useReducer 사용법
- [사이드 이펙트](./side-effects.md) - useEffect 사용법
- [커스텀 Hooks](./custom-hooks.md) - 재사용 가능한 Hook 만들기

React Hooks에 대한 더 자세한 정보는 [React 공식 문서](https://reactjs.org/docs/hooks-intro.html)를 참조하세요.