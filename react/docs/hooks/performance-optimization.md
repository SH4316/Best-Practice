# 성능 최적화

React 애플리케이션의 성능을 최적화하기 위해 `useMemo`와 `useCallback` Hook을 사용할 수 있습니다. 이 Hook들은 불필요한 재계산과 함수 재생성을 방지하여 렌더링 성능을 향상시킵니다.

## useMemo

`useMemo`는 비용이 많이 드는 계산 결과를 메모이제이션(memoization)할 때 사용합니다. 의존성이 변경되지 않으면 이전에 계산된 값을 재사용합니다.

### 기본 사용법

```typescript
// ❌ 나쁜 예시: 불필요한 재계산
interface Item {
  id: number;
  name: string;
  type: string;
}

interface ExpensiveComponentProps {
  items: Item[];
  filter: string;
}

const ExpensiveComponent: React.FC<ExpensiveComponentProps> = ({ items, filter }) => {
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
const ExpensiveComponent: React.FC<ExpensiveComponentProps> = ({ items, filter }) => {
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
```

### 복잡한 객체 참조 안정화

```typescript
// ✅ 좋은 예시: 복잡한 객체 참조 안정화
interface UserPreferences {
  theme: string;
  language: string;
  notifications: boolean;
}

interface User {
  preferences: UserPreferences;
}

interface UserProfileProps {
  user: User;
}

interface UserSettings {
  theme: string;
  language: string;
  notifications: boolean;
}

interface SettingsPanelProps {
  settings: UserSettings;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ settings }) => {
  return <div>Settings Panel</div>; // Simplified for example
};

const UserProfile: React.FC<UserProfileProps> = ({ user }) => {
  const userSettings = useMemo<UserSettings>(() => ({
    theme: user.preferences.theme,
    language: user.preferences.language,
    notifications: user.preferences.notifications,
  }), [user.preferences]);
  
  return <SettingsPanel settings={userSettings} />;
};
```

## useCallback

`useCallback`은 함수 참조를 안정화시켜 불필요한 리렌더링을 방지할 때 사용합니다. 특히 자식 컴포넌트에 콜백 함수를 전달할 때 유용합니다.

### 기본 사용법

```typescript
// ❌ 나쁜 예시: 매 렌더링마다 새 함수 생성
interface ChildComponentProps {
  onClick: () => void;
}

const ChildComponent: React.FC<ChildComponentProps> = ({ onClick }) => {
  return <button onClick={onClick}>Child Button</button>; // Simplified for example
};

const ParentComponent: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  
  const handleClick = (): void => {
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
const ParentComponent: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  
  const handleClick = useCallback((): void => {
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
```

### 의존성이 있는 함수

```typescript
// ✅ 좋은 예시: 의존성이 있는 함수
interface TodoItemData {
  id: number;
  text: string;
  completed: boolean;
}

interface TodoUpdate {
  completed?: boolean;
  deleted?: boolean;
}

interface TodoItemProps {
  todo: TodoItemData;
  onUpdate: (id: number, updates: TodoUpdate) => void;
}

const TodoItem: React.FC<TodoItemProps> = ({ todo, onUpdate }) => {
  const handleToggle = useCallback((): void => {
    onUpdate(todo.id, { completed: !todo.completed });
  }, [todo.id, todo.completed, onUpdate]);
  
  const handleDelete = useCallback((): void => {
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

## 성능 최적화 전략

### 1. React.memo와 함께 사용

```typescript
// ✅ 좋은 예시: React.memo와 useCallback 함께 사용
const ExpensiveChildComponent = React.FC<{ data: number[]; onItemClick: (id: number) => void }> = ({ 
  data, 
  onItemClick 
}) => {
  console.log('ExpensiveChildComponent rendered');
  
  return (
    <ul>
      {data.map(item => (
        <li key={item} onClick={() => onItemClick(item)}>
          Item {item}
        </li>
      ))}
    </ul>
  );
};

// 자식 컴포넌트 메모이제이션
const MemoizedExpensiveChild = React.memo(ExpensiveChildComponent);

const ParentComponent: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  const [data] = useState<number[]>([1, 2, 3, 4, 5]);
  
  const handleItemClick = useCallback((id: number): void => {
    console.log('Item clicked:', id);
  }, []);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <MemoizedExpensiveChild data={data} onItemClick={handleItemClick} />
    </div>
  );
};
```

### 2. 복잡한 계산 메모이제이션

```typescript
// ✅ 좋은 예시: 복잡한 계산 메모이제이션
const FibonacciCalculator: React.FC<{ n: number }> = ({ n }) => {
  const fibonacci = useMemo(() => {
    console.log('Calculating Fibonacci...');
    
    if (n <= 1) return n;
    
    let prev = 0;
    let curr = 1;
    
    for (let i = 2; i <= n; i++) {
      const next = prev + curr;
      prev = curr;
      curr = next;
    }
    
    return curr;
  }, [n]);
  
  return (
    <div>
      <p>Fibonacci({n}) = {fibonacci}</p>
    </div>
  );
};
```

### 3. API 데이터 변환

```typescript
// ✅ 좋은 예시: API 데이터 변환 메모이제이션
interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

interface UserListProps {
  users: User[];
  filterRole?: 'admin' | 'user';
}

const UserList: React.FC<UserListProps> = ({ users, filterRole }) => {
  const filteredUsers = useMemo(() => {
    if (!filterRole) return users;
    
    return users.filter(user => user.role === filterRole);
  }, [users, filterRole]);
  
  const userOptions = useMemo(() => {
    return filteredUsers.map(user => ({
      value: user.id,
      label: user.name,
      email: user.email,
    }));
  }, [filteredUsers]);
  
  return (
    <select>
      {userOptions.map(option => (
        <option key={option.value} value={option.value}>
          {option.label} ({option.email})
        </option>
      ))}
    </select>
  );
};
```

## 주의사항

### 1. 과도한 최적화 피하기

```typescript
// ❌ 나쁜 예시: 불필요한 useMemo 사용
const SimpleComponent: React.FC<{ name: string }> = ({ name }) => {
  const displayName = useMemo(() => name, [name]); // 간단한 값은 useMemo가 필요 없음
  
  return <div>{displayName}</div>;
};

// ✅ 좋은 예시: 간단한 값은 직접 사용
const SimpleComponent: React.FC<{ name: string }> = ({ name }) => {
  return <div>{name}</div>;
};
```

### 2. 의존성 배열 정확성

```typescript
// ❌ 나쁜 예시: 의존성 배열 누락
const Component: React.FC<{ data: any[] }> = ({ data }) => {
  const processedData = useMemo(() => {
    return data.map(item => ({ ...item, processed: true }));
  }, []); // data를 의존성 배열에 포함하지 않음
  
  return <div>{/* ... */}</div>;
};

// ✅ 좋은 예시: 모든 의존성 포함
const Component: React.FC<{ data: any[] }> = ({ data }) => {
  const processedData = useMemo(() => {
    return data.map(item => ({ ...item, processed: true }));
  }, [data]); // data를 의존성 배열에 포함
  
  return <div>{/* ... */}</div>;
};
```

### 3. 객체와 배열 참조

```typescript
// ❌ 나쁜 예시: 매번 새 객체/배열 생성
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  
  const config = useMemo(() => ({
    enabled: true,
    count: count,
  }), [count]); // count가 변경될 때마다 새 객체 생성
  
  return <div>{/* ... */}</div>;
};

// ✅ 좋은 예시: 필요한 경우에만 객체 생성
const Component: React.FC = () => {
  const [count, setCount] = useState(0);
  
  const config = useMemo(() => ({
    enabled: true,
    count: count,
  }), [count]);
  
  return <div>{/* ... */}</div>;
};
```

## 다음 단계

DOM 상호작용에 대해서는 [DOM 상호작용](./dom-interaction.md) 문서를 확인하세요.

커스텀 Hook 만들기에 대해서는 [커스텀 Hooks](./custom-hooks.md) 문서를 참조하세요.