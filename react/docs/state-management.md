# React 상태 관리

React 애플리케이션에서 상태 관리는 매우 중요합니다. 적절한 상태 관리 전략을 선택하면 애플리케이션의 성능, 유지보수성, 확장성이 크게 향상됩니다.

## 상태의 종류

### 1. 로컬 상태 (Local State)

컴포넌트 내부에서만 사용되는 상태로, `useState` Hook으로 관리합니다.

```typescript
// ✅ 좋은 예시: 로컬 상태 사용
const ToggleButton = () => {
  const [isToggled, setIsToggled] = useState(false);
  
  return (
    <button onClick={() => setIsToggled(!isToggled)}>
      {isToggled ? 'ON' : 'OFF'}
    </button>
  );
};
```

### 2. 전역 상태 (Global State)

애플리케이션 전체에서 공유되는 상태로, 여러 컴포넌트에서 접근해야 합니다.

```typescript
// ✅ 좋은 예시: Context API를 사용한 전역 상태
const AuthContext = createContext<{
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}>({
  user: null,
  login: async () => {},
  logout: () => {},
});

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  
  const login = async (email: string, password: string) => {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    
    const userData = await response.json();
    setUser(userData);
  };
  
  const logout = () => {
    setUser(null);
  };
  
  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
```

### 3. 서버 상태 (Server State)

서버에서 가져온 데이터로, 클라이언트 측에서 캐싱, 동기화, 업데이트가 필요합니다.

```typescript
// ✅ 좋은 예시: React Query를 사용한 서버 상태 관리
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

const fetchUsers = async (): Promise<User[]> => {
  const response = await fetch('/api/users');
  return response.json();
};

const updateUser = async (user: User): Promise<User> => {
  const response = await fetch(`/api/users/${user.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(user),
  });
  return response.json();
};

export const useUsers = () => {
  return useQuery({
    queryKey: ['users'],
    queryFn: fetchUsers,
    staleTime: 5 * 60 * 1000, // 5분
  });
};

export const useUpdateUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: updateUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
};
```

### 4. 폼 상태 (Form State)

폼 입력과 관련된 상태로, 복잡한 폼은专门的 라이브러리를 사용하는 것이 좋습니다.

```typescript
// ✅ 좋은 예시: React Hook Form을 사용한 폼 상태 관리
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const userSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  email: z.string().email('Invalid email'),
  age: z.number().min(18, 'Must be at least 18'),
});

type UserFormData = z.infer<typeof userSchema>;

export const UserForm = () => {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
  });
  
  const onSubmit = async (data: UserFormData) => {
    await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  };
  
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('name')} />
      {errors.name && <p>{errors.name.message}</p>}
      
      <input {...register('email')} />
      {errors.email && <p>{errors.email.message}</p>}
      
      <input type="number" {...register('age', { valueAsNumber: true })} />
      {errors.age && <p>{errors.age.message}</p>}
      
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Submitting...' : 'Submit'}
      </button>
    </form>
  );
};
```

## 상태 관리 원칙

### 1. 상태 최소화 원칙

불필요한 상태는 피하고, 파생된 값은 상태로 저장하지 않습니다.

```typescript
// ❌ 나쁜 예시: 불필요한 상태
const UserList = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [filteredUsers, setFilteredUsers] = useState<User[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  
  useEffect(() => {
    setFilteredUsers(
      users.filter(user => 
        user.name.toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  }, [users, searchTerm]);
  
  // ...
};

// ✅ 좋은 예시: 파생된 값은 상태로 저장하지 않음
const UserList = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  
  const filteredUsers = useMemo(() => 
    users.filter(user => 
      user.name.toLowerCase().includes(searchTerm.toLowerCase())
    ), [users, searchTerm]
  );
  
  // ...
};
```

### 2. 상태 위치 규칙

상태는 가능한 한 사용되는 곳에 가깝게 위치시킵니다.

```typescript
// ❌ 나쁜 예시: 불필요하게 상태를 상위 컴포넌트에 위치
const App = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  
  return (
    <div>
      <Header />
      <Dropdown isOpen={isDropdownOpen} setIsOpen={setIsDropdownOpen} />
      <Content />
    </div>
  );
};

// ✅ 좋은 예시: 상태를 사용하는 곳에 위치
const App = () => {
  return (
    <div>
      <Header />
      <Dropdown />
      <Content />
    </div>
  );
};

const Dropdown = () => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div>
      <button onClick={() => setIsOpen(!isOpen)}>
        Toggle
      </button>
      {isOpen && <DropdownContent />}
    </div>
  );
};
```

### 3. 상태 업데이트 원칙

상태 업데이트는 불변성을 유지해야 합니다.

```typescript
// ❌ 나쁜 예시: 가변 상태 업데이트
const TodoList = () => {
  const [todos, setTodos] = useState<Todo[]>([]);
  
  const addTodo = (todo: Todo) => {
    todos.push(todo);
    setTodos(todos);
  };
  
  const toggleTodo = (id: number) => {
    const todo = todos.find(t => t.id === id);
    if (todo) {
      todo.completed = !todo.completed;
      setTodos(todos);
    }
  };
};

// ✅ 좋은 예시: 불변 상태 업데이트
const TodoList = () => {
  const [todos, setTodos] = useState<Todo[]>([]);
  
  const addTodo = (todo: Todo) => {
    setTodos(prevTodos => [...prevTodos, todo]);
  };
  
  const toggleTodo = (id: number) => {
    setTodos(prevTodos => 
      prevTodos.map(todo => 
        todo.id === id 
          ? { ...todo, completed: !todo.completed }
          : todo
      )
    );
  };
};
```

## 상태 관리 라이브러리 선택 가이드

### 1. 작은 애플리케이션

- **useState + useContext**: 대부분의 간단한 애플리케이션에 충분합니다.
- **장점**: 추가 의존성 없음, React 기본 기능
- **단점**: 복잡한 상태 관리에는 한계가 있음

### 2. 중간 규모 애플리케이션

- **Zustand**: 간단하고 직관적인 전역 상태 관리
- **Jotai**: 원자적 상태 관리, 세밀한 제어 가능

```typescript
// Zustand 예시
import { create } from 'zustand';

interface UserStore {
  user: User | null;
  setUser: (user: User | null) => void;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

export const useUserStore = create<UserStore>((set) => ({
  user: null,
  setUser: (user) => set({ user }),
  login: async (email, password) => {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const user = await response.json();
    set({ user });
  },
  logout: () => set({ user: null }),
}));
```

### 3. 대규모 애플리케이션

- **Redux Toolkit**: 복잡한 상태 관리, 미들웨어, 개발자 도구
- **React Query + Zustand**: 서버 상태와 클라이언트 상태 분리 관리

```typescript
// Redux Toolkit 예시
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

export const fetchUsers = createAsyncThunk(
  'users/fetchUsers',
  async () => {
    const response = await fetch('/api/users');
    return response.json();
  }
);

const usersSlice = createSlice({
  name: 'users',
  initialState: {
    data: [],
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchUsers.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchUsers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

export default usersSlice.reducer;
```

## 상태 관리 모범 사례

### 1. 상태 정규화

중첩된 데이터 구조는 정규화하여 저장합니다.

```typescript
// ❌ 나쁜 예시: 중첩된 상태
const [state, setState] = useState({
  users: [
    {
      id: 1,
      name: 'John',
      posts: [
        { id: 101, title: 'Post 1' },
        { id: 102, title: 'Post 2' },
      ],
    },
  ],
});

// ✅ 좋은 예시: 정규화된 상태
const [state, setState] = useState({
  users: {
    byId: {
      1: { id: 1, name: 'John', postIds: [101, 102] },
    },
    allIds: [1],
  },
  posts: {
    byId: {
      101: { id: 101, title: 'Post 1', userId: 1 },
      102: { id: 102, title: 'Post 2', userId: 1 },
    },
    allIds: [101, 102],
  },
});
```

### 2. 상태 분리

관련 없는 상태는 분리하여 관리합니다.

```typescript
// ❌ 나쁜 예시: 관련 없는 상태를 함께 관리
const [state, setState] = useState({
  user: null,
  theme: 'light',
  notifications: [],
  sidebarOpen: false,
});

// ✅ 좋은 예시: 상태 분리
const [user, setUser] = useState(null);
const [theme, setTheme] = useState('light');
const [notifications, setNotifications] = useState([]);
const [sidebarOpen, setSidebarOpen] = useState(false);
```

### 3. 비동기 상태 처리

로딩, 에러, 데이터 상태를 명확하게 관리합니다.

```typescript
// ✅ 좋은 예시: 비동기 상태 처리
const useAsyncData = <T,>(fetchFn: () => Promise<T>) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const execute = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await fetchFn();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };
  
  return { data, loading, error, execute };
};
```

## 결론

효과적인 상태 관리는 다음 원칙을 따라야 합니다:
- 상태 종류에 따라 적절한 관리 방법 선택
- 불필요한 상태 최소화
- 상태는 가능한 한 사용되는 곳에 가깝게 위치
- 불변성을 유지하는 상태 업데이트
- 복잡도에 맞는 상태 관리 라이브러리 선택

이러한 원칙을 따르면 유지보수가 쉽고 확장 가능한 React 애플리케이션을 만들 수 있습니다.