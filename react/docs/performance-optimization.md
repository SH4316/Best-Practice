# React 성능 최적화

React 애플리케이션의 성능을 최적화하는 것은 사용자 경험을 향상시키는 데 중요합니다. 이 문서에서는 React 애플리케이션의 성능을 최적화하는 다양한 기법과 모범 사례를 설명합니다.

## 리렌더링 최적화

### 1. React.memo

컴포넌트의 props가 변경되지 않았을 때 리렌더링을 방지합니다.

```typescript
// ❌ 나쁜 예시: 불필요한 리렌더링
const UserAvatar = ({ user, size }) => {
  console.log('UserAvatar rendered');
  return (
    <img 
      src={user.avatar} 
      alt={user.name}
      width={size}
      height={size}
    />
  );
};

// ✅ 좋은 예시: React.memo로 리렌더링 방지
const UserAvatar = React.memo(({ user, size }) => {
  console.log('UserAvatar rendered');
  return (
    <img 
      src={user.avatar} 
      alt={user.name}
      width={size}
      height={size}
    />
  );
});

// ✅ 좋은 예시: 커스텀 비교 함수 사용
const UserAvatar = React.memo(({ user, size }) => {
  console.log('UserAvatar rendered');
  return (
    <img 
      src={user.avatar} 
      alt={user.name}
      width={size}
      height={size}
    />
  );
}, (prevProps, nextProps) => {
  // user 객체의 avatar와 name만 비교
  return (
    prevProps.user.avatar === nextProps.user.avatar &&
    prevProps.user.name === nextProps.user.name &&
    prevProps.size === nextProps.size
  );
});
```

### 2. useMemo

비용이 많이 드는 계산 결과를 메모이제이션합니다.

```typescript
// ❌ 나쁜 예시: 매 렌더링마다 재계산
const ExpensiveComponent = ({ items, filter }) => {
  const filteredItems = items
    .filter(item => item.category === filter)
    .sort((a, b) => a.name.localeCompare(b.name));
  
  return (
    <ul>
      {filteredItems.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
};

// ✅ 좋은 예시: useMemo로 재계산 방지
const ExpensiveComponent = ({ items, filter }) => {
  const filteredItems = useMemo(() => {
    return items
      .filter(item => item.category === filter)
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [items, filter]);
  
  return (
    <ul>
      {filteredItems.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
};
```

### 3. useCallback

함수 참조를 안정화시켜 불필요한 리렌더링을 방지합니다.

```typescript
// ❌ 나쁜 예시: 매 렌더링마다 새 함수 생성
const ParentComponent = () => {
  const [count, setCount] = useState(0);
  
  const handleIncrement = () => {
    setCount(count + 1);
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <ChildComponent onIncrement={handleIncrement} />
    </div>
  );
};

// ✅ 좋은 예시: useCallback으로 함수 참조 안정화
const ParentComponent = () => {
  const [count, setCount] = useState(0);
  
  const handleIncrement = useCallback(() => {
    setCount(prevCount => prevCount + 1);
  }, []);
  
  return (
    <div>
      <p>Count: {count}</p>
      <ChildComponent onIncrement={handleIncrement} />
    </div>
  );
};
```

## 상태 관리 최적화

### 1. 상태 분리

관련 없는 상태는 분리하여 불필요한 리렌더링을 방지합니다.

```typescript
// ❌ 나쁜 예시: 관련 없는 상태를 함께 관리
const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');
  const [notifications, setNotifications] = useState([]);
  
  // user가 변경되면 theme과 notifications과 관련 없는 UI도 리렌더링됨
  
  return (
    <div>
      <UserInfo user={user} />
      <ThemeSelector theme={theme} setTheme={setTheme} />
      <NotificationList notifications={notifications} />
    </div>
  );
};

// ✅ 좋은 예시: 상태 분리
const UserProfile = () => {
  const [user, setUser] = useState(null);
  
  return (
    <div>
      <UserInfo user={user} setUser={setUser} />
      <ThemeSettings />
      <NotificationSettings />
    </div>
  );
};

const ThemeSettings = () => {
  const [theme, setTheme] = useState('light');
  
  return (
    <ThemeSelector theme={theme} setTheme={setTheme} />
  );
};

const NotificationSettings = () => {
  const [notifications, setNotifications] = useState([]);
  
  return (
    <NotificationList notifications={notifications} />
  );
};
```

### 2. 상태 구조 최적화

불필요한 객체 생성을 피하고 상태를 평탄하게 유지합니다.

```typescript
// ❌ 나쁜 예시: 중첩된 상태 구조
const UserProfile = () => {
  const [profile, setProfile] = useState({
    user: {
      name: '',
      email: '',
      avatar: '',
    },
    settings: {
      theme: 'light',
      language: 'en',
    },
    stats: {
      posts: 0,
      followers: 0,
      following: 0,
    },
  });
  
  const updateName = (name) => {
    setProfile({
      ...profile,
      user: {
        ...profile.user,
        name,
      },
    });
  };
  
  return (
    <div>
      <input 
        value={profile.user.name} 
        onChange={(e) => updateName(e.target.value)} 
      />
    </div>
  );
};

// ✅ 좋은 예시: 평탄한 상태 구조
const UserProfile = () => {
  const [user, setUser] = useState({
    name: '',
    email: '',
    avatar: '',
  });
  
  const [settings, setSettings] = useState({
    theme: 'light',
    language: 'en',
  });
  
  const [stats, setStats] = useState({
    posts: 0,
    followers: 0,
    following: 0,
  });
  
  const updateName = (name) => {
    setUser(prevUser => ({ ...prevUser, name }));
  };
  
  return (
    <div>
      <input 
        value={user.name} 
        onChange={(e) => updateName(e.target.value)} 
      />
    </div>
  );
};
```

## 리스트 렌더링 최적화

### 1. 가상화

대용량 리스트를 렌더링할 때는 가상화를 사용합니다.

```typescript
// ❌ 나쁜 예시: 모든 항목을 한 번에 렌더링
const LargeList = ({ items }) => {
  return (
    <div>
      {items.map(item => (
        <div key={item.id} style={{ height: 50 }}>
          {item.name}
        </div>
      ))}
    </div>
  );
};

// ✅ 좋은 예시: react-window를 사용한 가상화
import { FixedSizeList as List } from 'react-window';

const LargeList = ({ items }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      {items[index].name}
    </div>
  );
  
  return (
    <List
      height={500}
      itemCount={items.length}
      itemSize={50}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

### 2. 키 최적화

리스트의 키로 안정적인 식별자를 사용합니다.

```typescript
// ❌ 나쁜 예시: 인덱스를 키로 사용
const TodoList = ({ todos, onToggle }) => {
  return (
    <ul>
      {todos.map((todo, index) => (
        <TodoItem 
          key={index} // 인덱스를 키로 사용
          todo={todo}
          onToggle={onToggle}
        />
      ))}
    </ul>
  );
};

// ✅ 좋은 예시: 고유한 식별자를 키로 사용
const TodoList = ({ todos, onToggle }) => {
  return (
    <ul>
      {todos.map(todo => (
        <TodoItem 
          key={todo.id} // 고유한 ID를 키로 사용
          todo={todo}
          onToggle={onToggle}
        />
      ))}
    </ul>
  );
};
```

## 코드 분할 및 지연 로딩

### 1. 동적 import

컴포넌트를 동적으로 import하여 초기 로딩 시간을 줄입니다.

```typescript
// ❌ 나쁜 예시: 모든 컴포넌트를 정적으로 import
import Home from './pages/Home';
import About from './pages/About';
import Contact from './pages/Contact';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
      </Routes>
    </Router>
  );
};

// ✅ 좋은 예시: 동적 import를 사용한 코드 분할
import { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

const Home = lazy(() => import('./pages/Home'));
const About = lazy(() => import('./pages/About'));
const Contact = lazy(() => import('./pages/Contact'));

const App = () => {
  return (
    <Router>
      <Suspense fallback={<div>Loading...</div>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </Suspense>
    </Router>
  );
};
```

### 2. 라우트 기반 코드 분할

라우트에 따라 컴포넌트를 분할합니다.

```typescript
// ✅ 좋은 예시: 라우트 기반 코드 분할
import { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Profile = lazy(() => import('./pages/Profile'));
const Settings = lazy(() => import('./pages/Settings'));

const App = () => {
  return (
    <Router>
      <Suspense fallback={<div>Loading page...</div>}>
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Suspense>
    </Router>
  );
};
```

## 이미지 최적화

### 1. 지연 로딩

이미지를 지연 로딩하여 초기 페이지 로딩 속도를 향상시킵니다.

```typescript
// ❌ 나쁜 예시: 모든 이미지를 즉시 로드
const Gallery = ({ images }) => {
  return (
    <div>
      {images.map(image => (
        <img 
          key={image.id}
          src={image.url}
          alt={image.alt}
        />
      ))}
    </div>
  );
};

// ✅ 좋은 예시: Intersection Observer를 사용한 지연 로딩
const LazyImage = ({ src, alt }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef();
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    
    if (imgRef.current) {
      observer.observe(imgRef.current);
    }
    
    return () => observer.disconnect();
  }, []);
  
  return (
    <div ref={imgRef}>
      {isInView && (
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          style={{ opacity: isLoaded ? 1 : 0 }}
        />
      )}
    </div>
  );
};

const Gallery = ({ images }) => {
  return (
    <div>
      {images.map(image => (
        <LazyImage 
          key={image.id}
          src={image.url}
          alt={image.alt}
        />
      ))}
    </div>
  );
};
```

### 2. 이미지 최적화

적절한 이미지 포맷과 크기를 사용합니다.

```typescript
// ✅ 좋은 예시: 반응형 이미지와 최적화된 포맷 사용
const OptimizedImage = ({ src, alt, sizes }) => {
  return (
    <picture>
      <source 
        srcSet={`${src}.webp`} 
        type="image/webp" 
      />
      <source 
        srcSet={`${src}.avif`} 
        type="image/avif" 
      />
      <img
        src={`${src}.jpg`}
        alt={alt}
        sizes={sizes}
        loading="lazy"
      />
    </picture>
  );
};
```

## 디버깅 및 프로파일링

### 1. React DevTools Profiler

React DevTools Profiler를 사용하여 성능 병목 현상을 식별합니다.

```typescript
// ✅ 좋은 예시: Profiler로 컴포넌트 성능 측정
import { Profiler } from 'react';

const onRenderCallback = (id, phase, actualDuration) => {
  console.log(`${id} ${phase} took ${actualDuration}ms`);
};

const App = () => {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Router>
    </Profiler>
  );
};
```

### 2. 성능 메트릭

성능 메트릭을 수집하고 모니터링합니다.

```typescript
// ✅ 좋은 예시: 성능 메트릭 수집
const usePerformanceMetrics = (componentName) => {
  const renderStartTime = useRef();
  const renderCount = useRef(0);
  
  useEffect(() => {
    renderCount.current += 1;
    
    if (process.env.NODE_ENV === 'development') {
      console.log(
        `${componentName} rendered ${renderCount.current} times`
      );
    }
  });
  
  useEffect(() => {
    renderStartTime.current = performance.now();
    
    return () => {
      if (process.env.NODE_ENV === 'development') {
        const renderTime = performance.now() - renderStartTime.current;
        console.log(
          `${componentName} render time: ${renderTime.toFixed(2)}ms`
        );
      }
    };
  });
};

const MyComponent = () => {
  usePerformanceMetrics('MyComponent');
  
  // 컴포넌트 로직
};
```

## 결론

React 성능 최적화는 다음 원칙을 따라야 합니다:
- 불필요한 리렌더링 방지
- 비용이 많이 드는 계산 메모이제이션
- 대용량 데이터 가상화
- 코드 분할 및 지연 로딩
- 이미지 최적화
- 성능 모니터링 및 프로파일링

이러한 기법을 적절히 조합하여 사용하면 빠르고 반응성이 좋은 React 애플리케이션을 만들 수 있습니다.