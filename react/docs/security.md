# React 보안

웹 애플리케이션 보안은 사용자 데이터를 보호하고 악의적인 공격으로부터 시스템을 safeguarding하는 데 중요합니다. React 애플리케이션에서 보안을 구현하는 것은 민감한 정보를 보호하고, 사용자의 신뢰를 유지하며, 법적 요구사항을 준수하는 데 필수적입니다. 이 문서에서는 React 애플리케이션의 보안을 강화하는 다양한 기법과 모범 사례를 설명합니다.

## XSS (Cross-Site Scripting) 방지

### 1. JSX 자동 이스케이프

React는 JSX에서 렌더링되는 모든 데이터를 자동으로 이스케이프하여 XSS 공격을 방지합니다.

```typescript
// ❌ 나쁜 예시: 직접 HTML 삽입 (위험)
const BadComponent = ({ userContent }) => {
  return (
    <div dangerouslySetInnerHTML={{ __html: userContent }} />
  );
};

// ✅ 좋은 예시: JSX 사용 (안전)
const GoodComponent = ({ userContent }) => {
  return (
    <div>{userContent}</div>
  );
};
```

### 2. 안전한 HTML 렌더링

`dangerouslySetInnerHTML`을 사용해야 하는 경우, 입력을 철저히 검증하고 정화해야 합니다.

```typescript
// ❌ 나쁜 예시: 검증 없이 HTML 렌더링
const UnsafeHtmlRenderer = ({ htmlContent }) => {
  return (
    <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
  );
};

// ✅ 좋은 예시: HTML 정화 라이브러리 사용
import DOMPurify from 'dompurify';

const SafeHtmlRenderer = ({ htmlContent }) => {
  const cleanHtml = DOMPurify.sanitize(htmlContent);
  
  return (
    <div dangerouslySetInnerHTML={{ __html: cleanHtml }} />
  );
};
```

## 인증 및 권한 부여

### 1. 안전한 인증 구현

인증은 사용자 신원을 확인하는 과정입니다. 안전한 인증을 구현하는 방법을 알아봅시다.

```typescript
// ❌ 나쁜 예시: 클라이언트 측에서만 인증 검증
const BadAuthComponent = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  
  const handleLogin = (username, password) => {
    // 클라이언트 측에서만 인증 검증 (매우 위험)
    if (username === 'admin' && password === 'password') {
      setIsLoggedIn(true);
      localStorage.setItem('isLoggedIn', 'true');
    }
  };
  
  return (
    <div>
      {isLoggedIn ? (
        <div>Welcome, admin!</div>
      ) : (
        <LoginForm onLogin={handleLogin} />
      )}
    </div>
  );
};

// ✅ 좋은 예시: 서버 측 인증 검증
const GoodAuthComponent = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleLogin = async (username, password) => {
    setIsLoading(true);
    
    try {
      // 서버에 인증 요청
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });
      
      if (!response.ok) {
        throw new Error('Authentication failed');
      }
      
      const { user: userData, token } = await response.json();
      
      // JWT 토큰 저장
      localStorage.setItem('authToken', token);
      
      setUser(userData);
    } catch (error) {
      console.error('Login error:', error);
      // 에러 처리
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div>
      {user ? (
        <div>Welcome, {user.name}!</div>
      ) : (
        <LoginForm onLogin={handleLogin} isLoading={isLoading} />
      )}
    </div>
  );
};
```

### 2. 권한 부여 구현

권한 부여는 인증된 사용자가 특정 리소스에 접근할 수 있는지 확인하는 과정입니다.

```typescript
// ✅ 좋은 예시: 역할 기반 권한 부여
const ProtectedComponent = ({ requiredRole, children }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('authToken');
        
        if (!token) {
          setUser(null);
          return;
        }
        
        // 토큰 검증 및 사용자 정보 가져오기
        const response = await fetch('/api/auth/me', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        
        if (!response.ok) {
          throw new Error('Invalid token');
        }
        
        const userData = await response.json();
        setUser(userData);
      } catch (error) {
        console.error('Auth check error:', error);
        localStorage.removeItem('authToken');
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };
    
    checkAuth();
  }, []);
  
  if (isLoading) {
    return <div>Loading...</div>;
  }
  
  if (!user) {
    return <div>Please log in to access this resource.</div>;
  }
  
  if (requiredRole && !user.roles.includes(requiredRole)) {
    return <div>You don't have permission to access this resource.</div>;
  }
  
  return <>{children}</>;
};

// 사용 예시
const AdminPanel = () => {
  return (
    <ProtectedComponent requiredRole="admin">
      <div>Admin Panel Content</div>
    </ProtectedComponent>
  );
};
```

## API 보안

### 1. 안전한 API 통신

API 통신 시 보안을 고려해야 합니다.

```typescript
// ❌ 나쁜 예시: 인증 없이 API 호출
const BadApiComponent = () => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    // 인증 없이 API 호출 (위험)
    fetch('/api/sensitive-data')
      .then(response => response.json())
      .then(setData)
      .catch(console.error);
  }, []);
  
  return <div>{JSON.stringify(data)}</div>;
};

// ✅ 좋은 예시: 인증 토큰과 함께 API 호출
const GoodApiComponent = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = localStorage.getItem('authToken');
        
        if (!token) {
          throw new Error('No authentication token');
        }
        
        const response = await fetch('/api/sensitive-data', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      }
    };
    
    fetchData();
  }, []);
  
  if (error) {
    return <div>Error: {error}</div>;
  }
  
  return <div>{JSON.stringify(data)}</div>;
};
```

### 2. CSRF 방지

CSRF(Cross-Site Request Forgery) 공격을 방지하기 위한 조치를 취해야 합니다.

```typescript
// ✅ 좋은 예시: CSRF 토큰 사용
const SecureFormComponent = () => {
  const [csrfToken, setCsrfToken] = useState('');
  const [formData, setFormData] = useState({});
  
  useEffect(() => {
    // CSRF 토큰 가져오기
    const fetchCsrfToken = async () => {
      try {
        const response = await fetch('/api/csrf-token');
        const { token } = await response.json();
        setCsrfToken(token);
      } catch (error) {
        console.error('Failed to fetch CSRF token:', error);
      }
    };
    
    fetchCsrfToken();
  }, []);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/submit-form', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error(`Form submission error: ${response.status}`);
      }
      
      // 성공 처리
    } catch (error) {
      console.error('Form submission error:', error);
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      {/* 폼 필드 */}
      <input type="hidden" name="csrf_token" value={csrfToken} />
      <button type="submit">Submit</button>
    </form>
  );
};
```

## 콘텐츠 보안 정책 (CSP)

### 1. CSP 헤더 설정

콘텐츠 보안 정책(CSP)은 XSS 공격을 방지하는 데 도움이 되는 추가 보안 계층입니다.

```typescript
// ✅ 좋은 예시: CSP 헤더 설정
// 서버 측에서 설정해야 함
const setCSPHeader = (res) => {
  res.setHeader(
    'Content-Security-Policy',
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';"
  );
};

// Next.js 예시 (next.config.js)
module.exports = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';",
          },
        ],
      },
    ];
  },
};
```

## 민감한 정보 처리

### 1. 민감한 데이터 노출 방지

민감한 정보를 클라이언트 측에 노출하지 않도록 주의해야 합니다.

```typescript
// ❌ 나쁜 예시: 민감한 정보를 클라이언트에 노출
const BadUserComponent = ({ user }) => {
  return (
    <div>
      <p>Name: {user.name}</p>
      <p>Email: {user.email}</p>
      <p>Password: {user.password}</p> {/* 비밀번호 노출 (위험) */}
      <p>API Key: {user.apiKey}</p> {/* API 키 노출 (위험) */}
    </div>
  );
};

// ✅ 좋은 예시: 민감한 정보를 숨김
const GoodUserComponent = ({ user }) => {
  return (
    <div>
      <p>Name: {user.name}</p>
      <p>Email: {user.email}</p>
      {/* 비밀번호, API 키 등 민감한 정보는 표시하지 않음 */}
    </div>
  );
};
```

### 2. 안전한 로깅

로그에 민감한 정보가 포함되지 않도록 해야 합니다.

```typescript
// ❌ 나쁜 예시: 민감한 정보 로깅
const badLoginHandler = (req, res) => {
  const { username, password } = req.body;
  
  // 비밀번호 로깅 (위험)
  console.log(`Login attempt: ${username}, password: ${password}`);
  
  // 로그인 처리
};

// ✅ 좋은 예시: 민감한 정보 제외 로깅
const goodLoginHandler = (req, res) => {
  const { username } = req.body;
  
  // 사용자 이름만 로깅
  console.log(`Login attempt: ${username}`);
  
  // 로그인 처리
};
```

## 보안 도구 및 라이브러리

### 1. 보안 라이브러리 사용

보안을 강화하기 위해 검증된 라이브러리를 사용해야 합니다.

```typescript
// ✅ 좋은 예시: 보안 라이브러리 사용
import DOMPurify from 'dompurify';
import { Helmet } from 'react-helmet';
import bcrypt from 'bcryptjs';

// HTML 정화
const cleanHtml = DOMPurify.sanitize(dirtyHtml);

// CSP 헤더 설정
const SecurePage = () => {
  return (
    <div>
      <Helmet>
        <meta
          httpEquiv="Content-Security-Policy"
          content="default-src 'self'; script-src 'self' 'unsafe-inline';"
        />
      </Helmet>
      {/* 페이지 콘텐츠 */}
    </div>
  );
};

// 비밀번호 해싱 (서버 측에서 사용)
const hashPassword = async (password) => {
  const saltRounds = 10;
  const hashedPassword = await bcrypt.hash(password, saltRounds);
  return hashedPassword;
};
```

## 결론

효과적인 보안 구현은 다음 원칙을 따라야 합니다:
- XSS 공격 방지를 위한 입력 검증 및 출력 이스케이프
- 안전한 인증 및 권한 부여 구현
- API 통신 시 적절한 인증 및 CSRF 방지
- 콘텐츠 보안 정책(CSP) 설정
- 민감한 정보의 적절한 처리 및 보호
- 검증된 보안 라이브러리 사용

이러한 원칙을 따르면 안전하고 신뢰할 수 있는 React 애플리케이션을 만들 수 있습니다.