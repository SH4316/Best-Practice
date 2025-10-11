# React 코드 조직

React 애플리케이션의 코드 조직은 유지보수성, 확장성, 협업 효율성에 큰 영향을 미칩니다. 이 문서에서는 React 프로젝트의 코드를 효과적으로 조직하는 방법을 설명합니다.

## 파일 및 폴더 구조

### 1. 기능별 구조 (Feature-based Structure)

대규모 애플리케이션에서는 기능별로 폴더를 구성하는 것이 좋습니다.

```
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   ├── utils/
│   │   ├── index.ts
│   │   └── Auth.module.css
│   ├── products/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   ├── utils/
│   │   ├── index.ts
│   │   └── Products.module.css
│   └── users/
├── shared/
│   ├── components/
│   ├── hooks/
│   ├── utils/
│   ├── types/
│   └── constants/
├── pages/
├── layouts/
└── App.tsx
```

### 2. 타입별 구조 (Type-based Structure)

소규모 애플리케이션에서는 타입별로 폴더를 구성하는 것이 좋습니다.

```
src/
├── components/
│   ├── common/
│   ├── forms/
│   └── layout/
├── hooks/
├── services/
├── utils/
├── types/
├── constants/
├── pages/
├── styles/
└── App.tsx
```

## 컴포넌트 구조

### 1. 컴포넌트 폴더 구조

각 컴포넌트는 다음과 같은 구조를 따르는 것이 좋습니다.

```
Button/
├── Button.tsx          # 메인 컴포넌트
├── Button.test.tsx     # 테스트 파일
├── Button.stories.tsx  # Storybook 스토리 (선택적)
├── Button.styles.ts    # 스타일 관련 코드
├── types.ts           # 타입 정의 (필요시)
└── index.ts           # 내보내기 파일
```

### 2. 컴포넌트 파일 구조

```typescript
// Button/types.ts
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
  className?: string;
}

// Button/Button.tsx
import React from 'react';
import { ButtonProps } from './types';
import { ButtonStyles } from './Button.styles';

const Button = ({
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  onClick,
  children,
  className,
}: ButtonProps) => {
  return (
    <button
      className={ButtonStyles.getButtonClass(variant, size, disabled, loading, className)}
      onClick={onClick}
      disabled={disabled || loading}
    >
      {loading ? 'Loading...' : children}
    </button>
  );
};

export default Button;

// Button/Button.styles.ts
export const ButtonStyles = {
  getButtonClass: (
    variant: string,
    size: string,
    disabled: boolean,
    loading: boolean,
    className?: string
  ) => {
    const baseClass = 'btn';
    const variantClass = `btn--${variant}`;
    const sizeClass = `btn--${size}`;
    const stateClasses = [
      disabled && 'btn--disabled',
      loading && 'btn--loading',
    ]
      .filter(Boolean)
      .join(' ');
    
    return [
      baseClass,
      variantClass,
      sizeClass,
      stateClasses,
      className,
    ]
      .filter(Boolean)
      .join(' ');
  },
};

// Button/index.ts
export { default } from './Button';
export type { ButtonProps } from './types';
```

## Hooks 구조

### 1. 커스텀 Hooks 폴더 구조

```
hooks/
├── useAuth.ts
├── useApi.ts
├── useLocalStorage.ts
├── useDebounce.ts
└── index.ts
```

### 2. 커스텀 Hooks 파일 구조

```typescript
// hooks/useAuth.ts
import { useState, useEffect, useContext, createContext } from 'react';
import type { User, AuthState } from '../types/auth';

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    loading: false,
    error: null,
  });

  const login = async (email: string, password: string) => {
    setAuthState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      
      if (!response.ok) {
        throw new Error('Login failed');
      }
      
      const user = await response.json();
      setAuthState({ user, loading: false, error: null });
    } catch (error) {
      setAuthState({
        user: null,
        loading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  };

  const logout = () => {
    setAuthState({ user: null, loading: false, error: null });
  };

  return (
    <AuthContext.Provider value={{ ...authState, login, logout }}>
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

// hooks/index.ts
export { useAuth } from './useAuth';
export { useApi } from './useApi';
export { useLocalStorage } from './useLocalStorage';
export { useDebounce } from './useDebounce';
```

## 서비스 구조

### 1. API 서비스 폴더 구조

```
services/
├── api/
│   ├── auth.ts
│   ├── users.ts
│   ├── products.ts
│   └── index.ts
├── storage/
│   ├── localStorage.ts
│   ├── sessionStorage.ts
│   └── index.ts
└── index.ts
```

### 2. API 서비스 파일 구조

```typescript
// services/api/auth.ts
import { apiClient } from './client';
import type { LoginRequest, LoginResponse, User } from '../../types/auth';

export const authApi = {
  login: async (data: LoginRequest): Promise<LoginResponse> => {
    const response = await apiClient.post<LoginResponse>('/auth/login', data);
    return response.data;
  },
  
  logout: async (): Promise<void> => {
    await apiClient.post('/auth/logout');
  },
  
  getCurrentUser: async (): Promise<User> => {
    const response = await apiClient.get<User>('/auth/me');
    return response.data;
  },
  
  refreshToken: async (refreshToken: string): Promise<LoginResponse> => {
    const response = await apiClient.post<LoginResponse>('/auth/refresh', {
      refreshToken,
    });
    return response.data;
  },
};

// services/api/client.ts
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('accessToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 응답 인터셉터
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        const refreshToken = localStorage.getItem('refreshToken');
        const response = await axios.post('/auth/refresh', {
          refreshToken,
        });
        
        const { accessToken } = response.data;
        localStorage.setItem('accessToken', accessToken);
        
        return apiClient(originalRequest);
      } catch (refreshError) {
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        window.location.href = '/login';
      }
    }
    
    return Promise.reject(error);
  }
);

// services/api/index.ts
export { authApi } from './auth';
export { usersApi } from './users';
export { productsApi } from './products';
export { apiClient } from './client';
```

## 유틸리티 구조

### 1. 유틸리티 폴더 구조

```
utils/
├── helpers/
│   ├── date.ts
│   ├── string.ts
│   ├── array.ts
│   └── index.ts
├── validators/
│   ├── form.ts
│   ├── auth.ts
│   └── index.ts
├── constants/
│   ├── api.ts
│   ├── app.ts
│   └── index.ts
└── index.ts
```

### 2. 유틸리티 파일 구조

```typescript
// utils/helpers/date.ts
export const formatDate = (date: Date | string, format = 'YYYY-MM-DD'): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  
  return format
    .replace('YYYY', String(year))
    .replace('MM', month)
    .replace('DD', day);
};

export const isDateToday = (date: Date | string): boolean => {
  const d = typeof date === 'string' ? new Date(date) : date;
  const today = new Date();
  
  return (
    d.getFullYear() === today.getFullYear() &&
    d.getMonth() === today.getMonth() &&
    d.getDate() === today.getDate()
  );
};

export const addDays = (date: Date, days: number): Date => {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
};

// utils/helpers/index.ts
export * from './date';
export * from './string';
export * from './array';

// utils/validators/form.ts
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const validatePassword = (password: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];
  
  if (password.length < 8) {
    errors.push('Password must be at least 8 characters long');
  }
  
  if (!/[A-Z]/.test(password)) {
    errors.push('Password must contain at least one uppercase letter');
  }
  
  if (!/[a-z]/.test(password)) {
    errors.push('Password must contain at least one lowercase letter');
  }
  
  if (!/[0-9]/.test(password)) {
    errors.push('Password must contain at least one number');
  }
  
  return {
    isValid: errors.length === 0,
    errors,
  };
};

// utils/validators/index.ts
export * from './form';
export * from './auth';

// utils/index.ts
export * from './helpers';
export * from './validators';
export * from './constants';
```

## 타입 구조

### 1. 타입 폴더 구조

```
types/
├── api/
│   ├── auth.ts
│   ├── users.ts
│   ├── products.ts
│   └── index.ts
├── components/
│   ├── button.ts
│   ├── form.ts
│   └── index.ts
├── global.d.ts
└── index.ts
```

### 2. 타입 파일 구조

```typescript
// types/api/auth.ts
export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  role: 'admin' | 'user';
  createdAt: string;
  updatedAt: string;
}

export interface AuthState {
  user: User | null;
  loading: boolean;
  error: string | null;
}

// types/api/index.ts
export * from './auth';
export * from './users';
export * from './products';

// types/components/button.ts
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
  className?: string;
}

// types/components/index.ts
export * from './button';
export * from './form';

// types/index.ts
export * from './api';
export * from './components';
```

## 상수 구조

### 1. 상수 폴더 구조

```
constants/
├── api.ts
├── routes.ts
├── storage.ts
├── app.ts
└── index.ts
```

### 2. 상수 파일 구조

```typescript
// constants/api.ts
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    LOGOUT: '/auth/logout',
    REGISTER: '/auth/register',
    REFRESH: '/auth/refresh',
    ME: '/auth/me',
  },
  USERS: {
    LIST: '/users',
    DETAIL: (id: string) => `/users/${id}`,
    CREATE: '/users',
    UPDATE: (id: string) => `/users/${id}`,
    DELETE: (id: string) => `/users/${id}`,
  },
  PRODUCTS: {
    LIST: '/products',
    DETAIL: (id: string) => `/products/${id}`,
    CREATE: '/products',
    UPDATE: (id: string) => `/products/${id}`,
    DELETE: (id: string) => `/products/${id}`,
    SEARCH: '/products/search',
  },
} as const;

export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  INTERNAL_SERVER_ERROR: 500,
} as const;

// constants/routes.ts
export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  REGISTER: '/register',
  DASHBOARD: '/dashboard',
  PROFILE: '/profile',
  PRODUCTS: '/products',
  PRODUCT_DETAIL: (id: string) => `/products/${id}`,
  CART: '/cart',
  CHECKOUT: '/checkout',
} as const;

// constants/storage.ts
export const STORAGE_KEYS = {
  ACCESS_TOKEN: 'accessToken',
  REFRESH_TOKEN: 'refreshToken',
  USER: 'user',
  CART: 'cart',
  THEME: 'theme',
  LANGUAGE: 'language',
} as const;

// constants/index.ts
export * from './api';
export * from './routes';
export * from './storage';
export * from './app';
```

## 명명 규칙

### 1. 파일 명명 규칙

- **컴포넌트 파일**: PascalCase (예: `Button.tsx`, `UserProfile.tsx`)
- **Hook 파일**: camelCase with 'use' prefix (예: `useAuth.ts`, `useApi.ts`)
- **유틸리티 파일**: camelCase (예: `dateHelpers.ts`, `formValidators.ts`)
- **타입 파일**: camelCase (예: `authTypes.ts`, `buttonTypes.ts`)
- **상수 파일**: camelCase (예: `apiConstants.ts`, `routeConstants.ts`)

### 2. 폴더 명명 규칙

- **컴포넌트 폴더**: PascalCase (예: `Button/`, `UserProfile/`)
- **기능 폴더**: camelCase (예: `auth/`, `products/`)
- **공유 폴더**: lowercase (예: `components/`, `hooks/`, `utils/`)

### 3. 변수 및 함수 명명 규칙

- **컴포넌트**: PascalCase (예: `Button`, `UserProfile`)
- **Hook**: camelCase with 'use' prefix (예: `useAuth`, `useApi`)
- **유틸리티 함수**: camelCase (예: `formatDate`, `validateEmail`)
- **상수**: UPPER_SNAKE_CASE (예: `API_ENDPOINTS`, `STORAGE_KEYS`)
- **타입/인터페이스**: PascalCase (예: `User`, `ButtonProps`)

## 인덱스 파일 활용

인덱스 파일을 사용하여 import 경로를 간결하게 유지합니다.

```typescript
// components/index.ts
export { default as Button } from './Button';
export { default as Input } from './Input';
export { default as Modal } from './Modal';

// hooks/index.ts
export { useAuth } from './useAuth';
export { useApi } from './useApi';
export { useLocalStorage } from './useLocalStorage';

// 사용 시
import { Button, Input, Modal } from '@/components';
import { useAuth, useApi } from '@/hooks';
```

## 절대 경로 설정

Vite나 Webpack에서 절대 경로를 설정하여 상대 경로의 복잡성을 줄입니다.

```typescript
// vite.config.ts
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@types': path.resolve(__dirname, './src/types'),
      '@services': path.resolve(__dirname, './src/services'),
      '@constants': path.resolve(__dirname, './src/constants'),
    },
  },
});
```

## 결론

효과적인 코드 조직은 다음 원칙을 따라야 합니다:
- 일관된 폴더 및 파일 구조 유지
- 관련 코드를 함께 그룹화
- 명확한 명명 규칙 사용
- 인덱스 파일로 import 경로 간소화
- 절대 경로 설정으로 상대 경로 복잡성 감소

이러한 원칙을 따르면 유지보수가 쉽고 확장 가능한 React 애플리케이션을 만들 수 있습니다.