# React 스타일링

스타일링은 React 애플리케이션의 사용자 경험에 큰 영향을 미칩니다. 적절한 스타일링 전략을 사용하면 유지보수가 쉽고, 확장 가능하며, 성능이 뛰어난 애플리케이션을 만들 수 있습니다. 이 문서에서는 React 애플리케이션을 스타일링하는 다양한 방법과 모범 사례를 설명합니다.

## 스타일링 방법론

### 1. 일반 CSS

가장 기본적인 방법은 일반 CSS 파일을 사용하는 것입니다.

```css
/* ❌ 나쁜 예시: 전역 범위의 CSS */
.button {
  background-color: blue;
  color: white;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
}

/* 다른 컴포넌트에 영향을 줄 수 있음 */
.card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
}
```

```css
/* ✅ 좋은 예시: BEM 방법론 사용 */
.button {
  background-color: blue;
  color: white;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
}

.button--primary {
  background-color: #007bff;
}

.button--secondary {
  background-color: #6c757d;
}

.button--large {
  padding: 12px 24px;
  font-size: 16px;
}

.card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
}

.card__header {
  border-bottom: 1px solid #eee;
  padding-bottom: 8px;
  margin-bottom: 16px;
}

.card__content {
  font-size: 14px;
  line-height: 1.5;
}
```

### 2. CSS 모듈

CSS 모듈은 CSS를 컴포넌트 범위로 제한하여 클래스 이름 충돌을 방지합니다.

```css
/* ✅ 좋은 예시: CSS 모듈 사용 */
/* Button.module.css */
.button {
  background-color: blue;
  color: white;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.button:hover {
  background-color: #0056b3;
}

.buttonPrimary {
  background-color: #007bff;
}

.buttonPrimary:hover {
  background-color: #0069d9;
}
```

```typescript
// Button.tsx
import styles from './Button.module.css';

interface ButtonProps {
  children: React.ReactNode;
  variant?: 'default' | 'primary';
  onClick?: () => void;
}

const Button = ({ 
  children, 
  variant = 'default', 
  onClick 
}: ButtonProps) => {
  const buttonClass = variant === 'primary' 
    ? `${styles.button} ${styles.buttonPrimary}`
    : styles.button;
    
  return (
    <button className={buttonClass} onClick={onClick}>
      {children}
    </button>
  );
};

export default Button;
```

### 3. CSS-in-JS

CSS-in-JS는 JavaScript를 사용하여 스타일을 작성하는 방법입니다.

```typescript
// ❌ 나쁜 예시: 인라인 스타일 사용
const BadButton = ({ children, onClick }) => {
  return (
    <button 
      style={{
        backgroundColor: 'blue',
        color: 'white',
        padding: '8px 16px',
        border: 'none',
        borderRadius: '4px',
      }}
      onClick={onClick}
    >
      {children}
    </button>
  );
};

// ✅ 좋은 예시: styled-components 사용
import styled from 'styled-components';

const StyledButton = styled.button<{ variant?: 'default' | 'primary' }>`
  background-color: ${props => 
    props.variant === 'primary' ? '#007bff' : 'blue'
  };
  color: white;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: ${props => 
      props.variant === 'primary' ? '#0069d9' : '#0056b3'
    };
  }
`;

const Button = ({ 
  children, 
  variant = 'default', 
  onClick 
}: ButtonProps) => {
  return (
    <StyledButton variant={variant} onClick={onClick}>
      {children}
    </StyledButton>
  );
};

export default Button;
```

## 컴포넌트 스타일링 패턴

### 1. 컴포지션 패턴

컴포넌트를 조합하여 스타일을 재사용하는 패턴입니다.

```typescript
// ✅ 좋은 예시: 컴포지션 패턴 사용
import styled from 'styled-components';

// 기본 스타일을 가진 컴포넌트
const BaseButton = styled.button`
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s;
`;

// 스타일을 확장한 컴포넌트
const PrimaryButton = styled(BaseButton)`
  background-color: #007bff;
  color: white;
  
  &:hover {
    background-color: #0069d9;
  }
`;

const SecondaryButton = styled(BaseButton)`
  background-color: transparent;
  color: #007bff;
  border: 1px solid #007bff;
  
  &:hover {
    background-color: #f8f9fa;
  }
`;

const Button = ({ 
  children, 
  variant = 'primary', 
  onClick 
}: ButtonProps) => {
  const ButtonComponent = variant === 'primary' 
    ? PrimaryButton 
    : SecondaryButton;
    
  return (
    <ButtonComponent onClick={onClick}>
      {children}
    </ButtonComponent>
  );
};

export default Button;
```

### 2. 테마 및 디자인 시스템

테마를 사용하여 일관된 디자인을 유지하는 패턴입니다.

```typescript
// ✅ 좋은 예시: 테마 및 디자인 시스템 사용
import styled, { ThemeProvider } from 'styled-components';

// 테마 정의
const theme = {
  colors: {
    primary: '#007bff',
    secondary: '#6c757d',
    success: '#28a745',
    danger: '#dc3545',
    warning: '#ffc107',
    info: '#17a2b8',
    light: '#f8f9fa',
    dark: '#343a40',
  },
  fonts: {
    primary: '"Helvetica Neue", Helvetica, Arial, sans-serif',
    monospace: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace',
  },
  fontSizes: {
    xs: '0.75rem',
    sm: '0.875rem',
    md: '1rem',
    lg: '1.25rem',
    xl: '1.5rem',
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '3rem',
  },
  breakpoints: {
    mobile: '480px',
    tablet: '768px',
    desktop: '1024px',
  },
};

// 테마를 사용하는 스타일드 컴포넌트
const Button = styled.button<{ variant?: 'primary' | 'secondary' }>`
  background-color: ${props => 
    props.variant === 'primary' 
      ? props.theme.colors.primary 
      : props.theme.colors.secondary
  };
  color: white;
  padding: ${props.theme.spacing.sm} ${props.theme.spacing.md};
  border: none;
  border-radius: 4px;
  font-size: ${props.theme.fontSizes.md};
  font-family: ${props.theme.fonts.primary};
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: ${props => 
      props.variant === 'primary' 
        ? '#0069d9' 
        : '#5a6268'
    };
  }
  
  @media (max-width: ${props.theme.breakpoints.mobile}) {
    padding: ${props.theme.spacing.xs} ${props.theme.spacing.sm};
    font-size: ${props.theme.fontSizes.sm};
  }
`;

// 앱에서 테마 제공자 사용
const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <div>
        <Button variant="primary">Primary Button</Button>
        <Button variant="secondary">Secondary Button</Button>
      </div>
    </ThemeProvider>
  );
};

export default App;
```

## 반응형 디자인

### 1. 미디어 쿼리

미디어 쿼리를 사용하여 다양한 화면 크기에 맞게 스타일을 조정합니다.

```css
/* ✅ 좋은 예시: 미디어 쿼리 사용 */
/* Card.module.css */
.card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

/* 모바일 화면 */
@media (max-width: 768px) {
  .card {
    padding: 12px;
    margin-bottom: 12px;
  }
}

/* 태블릿 화면 */
@media (min-width: 769px) and (max-width: 1024px) {
  .card {
    padding: 14px;
    margin-bottom: 14px;
  }
}

/* 데스크톱 화면 */
@media (min-width: 1025px) {
  .card {
    padding: 16px;
    margin-bottom: 16px;
  }
}
```

```typescript
// ✅ 좋은 예시: styled-components에서 미디어 쿼리 사용
import styled from 'styled-components';

const Card = styled.div`
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
  
  @media (max-width: 768px) {
    padding: 12px;
    margin-bottom: 12px;
  }
  
  @media (min-width: 769px) and (max-width: 1024px) {
    padding: 14px;
    margin-bottom: 14px;
  }
`;

export default Card;
```

### 2. 컨테이너 쿼리

컨테이너 쿼리는 컴포넌트의 크기에 따라 스타일을 조정하는 방법입니다.

```typescript
// ✅ 좋은 예시: 컨테이너 쿼리 사용
import { useContainerQuery } from 'react-container-query';
import styled from 'styled-components';

// 쿼리 정의
const query = {
  'small': {
    maxWidth: 300,
  },
  'medium': {
    minWidth: 301,
    maxWidth: 600,
  },
  'large': {
    minWidth: 601,
  },
};

// 스타일드 컴포넌트
const ResponsiveCard = styled.div<{ size: string }>`
  padding: ${props => {
    switch (props.size) {
      case 'small': return '8px';
      case 'medium': return '12px';
      case 'large': return '16px';
      default: return '16px';
    }
  }};
  
  font-size: ${props => {
    switch (props.size) {
      case 'small': return '12px';
      case 'medium': return '14px';
      case 'large': return '16px';
      default: return '16px';
    }
  }};
`;

// 컴포넌트
const Card = ({ children }) => {
  const [containerRef, size] = useContainerQuery(query);
  
  return (
    <ResponsiveCard ref={containerRef} size={size}>
      {children}
    </ResponsiveCard>
  );
};

export default Card;
```

## 성능 최적화

### 1. 스타일 최적화

불필요한 스타일 계산을 피하고 스타일을 효율적으로 로드합니다.

```typescript
// ❌ 나쁜 예시: 렌더링마다 스타일 계산
const BadComponent = ({ isActive }) => {
  return (
    <div 
      style={{
        backgroundColor: isActive ? 'blue' : 'gray',
        color: 'white',
        padding: '8px 16px',
        borderRadius: '4px',
      }}
    >
      Button
    </div>
  );
};

// ✅ 좋은 예시: 스타일을 미리 계산
import { useMemo } from 'react';

const GoodComponent = ({ isActive }) => {
  const buttonStyle = useMemo(() => ({
    backgroundColor: isActive ? 'blue' : 'gray',
    color: 'white',
    padding: '8px 16px',
    borderRadius: '4px',
  }), [isActive]);
  
  return (
    <div style={buttonStyle}>
      Button
    </div>
  );
};
```

### 2. CSS-in-JS 최적화

CSS-in-JS 라이브러리의 최적화 기능을 활용합니다.

```typescript
// ✅ 좋은 예시: styled-components 최적화
import styled, { css } from 'styled-components';

// 재사용 가능한 스타일 조각
const buttonStyles = css`
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s;
`;

// 스타일 조각을 사용하는 컴포넌트
const PrimaryButton = styled.button`
  ${buttonStyles}
  background-color: #007bff;
  color: white;
  
  &:hover {
    background-color: #0069d9;
  }
`;

const SecondaryButton = styled.button`
  ${buttonStyles}
  background-color: transparent;
  color: #007bff;
  border: 1px solid #007bff;
  
  &:hover {
    background-color: #f8f9fa;
  }
`;
```

## 접근성 고려사항

### 1. 색상 대비

충분한 색상 대비를 제공하여 가독성을 높입니다.

```css
/* ✅ 좋은 예시: 충분한 색상 대비 */
.button {
  background-color: #007bff;
  color: #ffffff;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
}

.button:hover {
  background-color: #0056b3;
}

.button:focus {
  outline: 2px solid #80bdff;
  outline-offset: 2px;
}

/* 대비율이 낮은 조합 피하기 */
.button-low-contrast {
  background-color: #f0f0f0;
  color: #e0e0e0; /* 대비율이 낮아 접근성 문제 */
}
```

### 2. 다크 모드 지원

다크 모드를 지원하여 다양한 환경에서 사용할 수 있도록 합니다.

```typescript
// ✅ 좋은 예시: 다크 모드 지원
import styled, { ThemeProvider } from 'styled-components';

// 라이트 및 다크 테마 정의
const lightTheme = {
  colors: {
    background: '#ffffff',
    text: '#333333',
    primary: '#007bff',
  },
};

const darkTheme = {
  colors: {
    background: '#333333',
    text: '#ffffff',
    primary: '#0d6efd',
  },
};

// 테마를 사용하는 컴포넌트
const Container = styled.div`
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.text};
  min-height: 100vh;
  padding: 16px;
`;

const Button = styled.button`
  background-color: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
`;

// 테마 전환 기능이 있는 앱
const App = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const theme = isDarkMode ? darkTheme : lightTheme;
  
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };
  
  return (
    <ThemeProvider theme={theme}>
      <Container>
        <h1>{isDarkMode ? 'Dark Mode' : 'Light Mode'}</h1>
        <Button onClick={toggleTheme}>
          Toggle Theme
        </Button>
      </Container>
    </ThemeProvider>
  );
};

export default App;
```

## 결론

효과적인 스타일링 전략은 다음 원칙을 따라야 합니다:
- 컴포넌트 범위의 스타일링을 위해 CSS 모듈 또는 CSS-in-JS 사용
- 일관된 디자인을 위해 테마 및 디자인 시스템 구축
- 다양한 화면 크기에 대응하는 반응형 디자인 구현
- 불필요한 스타일 계산을 피하는 성능 최적화
- 충분한 색상 대비와 다크 모드 지원을 통한 접근성 향상

이러한 원칙을 따르면 유지보수가 쉽고, 확장 가능하며, 사용자 경험이 뛰어난 React 애플리케이션을 만들 수 있습니다.