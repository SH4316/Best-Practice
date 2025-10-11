# React 컴포넌트 구조

React 컴포넌트를 효과적으로 구조화하는 것은 유지보수성, 재사용성, 테스트 용이성에 매우 중요합니다.

## 컴포넌트 구조 원칙

### 1. 단일 책임 원칙 (Single Responsibility Principle)

각 컴포넌트는 하나의 명확한 책임만 가져야 합니다.

```typescript
// ❌ 나쁜 예시: 여러 책임을 가진 컴포넌트
const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // 데이터 페칭, UI 렌더링, 스타일링까지 모두 처리
  useEffect(() => {
    fetch('/api/user')
      .then(res => res.json())
      .then(data => setUser(data))
      .finally(() => setIsLoading(false));
  }, []);

  if (isLoading) return <div>Loading...</div>;
  
  return (
    <div className="user-profile">
      <img src={user.avatar} alt={user.name} />
      <h2>{user.name}</h2>
      <p>{user.email}</p>
      <button onClick={() => {/* 복잡한 로직 */}}>
        Update Profile
      </button>
    </div>
  );
};
```

```typescript
// ✅ 좋은 예시: 책임 분리
// 데이터 페칭 전담 Hook
const useUser = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    setIsLoading(true);
    fetch('/api/user')
      .then(res => res.json())
      .then(data => setUser(data))
      .finally(() => setIsLoading(false));
  }, []);

  return { user, isLoading };
};

// UI 렌더링 전담 컴포넌트
const UserProfile = () => {
  const { user, isLoading } = useUser();
  
  if (isLoading) return <LoadingSpinner />;
  
  return (
    <div className="user-profile">
      <Avatar src={user.avatar} alt={user.name} />
      <UserInfo name={user.name} email={user.email} />
      <UpdateProfileButton userId={user.id} />
    </div>
  );
};
```

### 2. 컴포넌트 합성 (Component Composition)

작은 컴포넌트를 조합하여 복잡한 UI를 구축합니다.

```typescript
// ❌ 나쁜 예시: 모든 것을 하나의 컴포넌트에서 처리
const ProductCard = ({ product }) => {
  return (
    <div className="product-card">
      <div className="product-image">
        <img src={product.image} alt={product.name} />
        {product.isNew && <span className="new-badge">NEW</span>}
        {product.discount && <span className="discount-badge">-{product.discount}%</span>}
      </div>
      <div className="product-info">
        <h3 className="product-name">{product.name}</h3>
        <p className="product-description">{product.description}</p>
        <div className="product-price">
          {product.originalPrice && (
            <span className="original-price">${product.originalPrice}</span>
          )}
          <span className="current-price">${product.price}</span>
        </div>
        <div className="product-rating">
          <div className="stars">
            {[...Array(5)].map((_, i) => (
              <span key={i} className={i < product.rating ? "filled" : "empty"}>
                ★
              </span>
            ))}
          </div>
          <span className="rating-count">({product.reviewCount})</span>
        </div>
        <button 
          className="add-to-cart-btn"
          onClick={() => {/* 복잡한 장바구니 로직 */}}
          disabled={!product.inStock}
        >
          {product.inStock ? "Add to Cart" : "Out of Stock"}
        </button>
      </div>
    </div>
  );
};
```

```typescript
// ✅ 좋은 예시: 컴포넌트 합성
const ProductCard = ({ product }) => {
  return (
    <Card>
      <ProductImage 
        src={product.image} 
        alt={product.name}
        badges={
          <>
            {product.isNew && <Badge type="new">NEW</Badge>}
            {product.discount && <Badge type="discount">-{product.discount}%</Badge>}
          </>
        }
      />
      <CardBody>
        <ProductTitle name={product.name} />
        <ProductDescription description={product.description} />
        <ProductPrice 
          originalPrice={product.originalPrice}
          currentPrice={product.price}
        />
        <ProductRating 
          rating={product.rating}
          reviewCount={product.reviewCount}
        />
        <AddToCartButton 
          productId={product.id}
          inStock={product.inStock}
        />
      </CardBody>
    </Card>
  );
};
```

### 3. Props 인터페이스 명확화

컴포넌트의 Props를 명확하게 정의하고 타입을 지정합니다.

```typescript
// ❌ 나쁜 예시: 불명확한 Props
const Button = (props) => {
  return (
    <button 
      className={props.className}
      onClick={props.onClick}
      disabled={props.disabled}
    >
      {props.children}
    </button>
  );
};
```

```typescript
// ✅ 좋은 예시: 명확한 Props 인터페이스
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
  className?: string;
  'data-testid'?: string;
}

const Button = ({
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  onClick,
  children,
  className,
  'data-testid': testId,
}: ButtonProps) => {
  return (
    <button
      className={cn(
        'btn',
        `btn--${variant}`,
        `btn--${size}`,
        { 'btn--loading': loading },
        className
      )}
      onClick={onClick}
      disabled={disabled || loading}
      data-testid={testId}
    >
      {loading ? <Spinner /> : children}
    </button>
  );
};
```

### 4. 컴포넌트 파일 구조

각 컴포넌트는 다음과 같은 파일 구조를 따르는 것이 좋습니다:

```
Button/
├── Button.tsx          # 메인 컴포넌트
├── Button.styles.ts    # 스타일 관련 코드
├── Button.test.tsx     # 테스트 코드
├── Button.stories.tsx  # Storybook 스토리 (선택적)
├── index.ts            # 내보내기 파일
└── types.ts            # 타입 정의 (필요시)
```

## 컴포넌트 종류별 구조 가이드

### 1. 프레젠테이션 컴포넌트 (Presentational Components)

데이터 표시에만 집중하고, 비즈니스 로직은 포함하지 않습니다.

```typescript
// Avatar.tsx
interface AvatarProps {
  src?: string;
  alt: string;
  size?: 'small' | 'medium' | 'large';
  fallback?: string;
}

const Avatar = ({ 
  src, 
  alt, 
  size = 'medium', 
  fallback 
}: AvatarProps) => {
  const [hasError, setHasError] = useState(false);
  
  const handleImageError = () => {
    setHasError(true);
  };
  
  if (hasError || !src) {
    return (
      <div className={`avatar avatar--${size} avatar--fallback`}>
        {fallback?.charAt(0).toUpperCase()}
      </div>
    );
  }
  
  return (
    <img 
      src={src} 
      alt={alt}
      className={`avatar avatar--${size}`}
      onError={handleImageError}
    />
  );
};
```

### 2. 컨테이너 컴포넌트 (Container Components)

데이터 페칭, 상태 관리 등 비즈니스 로직을 처리합니다.

```typescript
// UserContainer.tsx
const UserContainer = () => {
  const { user, isLoading, error } = useUser();
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error.message} />;
  if (!user) return <EmptyState message="User not found" />;
  
  return <UserProfile user={user} />;
};
```

### 3. 페이지 컴포넌트 (Page Components)

페이지 전체의 레이아웃과 여러 컴포넌트를 조합합니다.

```typescript
// HomePage.tsx
const HomePage = () => {
  return (
    <PageLayout>
      <Hero />
      <FeaturedProducts />
      <Testimonials />
      <NewsletterSignup />
    </PageLayout>
  );
};
```

## 컴포넌트 명명 규칙

1. **컴포넌트 이름**: PascalCase (예: `UserProfile`, `DataTable`)
2. **Props 인터페이스**: 컴포넌트 이름 + Props (예: `UserProfileProps`)
3. **컴포넌트 폴더**: 컴포넌트 이름과 동일 (예: `UserProfile/`)
4. **스타일 관련**: 컴포넌트 이름 + Styles (예: `UserProfileStyles`)

## 컴포넌트 재사용성 향상 팁

### 1. Children Prop 활용

```typescript
// ❌ 유연하지 않은 컴포넌트
const Card = ({ title, content, footer }) => {
  return (
    <div className="card">
      <div className="card-header">{title}</div>
      <div className="card-body">{content}</div>
      <div className="card-footer">{footer}</div>
    </div>
  );
};
```

```typescript
// ✅ 유연한 컴포넌트
interface CardProps {
  header?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

const Card = ({ header, footer, children, className }: CardProps) => {
  return (
    <div className={cn('card', className)}>
      {header && <div className="card-header">{header}</div>}
      <div className="card-body">{children}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
};
```

### 2. Render Props 패턴

```typescript
interface DataFetcherProps<T> {
  url: string;
  children: (data: T | null, isLoading: boolean, error: Error | null) => React.ReactNode;
}

const DataFetcher = <T,>({ url, children }: DataFetcherProps<T>) => {
  const { data, isLoading, error } = useFetch<T>(url);
  
  return <>{children(data, isLoading, error)}</>;
};

// 사용 예시
const UserList = () => {
  return (
    <DataFetcher<User[]> url="/api/users">
      {(users, isLoading, error) => {
        if (isLoading) return <LoadingSpinner />;
        if (error) return <ErrorMessage error={error} />;
        if (!users) return <EmptyState />;
        
        return (
          <ul>
            {users.map(user => (
              <li key={user.id}>{user.name}</li>
            ))}
          </ul>
        );
      }}
    </DataFetcher>
  );
};
```

### 3. 커스텀 Hook으로 로직 분리

```typescript
// useModal.ts
interface UseModalReturn {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

const useModal = (initialState = false): UseModalReturn => {
  const [isOpen, setIsOpen] = useState(initialState);
  
  const open = () => setIsOpen(true);
  const close = () => setIsOpen(false);
  const toggle = () => setIsOpen(prev => !prev);
  
  return { isOpen, open, close, toggle };
};

// Modal.tsx
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

const Modal = ({ isOpen, onClose, children }: ModalProps) => {
  if (!isOpen) return null;
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          ×
        </button>
        {children}
      </div>
    </div>
  );
};

// 사용 예시
const App = () => {
  const { isOpen, open, close } = useModal();
  
  return (
    <div>
      <button onClick={open}>Open Modal</button>
      <Modal isOpen={isOpen} onClose={close}>
        <h2>Modal Content</h2>
        <p>This is the modal content</p>
      </Modal>
    </div>
  );
};
```

## 결론

효과적인 컴포넌트 구조는 다음 원칙을 따라야 합니다:
- 단일 책임 원칙 준수
- 작은 컴포넌트의 합성으로 복잡한 UI 구축
- 명확한 Props 인터페이스 정의
- 일관된 파일 구조 유지
- 재사용성을 높이는 패턴 활용

이러한 원칙을 따르면 유지보수가 쉽고 재사용 가능한 컴포넌트를 만들 수 있습니다.