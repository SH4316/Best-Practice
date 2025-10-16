# DOM 상호작용

`useRef`는 DOM 요소에 직접 접근하거나 렌더링을 유발하지 않는 값을 저장할 때 사용하는 Hook입니다. `useRef`는 일반 JavaScript 변수와 달리 값이 변경되어도 컴포넌트가 다시 렌더링되지 않습니다.

## 기본 사용법

### DOM 요소 참조

```typescript
// ✅ 좋은 예시: DOM 요소 참조
const TextInputWithFocusButton: React.FC = () => {
  const inputRef = useRef<HTMLInputElement>(null);
  
  const onButtonClick = (): void => {
    inputRef.current?.focus();
  };
  
  return (
    <>
      <input ref={inputRef} type="text" />
      <button onClick={onButtonClick}>Focus the input</button>
    </>
  );
};
```

### 렌더링을 유발하지 않는 값 저장

```typescript
// ✅ 좋은 예시: 렌더링을 유발하지 않는 값 저장
const Timer: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const startTimer = (): void => {
    if (intervalRef.current) return; // 이미 실행 중이면 시작하지 않음
    
    intervalRef.current = setInterval(() => {
      setCount(prevCount => prevCount + 1);
    }, 1000);
  };
  
  const stopTimer = (): void => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={startTimer}>Start</button>
      <button onClick={stopTimer}>Stop</button>
    </div>
  );
};
```

### 이전 값 저장

```typescript
// ✅ 좋은 예시: 이전 값 저장
interface PreviousValueProps<T> {
  value: T;
}

const PreviousValue = <T,>({ value }: PreviousValueProps<T>) => {
  const prevValueRef = useRef<T>();
  
  useEffect(() => {
    prevValueRef.current = value;
  }); // 렌더링 후 실행되어 이전 값을 저장
  
  return (
    <div>
      <p>Current: {String(value)}</p>
      <p>Previous: {String(prevValueRef.current)}</p>
    </div>
  );
};
```

## 고급 사용 패턴

### 스크롤 위치 제어

```typescript
const ScrollToTop: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const scrollToTop = (): void => {
    containerRef.current?.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };
  
  return (
    <div>
      <div 
        ref={containerRef}
        style={{ height: '200px', overflow: 'auto', border: '1px solid #ccc' }}
      >
        <div style={{ height: '1000px', padding: '20px' }}>
          <p>Long content here...</p>
          <p>Scroll down and then click the button</p>
        </div>
      </div>
      <button onClick={scrollToTop}>Scroll to Top</button>
    </div>
  );
};
```

### 캔버스 조작

```typescript
const CanvasDrawing: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // 간단한 그리기
    ctx.fillStyle = '#3B82F6';
    ctx.fillRect(10, 10, 100, 100);
    
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(150, 60, 50, 0, 2 * Math.PI);
    ctx.stroke();
  }, []);
  
  return (
    <div>
      <canvas 
        ref={canvasRef}
        width={300}
        height={200}
        style={{ border: '1px solid #ccc' }}
      />
    </div>
  );
};
```

### 외부 라이브러리와 통합

```typescript
// 외부 라이브러리 (예: 차트 라이브러리)와 통합
const ChartComponent: React.FC<{ data: number[] }> = ({ data }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  
  useEffect(() => {
    if (!chartRef.current) return;
    
    // 가상의 차트 라이브러리 초기화
    // chartInstanceRef.current = new ChartLib(chartRef.current, {
    //   data: data,
    //   options: { /* 차트 옵션 */ }
    // });
    
    // 실제로는 여기서 외부 라이브러리를 초기화합니다
    
    return () => {
      // 컴포넌트 언마운트 시 정리
      if (chartInstanceRef.current) {
        // chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [data]);
  
  return <div ref={chartRef} style={{ width: '100%', height: '400px' }} />;
};
```

### 커스텀 Hook과 함께 사용

```typescript
// 커스텀 Hook: useClickOutside
const useClickOutside = (
  ref: RefObject<HTMLElement>,
  handler: () => void
) => {
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        handler();
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [ref, handler]);
};

// 사용 예시: 모달 컴포넌트
const Modal: React.FC<{ isOpen: boolean; onClose: () => void; children: ReactNode }> = ({ 
  isOpen, 
  onClose, 
  children 
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  
  useClickOutside(modalRef, onClose);
  
  if (!isOpen) return null;
  
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <div 
        ref={modalRef}
        style={{
          backgroundColor: 'white',
          padding: '20px',
          borderRadius: '8px',
          maxWidth: '500px',
          width: '100%'
        }}
      >
        {children}
      </div>
    </div>
  );
};
```

## useRef 사용 시 주의사항

### 1. ref 초기화

```typescript
// ✅ 좋은 예시: 타입과 함께 ref 초기화
const inputRef = useRef<HTMLInputElement>(null);
const countRef = useRef<number>(0);
const dataRef = useRef<any[]>([]);

// ❌ 나쁜 예시: 타입 없이 ref 사용
const inputRef = useRef(null); // 타입 추론이 어려움
```

### 2. ref 값 변경

```typescript
// ✅ 좋은 예시: ref 값 직접 변경
const Component: React.FC = () => {
  const countRef = useRef<number>(0);
  
  const increment = (): void => {
    countRef.current += 1; // 렌더링을 유발하지 않음
    console.log('Count:', countRef.current);
  };
  
  return <button onClick={increment}>Increment</button>;
};

// ❌ 나쁜 예시: ref 값 변경 후 렌더링 시도
const Component: React.FC = () => {
  const countRef = useRef<number>(0);
  const [, forceUpdate] = useState({});
  
  const increment = (): void => {
    countRef.current += 1;
    forceUpdate({}); // 강제 리렌더링은 안티패턴
  };
  
  return <button onClick={increment}>Increment</button>;
};
```

### 3. ref와 state 구분

```typescript
// ✅ 좋은 예시: 적절한 상황에 ref와 state 사용
const Component: React.FC = () => {
  const [count, setCount] = useState<number>(0); // 렌더링에 필요한 값
  const renderCountRef = useRef<number>(0); // 렌더링에 필요 없는 값
  
  renderCountRef.current += 1;
  
  return (
    <div>
      <p>Count: {count}</p>
      <p>Render count: {renderCountRef.current}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
};
```

## 일반적인 사용 사례

### 1. 포커스 관리

```typescript
const FormWithAutoFocus: React.FC = () => {
  const nameInputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    // 컴포넌트 마운트 시 자동 포커스
    nameInputRef.current?.focus();
  }, []);
  
  const handleSubmit = (e: FormEvent): void => {
    e.preventDefault();
    // 폼 제출 후 다른 입력 필드로 포커스 이동
    nameInputRef.current?.focus();
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input ref={nameInputRef} type="text" placeholder="Name" />
      <input type="email" placeholder="Email" />
      <button type="submit">Submit</button>
    </form>
  );
};
```

### 2. 미디어 요소 제어

```typescript
const VideoPlayer: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  
  const togglePlay = (): void => {
    if (!videoRef.current) return;
    
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
    
    setIsPlaying(!isPlaying);
  };
  
  return (
    <div>
      <video ref={videoRef} width="400" height="300">
        <source src="video.mp4" type="video/mp4" />
      </video>
      <button onClick={togglePlay}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
    </div>
  );
};
```

## 다음 단계

커스텀 Hook 만들기에 대해서는 [커스텀 Hooks](./custom-hooks.md) 문서를 확인하세요.

Hooks 사용 규칙에 대해서는 [Hooks 사용 규칙](./hooks-rules.md) 문서를 참조하세요.