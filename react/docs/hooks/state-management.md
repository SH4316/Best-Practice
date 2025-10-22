# 상태 관리

React에서 상태 관리는 컴포넌트의 데이터를 다루는 핵심적인 방법입니다. 이 섹션에서는 `useState`와 `useReducer`를 사용하여 컴포넌트 상태를 효과적으로 관리하는 방법을 설명합니다.

## useState

`useState`는 함수형 컴포넌트에 상태를 추가할 때 사용하는 가장 기본적인 Hook입니다.

### 기본 사용법

```typescript
// ✅ 좋은 예시: 기본 useState 사용
const Counter: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
};
```

### 함수형 업데이트

이전 상태에 의존하는 상태 업데이트에서는 함수형 업데이트를 사용하는 것이 안전합니다.

```typescript
// ✅ 좋은 예시: 함수형 업데이트 사용
const Counter: React.FC = () => {
  const [count, setCount] = useState<number>(0);
  
  const increment = (): void => {
    setCount(prevCount => prevCount + 1);
  };
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>Increment</button>
    </div>
  );
};
```

### 객체 상태 관리

복잡한 데이터 구조를 다룰 때는 객체 상태를 사용할 수 있습니다.

```typescript
// ✅ 좋은 예시: 객체 상태 관리
interface User {
  name: string;
  email: string;
  age: number;
}

const UserProfile: React.FC = () => {
  const [user, setUser] = useState<User>({
    name: '',
    email: '',
    age: 0,
  });
  
  const updateName = (name: string): void => {
    setUser(prevUser => ({ ...prevUser, name }));
  };
  
  return (
    <div>
      <input
        value={user.name}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateName(e.target.value)}
      />
      <p>Name: {user.name}</p>
    </div>
  );
};
```

## useReducer

`useReducer`는 복잡한 상태 로직을 관리할 때 `useState`보다 유용한 대안입니다. 특히 상태 업데이트 로직이 여러 개이거나, 이전 상태에 의존적인 경우에 적합합니다.

### 기본 사용법

```typescript
// ✅ 좋은 예시: useReducer를 사용한 TODO 상태 관리
interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

type TodoAction =
  | { type: 'ADD_TODO'; text: string }
  | { type: 'TOGGLE_TODO'; id: number }
  | { type: 'DELETE_TODO'; id: number };

const todoReducer = (todos: Todo[], action: TodoAction): Todo[] => {
  switch (action.type) {
    case 'ADD_TODO':
      return [
        ...todos,
        { id: Date.now(), text: action.text, completed: false },
      ];
    case 'TOGGLE_TODO':
      return todos.map(todo =>
        todo.id === action.id ? { ...todo, completed: !todo.completed } : todo
      );
    case 'DELETE_TODO':
      return todos.filter(todo => todo.id !== action.id);
    default:
      return todos;
  }
};

const TodoApp: React.FC = () => {
  const [todos, dispatch] = useReducer(todoReducer, []);
  const [inputValue, setInputValue] = useState<string>('');
  
  const addTodo = (): void => {
    if (inputValue.trim()) {
      dispatch({ type: 'ADD_TODO', text: inputValue });
      setInputValue('');
    }
  };
  
  return (
    <div>
      <input
        value={inputValue}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputValue(e.target.value)}
        placeholder="Add a todo"
      />
      <button onClick={addTodo}>Add</button>
      
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => dispatch({ type: 'TOGGLE_TODO', id: todo.id })}
            />
            {todo.text}
            <button onClick={() => dispatch({ type: 'DELETE_TODO', id: todo.id })}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};
```

## useState vs useReducer

### useState를 사용해야 하는 경우:
- 간단한 상태 값 (숫자, 문자열, boolean 등)
- 독립적인 상태 값들
- 상태 업데이트 로직이 단순한 경우

### useReducer를 사용해야 하는 경우:
- 복잡한 상태 로직이 있는 경우
- 여러 개의 상태 값이 서로 의존적인 경우
- 상태 업데이트 로직을 재사용해야 하는 경우
- 상태 변경을 추적하거나 디버깅이 필요한 경우

## 상태 관리 모범 사례

1. **상태 구조화**: 관련된 데이터는 객체로 묶어 관리하세요
2. **불변성 유지**: 상태를 직접 수정하지 말고, 항상 새로운 상태를 생성하세요
3. **상태 최소화**: 파생될 수 있는 값은 상태로 저장하지 마세요
4. **적절한 Hook 선택**: 상태 복잡도에 따라 `useState`와 `useReducer`를 적절히 선택하세요

## 다음 단계

상태 관리에 대한 더 자세한 내용은 [전역 상태 관리](../state-management.md) 문서를 참조하세요.

사이드 이펙트 처리에 대해서는 [사이드 이펙트](./side-effects.md) 문서를 확인하세요.