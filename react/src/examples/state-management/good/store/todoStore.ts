import { createContext, useContext, useReducer, createElement } from 'react';
import type { ReactNode } from 'react';
import type { Todo, TodoFilter, TodoState } from '../types';

// Action types
type TodoAction =
  | { type: 'SET_TODOS'; payload: Todo[] }
  | { type: 'ADD_TODO'; payload: Omit<Todo, 'id' | 'createdAt'> }
  | { type: 'TOGGLE_TODO'; payload: number }
  | { type: 'DELETE_TODO'; payload: number }
  | { type: 'UPDATE_TODO_PRIORITY'; payload: { id: number; priority: Todo['priority'] } }
  | { type: 'SET_FILTER'; payload: Partial<TodoFilter> }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null };

// Initial state
const initialState: TodoState = {
  todos: [],
  filter: {
    status: 'all',
    searchTerm: '',
    category: 'all',
  },
  isLoading: false,
  error: null,
};

// Reducer
const todoReducer = (state: TodoState, action: TodoAction): TodoState => {
  switch (action.type) {
    case 'SET_TODOS':
      return { ...state, todos: action.payload };
    
    case 'ADD_TODO':
      return {
        ...state,
        todos: [
          ...state.todos,
          {
            ...action.payload,
            id: Date.now(),
            createdAt: new Date(),
          },
        ],
      };
    
    case 'TOGGLE_TODO':
      return {
        ...state,
        todos: state.todos.map((todo) =>
          todo.id === action.payload
            ? { ...todo, completed: !todo.completed }
            : todo
        ),
      };
    
    case 'DELETE_TODO':
      return {
        ...state,
        todos: state.todos.filter((todo) => todo.id !== action.payload),
      };
    
    case 'UPDATE_TODO_PRIORITY':
      return {
        ...state,
        todos: state.todos.map((todo) =>
          todo.id === action.payload.id
            ? { ...todo, priority: action.payload.priority }
            : todo
        ),
      };
    
    case 'SET_FILTER':
      return {
        ...state,
        filter: { ...state.filter, ...action.payload },
      };
    
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    
    default:
      return state;
  }
};

// Context
const TodoContext = createContext<{
  state: TodoState;
  dispatch: React.Dispatch<TodoAction>;
} | null>(null);

// Provider
export const TodoProvider = ({ children }: { children: ReactNode }) => {
  const [state, dispatch] = useReducer(todoReducer, initialState);

  return createElement(
    TodoContext.Provider,
    { value: { state, dispatch } },
    children
  );
};

// Hook
export const useTodoContext = () => {
  const context = useContext(TodoContext);
  if (!context) {
    throw new Error('useTodoContext must be used within a TodoProvider');
  }
  return context;
};

// ✅ 좋은 예시: 파생된 상태를 위한 선택자 Hook
export const useFilteredTodos = () => {
  const { state } = useTodoContext();
  
  return state.todos.filter((todo) => {
    // 상태 필터
    if (state.filter.status === 'active' && todo.completed) return false;
    if (state.filter.status === 'completed' && !todo.completed) return false;
    
    // 검색어 필터
    if (state.filter.searchTerm && 
        !todo.text.toLowerCase().includes(state.filter.searchTerm.toLowerCase())) {
      return false;
    }
    
    // 카테고리 필터
    if (state.filter.category !== 'all' && todo.category !== state.filter.category) {
      return false;
    }
    
    return true;
  });
};

// ✅ 좋은 예시: 파생된 상태를 위한 선택자 Hook
export const useTodoStats = () => {
  const { state } = useTodoContext();
  
  return {
    total: state.todos.length,
    completed: state.todos.filter((todo) => todo.completed).length,
    active: state.todos.filter((todo) => !todo.completed).length,
  };
};

// ✅ 좋은 예시: 파생된 상태를 위한 선택자 Hook
export const useCategories = () => {
  const { state } = useTodoContext();
  
  return Array.from(new Set(state.todos.map((todo) => todo.category)));
};

// ✅ 좋은 예시: 비동기 액션을 위한 Hook
export const useTodoActions = () => {
  const { dispatch } = useTodoContext();
  
  const fetchTodos = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    
    try {
      const response = await fetch('/api/todos');
      if (!response.ok) {
        throw new Error('Failed to fetch todos');
      }
      const data = await response.json();
      dispatch({ type: 'SET_TODOS', payload: data });
    } catch (err) {
      dispatch({ 
        type: 'SET_ERROR', 
        payload: err instanceof Error ? err.message : 'An error occurred' 
      });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };
  
  const addTodo = (todoData: Omit<Todo, 'id' | 'createdAt'>) => {
    dispatch({ type: 'ADD_TODO', payload: todoData });
  };
  
  const toggleTodo = (id: number) => {
    dispatch({ type: 'TOGGLE_TODO', payload: id });
  };
  
  const deleteTodo = (id: number) => {
    dispatch({ type: 'DELETE_TODO', payload: id });
  };
  
  const updateTodoPriority = (id: number, priority: Todo['priority']) => {
    dispatch({ 
      type: 'UPDATE_TODO_PRIORITY', 
      payload: { id, priority } 
    });
  };
  
  const setFilter = (filterUpdate: Partial<TodoFilter>) => {
    dispatch({ type: 'SET_FILTER', payload: filterUpdate });
  };
  
  return {
    fetchTodos,
    addTodo,
    toggleTodo,
    deleteTodo,
    updateTodoPriority,
    setFilter,
  };
};