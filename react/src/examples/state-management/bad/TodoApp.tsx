import { useState, useEffect } from 'react';

interface Todo {
  id: number;
  text: string;
  completed: boolean;
  category: string;
  priority: 'low' | 'medium' | 'high';
  createdAt: Date;
}

// ❌ 나쁜 예시: 상태 관리가 복잡하고 비효율적
const TodoApp = () => {
  // 불필요하게 많은 상태 변수
  const [todos, setTodos] = useState<Todo[]>([]);
  const [filteredTodos, setFilteredTodos] = useState<Todo[]>([]);
  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [newTodoText, setNewTodoText] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [categories, setCategories] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({
    total: 0,
    completed: 0,
    active: 0,
  });

  // 불필요한 useEffect로 파생된 상태 계산
  useEffect(() => {
    let filtered = todos;
    
    // 필터 적용
    if (filter === 'active') {
      filtered = filtered.filter(todo => !todo.completed);
    } else if (filter === 'completed') {
      filtered = filtered.filter(todo => todo.completed);
    }
    
    // 검색어 적용
    if (searchTerm) {
      filtered = filtered.filter(todo =>
        todo.text.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // 카테고리 필터 적용
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(todo => todo.category === selectedCategory);
    }
    
    setFilteredTodos(filtered);
  }, [todos, filter, searchTerm, selectedCategory]);

  // 불필요한 useEffect로 통계 계산
  useEffect(() => {
    const total = todos.length;
    const completed = todos.filter(todo => todo.completed).length;
    const active = total - completed;
    
    setStats({ total, completed, active });
  }, [todos]);

  // 불필요한 useEffect로 카테고리 추출
  useEffect(() => {
    const uniqueCategories = Array.from(
      new Set(todos.map(todo => todo.category))
    );
    setCategories(uniqueCategories);
  }, [todos]);

  // 데이터 페칭 로직이 컴포넌트에 혼재
  useEffect(() => {
    const fetchTodos = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch('/api/todos');
        if (!response.ok) {
          throw new Error('Failed to fetch todos');
        }
        const data = await response.json();
        setTodos(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    fetchTodos();
  }, []);

  // 가변 상태 업데이트
  const addTodo = () => {
    if (!newTodoText.trim()) return;
    
    const newTodo: Todo = {
      id: Date.now(),
      text: newTodoText,
      completed: false,
      category: selectedCategory === 'all' ? 'general' : selectedCategory,
      priority: 'medium',
      createdAt: new Date(),
    };
    
    // 잘못된 상태 업데이트 방식
    todos.push(newTodo);
    setTodos(todos);
    setNewTodoText('');
  };

  // 가변 상태 업데이트
  const toggleTodo = (id: number) => {
    const todo = todos.find(t => t.id === id);
    if (todo) {
      todo.completed = !todo.completed;
      setTodos([...todos]);
    }
  };

  // 비효율적인 상태 업데이트
  const deleteTodo = (id: number) => {
    const updatedTodos = todos.filter(todo => todo.id !== id);
    setTodos(updatedTodos);
    
    // 불필요한 재계산
    const uniqueCategories = Array.from(
      new Set(updatedTodos.map(todo => todo.category))
    );
    setCategories(uniqueCategories);
  };

  // 복잡한 상태 업데이트 로직
  const updateTodoPriority = (id: number, priority: 'low' | 'medium' | 'high') => {
    const updatedTodos = todos.map(todo => {
      if (todo.id === id) {
        return { ...todo, priority };
      }
      return todo;
    });
    setTodos(updatedTodos);
  };

  return (
    <div className="todo-app">
      <h1>Todo App</h1>
      
      {/* 통계 표시 */}
      <div className="stats">
        <div>Total: {stats.total}</div>
        <div>Active: {stats.active}</div>
        <div>Completed: {stats.completed}</div>
      </div>

      {/* 입력 폼 */}
      <div className="todo-input">
        <input
          type="text"
          value={newTodoText}
          onChange={(e) => setNewTodoText(e.target.value)}
          placeholder="Add a new todo..."
        />
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
        >
          <option value="all">All Categories</option>
          {categories.map(category => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>
        <button onClick={addTodo}>Add</button>
      </div>

      {/* 필터 */}
      <div className="filters">
        <button
          className={filter === 'all' ? 'active' : ''}
          onClick={() => setFilter('all')}
        >
          All
        </button>
        <button
          className={filter === 'active' ? 'active' : ''}
          onClick={() => setFilter('active')}
        >
          Active
        </button>
        <button
          className={filter === 'completed' ? 'active' : ''}
          onClick={() => setFilter('completed')}
        >
          Completed
        </button>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search todos..."
        />
      </div>

      {/* 로딩 및 에러 상태 */}
      {isLoading && <div>Loading...</div>}
      {error && <div className="error">Error: {error}</div>}

      {/* Todo 리스트 */}
      <div className="todo-list">
        {filteredTodos.map(todo => (
          <div key={todo.id} className={`todo ${todo.completed ? 'completed' : ''}`}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo(todo.id)}
            />
            <span>{todo.text}</span>
            <span className="category">{todo.category}</span>
            <span className={`priority ${todo.priority}`}>{todo.priority}</span>
            <select
              value={todo.priority}
              onChange={(e) => updateTodoPriority(
                todo.id, 
                e.target.value as 'low' | 'medium' | 'high'
              )}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
            <button onClick={() => deleteTodo(todo.id)}>Delete</button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TodoApp;