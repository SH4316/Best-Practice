import { useEffect, useState } from 'react';
import { TodoProvider, useTodoContext, useFilteredTodos, useTodoStats, useCategories, useTodoActions } from './store/todoStore';
import { TodoInput } from './components/TodoInput';
import { TodoList } from './components/TodoList';
import { TodoFilters } from './components/TodoFilters';
import { TodoStats } from './components/TodoStats';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import './TodoApp.css';

// ✅ 좋은 예시: 상태 관리가 분리되고 효율적인 TodoApp
const TodoAppContent = () => {
  const { state } = useTodoContext();
  const filteredTodos = useFilteredTodos();
  const stats = useTodoStats();
  const categories = useCategories();
  const { fetchTodos } = useTodoActions();
  const [newTodoText, setNewTodoText] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  // 데이터 페칭은 컴포넌트 마운트 시 한 번만 실행
  useEffect(() => {
    fetchTodos();
  }, [fetchTodos]);

  return (
    <div className="todo-app">
      <h1>Todo App</h1>
      
      {/* 통계 표시 */}
      <TodoStats stats={stats} />

      {/* 입력 폼 */}
      <TodoInput
        newTodoText={newTodoText}
        setNewTodoText={setNewTodoText}
        selectedCategory={selectedCategory}
        setSelectedCategory={setSelectedCategory}
        categories={categories}
      />

      {/* 필터 */}
      <TodoFilters
        filter={state.filter}
        categories={categories}
      />

      {/* 로딩 및 에러 상태 */}
      {state.isLoading && <LoadingSpinner />}
      {state.error && <ErrorMessage message={state.error} />}

      {/* Todo 리스트 */}
      {!state.isLoading && !state.error && (
        <TodoList todos={filteredTodos} />
      )}
    </div>
  );
};

// Provider로 감싸서 내보내기
export const TodoApp = () => {
  return (
    <TodoProvider>
      <TodoAppContent />
    </TodoProvider>
  );
};

export default TodoApp;