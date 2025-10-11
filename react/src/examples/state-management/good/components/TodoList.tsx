import type { Todo } from '../types';
import { useTodoActions } from '../store/todoStore';
import { TodoItem } from './TodoItem';

interface TodoListProps {
  todos: Todo[];
}

// ✅ 좋은 예시: 리스트 렌더링 로직을 분리한 컴포넌트
export const TodoList = ({ todos }: TodoListProps) => {
  const { toggleTodo, deleteTodo, updateTodoPriority } = useTodoActions();

  if (todos.length === 0) {
    return (
      <div className="empty-state">
        <p>No todos found. Add one above!</p>
      </div>
    );
  }

  return (
    <div className="todo-list">
      {todos.map((todo) => (
        <TodoItem
          key={todo.id}
          todo={todo}
          onToggle={() => toggleTodo(todo.id)}
          onDelete={() => deleteTodo(todo.id)}
          onUpdatePriority={(priority: Todo['priority']) => updateTodoPriority(todo.id, priority)}
        />
      ))}
    </div>
  );
};