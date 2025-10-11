import type { Todo } from '../types';

interface TodoItemProps {
  todo: Todo;
  onToggle: () => void;
  onDelete: () => void;
  onUpdatePriority: (priority: Todo['priority']) => void;
}

// ✅ 좋은 예시: 개별 항목 렌더링을 분리한 컴포넌트
export const TodoItem = ({
  todo,
  onToggle,
  onDelete,
  onUpdatePriority,
}: TodoItemProps) => {
  return (
    <div className={`todo-item ${todo.completed ? 'completed' : ''}`}>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={onToggle}
      />
      <span className="todo-text">{todo.text}</span>
      <span className="todo-category">{todo.category}</span>
      <span className={`todo-priority ${todo.priority}`}>
        {todo.priority}
      </span>
      <select
        value={todo.priority}
        onChange={(e) => onUpdatePriority(e.target.value as Todo['priority'])}
        className="priority-select"
      >
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High</option>
      </select>
      <button onClick={onDelete} className="delete-btn">
        Delete
      </button>
    </div>
  );
};