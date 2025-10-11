import { useTodoActions } from '../store/todoStore';

interface TodoInputProps {
  newTodoText: string;
  setNewTodoText: (text: string) => void;
  selectedCategory: string;
  setSelectedCategory: (category: string) => void;
  categories: string[];
}

// ✅ 좋은 예시: 입력 로직을 분리한 컴포넌트
export const TodoInput = ({
  newTodoText,
  setNewTodoText,
  selectedCategory,
  setSelectedCategory,
  categories,
}: TodoInputProps) => {
  const { addTodo } = useTodoActions();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTodoText.trim()) return;

    addTodo({
      text: newTodoText,
      completed: false,
      category: selectedCategory === 'all' ? 'general' : selectedCategory,
      priority: 'medium',
    });

    setNewTodoText('');
  };

  return (
    <form className="todo-input" onSubmit={handleSubmit}>
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
        {categories.map((category) => (
          <option key={category} value={category}>
            {category}
          </option>
        ))}
      </select>
      <button type="submit">Add</button>
    </form>
  );
};