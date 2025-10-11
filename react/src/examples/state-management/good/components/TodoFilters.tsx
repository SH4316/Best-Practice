import type { TodoFilter } from '../types';
import { useTodoActions } from '../store/todoStore';

interface TodoFiltersProps {
  filter: TodoFilter;
  categories: string[];
}

// ✅ 좋은 예시: 필터 로직을 분리한 컴포넌트
export const TodoFilters = ({ filter, categories }: TodoFiltersProps) => {
  const { setFilter } = useTodoActions();

  const handleStatusChange = (status: TodoFilter['status']) => {
    setFilter({ status });
  };

  const handleSearchChange = (searchTerm: string) => {
    setFilter({ searchTerm });
  };

  const handleCategoryChange = (category: string) => {
    setFilter({ category });
  };

  return (
    <div className="todo-filters">
      <div className="status-filters">
        <button
          className={filter.status === 'all' ? 'active' : ''}
          onClick={() => handleStatusChange('all')}
        >
          All
        </button>
        <button
          className={filter.status === 'active' ? 'active' : ''}
          onClick={() => handleStatusChange('active')}
        >
          Active
        </button>
        <button
          className={filter.status === 'completed' ? 'active' : ''}
          onClick={() => handleStatusChange('completed')}
        >
          Completed
        </button>
      </div>
      
      <div className="search-filter">
        <input
          type="text"
          value={filter.searchTerm}
          onChange={(e) => handleSearchChange(e.target.value)}
          placeholder="Search todos..."
        />
      </div>
      
      <div className="category-filter">
        <select
          value={filter.category}
          onChange={(e) => handleCategoryChange(e.target.value)}
        >
          <option value="all">All Categories</option>
          {categories.map((category) => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};