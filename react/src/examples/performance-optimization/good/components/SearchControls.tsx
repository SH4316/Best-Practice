import React from 'react';

interface SearchControlsProps {
  filter: string;
  sortBy: 'name' | 'price';
  onFilterChange: (filter: string) => void;
  onSortChange: (sortBy: 'name' | 'price') => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}

// ✅ 좋은 예시: React.memo로 리렌더링 최적화
const SearchControls = React.memo(({
  filter,
  sortBy,
  onFilterChange,
  onSortChange,
  theme,
  onToggleTheme,
}: SearchControlsProps) => {
  return (
    <div className="controls">
      <input
        type="text"
        value={filter}
        onChange={(e) => onFilterChange(e.target.value)}
        placeholder="Search products..."
        className="search-input"
      />
      
      <select
        value={sortBy}
        onChange={(e) => onSortChange(e.target.value as 'name' | 'price')}
        className="sort-select"
      >
        <option value="name">Sort by Name</option>
        <option value="price">Sort by Price</option>
      </select>
      
      <button onClick={onToggleTheme} className="theme-toggle">
        Toggle Theme ({theme})
      </button>
    </div>
  );
});

SearchControls.displayName = 'SearchControls';

export default SearchControls;