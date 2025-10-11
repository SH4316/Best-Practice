import React from 'react';
import { FILE_SORT_OPTIONS } from '../constants';
import type { FileViewMode } from '../types';

interface FileControlsProps {
  searchTerm: string;
  sortBy: 'name' | 'size' | 'date';
  viewMode: FileViewMode;
  selectedCount: number;
  totalCount: number;
  onSearchChange: (term: string) => void;
  onSortChange: (sortBy: 'name' | 'size' | 'date') => void;
  onViewModeChange: (viewMode: FileViewMode) => void;
  onToggleSelectAll: () => void;
  onDeleteSelected: () => void;
  onUploadClick: () => void;
}

// ✅ 좋은 예시: 파일 컨트롤 컴포넌트를 별도 파일로 분리
const FileControls = React.memo(({
  searchTerm,
  sortBy,
  viewMode,
  selectedCount,
  totalCount,
  onSearchChange,
  onSortChange,
  onViewModeChange,
  onToggleSelectAll,
  onDeleteSelected,
  onUploadClick,
}: FileControlsProps) => {
  return (
    <div className="file-controls">
      <div className="search-box">
        <input
          type="text"
          placeholder="Search files..."
          value={searchTerm}
          onChange={(e) => onSearchChange(e.target.value)}
          className="search-input"
        />
      </div>
      
      <div className="filter-controls">
        <div className="sort-control">
          <select
            value={sortBy}
            onChange={(e) => onSortChange(e.target.value as 'name' | 'size' | 'date')}
            className="sort-select"
          >
            {Object.values(FILE_SORT_OPTIONS).map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        
        <div className="view-controls">
          <button
            className={viewMode === 'grid' ? 'active' : ''}
            onClick={() => onViewModeChange('grid')}
            title="Grid View"
          >
            Grid
          </button>
          <button
            className={viewMode === 'list' ? 'active' : ''}
            onClick={() => onViewModeChange('list')}
            title="List View"
          >
            List
          </button>
        </div>
      </div>
      
      <div className="action-controls">
        <button
          onClick={onUploadClick}
          className="upload-button"
        >
          Upload Files
        </button>
        
        {totalCount > 0 && (
          <div className="selection-controls">
            <label className="select-all">
              <input
                type="checkbox"
                checked={selectedCount === totalCount && totalCount > 0}
                onChange={onToggleSelectAll}
              />
              Select All
            </label>
            
            {selectedCount > 0 && (
              <button
                onClick={onDeleteSelected}
                className="delete-selected-button"
              >
                Delete Selected ({selectedCount})
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

FileControls.displayName = 'FileControls';

export default FileControls;