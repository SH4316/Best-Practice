import { useFileManager } from './hooks';
import { FileList, FileControls, FileDropArea, UploadProgress } from './components';
import './FileManager.css';

// ✅ 좋은 예시: 코드가 잘 조직된 파일 관리자 컴포넌트
const FileManager = () => {
  const {
    loading,
    error,
    selectedFiles,
    viewMode,
    sortBy,
    searchTerm,
    uploadProgress,
    isDragging,
    filteredFiles,
    fileInputRef,
    deleteFile,
    toggleFileSelection,
    toggleSelectAll,
    deleteSelectedFiles,
    setViewMode,
    setSortBy,
    setSearchTerm,
    handleFileSelect,
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    triggerFileSelect,
  } = useFileManager();

  return (
    <div className="file-manager">
      <div className="file-manager-header">
        <h1>File Manager</h1>
        
        <FileControls
          searchTerm={searchTerm}
          sortBy={sortBy}
          viewMode={viewMode}
          selectedCount={selectedFiles.length}
          totalCount={filteredFiles.length}
          onSearchChange={setSearchTerm}
          onSortChange={setSortBy}
          onViewModeChange={setViewMode}
          onToggleSelectAll={toggleSelectAll}
          onDeleteSelected={deleteSelectedFiles}
          onUploadClick={triggerFileSelect}
        />
      </div>
      
      <div className="file-manager-content">
        {uploadProgress > 0 && uploadProgress < 100 && (
          <UploadProgress progress={uploadProgress} />
        )}
        
        {loading && <div className="loading">Loading...</div>}
        
        {error && <div className="error">{error}</div>}
        
        <FileDropArea
          isDragging={isDragging}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        />
        
        <FileList
          files={filteredFiles}
          selectedFiles={selectedFiles}
          viewMode={viewMode}
          onToggleSelection={toggleFileSelection}
          onDeleteFile={deleteFile}
        />
      </div>
      
      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />
    </div>
  );
};

export default FileManager;