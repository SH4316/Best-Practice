import React from 'react';

interface FileDropAreaProps {
  isDragging: boolean;
  onDragEnter: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
}

// ✅ 좋은 예시: 파일 드롭 영역 컴포넌트를 별도 파일로 분리
const FileDropArea = React.memo(({
  isDragging,
  onDragEnter,
  onDragLeave,
  onDragOver,
  onDrop,
}: FileDropAreaProps) => {
  return (
    <div
      className={`file-drop-area ${isDragging ? 'dragging' : ''}`}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <div className="drop-area-content">
        <div className="drop-icon">📁</div>
        <p>Drag and drop files here</p>
        <p className="drop-hint">or click to browse</p>
      </div>
    </div>
  );
});

FileDropArea.displayName = 'FileDropArea';

export default FileDropArea;