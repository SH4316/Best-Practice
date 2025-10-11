import React from 'react';
import { formatFileSize, formatDate, getFileIcon } from '../utils';
import { fileService } from '../services';
import type { FileItem } from '../types';

interface FileListProps {
  files: FileItem[];
  selectedFiles: string[];
  viewMode: 'grid' | 'list';
  onToggleSelection: (fileId: string) => void;
  onDeleteFile: (fileId: string) => void;
}

// ✅ 좋은 예시: 파일 리스트 컴포넌트를 별도 파일로 분리
const FileList = React.memo(({
  files,
  selectedFiles,
  viewMode,
  onToggleSelection,
  onDeleteFile,
}: FileListProps) => {
  const handleDelete = (fileId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    onDeleteFile(fileId);
  };

  const handleDownloadClick = (file: FileItem, event: React.MouseEvent) => {
    event.stopPropagation();
    const downloadUrl = fileService.getDownloadUrl(file.id);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (files.length === 0) {
    return (
      <div className="empty-state">
        <p>No files found</p>
      </div>
    );
  }

  return (
    <div className={`file-list ${viewMode}`}>
      {files.map(file => (
        <div
          key={file.id}
          className={`file-item ${selectedFiles.includes(file.id) ? 'selected' : ''}`}
          onClick={() => onToggleSelection(file.id)}
        >
          <div className="file-icon">
            {getFileIcon(file.type)}
          </div>
          
          <div className="file-info">
            <div className="file-name" title={file.name}>
              {file.name}
            </div>
            <div className="file-meta">
              <span className="file-size">{formatFileSize(file.size)}</span>
              <span className="file-date">{formatDate(file.createdAt)}</span>
            </div>
          </div>
          
          <div className="file-actions">
            <button
              onClick={(e) => handleDownloadClick(file, e)}
              className="download-button"
              title="Download"
            >
              Download
            </button>
            
            <button
              onClick={(e) => handleDelete(file.id, e)}
              className="delete-button"
              title="Delete"
            >
              Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  );
});

FileList.displayName = 'FileList';

export default FileList;