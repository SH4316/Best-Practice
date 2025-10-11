import { useState, useEffect, useRef } from 'react';
// axios를 직접 사용하지 않고 fetch API로 대체

// ❌ 나쁜 예시: 모든 것이 하나의 파일에 혼재
const FileManager = () => {
  // 타입 정의를 컴포넌트 밖으로 이동
  interface FileItem {
    id: string;
    name: string;
    size: number;
    type: string;
    url: string;
    createdAt: string;
  }

  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState('grid');
  const [sortBy, setSortBy] = useState('name');
  const [searchTerm, setSearchTerm] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);

  // API 엔드포인트와 상수가 컴포넌트 내에 정의됨
  const API_BASE_URL = 'https://api.example.com';
  const FILES_ENDPOINT = '/files';
  const UPLOAD_ENDPOINT = '/upload';

  // 타입 정의를 위로 이동

  // 복잡한 유틸리티 함수가 컴포넌트 내에 있음
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    const options: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  const getFileIcon = (fileType: string): string => {
    if (fileType.startsWith('image/')) return '🖼️';
    if (fileType.startsWith('video/')) return '🎥';
    if (fileType.startsWith('audio/')) return '🎵';
    if (fileType.includes('pdf')) return '📄';
    if (fileType.includes('word') || fileType.includes('document')) return '📝';
    if (fileType.includes('excel') || fileType.includes('spreadsheet')) return '📊';
    if (fileType.includes('powerpoint') || fileType.includes('presentation')) return '📋';
    if (fileType.includes('zip') || fileType.includes('rar') || fileType.includes('tar')) return '🗜️';
    return '📄';
  };

  // API 호출 로직이 컴포넌트 내에 있음
  const fetchFiles = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE_URL}${FILES_ENDPOINT}`);
      if (!response.ok) {
        throw new Error('Failed to fetch files');
      }
      const data = await response.json();
      setFiles(data);
    } catch (err) {
      setError('Failed to fetch files');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // XMLHttpRequest를 사용하여 업로드 진행률 추적
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(progress);
        }
      });
      
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const newFile = JSON.parse(xhr.responseText);
          setFiles(prevFiles => [...prevFiles, newFile]);
        } else {
          setError('Failed to upload file');
        }
        setUploadProgress(0);
      });
      
      xhr.addEventListener('error', () => {
        setError('Failed to upload file');
        setUploadProgress(0);
      });
      
      xhr.open('POST', `${API_BASE_URL}${UPLOAD_ENDPOINT}`);
      xhr.send(formData);
    } catch (err) {
      setError('Failed to upload file');
      console.error(err);
    }
  };

  const deleteFile = async (fileId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}${FILES_ENDPOINT}/${fileId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete file');
      }
      
      setFiles(prevFiles => prevFiles.filter(file => file.id !== fileId));
    } catch (err) {
      setError('Failed to delete file');
      console.error(err);
    }
  };

  // 복잡한 이벤트 핸들러가 컴포넌트 내에 있음
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      Array.from(files).forEach(file => uploadFile(file));
    }
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      Array.from(e.dataTransfer.files).forEach(file => uploadFile(file));
    }
  };

  const handleFileClick = (file: FileItem) => {
    if (selectedFiles.includes(file.id)) {
      setSelectedFiles(selectedFiles.filter(id => id !== file.id));
    } else {
      setSelectedFiles([...selectedFiles, file.id]);
    }
  };

  const handleSelectAll = () => {
    if (selectedFiles.length === filteredFiles.length) {
      setSelectedFiles([]);
    } else {
      setSelectedFiles(filteredFiles.map(file => file.id));
    }
  };

  const handleDeleteSelected = () => {
    selectedFiles.forEach(fileId => deleteFile(fileId));
    setSelectedFiles([]);
  };

  // 복잡한 필터링 및 정렬 로직이 컴포넌트 내에 있음
  const filteredFiles = files
    .filter(file => 
      file.name.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      } else if (sortBy === 'size') {
        return b.size - a.size;
      } else if (sortBy === 'date') {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      }
      return 0;
    });

  // 사이드 이펙트가 컴포넌트 내에 있음
  useEffect(() => {
    fetchFiles();
  }, []);

  return (
    <div className="file-manager">
      <div className="file-manager-header">
        <h1>File Manager</h1>
        
        <div className="file-manager-controls">
          <div className="search-box">
            <input
              type="text"
              placeholder="Search files..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          
          <div className="sort-controls">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="name">Sort by Name</option>
              <option value="size">Sort by Size</option>
              <option value="date">Sort by Date</option>
            </select>
          </div>
          
          <div className="view-controls">
            <button
              className={viewMode === 'grid' ? 'active' : ''}
              onClick={() => setViewMode('grid')}
            >
              Grid View
            </button>
            <button
              className={viewMode === 'list' ? 'active' : ''}
              onClick={() => setViewMode('list')}
            >
              List View
            </button>
          </div>
        </div>
        
        <div className="file-actions">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="upload-button"
          >
            Upload Files
          </button>
          
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          
          {selectedFiles.length > 0 && (
            <button
              onClick={handleDeleteSelected}
              className="delete-button"
            >
              Delete Selected ({selectedFiles.length})
            </button>
          )}
        </div>
      </div>
      
      <div className="file-manager-content">
        {uploadProgress > 0 && uploadProgress < 100 && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <span>{uploadProgress}%</span>
          </div>
        )}
        
        {loading && <div className="loading">Loading...</div>}
        
        {error && <div className="error">{error}</div>}
        
        <div
          className={`file-drop-area ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <div className="drop-area-content">
            <p>Drag and drop files here or click to upload</p>
          </div>
        </div>
        
        {filteredFiles.length > 0 && (
          <div className="file-selection">
            <label>
              <input
                type="checkbox"
                checked={selectedFiles.length === filteredFiles.length && filteredFiles.length > 0}
                onChange={handleSelectAll}
              />
              Select All
            </label>
          </div>
        )}
        
        <div className={`file-list ${viewMode}`}>
          {filteredFiles.map(file => (
            <div
              key={file.id}
              className={`file-item ${selectedFiles.includes(file.id) ? 'selected' : ''}`}
              onClick={() => handleFileClick(file)}
            >
              <div className="file-icon">
                {getFileIcon(file.type)}
              </div>
              
              <div className="file-info">
                <div className="file-name">{file.name}</div>
                <div className="file-meta">
                  <span className="file-size">{formatFileSize(file.size)}</span>
                  <span className="file-date">{formatDate(file.createdAt)}</span>
                </div>
              </div>
              
              <div className="file-actions">
                <a
                  href={file.url}
                  download={file.name}
                  onClick={(e) => e.stopPropagation()}
                  className="download-button"
                >
                  Download
                </a>
                
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteFile(file.id);
                  }}
                  className="delete-button"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FileManager;