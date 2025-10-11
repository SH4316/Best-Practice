import { useState, useEffect, useCallback, useRef } from 'react';
import { fileService } from '../services';
import { sortFiles, filterFiles } from '../utils';
import type { FileItem, FileViewMode } from '../types';

// ✅ 좋은 예시: 파일 관리 로직을 커스텀 Hook으로 분리
export const useFileManager = () => {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<FileViewMode>('grid');
  const [sortBy, setSortBy] = useState<'name' | 'size' | 'date'>('name');
  const [searchTerm, setSearchTerm] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);

  // 파일 목록 가져오기
  const fetchFiles = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await fileService.getFiles();
      setFiles(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch files');
    } finally {
      setLoading(false);
    }
  }, []);

  // 파일 업로드
  const uploadFile = useCallback(async (file: File) => {
    setError(null);
    
    try {
      const newFile = await fileService.uploadFile(file, (progress) => {
        setUploadProgress(progress);
      });
      
      setFiles(prevFiles => [...prevFiles, newFile]);
      setUploadProgress(0);
      
      return newFile;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file');
      setUploadProgress(0);
      throw err;
    }
  }, []);

  // 파일 삭제
  const deleteFile = useCallback(async (fileId: string) => {
    setError(null);
    
    try {
      await fileService.deleteFile(fileId);
      setFiles(prevFiles => prevFiles.filter(file => file.id !== fileId));
      
      // 선택된 파일 목록에서도 제거
      setSelectedFiles(prevSelected => 
        prevSelected.filter(id => id !== fileId)
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete file');
      throw err;
    }
  }, []);

  // 파일 선택 토글
  const toggleFileSelection = useCallback((fileId: string) => {
    setSelectedFiles(prevSelected => {
      if (prevSelected.includes(fileId)) {
        return prevSelected.filter(id => id !== fileId);
      } else {
        return [...prevSelected, fileId];
      }
    });
  }, []);

  // 전체 선택 토글
  const toggleSelectAll = useCallback(() => {
    const currentFilteredFiles = sortFiles(filterFiles(files, searchTerm), sortBy);
    
    setSelectedFiles(prevSelected => {
      if (prevSelected.length === currentFilteredFiles.length) {
        return [];
      } else {
        return currentFilteredFiles.map(file => file.id);
      }
    });
  }, [files, searchTerm, sortBy]);

  // 선택된 파일 삭제
  const deleteSelectedFiles = useCallback(async () => {
    if (selectedFiles.length === 0) return;
    
    setError(null);
    
    try {
      // 병렬로 삭제 시도
      await Promise.all(
        selectedFiles.map(fileId => fileService.deleteFile(fileId))
      );
      
      setFiles(prevFiles => 
        prevFiles.filter(file => !selectedFiles.includes(file.id))
      );
      setSelectedFiles([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete files');
    }
  }, [selectedFiles]);

  // 파일 선택 처리
  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files;
    if (selectedFiles) {
      // 순차적으로 업로드
      const uploadPromises = Array.from(selectedFiles).map(file => 
        uploadFile(file)
      );
      
      // 모든 업로드가 완료될 때까지 기다리지 않음 (비동기 처리)
      uploadPromises.forEach(promise => {
        promise.catch(err => console.error('Upload error:', err));
      });
    }
  }, [uploadFile]);

  // 드래그 앤 드롭 이벤트 핸들러
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      
      // 순차적으로 업로드
      droppedFiles.forEach(file => {
        uploadFile(file).catch(err => console.error('Upload error:', err));
      });
    }
  }, [uploadFile]);

  // 파일 입력창 클릭
  const triggerFileSelect = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // 필터링 및 정렬된 파일 목록
  const filteredFiles = sortFiles(filterFiles(files, searchTerm), sortBy);

  // 초기 데이터 로드
  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  return {
    // 상태
    files,
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
    
    // 액션
    fetchFiles,
    uploadFile,
    deleteFile,
    toggleFileSelection,
    toggleSelectAll,
    deleteSelectedFiles,
    setViewMode,
    setSortBy,
    setSearchTerm,
    
    // 이벤트 핸들러
    handleFileSelect,
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    triggerFileSelect,
  };
};