// ✅ 좋은 예시: 타입 정의를 별도 파일로 분리
export interface FileItem {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
  createdAt: string;
}

export interface FileUploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
}

export interface FileSortOption {
  value: 'name' | 'size' | 'date';
  label: string;
}

export type FileViewMode = 'grid' | 'list';