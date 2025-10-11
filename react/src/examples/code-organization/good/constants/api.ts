// ✅ 좋은 예시: API 관련 상수를 별도 파일로 분리
export const API_BASE_URL = 'https://api.example.com';

export const API_ENDPOINTS = {
  FILES: '/files',
  UPLOAD: '/upload',
} as const;

export const FILE_SORT_OPTIONS = {
  NAME: { value: 'name', label: 'Sort by Name' },
  SIZE: { value: 'size', label: 'Sort by Size' },
  DATE: { value: 'date', label: 'Sort by Date' },
} as const;