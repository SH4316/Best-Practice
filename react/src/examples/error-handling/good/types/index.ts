// ✅ 좋은 예시: 타입 정의를 별도 파일로 분리
export interface User {
  id: string;
  name: string;
  email: string;
  avatar: string;
}

export interface Post {
  id: string;
  title: string;
  content: string;
  createdAt: string;
}

export interface ApiError {
  message: string;
  status?: number;
  code?: string;
}

export interface ErrorState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}