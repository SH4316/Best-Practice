// ✅ 좋은 예시: 타입 정의를 별도 파일로 분리
export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}

export interface UserFormData {
  name: string;
  email: string;
}

export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}