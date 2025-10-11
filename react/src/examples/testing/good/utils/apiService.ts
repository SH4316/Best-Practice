import type { User } from '../types';

// ✅ 좋은 예시: 의존성 주입이 가능한 API 서비스
class ApiService {
  private baseUrl: string;
  
  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }
  
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    
    return response.json() as Promise<T>;
  }
  
  async getUser(id: string): Promise<User> {
    return this.request<User>(`/users/${id}`);
  }
  
  async updateUser(id: string, data: Partial<User>): Promise<User> {
    return this.request<User>(`/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }
  
  async createUser(data: Omit<User, 'id'>): Promise<User> {
    return this.request<User>('/users', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
  
  async deleteUser(id: string): Promise<void> {
    return this.request<void>(`/users/${id}`, {
      method: 'DELETE',
    });
  }
}

// 기본 인스턴스
const defaultApiService = new ApiService();

export default ApiService;
export { defaultApiService };

// 타입 내보내기
export type { ApiService };