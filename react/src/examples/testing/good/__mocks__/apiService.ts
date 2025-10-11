import type { User } from '../types';
import ApiService from '../utils/apiService';

// ✅ 좋은 예시: API 서비스 모킹
class MockApiService extends ApiService {
  private users: Record<string, User> = {};
  private delay = 100;
  
  constructor() {
    super('/mock-api');
    
    // 초기 테스트 데이터 설정
    this.users['user-123'] = {
      id: 'user-123',
      name: 'John Doe',
      email: 'john@example.com',
      avatar: 'https://example.com/avatar.jpg',
    };
  }
  
  setDelay(ms: number) {
    this.delay = ms;
  }
  
  setUser(user: User) {
    this.users[user.id] = user;
  }
  
  clearUsers() {
    this.users = {};
  }
  
  async getUser(id: string): Promise<User> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        const user = this.users[id];
        if (user) {
          resolve(user);
        } else {
          reject(new Error(`User with id ${id} not found`));
        }
      }, this.delay);
    });
  }
  
  async updateUser(id: string, data: Partial<User>): Promise<User> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        const user = this.users[id];
        if (user) {
          this.users[id] = { ...user, ...data };
          resolve(this.users[id]);
        } else {
          reject(new Error(`User with id ${id} not found`));
        }
      }, this.delay);
    });
  }
  
  async createUser(data: Omit<User, 'id'>): Promise<User> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const id = `user-${Date.now()}`;
        const newUser = { ...data, id };
        this.users[id] = newUser;
        resolve(newUser);
      }, this.delay);
    });
  }
  
  async deleteUser(id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (this.users[id]) {
          delete this.users[id];
          resolve();
        } else {
          reject(new Error(`User with id ${id} not found`));
        }
      }, this.delay);
    });
  }
}

// 모킹된 인스턴스 내보내기
export const mockApiService = new MockApiService();
export default MockApiService;