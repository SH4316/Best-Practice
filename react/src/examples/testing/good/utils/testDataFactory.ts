import type { User } from '../types';

// ✅ 좋은 예시: 테스트 데이터 팩토리
export const createUser = (overrides: Partial<User> = {}): User => {
  return {
    id: 'user-123',
    name: 'John Doe',
    email: 'john@example.com',
    avatar: 'https://example.com/avatar.jpg',
    ...overrides,
  };
};

export const createUsers = (count: number, overrides: Partial<User> = {}): User[] => {
  return Array.from({ length: count }, (_, index) => 
    createUser({
      id: `user-${index + 1}`,
      name: `User ${index + 1}`,
      email: `user${index + 1}@example.com`,
      ...overrides,
    })
  );
};