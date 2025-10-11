import { useState, useEffect } from 'react';

export interface User {
  id: string;
  name: string;
  email: string;
  bio: string;
  avatar?: string;
  postsCount?: number;
  followersCount?: number;
  followingCount?: number;
}

export interface UseUserReturn {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  updateUser: (data: { name: string; email: string; bio: string }) => Promise<void>;
}

// ✅ 좋은 예시: 데이터 페칭 로직을 커스텀 Hook으로 분리
export const useUser = (userId: string): UseUserReturn => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 사용자 데이터 페칭
  useEffect(() => {
    const fetchUser = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`/api/user/${userId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch user');
        }
        
        const userData = await response.json();
        setUser(userData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    fetchUser();
  }, [userId]);

  // 사용자 데이터 업데이트
  const updateUser = async (data: { name: string; email: string; bio: string }) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/user/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        throw new Error('Failed to update user');
      }
      
      const updatedUser = await response.json();
      setUser(updatedUser);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return { user, isLoading, error, updateUser };
};