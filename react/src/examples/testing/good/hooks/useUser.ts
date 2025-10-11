import { useState, useEffect, useCallback } from 'react';
import type { User } from '../types';
import { type ApiService } from '../utils/apiService';

// ✅ 좋은 예시: 의존성 주입이 가능한 커스텀 Hook
interface UseUserOptions {
  apiService: ApiService;
  autoFetch?: boolean;
}

interface UseUserResult {
  user: User | null;
  loading: boolean;
  error: string | null;
  fetchUser: () => Promise<void>;
  updateUser: (data: Partial<User>) => Promise<void>;
  clearUser: () => void;
}

export const useUser = (userId: string, options: UseUserOptions): UseUserResult => {
  const { apiService, autoFetch = true } = options;
  
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUser = useCallback(async () => {
    if (!userId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const userData = await apiService.getUser(userId);
      setUser(userData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch user');
    } finally {
      setLoading(false);
    }
  }, [userId, apiService]);

  const updateUser = useCallback(async (data: Partial<User>) => {
    if (!userId || !user) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const updatedUser = await apiService.updateUser(userId, data);
      setUser(updatedUser);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update user');
    } finally {
      setLoading(false);
    }
  }, [userId, user, apiService]);

  const clearUser = useCallback(() => {
    setUser(null);
    setError(null);
  }, []);

  useEffect(() => {
    if (autoFetch) {
      fetchUser();
    }
  }, [autoFetch, fetchUser]);

  return {
    user,
    loading,
    error,
    fetchUser,
    updateUser,
    clearUser,
  };
};