import { useState, useEffect, useCallback } from 'react';
import { ApiService, ErrorLogger } from '../utils';
import type { User, Post, ApiError } from '../types';

// ✅ 좋은 예시: API 호출을 위한 커스텀 Hook
interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export const useApi = <T,>(url: string): UseApiResult<T> => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.get<T>(url);
      setData(result);
    } catch (err) {
      const errorMessage = ErrorLogger.getErrorMessage(err as Error | ApiError);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [url]);
  
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  
  return { data, loading, error, refetch: fetchData };
};

// 사용자 데이터를 위한 Hook
export const useUser = (userId: string): UseApiResult<User> => {
  return useApi<User>(`/api/users/${userId}`);
};

// 포스트 데이터를 위한 Hook
export const useUserPosts = (userId: string): UseApiResult<Post[]> => {
  return useApi<Post[]>(`/api/users/${userId}/posts`);
};

// 포스트 생성을 위한 Hook
export const useCreatePost = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const createPost = useCallback(async (userId: string, title: string, content: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const newPost = await ApiService.post<Post>(`/api/users/${userId}/posts`, {
        title,
        content,
      });
      
      return newPost;
    } catch (err) {
      const errorMessage = ErrorLogger.getErrorMessage(err as Error | ApiError);
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  return { createPost, loading, error };
};

// 포스트 삭제를 위한 Hook
export const useDeletePost = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const deletePost = useCallback(async (postId: string) => {
    setLoading(true);
    setError(null);
    
    try {
      await ApiService.delete(`/api/posts/${postId}`);
    } catch (err) {
      const errorMessage = ErrorLogger.getErrorMessage(err as Error | ApiError);
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  return { deletePost, loading, error };
};