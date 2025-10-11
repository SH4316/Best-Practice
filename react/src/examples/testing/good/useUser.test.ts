import { renderHook, act } from '@testing-library/react';
import { useUser } from './hooks';
import { mockApiService } from './__mocks__/apiService';
import { createUser } from './utils';
import type { User } from './types';

// ✅ 좋은 예시: 커스텀 Hook 테스트
describe('useUser Hook', () => {
  // 각 테스트 전에 모킹된 API 서비스 초기화
  beforeEach(() => {
    mockApiService.clearUsers();
    mockApiService.setDelay(0);
  });

  test('should initialize with loading state', () => {
    // API 호출이 지연되도록 설정
    mockApiService.setDelay(100);
    
    const { result } = renderHook(() => 
      useUser('user-123', { apiService: mockApiService })
    );
    
    // 초기 상태 확인
    expect(result.current.user).toBeNull();
    expect(result.current.loading).toBe(true);
    expect(result.current.error).toBeNull();
  });

  test('should fetch user data successfully', async () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    const { result } = renderHook(() => 
      useUser('user-123', { apiService: mockApiService })
    );
    
    // 데이터 로딩 확인
    await waitFor(() => {
      expect(result.current.user).toEqual(testUser);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  test('should handle API error', async () => {
    // 존재하지 않는 사용자 ID로 API 실패 시뮬레이션
    const { result } = renderHook(() => 
      useUser('non-existent-user', { apiService: mockApiService })
    );
    
    // 에러 상태 확인
    await waitFor(() => {
      expect(result.current.user).toBeNull();
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBe('User with id non-existent-user not found');
    });
  });

  test('should update user data successfully', async () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    const { result } = renderHook(() => 
      useUser('user-123', { apiService: mockApiService })
    );
    
    // 데이터 로딩 확인
    await waitFor(() => {
      expect(result.current.user).toEqual(testUser);
    });
    
    // 사용자 업데이트
    await act(async () => {
      await result.current.updateUser({ name: 'Jane Doe' });
    });
    
    // 업데이트된 데이터 확인
    expect(result.current.user?.name).toBe('Jane Doe');
  });

  test('should handle update error', async () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    const { result } = renderHook(() => 
      useUser('user-123', { apiService: mockApiService })
    );
    
    // 데이터 로딩 확인
    await waitFor(() => {
      expect(result.current.user).toEqual(testUser);
    });
    
    // 사용자 제거하여 업데이트 실패 시뮬레이션
    mockApiService.clearUsers();
    
    // 사용자 업데이트 시도
    await act(async () => {
      await result.current.updateUser({ name: 'Jane Doe' });
    });
    
    // 에러 상태 확인
    expect(result.current.error).toBe('User with id user-123 not found');
  });

  test('should clear user data', () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    const { result } = renderHook(() => 
      useUser('user-123', { apiService: mockApiService, autoFetch: false })
    );
    
    // 사용자 데이터 설정
    act(() => {
      result.current.user = testUser;
    });
    
    // 사용자 데이터 확인
    expect(result.current.user).toEqual(testUser);
    
    // 사용자 데이터 초기화
    act(() => {
      result.current.clearUser();
    });
    
    // 초기화된 데이터 확인
    expect(result.current.user).toBeNull();
    expect(result.current.error).toBeNull();
  });

  test('should not fetch data when autoFetch is false', () => {
    const { result } = renderHook(() => 
      useUser('user-123', { apiService: mockApiService, autoFetch: false })
    );
    
    // 초기 상태 확인 (로딩 상태가 아님)
    expect(result.current.user).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  test('should fetch data when fetchUser is called', async () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    const { result } = renderHook(() => 
      useUser('user-123', { apiService: mockApiService, autoFetch: false })
    );
    
    // 초기 상태 확인
    expect(result.current.user).toBeNull();
    expect(result.current.loading).toBe(false);
    
    // 수동 데이터 페칭
    await act(async () => {
      await result.current.fetchUser();
    });
    
    // 데이터 로딩 확인
    expect(result.current.user).toEqual(testUser);
    expect(result.current.loading).toBe(false);
  });
});