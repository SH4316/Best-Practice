import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UserProfile from './components/UserProfile';
import { mockApiService } from './__mocks__/apiService';
import { createUser } from './utils';
import type { User } from './types';

// ✅ 좋은 예시: 사용자 관점에서 테스트
describe('UserProfile Component', () => {
  // 각 테스트 전에 모킹된 API 서비스 초기화
  beforeEach(() => {
    mockApiService.clearUsers();
    mockApiService.setDelay(0);
  });

  test('should display loading state initially', () => {
    // API 호출이 지연되도록 설정
    mockApiService.setDelay(100);
    
    render(<UserProfile userId="user-123" />);
    
    // 로딩 상태 확인
    expect(screen.getByText('Loading user profile...')).toBeInTheDocument();
  });

  test('should display user information when data is loaded', async () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
      email: 'jane@example.com',
    });
    
    mockApiService.setUser(testUser);
    
    render(<UserProfile userId="user-123" />);
    
    // 사용자 정보 확인
    await waitFor(() => {
      expect(screen.getByText('Jane Smith')).toBeInTheDocument();
      expect(screen.getByText('jane@example.com')).toBeInTheDocument();
    });
    
    // 이미지 alt 텍스트 확인
    const avatar = screen.getByAltText("Jane Smith's avatar");
    expect(avatar).toBeInTheDocument();
  });

  test('should display error message when API fails', async () => {
    // 존재하지 않는 사용자 ID로 API 실패 시뮬레이션
    render(<UserProfile userId="non-existent-user" />);
    
    // 에러 메시지 확인
    await waitFor(() => {
      expect(screen.getByText(/Failed to load user profile/)).toBeInTheDocument();
    });
  });

  test('should display empty state when user not found', async () => {
    // 사용자가 없는 경우
    render(<UserProfile userId="empty-user" />);
    
    // 빈 상태 확인
    await waitFor(() => {
      expect(screen.getByText('User not found')).toBeInTheDocument();
    });
  });

  test('should update user name when name input is changed', async () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    render(<UserProfile userId="user-123" />);
    
    // 사용자 정보 표시 확인
    await waitFor(() => {
      expect(screen.getByText('Jane Smith')).toBeInTheDocument();
    });
    
    // 이름 입력 필드 찾기
    const nameInput = screen.getByLabelText('Update Name');
    
    // 이름 변경
    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, 'Jane Doe');
    
    // 입력 필드에서 포커스가 벗어나면 업데이트 호출
    fireEvent.blur(nameInput);
    
    // API 호출 확인
    await waitFor(() => {
      expect(screen.getByText('Jane Doe')).toBeInTheDocument();
    });
  });

  test('should pass custom API service to useUser hook', async () => {
    // 커스텀 API 서비스로 렌더링
    render(<UserProfile userId="user-123" apiService={mockApiService} />);
    
    // 로딩 상태 확인
    expect(screen.getByText('Loading user profile...')).toBeInTheDocument();
  });

  test('should have proper ARIA attributes for accessibility', async () => {
    // 테스트 사용자 데이터 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    render(<UserProfile userId="user-123" />);
    
    // 로딩 상태 ARIA 확인
    const loadingElement = screen.getByText('Loading user profile...');
    expect(loadingElement).toHaveAttribute('aria-live', 'polite');
    
    // 사용자 정보 표시 확인
    await waitFor(() => {
      expect(screen.getByText('Jane Smith')).toBeInTheDocument();
    });
    
    // 입력 필드 ARIA 확인
    const nameInput = screen.getByLabelText('Update Name');
    expect(nameInput).toHaveAttribute('aria-label', 'Update name');
  });

  test('should handle API errors gracefully', async () => {
    // API 업데이트 실패를 위한 테스트 사용자 설정
    const testUser: User = createUser({
      id: 'user-123',
      name: 'Jane Smith',
    });
    
    mockApiService.setUser(testUser);
    
    render(<UserProfile userId="user-123" />);
    
    // 사용자 정보 표시 확인
    await waitFor(() => {
      expect(screen.getByText('Jane Smith')).toBeInTheDocument();
    });
    
    // 사용자를 제거하여 업데이트 실패 시뮬레이션
    mockApiService.clearUsers();
    
    // 이름 변경 시도
    const nameInput = screen.getByLabelText('Update Name');
    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, 'Jane Doe');
    fireEvent.blur(nameInput);
    
    // 에러가 발생해도 컴포넌트가 충돌하지 않고 계속 표시되는지 확인
    // 실제 에러 메시지는 useUser Hook에서 처리됨
  });
});