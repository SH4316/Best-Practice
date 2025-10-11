import { useUser } from '../hooks';
import { defaultApiService } from '../utils';
import type { ApiService } from '../utils/apiService';

// ✅ 좋은 예시: 테스트하기 쉬운 컴포넌트
interface UserProfileProps {
  userId: string;
  apiService?: ApiService;
}

const UserProfile = ({ userId, apiService = defaultApiService }: UserProfileProps) => {
  const { user, loading, error, updateUser } = useUser(userId, { apiService });
  
  // 이름 업데이트 핸들러
  const handleNameChange = async (newName: string) => {
    try {
      await updateUser({ name: newName });
    } catch (err) {
      // 에러는 useUser Hook에서 처리됨
      console.error('Failed to update name:', err);
    }
  };
  
  // 로딩 상태 표시
  if (loading) {
    return <div className="loading" aria-live="polite">Loading user profile...</div>;
  }
  
  // 에러 상태 표시
  if (error) {
    return (
      <div className="error" role="alert" aria-live="assertive">
        Failed to load user profile: {error}
      </div>
    );
  }
  
  // 사용자 데이터가 없는 경우
  if (!user) {
    return <div className="empty">User not found</div>;
  }
  
  // 사용자 정보 표시
  return (
    <div className="user-profile">
      <div className="user-avatar">
        <img src={user.avatar || 'https://via.placeholder.com/150'} alt={`${user.name}'s avatar`} />
      </div>
      
      <div className="user-info">
        <h1 className="user-name">{user.name}</h1>
        <p className="user-email">{user.email}</p>
      </div>
      
      <div className="user-actions">
        <label htmlFor="name-input">Update Name:</label>
        <input
          id="name-input"
          type="text"
          defaultValue={user.name}
          onBlur={(e) => handleNameChange(e.target.value)}
          aria-label="Update name"
        />
      </div>
    </div>
  );
};

export default UserProfile;