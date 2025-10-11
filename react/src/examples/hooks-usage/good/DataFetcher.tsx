import { useState } from 'react';
import { useFetch } from './hooks/useFetch';
import { useUserInfo } from './hooks/useUserInfo';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import './DataFetcher.css';

interface User {
  id: number;
  name: string;
  email: string;
}

// ✅ 좋은 예시: Hooks를 올바르게 사용한 데이터 페칭 컴포넌트
const DataFetcher = ({ userId }: { userId: number }) => {
  const [retryCount, setRetryCount] = useState(0);
  
  // 커스텀 Hook을 사용하여 데이터 페칭 로직 분리
  const { data: user, loading, error, refetch } = useFetch<User>(`/api/users/${userId}`);
  
  // 커스텀 Hook을 사용하여 계산 로직 분리
  const userInfo = useUserInfo(user);

  // useCallback으로 함수 참조 안정화
  const handleRetry = () => {
    setRetryCount(prevCount => prevCount + 1);
    refetch();
  };

  // 항상 최상위 레벨에서 Hook 호출
  if (loading && !user) {
    return <LoadingSpinner />;
  }

  if (error && !user) {
    return (
      <ErrorMessage 
        message={error} 
        onRetry={handleRetry}
      />
    );
  }

  if (!user) {
    return (
      <div className="empty-state">
        <p>No user data available</p>
        <button onClick={handleRetry}>
          Retry ({retryCount})
        </button>
      </div>
    );
  }

  return (
    <div className="user-profile">
      {loading && <LoadingSpinner />}
      
      {error && (
        <ErrorMessage 
          message={error} 
          onRetry={handleRetry}
        />
      )}
      
      <div className="user-info">
        <h2>{userInfo.fullName}</h2>
        <p>Initials: {userInfo.initials}</p>
        <p>Domain: {userInfo.domain}</p>
      </div>
    </div>
  );
};

export default DataFetcher;