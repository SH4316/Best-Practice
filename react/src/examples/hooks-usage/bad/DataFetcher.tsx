import { useState, useEffect } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
}

// ❌ 나쁜 예시: Hooks를 잘못 사용한 데이터 페칭 컴포넌트
const DataFetcher = ({ userId }: { userId: number }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 나쁜 예시: 의존성 배열을 잘못 사용
  useEffect(() => {
    setLoading(true);
    
    fetch(`/api/users/${userId}`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch user');
        }
        return response.json();
      })
      .then(data => {
        setUser(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []); // userId가 변경되어도 다시 fetch하지 않음

  // 나쁜 예시: 매 렌더링마다 새 함수 생성
  const handleRetry = () => {
    setLoading(true);
    setError(null);
    
    fetch(`/api/users/${userId}`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch user');
        }
        return response.json();
      })
      .then(data => {
        setUser(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  };

  // 나쁜 예시: 불필요한 재계산
  const userInfo = {
    fullName: user ? `${user.name} (${user.email})` : '',
    initials: user ? user.name.split(' ').map(n => n[0]).join('') : '',
    domain: user ? user.email.split('@')[1] : '',
  };

  // 나쁜 예시: 조건부 Hook 호출
  if (!user) {
    const [retryCount, setRetryCount] = useState(0); // 조건부 Hook 호출
    
    return (
      <div>
        <p>No user data available</p>
        <button onClick={() => setRetryCount(retryCount + 1)}>
          Retry ({retryCount})
        </button>
      </div>
    );
  }

  return (
    <div>
      {loading && <p>Loading...</p>}
      {error && (
        <div>
          <p>Error: {error}</p>
          <button onClick={handleRetry}>Retry</button>
        </div>
      )}
      {user && (
        <div>
          <h2>{userInfo.fullName}</h2>
          <p>Initials: {userInfo.initials}</p>
          <p>Domain: {userInfo.domain}</p>
        </div>
      )}
    </div>
  );
};

export default DataFetcher;