import { useState, useEffect } from 'react';

// 타입 정의
interface User {
  id: string;
  name: string;
  email: string;
}

// ❌ 나쁜 예시: 테스트하기 어려운 컴포넌트
const UserProfile = ({ userId }: { userId: string }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 직접 API 호출을 하고 있어 테스트하기 어려움
    setLoading(true);
    
    fetch(`/api/users/${userId}`)
      .then(response => response.json())
      .then(data => {
        setUser(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [userId]);

  // data-testid를 사용하여 테스트에 의존하고 있음
  return (
    <div data-testid="user-profile">
      {loading && <div data-testid="loading">Loading...</div>}
      {error && <div data-testid="error">Error: {error}</div>}
      {user && (
        <div data-testid="user-info">
          <h1 data-testid="user-name">{user.name}</h1>
          <p data-testid="user-email">{user.email}</p>
        </div>
      )}
    </div>
  );
};

export default UserProfile;