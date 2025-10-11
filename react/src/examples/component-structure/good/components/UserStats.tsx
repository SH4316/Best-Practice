interface UserStatsProps {
  postsCount: number;
  followersCount: number;
  followingCount: number;
}

// ✅ 좋은 예시: 단일 책임을 가진 프레젠테이션 컴포넌트
export const UserStats = ({ 
  postsCount, 
  followersCount, 
  followingCount 
}: UserStatsProps) => {
  return (
    <div className="user-stats">
      <div className="stat-item">
        <div className="stat-value">{postsCount}</div>
        <div className="stat-label">Posts</div>
      </div>
      <div className="stat-item">
        <div className="stat-value">{followersCount}</div>
        <div className="stat-label">Followers</div>
      </div>
      <div className="stat-item">
        <div className="stat-value">{followingCount}</div>
        <div className="stat-label">Following</div>
      </div>
    </div>
  );
};