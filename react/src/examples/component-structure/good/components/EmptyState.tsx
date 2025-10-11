interface EmptyStateProps {
  message: string;
}

// ✅ 좋은 예시: 단일 책임을 가진 프레젠테이션 컴포넌트
export const EmptyState = ({ message }: EmptyStateProps) => {
  return (
    <div className="empty-state">
      <div className="empty-icon">📭</div>
      <p>{message}</p>
    </div>
  );
};