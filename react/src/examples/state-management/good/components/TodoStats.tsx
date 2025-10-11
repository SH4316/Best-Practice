interface TodoStatsProps {
  total: number;
  completed: number;
  active: number;
}

// ✅ 좋은 예시: 통계 표시 로직을 분리한 컴포넌트
export const TodoStats = ({ total, completed, active }: TodoStatsProps) => {
  return (
    <div className="todo-stats">
      <div className="stat-item">
        <span className="stat-value">{total}</span>
        <span className="stat-label">Total</span>
      </div>
      <div className="stat-item">
        <span className="stat-value">{active}</span>
        <span className="stat-label">Active</span>
      </div>
      <div className="stat-item">
        <span className="stat-value">{completed}</span>
        <span className="stat-label">Completed</span>
      </div>
    </div>
  );
};