// ✅ 좋은 예시: 로딩 상태를 표시하는 컴포넌트
export const LoadingSpinner = () => {
  return (
    <div className="loading-spinner" aria-label="Loading">
      <div className="spinner"></div>
      <span>Loading...</span>
    </div>
  );
};