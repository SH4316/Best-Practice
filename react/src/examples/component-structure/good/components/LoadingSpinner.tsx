// ✅ 좋은 예시: 단일 책임을 가진 프레젠테이션 컴포넌트
export const LoadingSpinner = () => {
  return (
    <div className="loading-spinner" aria-label="Loading">
      <div className="spinner"></div>
      <span>Loading...</span>
    </div>
  );
};