// ✅ 좋은 예시: 재사용 가능한 로딩 스피너 컴포넌트
export const LoadingSpinner = () => {
  return (
    <div className="loading-spinner" aria-label="Loading">
      <div className="spinner"></div>
      <span>Loading...</span>
    </div>
  );
};