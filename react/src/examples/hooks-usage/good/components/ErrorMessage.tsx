interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

// ✅ 좋은 예시: 재사용 가능한 에러 메시지 컴포넌트
export const ErrorMessage = ({ message, onRetry }: ErrorMessageProps) => {
  return (
    <div className="error-message" role="alert">
      <span className="error-icon">⚠️</span>
      <span className="error-text">Error: {message}</span>
      {onRetry && (
        <button className="retry-button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
};