interface ErrorMessageProps {
  message: string;
}

// ✅ 좋은 예시: 에러 메시지를 표시하는 컴포넌트
export const ErrorMessage = ({ message }: ErrorMessageProps) => {
  return (
    <div className="error-message" role="alert">
      <span className="error-icon">⚠️</span>
      <span className="error-text">Error: {message}</span>
    </div>
  );
};