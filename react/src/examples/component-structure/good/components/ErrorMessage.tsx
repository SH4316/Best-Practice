interface ErrorMessageProps {
  message: string;
}

// ✅ 좋은 예시: 단일 책임을 가진 프레젠테이션 컴포넌트
export const ErrorMessage = ({ message }: ErrorMessageProps) => {
  return (
    <div className="error-message" role="alert">
      <span className="error-icon">⚠️</span>
      <span className="error-text">Error: {message}</span>
    </div>
  );
};