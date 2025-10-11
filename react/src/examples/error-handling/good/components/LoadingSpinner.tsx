import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'small' | 'medium' | 'large';
}

// ✅ 좋은 예시: 로딩 스피너 컴포넌트
const LoadingSpinner = ({ message = 'Loading...', size = 'medium' }: LoadingSpinnerProps) => {
  return (
    <div className="loading-spinner" aria-live="polite">
      <div className={`spinner spinner--${size}`} aria-hidden="true"></div>
      <p className="loading-message">{message}</p>
    </div>
  );
};

export default LoadingSpinner;