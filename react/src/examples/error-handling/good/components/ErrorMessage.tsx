import React from 'react';
import { ErrorLogger } from '../utils';
import type { ApiError } from '../types';

interface ErrorMessageProps {
  error: Error | ApiError | string | null;
  onRetry?: () => void;
  onDismiss?: () => void;
}

// ✅ 좋은 예시: 에러 메시지 컴포넌트
const ErrorMessage = ({ error, onRetry, onDismiss }: ErrorMessageProps) => {
  if (!error) return null;

  const getErrorMessage = () => {
    if (typeof error === 'string') {
      return error;
    }
    
    return ErrorLogger.getErrorMessage(error);
  };

  const handleRetry = () => {
    if (onRetry) {
      onRetry();
    }
  };

  const handleDismiss = () => {
    if (onDismiss) {
      onDismiss();
    }
  };

  return (
    <div className="error-message" role="alert">
      <div className="error-icon">⚠️</div>
      <div className="error-content">
        <p className="error-text">{getErrorMessage()}</p>
        
        <div className="error-actions">
          {onRetry && (
            <button 
              className="retry-button" 
              onClick={handleRetry}
              aria-label="Retry"
            >
              Try Again
            </button>
          )}
          
          {onDismiss && (
            <button 
              className="dismiss-button" 
              onClick={handleDismiss}
              aria-label="Dismiss error"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorMessage;