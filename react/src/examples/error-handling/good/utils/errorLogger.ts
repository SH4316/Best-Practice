import type { ApiError } from '../types';

// ✅ 좋은 예시: 에러 로깅 유틸리티
class ErrorLogger {
  static log(error: Error | ApiError, context?: string, errorInfo?: React.ErrorInfo) {
    const errorData = {
      message: error.message,
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      context,
      errorInfo,
      // API 에러인 경우 추가 정보
      ...(error as ApiError).status && { status: (error as ApiError).status },
      ...(error as ApiError).code && { code: (error as ApiError).code },
    };
    
    // 개발 환경에서는 콘솔에 출력
    if (import.meta.env.DEV) {
      console.error('Error logged:', errorData);
      return;
    }
    
    // 프로덕션 환경에서는 로깅 서비스로 전송
    this.sendToLoggingService(errorData);
  }
  
  private static async sendToLoggingService(errorData: Record<string, unknown>) {
    try {
      await fetch('/api/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(errorData),
      });
    } catch (e) {
      console.error('Failed to log error:', e);
    }
  }
  
  static getErrorMessage(error: Error | ApiError): string {
    // API 에러인 경우 사용자 친화적인 메시지 반환
    if ('status' in error) {
      const apiError = error as ApiError;
      
      switch (apiError.status) {
        case 400:
          return 'Invalid request. Please check your input and try again.';
        case 401:
          return 'You are not authorized. Please log in and try again.';
        case 403:
          return 'You do not have permission to perform this action.';
        case 404:
          return 'The requested resource was not found.';
        case 500:
          return 'Server error. Please try again later.';
        default:
          return apiError.message || 'An unexpected error occurred.';
      }
    }
    
    // 일반 에러인 경우
    return error.message || 'An unexpected error occurred.';
  }
}

export default ErrorLogger;