import ErrorLogger from './errorLogger';
import type { ApiError } from '../types';

// ✅ 좋은 예시: API 서비스 유틸리티
class ApiService {
  private static async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      let errorMessage: string;
      let errorCode: string | undefined;
      
      try {
        const errorData = await response.json();
        errorMessage = errorData.message || response.statusText;
        errorCode = errorData.code;
      } catch {
        errorMessage = response.statusText || 'Unknown error';
      }
      
      const error: ApiError = new Error(errorMessage) as ApiError;
      error.status = response.status;
      error.code = errorCode;
      
      throw error;
    }
    
    return response.json() as Promise<T>;
  }
  
  static async get<T>(url: string): Promise<T> {
    try {
      const response = await fetch(url);
      return this.handleResponse<T>(response);
    } catch (error) {
      ErrorLogger.log(error as Error, `GET ${url}`);
      throw error;
    }
  }
  
  static async post<T>(url: string, data?: unknown): Promise<T> {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data ? JSON.stringify(data) : undefined,
      });
      
      return this.handleResponse<T>(response);
    } catch (error) {
      ErrorLogger.log(error as Error, `POST ${url}`);
      throw error;
    }
  }
  
  static async put<T>(url: string, data?: unknown): Promise<T> {
    try {
      const response = await fetch(url, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data ? JSON.stringify(data) : undefined,
      });
      
      return this.handleResponse<T>(response);
    } catch (error) {
      ErrorLogger.log(error as Error, `PUT ${url}`);
      throw error;
    }
  }
  
  static async delete<T>(url: string): Promise<T> {
    try {
      const response = await fetch(url, {
        method: 'DELETE',
      });
      
      return this.handleResponse<T>(response);
    } catch (error) {
      ErrorLogger.log(error as Error, `DELETE ${url}`);
      throw error;
    }
  }
}

export default ApiService;