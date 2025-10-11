import { API_BASE_URL, API_ENDPOINTS } from '../constants';
import type { FileItem } from '../types';

// ✅ 좋은 예시: API 관련 서비스를 별도 파일로 분리
export const fileService = {
  // 파일 목록 가져오기
  async getFiles(): Promise<FileItem[]> {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.FILES}`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch files');
    }
    
    return response.json();
  },
  
  // 파일 업로드
  async uploadFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<FileItem> {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const xhr = new XMLHttpRequest();
      
      // 업로드 진행률 추적
      if (onProgress) {
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded / event.total) * 100);
            onProgress(progress);
          }
        });
      }
      
      // 성공 처리
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          try {
            const newFile = JSON.parse(xhr.responseText);
            resolve(newFile);
          } catch {
            reject(new Error('Invalid response format'));
          }
        } else {
          reject(new Error(`Upload failed with status ${xhr.status}`));
        }
      });
      
      // 에러 처리
      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });
      
      // 요청 보내기
      xhr.open('POST', `${API_BASE_URL}${API_ENDPOINTS.UPLOAD}`);
      xhr.send(formData);
    });
  },
  
  // 파일 삭제
  async deleteFile(fileId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.FILES}/${fileId}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error('Failed to delete file');
    }
  },
  
  // 파일 다운로드 URL 생성
  getDownloadUrl(fileId: string): string {
    return `${API_BASE_URL}${API_ENDPOINTS.FILES}/${fileId}/download`;
  },
};