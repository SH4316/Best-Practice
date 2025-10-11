import { useEffect, useRef } from 'react';
import type { AnnouncerMessage } from '../types';

// ✅ 좋은 예시: 스크린 리더 알림 유틸리티
class ScreenReaderAnnouncer {
  private static announcer: HTMLElement | null = null;
  
  private static getAnnouncer(): HTMLElement {
    if (!this.announcer) {
      this.announcer = document.createElement('div');
      this.announcer.setAttribute('aria-live', 'polite');
      this.announcer.setAttribute('aria-atomic', 'true');
      this.announcer.style.position = 'absolute';
      this.announcer.style.left = '-10000px';
      this.announcer.style.width = '1px';
      this.announcer.style.height = '1px';
      this.announcer.style.overflow = 'hidden';
      document.body.appendChild(this.announcer);
    }
    
    return this.announcer;
  }
  
  static announce(message: string, politeness: 'polite' | 'assertive' | 'off' = 'polite'): void {
    const announcer = this.getAnnouncer();
    
    // politeness 속성 업데이트
    announcer.setAttribute('aria-live', politeness);
    
    // 메시지 설정
    announcer.textContent = message;
    
    // 메시지 지우기 (다음 메시지를 위해)
    setTimeout(() => {
      announcer.textContent = '';
    }, 1000);
  }
}

// 스크린 리더 알림을 위한 Hook
export const useScreenReaderAnnouncer = () => {
  const timeoutRef = useRef<number | null>(null);
  
  const announce = (message: string, options?: Partial<AnnouncerMessage>) => {
    const { politeness = 'polite', timeout } = options || {};
    
    // 이전 타임아웃 클리어
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    // 메시지 알림
    ScreenReaderAnnouncer.announce(message, politeness);
    
    // 타임아웃 설정
    if (timeout) {
      timeoutRef.current = setTimeout(() => {
        ScreenReaderAnnouncer.announce('');
      }, timeout);
    }
  };
  
  // 정리 함수
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);
  
  return { announce };
};

export default ScreenReaderAnnouncer;