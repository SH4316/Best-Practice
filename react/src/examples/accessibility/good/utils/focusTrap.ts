import { useEffect, useRef } from 'react';
import type { FocusTrapProps } from '../types';

// ✅ 좋은 예시: 포커스 트랩 유틸리티
export const useFocusTrap = ({ isActive, onEscape }: FocusTrapProps) => {
  const containerRef = useRef<HTMLElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  
  useEffect(() => {
    if (!isActive || !containerRef.current) return;
    
    // 현재 포커스된 요소 저장
    previousFocusRef.current = document.activeElement as HTMLElement;
    
    // 컨테이너 내의 포커스 가능한 요소 가져오기
    const focusableElements = containerRef.current.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as NodeListOf<HTMLElement>;
    
    if (focusableElements.length > 0) {
      // 첫 번째 포커스 가능 요소로 포커스 이동
      focusableElements[0].focus();
    }
    
    // 키보드 이벤트 핸들러
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onEscape?.();
        return;
      }
      
      if (event.key === 'Tab') {
        if (focusableElements.length === 0) return;
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        
        // Shift + Tab 키 처리
        if (event.shiftKey) {
          if (document.activeElement === firstElement) {
            event.preventDefault();
            lastElement.focus();
          }
        } 
        // Tab 키 처리
        else {
          if (document.activeElement === lastElement) {
            event.preventDefault();
            firstElement.focus();
          }
        }
      }
    };
    
    // 이벤트 리스너 추가
    document.addEventListener('keydown', handleKeyDown);
    
    // 정리 함수
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      
      // 이전 포커스된 요소로 포커스 복원
      if (previousFocusRef.current) {
        previousFocusRef.current.focus();
      }
    };
  }, [isActive, onEscape]);
  
  return containerRef;
};