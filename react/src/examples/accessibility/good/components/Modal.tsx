import React, { useEffect } from 'react';
import { useFocusTrap, useScreenReaderAnnouncer } from '../utils';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

// ✅ 좋은 예시: 접근성이 높은 모달 컴포넌트
const Modal = ({ isOpen, onClose, title, children }: ModalProps) => {
  const modalRef = useFocusTrap({
    isActive: isOpen,
    onEscape: onClose,
  });
  
  const { announce } = useScreenReaderAnnouncer();

  // 모달이 열릴 때 스크린 리더에 알림
  useEffect(() => {
    if (isOpen) {
      announce(`Modal opened: ${title}`);
      // 문서 본문 스크롤 방지
      document.body.style.overflow = 'hidden';
    } else {
      // 문서 본문 스크롤 복원
      document.body.style.overflow = '';
    }
    
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen, title, announce]);

  // 모달이 닫혀있으면 렌더링하지 않음
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div
        className="modal"
        ref={modalRef as React.RefObject<HTMLDivElement>}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        <div className="modal-header">
          <h2 id="modal-title">{title}</h2>
          <button
            className="modal-close-button"
            onClick={onClose}
            aria-label="Close modal"
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>
        
        <div className="modal-body">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;