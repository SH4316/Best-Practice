// ✅ 좋은 예시: 타입 정의를 별도 파일로 분리
export interface MenuItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  href?: string;
  onClick?: () => void;
  children?: MenuItem[];
}

export interface AnnouncerMessage {
  message: string;
  politeness?: 'polite' | 'assertive' | 'off';
  timeout?: number;
}

export interface FocusTrapProps {
  isActive: boolean;
  onEscape?: () => void;
  initialFocus?: React.RefObject<HTMLElement>;
  restoreFocus?: React.RefObject<HTMLElement>;
}