interface ButtonProps {
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
  children: React.ReactNode;
}

// ✅ 좋은 예시: 유연하고 재사용 가능한 컴포넌트
export const Button = ({ 
  onClick, 
  variant = 'primary', 
  disabled = false, 
  children 
}: ButtonProps) => {
  const className = `btn btn--${variant}`;
  
  return (
    <button 
      className={className}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};