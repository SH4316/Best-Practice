import { useState } from 'react';

interface AvatarProps {
  src?: string;
  alt: string;
  size?: 'small' | 'medium' | 'large';
  fallback?: string;
}

// ✅ 좋은 예시: 단일 책임을 가진 프레젠테이션 컴포넌트
export const Avatar = ({ 
  src, 
  alt, 
  size = 'medium', 
  fallback 
}: AvatarProps) => {
  const [hasError, setHasError] = useState(false);
  
  const handleImageError = () => {
    setHasError(true);
  };
  
  const sizeClass = `avatar--${size}`;
  
  if (hasError || !src) {
    return (
      <div className={`avatar avatar--fallback ${sizeClass}`}>
        {fallback?.charAt(0).toUpperCase()}
      </div>
    );
  }
  
  return (
    <img 
      src={src} 
      alt={alt}
      className={`avatar ${sizeClass}`}
      onError={handleImageError}
    />
  );
};