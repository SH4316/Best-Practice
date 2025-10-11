import { useState, useCallback, useEffect } from 'react';

// ✅ 좋은 예시: 테마 로직을 분리한 Hook
export const useTheme = () => {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  const toggleTheme = useCallback(() => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  }, []);

  // 테마 변경 시 DOM에 클래스 적용
  useEffect(() => {
    document.body.className = theme;
  }, [theme]);

  return {
    theme,
    toggleTheme,
  };
};