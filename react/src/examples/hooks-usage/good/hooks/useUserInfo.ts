import { useMemo } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
}

interface UserInfo {
  fullName: string;
  initials: string;
  domain: string;
}

// ✅ 좋은 예시: 계산 로직을 분리한 Hook
export const useUserInfo = (user: User | null): UserInfo => {
  return useMemo(() => {
    if (!user) {
      return {
        fullName: '',
        initials: '',
        domain: '',
      };
    }

    return {
      fullName: `${user.name} (${user.email})`,
      initials: user.name
        .split(' ')
        .map(n => n[0])
        .join(''),
      domain: user.email.split('@')[1] || '',
    };
  }, [user]);
};