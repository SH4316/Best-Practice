import { render } from '@testing-library/react';
import UserProfile from './UserProfile';

// ❌ 나쁜 예시: 구현 세부 사항에 의존하고 사용자 관점에서 테스트되지 않음
describe('UserProfile', () => {
  test('should render correctly', () => {
    // 컴포넌트 렌더링만 확인하고 무엇을 테스트하는지 명확하지 않음
    render(<UserProfile userId="123" />);
    
    // 구현 세부 사항에 의존 (data-testid)
    expect(document.querySelector('[data-testid="user-profile"]')).toBeTruthy();
  });

  test('should call getUser API', () => {
    // API 호출을 직접 테스트하고 있음 (통합 테스트여야 함)
    render(<UserProfile userId="123" />);
    
    // API 호출 확인을 위한 방법이 없음
    // 실제 API를 호출하게 되어 테스트가 느리고 불안정해짐
  });

  test('should update user state', () => {
    // 컴포넌트 내부 상태를 직접 테스트하는 안티패턴
    const { container } = render(<UserProfile userId="123" />);
    
    // 상태 변경을 확인하기 위한 인위적인 방법
    const setState = jest.fn();
    // 이 방식은 실제 사용자 시나리오를 반영하지 않음
  });

  test('should handle loading state', () => {
    // 로딩 상태를 확인하는 것이 아니라 그냥 렌더링만 확인
    render(<UserProfile userId="123" />);
    
    // 로딩 상태를 확인하지 않고 테스트가 통과됨
    expect(true).toBe(true);
  });

  test('should handle error state', () => {
    // 에러 상태를 확인하는 것이 아니라 그냥 렌더링만 확인
    render(<UserProfile userId="123" />);
    
    // 에러 상태를 확인하지 않고 테스트가 통과됨
    expect(true).toBe(true);
  });

  test('should match snapshot', () => {
    // 스냅샷 테스트만으로는 충분하지 않음
    const { asFragment } = render(<UserProfile userId="123" />);
    expect(asFragment()).toMatchSnapshot();
    
    // 스냅샷이 변경되어도 실제 기능이 올바르게 동작하는지 확인할 수 없음
  });
});