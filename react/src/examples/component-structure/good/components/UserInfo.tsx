interface UserInfoProps {
  name: string;
  email: string;
}

// ✅ 좋은 예시: 단일 책임을 가진 프레젠테이션 컴포넌트
export const UserInfo = ({ name, email }: UserInfoProps) => {
  return (
    <div className="user-info">
      <h1>{name}</h1>
      <p>{email}</p>
    </div>
  );
};