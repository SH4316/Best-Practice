import { useState } from 'react';
import { useUser } from './hooks/useUser';
import { Avatar } from './components/Avatar';
import { UserInfo } from './components/UserInfo';
import { UserStats } from './components/UserStats';
import { UserForm } from './components/UserForm';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import { EmptyState } from './components/EmptyState';
import { Button } from './components/Button';
import './UserProfile.styles.css';

// ✅ 좋은 예시: 책임 분리와 컴포넌트 합성
const UserProfile = () => {
  const { user, isLoading, error, updateUser } = useUser('123');
  const [isEditing, setIsEditing] = useState(false);

  const handleEditToggle = () => {
    setIsEditing(prev => !prev);
  };

  const handleFormSubmit = async (formData: { name: string; email: string; bio: string }) => {
    try {
      await updateUser(formData);
      setIsEditing(false);
    } catch (err) {
      // Error is handled in the hook
    }
  };

  if (isLoading && !user) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage message={error} />;
  }

  if (!user) {
    return <EmptyState message="User not found" />;
  }

  return (
    <div className="user-profile">
      <div className="user-header">
        <Avatar 
          src={user.avatar} 
          alt={user.name}
          size="large"
          fallback={user.name.charAt(0).toUpperCase()}
        />
        <UserInfo 
          name={user.name} 
          email={user.email} 
        />
      </div>

      {isEditing ? (
        <UserForm 
          initialData={{
            name: user.name,
            email: user.email,
            bio: user.bio
          }}
          onSubmit={handleFormSubmit}
          onCancel={handleEditToggle}
          isLoading={isLoading}
        />
      ) : (
        <div className="user-details">
          <div className="user-bio">
            <h2>Bio</h2>
            <p>{user.bio || 'No bio available'}</p>
          </div>
          
          <UserStats 
            postsCount={user.postsCount || 0}
            followersCount={user.followersCount || 0}
            followingCount={user.followingCount || 0}
          />
          
          <Button 
            onClick={handleEditToggle}
            variant="primary"
          >
            Edit Profile
          </Button>
        </div>
      )}
    </div>
  );
};

export default UserProfile;