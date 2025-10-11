import { useState, useEffect } from 'react';
import type { FormEvent, ChangeEvent } from 'react';

// 사용자 타입 정의
interface User {
  id: string;
  name: string;
  email: string;
  bio: string;
  avatar?: string;
  postsCount?: number;
  followersCount?: number;
  followingCount?: number;
}

interface FormData {
  name: string;
  email: string;
  bio: string;
}

// ❌ 나쁜 예시: 여러 책임을 가진 컴포넌트
const UserProfile = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: '',
    bio: ''
  });

  // 데이터 페칭 로직
  useEffect(() => {
    setIsLoading(true);
    fetch('/api/user/123')
      .then(res => {
        if (!res.ok) {
          throw new Error('Failed to fetch user');
        }
        return res.json();
      })
      .then(data => {
        setUser(data);
        setFormData({
          name: data.name,
          email: data.email,
          bio: data.bio
        });
      })
      .catch(err => {
        setError(err.message);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, []);

  // 폼 핸들러
  const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev: FormData) => ({
      ...prev,
      [name]: value
    }));
  };

  // 제출 핸들러
  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    
    fetch('/api/user/123', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(formData)
    })
      .then(res => {
        if (!res.ok) {
          throw new Error('Failed to update user');
        }
        return res.json();
      })
      .then(data => {
        setUser(data);
        setIsEditing(false);
      })
      .catch(err => {
        setError(err.message);
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  if (isLoading && !user) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!user) {
    return <div className="not-found">User not found</div>;
  }

  return (
    <div className="user-profile" style={{ 
      maxWidth: '600px', 
      margin: '0 auto', 
      padding: '20px',
      border: '1px solid #ddd',
      borderRadius: '8px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div className="user-header" style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
        <div className="avatar" style={{ 
          width: '80px', 
          height: '80px', 
          borderRadius: '50%', 
          backgroundColor: '#f0f0f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginRight: '20px',
          overflow: 'hidden'
        }}>
          {user.avatar ? (
            <img src={user.avatar} alt={user.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          ) : (
            <div style={{ fontSize: '24px', color: '#666' }}>
              {user.name.charAt(0).toUpperCase()}
            </div>
          )}
        </div>
        <div className="user-info">
          <h1 style={{ margin: '0 0 5px 0', fontSize: '24px' }}>{user.name}</h1>
          <p style={{ margin: '0', color: '#666' }}>{user.email}</p>
        </div>
      </div>

      {isEditing ? (
        <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Name:</label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '16px'
              }}
            />
          </div>
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Email:</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '16px'
              }}
            />
          </div>
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Bio:</label>
            <textarea
              name="bio"
              value={formData.bio}
              onChange={handleInputChange}
              rows={4}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '16px',
                resize: 'vertical'
              }}
            />
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              type="submit"
              disabled={isLoading}
              style={{
                padding: '8px 16px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                opacity: isLoading ? 0.7 : 1
              }}
            >
              {isLoading ? 'Saving...' : 'Save'}
            </button>
            <button
              type="button"
              onClick={() => setIsEditing(false)}
              style={{
                padding: '8px 16px',
                backgroundColor: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <div className="user-details">
          <div className="user-bio" style={{ marginBottom: '20px' }}>
            <h2 style={{ fontSize: '18px', marginBottom: '10px' }}>Bio</h2>
            <p style={{ lineHeight: '1.5' }}>{user.bio || 'No bio available'}</p>
          </div>
          
          <div className="user-stats" style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{user.postsCount || 0}</div>
              <div style={{ color: '#666' }}>Posts</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{user.followersCount || 0}</div>
              <div style={{ color: '#666' }}>Followers</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{user.followingCount || 0}</div>
              <div style={{ color: '#666' }}>Following</div>
            </div>
          </div>
          
          <button
            onClick={() => setIsEditing(true)}
            style={{
              padding: '8px 16px',
              backgroundColor: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Edit Profile
          </button>
        </div>
      )}
    </div>
  );
};

export default UserProfile;