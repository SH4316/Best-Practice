import { useState } from 'react';
import type { FormEvent, ChangeEvent } from 'react';

interface UserFormProps {
  initialData: {
    name: string;
    email: string;
    bio: string;
  };
  onSubmit: (data: { name: string; email: string; bio: string }) => Promise<void>;
  onCancel: () => void;
  isLoading: boolean;
}

// ✅ 좋은 예시: 폼 로직을 처리하는 컴포넌트
export const UserForm = ({ 
  initialData, 
  onSubmit, 
  onCancel, 
  isLoading 
}: UserFormProps) => {
  const [formData, setFormData] = useState(initialData);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    await onSubmit(formData);
  };

  return (
    <form className="user-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="name">Name:</label>
        <input
          type="text"
          id="name"
          name="name"
          value={formData.name}
          onChange={handleInputChange}
          disabled={isLoading}
        />
      </div>
      <div className="form-group">
        <label htmlFor="email">Email:</label>
        <input
          type="email"
          id="email"
          name="email"
          value={formData.email}
          onChange={handleInputChange}
          disabled={isLoading}
        />
      </div>
      <div className="form-group">
        <label htmlFor="bio">Bio:</label>
        <textarea
          id="bio"
          name="bio"
          value={formData.bio}
          onChange={handleInputChange}
          rows={4}
          disabled={isLoading}
        />
      </div>
      <div className="form-actions">
        <button
          type="submit"
          disabled={isLoading}
          className="btn btn-primary"
        >
          {isLoading ? 'Saving...' : 'Save'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={isLoading}
          className="btn btn-secondary"
        >
          Cancel
        </button>
      </div>
    </form>
  );
};