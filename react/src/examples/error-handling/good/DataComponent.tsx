import { useState, useCallback } from 'react';
import type { FormEvent } from 'react';
import { ErrorBoundary, ErrorMessage, LoadingSpinner } from './components';
import { useUser, useUserPosts, useCreatePost, useDeletePost } from './hooks';
import type { Post } from './types';
import './DataComponent.css';

// ✅ 좋은 예시: 에러 처리가 잘 된 데이터 컴포넌트
const DataComponent = ({ userId }: { userId: string }) => {
  const [isCreatingPost, setIsCreatingPost] = useState(false);
  const [createPostError, setCreatePostError] = useState<string | null>(null);
  const [isDeletingPost, setIsDeletingPost] = useState<string | null>(null);
  
  // 커스텀 Hook으로 데이터 페칭 및 에러 처리
  const { 
    data: user, 
    loading: userLoading, 
    error: userError, 
    refetch: refetchUser 
  } = useUser(userId);
  
  const { 
    data: posts, 
    loading: postsLoading, 
    error: postsError, 
    refetch: refetchPosts 
  } = useUserPosts(userId);
  
  const { createPost, loading: createPostLoading, error: createPostHookError } = useCreatePost();
  const { deletePost, loading: deletePostLoading, error: deletePostError } = useDeletePost();
  
  // 전체 로딩 상태
  const loading = userLoading || postsLoading;
  
  // 전체 에러 상태
  const error = userError || postsError;
  
  // 포스트 삭제 핸들러
  const handleDeletePost = useCallback(async (postId: string) => {
    setIsDeletingPost(postId);
    setCreatePostError(null);
    
    try {
      await deletePost(postId);
      // 성공하면 포스트 목록 새로고침
      refetchPosts();
    } catch (err) {
      // 에러는 useDeletePost Hook에서 처리됨
      console.error('Failed to delete post:', err);
    } finally {
      setIsDeletingPost(null);
    }
  }, [deletePost, refetchPosts]);
  
  // 포스트 생성 핸들러
  const handleCreatePost = useCallback(async (title: string, content: string) => {
    if (!title.trim() || !content.trim()) {
      setCreatePostError('Title and content are required');
      return;
    }
    
    setIsCreatingPost(true);
    setCreatePostError(null);
    
    try {
      await createPost(userId, title, content);
      // 성공하면 폼 초기화 및 포스트 목록 새로고침
      refetchPosts();
      return true;
    } catch (err) {
      // 에러는 useCreatePost Hook에서 처리됨
      console.error('Failed to create post:', err);
      return false;
    } finally {
      setIsCreatingPost(false);
    }
  }, [createPost, userId, refetchPosts]);
  
  // 폼 제출 핸들러
  const handleFormSubmit = useCallback(async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    
    const form = e.currentTarget;
    const title = form.elements.namedItem('title') as HTMLInputElement;
    const content = form.elements.namedItem('content') as HTMLTextAreaElement;
    
    const success = await handleCreatePost(title.value, content.value);
    
    if (success) {
      // 폼 초기화
      form.reset();
    }
  }, [handleCreatePost]);
  
  // 로딩 상태 표시
  if (loading) {
    return <LoadingSpinner message="Loading data..." />;
  }
  
  // 에러 상태 표시
  if (error) {
    return (
      <ErrorMessage 
        error={error} 
        onRetry={() => {
          refetchUser();
          refetchPosts();
        }} 
      />
    );
  }
  
  // 사용자 데이터가 없는 경우
  if (!user) {
    return (
      <div className="empty-state">
        <p>User not found</p>
      </div>
    );
  }
  
  return (
    <div className="data-component">
      <ErrorBoundary>
        <div className="user-profile">
          <h1>{user.name}</h1>
          <p>{user.email}</p>
          <img src={user.avatar} alt={user.name} />
        </div>
        
        <div className="posts">
          <h2>Posts</h2>
          
          {posts && posts.length > 0 ? (
            <ul className="posts-list">
              {posts.map((post: Post) => (
                <li key={post.id} className="post-item">
                  <h3>{post.title}</h3>
                  <p>{post.content}</p>
                  <div className="post-actions">
                    <button 
                      onClick={() => handleDeletePost(post.id)}
                      disabled={isDeletingPost === post.id || deletePostLoading}
                      className="delete-button"
                    >
                      {isDeletingPost === post.id ? 'Deleting...' : 'Delete'}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="empty-state">
              <p>No posts found</p>
            </div>
          )}
          
          <div className="create-post">
            <h3>Create Post</h3>
            
            <ErrorBoundary>
              <form onSubmit={handleFormSubmit} className="post-form">
                <div className="form-group">
                  <label htmlFor="title">Title</label>
                  <input 
                    type="text" 
                    id="title"
                    name="title" 
                    placeholder="Title" 
                    required
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="content">Content</label>
                  <textarea 
                    id="content"
                    name="content" 
                    placeholder="Content" 
                    required
                  />
                </div>
                
                {createPostError && (
                  <div className="form-error">
                    {createPostError}
                  </div>
                )}
                
                {createPostHookError && (
                  <ErrorMessage 
                    error={createPostHookError}
                    onDismiss={() => setCreatePostError(null)}
                  />
                )}
                
                {deletePostError && (
                  <ErrorMessage 
                    error={deletePostError}
                    onDismiss={() => setCreatePostError(null)}
                  />
                )}
                
                <button 
                  type="submit" 
                  disabled={isCreatingPost || createPostLoading}
                  className="submit-button"
                >
                  {isCreatingPost || createPostLoading ? 'Creating...' : 'Create'}
                </button>
              </form>
            </ErrorBoundary>
          </div>
        </div>
      </ErrorBoundary>
    </div>
  );
};

export default DataComponent;