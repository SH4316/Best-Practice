import { useState, useEffect } from 'react';
import type { FormEvent } from 'react';

// 타입 정의
interface User {
  id: string;
  name: string;
  email: string;
  avatar: string;
}

interface Post {
  id: string;
  title: string;
  content: string;
}

// ❌ 나쁜 예시: 에러 처리가 부족한 데이터 컴포넌트
const DataComponent = ({ userId }: { userId: string }) => {
  const [user, setUser] = useState<User | null>(null);
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(false);

  // 나쁜 예시: 에러 처리가 없는 데이터 페칭
  useEffect(() => {
    setLoading(true);
    
    // 사용자 데이터 페칭
    fetch(`/api/users/${userId}`)
      .then(response => response.json())
      .then(userData => {
        setUser(userData);
        
        // 포스트 데이터 페칭
        return fetch(`/api/users/${userId}/posts`);
      })
      .then(response => response.json())
      .then(postsData => {
        setPosts(postsData);
        setLoading(false);
      })
      .catch(error => {
        // 에러가 발생해도 사용자에게 알리지 않음
        console.error('Error fetching data:', error);
        setLoading(false);
      });
  }, [userId]);

  // 나쁜 예시: 에러 처리가 없는 포스트 삭제
  const handleDeletePost = (postId: string) => {
    fetch(`/api/posts/${postId}`, {
      method: 'DELETE',
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to delete post');
        }
        
        // 상태를 직접 수정하여 UI가 일치하지 않을 수 있음
        setPosts(posts.filter(post => post.id !== postId));
      })
      .catch(error => {
        // 에러가 발생해도 사용자에게 알리지 않음
        console.error('Error deleting post:', error);
      });
  };

  // 나쁜 예시: 에러 처리가 없는 포스트 생성
  const handleCreatePost = (title: string, content: string) => {
    fetch(`/api/users/${userId}/posts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ title, content }),
    })
      .then(response => response.json())
      .then(newPost => {
        // 상태를 직접 수정하여 UI가 일치하지 않을 수 있음
        setPosts([newPost, ...posts]);
      })
      .catch(error => {
        // 에러가 발생해도 사용자에게 알리지 않음
        console.error('Error creating post:', error);
      });
  };

  // 나쁜 예시: 잠재적인 에러가 있는 렌더링
  return (
    <div className="data-component">
      {loading ? (
        <div>Loading...</div>
      ) : (
        <>
          <div className="user-profile">
            {/* user가 null일 수 있으므로 에러 발생 가능 */}
            {user && (
              <>
                <h1>{user.name}</h1>
                <p>{user.email}</p>
                <img src={user.avatar} alt={user.name} />
              </>
            )}
          </div>
          
          <div className="posts">
            <h2>Posts</h2>
            
            {/* posts 배열이 비어있을 때 에러 발생 가능 */}
            {posts.length > 0 ? (
              <ul>
                {posts.map(post => (
                  <li key={post.id}>
                    <h3>{post.title}</h3>
                    <p>{post.content}</p>
                    <button onClick={() => handleDeletePost(post.id)}>
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No posts found</p>
            )}
            
            <div className="create-post">
              <h3>Create Post</h3>
              <form
                onSubmit={(e: FormEvent<HTMLFormElement>) => {
                  e.preventDefault();
                  const form = e.currentTarget;
                  const title = form.elements.namedItem('title') as HTMLInputElement;
                  const content = form.elements.namedItem('content') as HTMLTextAreaElement;
                  
                  // 유효성 검사 없이 API 호출
                  handleCreatePost(title.value, content.value);
                  
                  // 폼이 초기화되지 않음
                }}
              >
                <input type="text" name="title" placeholder="Title" />
                <textarea name="content" placeholder="Content"></textarea>
                <button type="submit">Create</button>
              </form>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default DataComponent;