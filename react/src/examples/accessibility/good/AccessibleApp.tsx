import { useState } from 'react';
import { NavigationMenu, Modal, FormField } from './components';
import type { MenuItem } from './types';
import './AccessibleApp.css';

// ✅ 좋은 예시: 접근성이 높은 애플리케이션
const AccessibleApp = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
  });
  const [formErrors, setFormErrors] = useState({
    name: '',
    email: '',
    message: '',
  });
  
  // 내비게이션 메뉴 항목
  const menuItems: MenuItem[] = [
    {
      id: 'home',
      label: 'Home',
      icon: <span aria-hidden="true">🏠</span>,
      href: '#home',
    },
    {
      id: 'about',
      label: 'About',
      icon: <span aria-hidden="true">ℹ️</span>,
      href: '#about',
    },
    {
      id: 'contact',
      label: 'Contact',
      icon: <span aria-hidden="true">📧</span>,
      onClick: () => setIsModalOpen(true),
    },
  ];
  
  // 모달 열기/닫기 핸들러
  const openModal = () => {
    setIsModalOpen(true);
  };
  
  const closeModal = () => {
    setIsModalOpen(false);
  };
  
  // 폼 입력 변경 핸들러
  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
    
    // 에러 상태 초기화
    if (formErrors[field as keyof typeof formErrors]) {
      setFormErrors(prev => ({
        ...prev,
        [field]: '',
      }));
    }
  };
  
  // 폼 유효성 검사
  const validateForm = () => {
    const errors = {
      name: '',
      email: '',
      message: '',
    };
    
    if (!formData.name.trim()) {
      errors.name = 'Name is required';
    }
    
    if (!formData.email.trim()) {
      errors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      errors.email = 'Please enter a valid email address';
    }
    
    if (!formData.message.trim()) {
      errors.message = 'Message is required';
    }
    
    setFormErrors(errors);
    
    return !Object.values(errors).some(error => error);
  };
  
  // 폼 제출 핸들러
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateForm()) {
      // 폼 제출 로직
      console.log('Form submitted:', formData);
      
      // 폼 초기화
      setFormData({
        name: '',
        email: '',
        message: '',
      });
      
      // 모달 닫기
      closeModal();
      
      // 성공 알림
      alert('Form submitted successfully!');
    }
  };
  
  return (
    <div className="accessible-app">
      <header className="app-header">
        <h1>Accessible React App</h1>
        <NavigationMenu items={menuItems} />
      </header>
      
      <main className="app-main">
        <section id="home" className="section">
          <h2>Welcome to our accessible app</h2>
          <p>
            This application demonstrates various accessibility best practices in React.
            Try navigating with your keyboard, using a screen reader, or testing with
            accessibility tools.
          </p>
        </section>
        
        <section id="about" className="section">
          <h2>About</h2>
          <p>
            Accessibility is about making your app usable by as many people as possible.
            This includes people with disabilities, people using mobile devices,
            and people with slow network connections.
          </p>
        </section>
        
        <section id="contact" className="section">
          <h2>Contact Us</h2>
          <button
            className="contact-button"
            onClick={openModal}
            aria-label="Open contact form"
          >
            Open Contact Form
          </button>
        </section>
      </main>
      
      <footer className="app-footer">
        <p>&copy; 2023 Accessible React App. All rights reserved.</p>
      </footer>
      
      {/* 연락처 모달 */}
      <Modal
        isOpen={isModalOpen}
        onClose={closeModal}
        title="Contact Form"
      >
        <form onSubmit={handleSubmit} className="contact-form">
          <FormField
            label="Name"
            value={formData.name}
            onChange={(value) => handleInputChange('name', value)}
            error={formErrors.name}
            required
          />
          
          <FormField
            label="Email"
            type="email"
            value={formData.email}
            onChange={(value) => handleInputChange('email', value)}
            error={formErrors.email}
            helpText="We'll never share your email with anyone else."
            required
          />
          
          <FormField
            label="Message"
            type="text"
            value={formData.message}
            onChange={(value) => handleInputChange('message', value)}
            error={formErrors.message}
            required
          />
          
          <div className="form-actions">
            <button type="button" onClick={closeModal}>
              Cancel
            </button>
            <button type="submit">
              Submit
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
};

export default AccessibleApp;