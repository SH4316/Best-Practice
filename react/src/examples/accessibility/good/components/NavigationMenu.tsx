import React, { useState, useRef, useEffect } from 'react';
import { useKeyboardNavigation } from '../hooks';
import { useScreenReaderAnnouncer } from '../utils';
import type { MenuItem } from '../types';

// ✅ 좋은 예시: 접근성이 높은 내비게이션 메뉴
const NavigationMenu = ({ items }: { items: MenuItem[] }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const menuItemsRef = useRef<(HTMLAnchorElement | HTMLButtonElement)[]>([]);
  const { announce } = useScreenReaderAnnouncer();

  // 키보드 네비게이션 설정
  useKeyboardNavigation({
    onEscape: () => {
      if (isOpen) {
        setIsOpen(false);
        setFocusedIndex(-1);
        menuButtonRef.current?.focus();
        announce('Menu closed');
      }
    },
    onArrowDown: () => {
      if (isOpen) {
        const nextIndex = focusedIndex < items.length - 1 ? focusedIndex + 1 : 0;
        setFocusedIndex(nextIndex);
        menuItemsRef.current[nextIndex]?.focus();
      }
    },
    onArrowUp: () => {
      if (isOpen) {
        const prevIndex = focusedIndex > 0 ? focusedIndex - 1 : items.length - 1;
        setFocusedIndex(prevIndex);
        menuItemsRef.current[prevIndex]?.focus();
      }
    },
    isActive: isOpen,
  });

  // 메뉴 열기/닫기 핸들러
  const toggleMenu = () => {
    setIsOpen(!isOpen);
    setFocusedIndex(-1);
    
    if (!isOpen) {
      announce('Menu opened');
    } else {
      announce('Menu closed');
    }
  };

  // 메뉴 항목 클릭 핸들러
  const handleItemClick = (item: MenuItem, index: number) => {
    setFocusedIndex(index);
    
    if (item.onClick) {
      item.onClick();
    }
    
    if (item.href) {
      // 링크 이동
      window.location.href = item.href;
    }
    
    setIsOpen(false);
    announce(`Selected ${item.label}`);
  };

  // 메뉴 외부 클릭 핸들러
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        isOpen &&
        menuButtonRef.current &&
        !menuButtonRef.current.contains(event.target as Node) &&
        menuItemsRef.current.every(
          ref => !ref || !ref.contains(event.target as Node)
        )
      ) {
        setIsOpen(false);
        setFocusedIndex(-1);
        announce('Menu closed');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, announce]);

  // 메뉴 항목 렌더링
  const renderMenuItem = (item: MenuItem, index: number) => {
    if (item.href) {
      return (
        <a
          key={item.id}
          href={item.href}
          ref={(el) => {
            if (el) menuItemsRef.current[index] = el;
          }}
          className="menu-item"
          onClick={() => handleItemClick(item, index)}
          aria-label={item.label}
        >
          {item.icon}
          <span>{item.label}</span>
        </a>
      );
    }

    return (
      <button
        key={item.id}
        type="button"
        ref={(el) => {
          if (el) menuItemsRef.current[index] = el;
        }}
        className="menu-item"
        onClick={() => handleItemClick(item, index)}
        aria-label={item.label}
      >
        {item.icon}
        <span>{item.label}</span>
      </button>
    );
  };

  return (
    <nav className="navigation" aria-label="Main navigation">
      <button
        ref={menuButtonRef}
        className="menu-toggle"
        onClick={toggleMenu}
        aria-expanded={isOpen}
        aria-controls="navigation-menu"
        aria-label="Toggle navigation menu"
      >
        <span className="menu-icon" aria-hidden="true">☰</span>
        <span className="menu-text">Menu</span>
      </button>
      
      <ul
        id="navigation-menu"
        className={`menu ${isOpen ? 'open' : ''}`}
        role="menu"
        aria-hidden={!isOpen}
      >
        {items.map((item, index) => (
          <li key={item.id} role="none">
            {renderMenuItem(item, index)}
          </li>
        ))}
      </ul>
      
      {/* 스크린 리더 전용 알림 */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {isOpen ? 'Menu is open' : 'Menu is closed'}
      </div>
    </nav>
  );
};

export default NavigationMenu;