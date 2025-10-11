import { useState } from 'react';

// 타입 정의
interface MenuItem {
  label: string;
  icon: React.ReactNode;
  href?: string;
}

// ❌ 나쁜 예시: 접근성이 부족한 내비게이션 메뉴
const NavigationMenu = ({ items }: { items: MenuItem[] }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="navigation">
      <div 
        className="menu-toggle"
        onClick={() => setIsOpen(!isOpen)}
      >
        ☰
      </div>
      
      <ul className={`menu ${isOpen ? 'open' : ''}`}>
        {items.map((item: MenuItem, index: number) => (
          <li key={index}>
            <div 
              className="menu-item"
              onClick={() => {
                // 메뉴 항목 클릭 처리
                console.log(`Clicked ${item.label}`);
              }}
            >
              {item.icon}
              <span>{item.label}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default NavigationMenu;