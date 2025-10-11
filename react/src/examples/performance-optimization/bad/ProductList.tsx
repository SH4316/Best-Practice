import { useState, useEffect } from 'react';

interface Product {
  id: number;
  name: string;
  price: number;
  category: string;
  image: string;
  description: string;
}

// ❌ 나쁜 예시: 성능 최적화가 되지 않은 상품 리스트
const ProductList = ({ products }: { products: Product[] }) => {
  const [cart, setCart] = useState<Record<number, number>>({});
  const [filter, setFilter] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [theme, setTheme] = useState('light');

  // 나쁜 예시: 매 렌더링마다 재계산
  const filteredProducts = products
    .filter(product => 
      product.name.toLowerCase().includes(filter.toLowerCase()) ||
      product.description.toLowerCase().includes(filter.toLowerCase())
    )
    .sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      } else if (sortBy === 'price') {
        return a.price - b.price;
      }
      return 0;
    });

  // 나쁜 예시: 매 렌더링마다 새 함수 생성
  const addToCart = (productId: number) => {
    setCart(prevCart => ({
      ...prevCart,
      [productId]: (prevCart[productId] || 0) + 1,
    }));
  };

  const removeFromCart = (productId: number) => {
    setCart(prevCart => {
      const newCart = { ...prevCart };
      if (newCart[productId] > 1) {
        newCart[productId]--;
      } else {
        delete newCart[productId];
      }
      return newCart;
    });
  };

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  // 나쁜 예시: 비효율적인 이펙트
  useEffect(() => {
    console.log('Products updated');
    console.log('Filter:', filter);
    console.log('Sort by:', sortBy);
    console.log('Theme:', theme);
  });

  return (
    <div className={`product-list ${theme}`}>
      <div className="controls">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Search products..."
        />
        
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
        >
          <option value="name">Sort by Name</option>
          <option value="price">Sort by Price</option>
        </select>
        
        <button onClick={toggleTheme}>
          Toggle Theme ({theme})
        </button>
      </div>

      <div className="products">
        {/* 나쁜 예시: 인덱스를 키로 사용 */}
        {filteredProducts.map((product, index) => (
          <div key={index} className="product-card">
            <img 
              src={product.image} 
              alt={product.name}
              width="200"
              height="200"
            />
            <h3>{product.name}</h3>
            <p className="price">${product.price.toFixed(2)}</p>
            <p className="category">{product.category}</p>
            <p className="description">{product.description}</p>
            
            <div className="cart-controls">
              <span>Quantity: {cart[product.id] || 0}</span>
              <button onClick={() => addToCart(product.id)}>
                Add to Cart
              </button>
              <button 
                onClick={() => removeFromCart(product.id)}
                disabled={!cart[product.id]}
              >
                Remove from Cart
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProductList;