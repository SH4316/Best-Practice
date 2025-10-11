import { useState, useCallback, useMemo } from 'react';
import { useFilteredProducts } from './hooks/useFilteredProducts';
import { useCart } from './hooks/useCart';
import { useTheme } from './hooks/useTheme';
import SearchControls from './components/SearchControls';
import ProductCard from './components/ProductCard';
import './ProductList.css';

interface Product {
  id: number;
  name: string;
  price: number;
  category: string;
  image: string;
  description: string;
}

// ✅ 좋은 예시: 성능 최적화된 상품 리스트
const ProductList = ({ products }: { products: Product[] }) => {
  const [filter, setFilter] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'price'>('name');
  
  // 커스텀 Hook으로 상태 관리 분리
  const { addToCart, removeFromCart, getQuantity } = useCart();
  const { theme, toggleTheme } = useTheme();
  
  // 커스텀 Hook으로 필터링과 정렬 로직 분리
  const filteredProducts = useFilteredProducts(products, filter, sortBy);
  
  // useCallback으로 함수 참조 안정화
  const handleFilterChange = useCallback((newFilter: string) => {
    setFilter(newFilter);
  }, []);
  
  const handleSortChange = useCallback((newSortBy: 'name' | 'price') => {
    setSortBy(newSortBy);
  }, []);
  
  // useMemo로 렌더링 최적화를 위한 데이터 구조화
  const productCards = useMemo(() => {
    return filteredProducts.map(product => (
      <ProductCard
        key={product.id} // 고유한 ID를 키로 사용
        product={product}
        quantity={getQuantity(product.id)}
        onAddToCart={addToCart}
        onRemoveFromCart={removeFromCart}
      />
    ));
  }, [filteredProducts, getQuantity, addToCart, removeFromCart]);

  return (
    <div className={`product-list ${theme}`}>
      <SearchControls
        filter={filter}
        sortBy={sortBy}
        onFilterChange={handleFilterChange}
        onSortChange={handleSortChange}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <div className="products">
        {filteredProducts.length === 0 ? (
          <div className="empty-state">
            <p>No products found matching your criteria.</p>
          </div>
        ) : (
          <div className="product-grid">
            {productCards}
          </div>
        )}
      </div>
    </div>
  );
};

export default ProductList;