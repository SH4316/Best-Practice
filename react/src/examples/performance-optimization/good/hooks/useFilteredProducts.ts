import { useMemo } from 'react';

interface Product {
  id: number;
  name: string;
  price: number;
  category: string;
  image: string;
  description: string;
}

// ✅ 좋은 예시: 필터링과 정렬 로직을 분리한 Hook
export const useFilteredProducts = (
  products: Product[],
  filter: string,
  sortBy: 'name' | 'price'
) => {
  return useMemo(() => {
    // 필터링
    const filtered = products.filter(product => 
      product.name.toLowerCase().includes(filter.toLowerCase()) ||
      product.description.toLowerCase().includes(filter.toLowerCase())
    );
    
    // 정렬
    return filtered.sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      } else if (sortBy === 'price') {
        return a.price - b.price;
      }
      return 0;
    });
  }, [products, filter, sortBy]);
};