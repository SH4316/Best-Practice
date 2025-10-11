import { useState, useCallback } from 'react';

// ✅ 좋은 예시: 장바구니 로직을 분리한 Hook
export const useCart = () => {
  const [cart, setCart] = useState<Record<number, number>>({});

  const addToCart = useCallback((productId: number) => {
    setCart(prevCart => ({
      ...prevCart,
      [productId]: (prevCart[productId] || 0) + 1,
    }));
  }, []);

  const removeFromCart = useCallback((productId: number) => {
    setCart(prevCart => {
      const newCart = { ...prevCart };
      if (newCart[productId] > 1) {
        newCart[productId]--;
      } else {
        delete newCart[productId];
      }
      return newCart;
    });
  }, []);

  const getQuantity = useCallback((productId: number) => {
    return cart[productId] || 0;
  }, [cart]);

  const clearCart = useCallback(() => {
    setCart({});
  }, []);

  return {
    cart,
    addToCart,
    removeFromCart,
    getQuantity,
    clearCart,
  };
};