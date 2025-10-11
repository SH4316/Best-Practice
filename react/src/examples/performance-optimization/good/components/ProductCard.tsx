import React from 'react';

interface Product {
  id: number;
  name: string;
  price: number;
  category: string;
  image: string;
  description: string;
}

interface ProductCardProps {
  product: Product;
  quantity: number;
  onAddToCart: (productId: number) => void;
  onRemoveFromCart: (productId: number) => void;
}

// ✅ 좋은 예시: React.memo로 리렌더링 최적화
const ProductCard = React.memo(({
  product,
  quantity,
  onAddToCart,
  onRemoveFromCart,
}: ProductCardProps) => {
  return (
    <div className="product-card">
      <LazyImage 
        src={product.image} 
        alt={product.name}
        width={200}
        height={200}
      />
      <h3>{product.name}</h3>
      <p className="price">${product.price.toFixed(2)}</p>
      <p className="category">{product.category}</p>
      <p className="description">{product.description}</p>
      
      <div className="cart-controls">
        <span>Quantity: {quantity}</span>
        <button onClick={() => onAddToCart(product.id)}>
          Add to Cart
        </button>
        <button 
          onClick={() => onRemoveFromCart(product.id)}
          disabled={quantity === 0}
        >
          Remove from Cart
        </button>
      </div>
    </div>
  );
});

// 커스텀 비교 함수로 불필요한 리렌더링 방지
ProductCard.displayName = 'ProductCard';

export default ProductCard;

// ✅ 좋은 예시: 지연 로딩 이미지 컴포넌트
interface LazyImageProps {
  src: string;
  alt: string;
  width: number;
  height: number;
}

const LazyImage = ({ src, alt, width, height }: LazyImageProps) => {
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [isInView, setIsInView] = React.useState(false);
  const imgRef = React.useRef<HTMLImageElement>(null);
  
  React.useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    
    if (imgRef.current) {
      observer.observe(imgRef.current);
    }
    
    return () => observer.disconnect();
  }, []);
  
  return (
    <div ref={imgRef} className="lazy-image-container">
      {isInView && (
        <img
          src={src}
          alt={alt}
          width={width}
          height={height}
          onLoad={() => setIsLoaded(true)}
          style={{ opacity: isLoaded ? 1 : 0 }}
        />
      )}
    </div>
  );
};