# NumPy 이미지 처리 응용

## 이미지 처리에서의 NumPy

NumPy는 이미지 처리의 기본 라이브러리로, 픽셀 데이터를 배열로 다루고 다양한 이미지 연산을 수행할 수 있습니다. OpenCV, PIL/Pillow, Scikit-image 등 다른 이미지 처리 라이브러리의 기반이 됩니다.

## 기본 이미지 연산

### 이미지 생성과 기본 속성

```python
import numpy as np

# 간단한 이미지 생성
width, height = 200, 200

# 검은색 이미지 (모든 픽셀이 0)
black_image = np.zeros((height, width), dtype=np.uint8)

# 흰색 이미지 (모든 픽셀이 255)
white_image = np.ones((height, width), dtype=np.uint8) * 255

# 회색 이미지 (모든 픽셀이 128)
gray_image = np.ones((height, width), dtype=np.uint8) * 128

# 컬러 이미지 (RGB)
color_image = np.zeros((height, width, 3), dtype=np.uint8)
color_image[:, :, 0] = 255  # 빨간색 채널
color_image[:, :, 1] = 0    # 초록색 채널
color_image[:, :, 2] = 0    # 파란색 채널

print("이미지 기본 속성:")
print(f"흑백 이미지 형태: {black_image.shape}")
print(f"컬러 이미지 형태: {color_image.shape}")
print(f"데이터 타입: {black_image.dtype}")
print(f"픽셀 값 범위: {np.min(black_image)} - {np.max(white_image)}")

# 그라데이션 이미지
gradient = np.zeros((height, width), dtype=np.uint8)
for i in range(width):
    gradient[:, i] = int(255 * i / width)

print(f"그라데이션 이미지 픽셀 값 범위: {np.min(gradient)} - {np.max(gradient)}")
```

### 이미지 변환

```python
# 이미지 회전
def rotate_image(image, angle_deg):
    """이미지 회전 (단순 구현)"""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # 회전 행렬
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # 새 이미지 크기 계산
    new_width = int(abs(width * cos_angle) + abs(height * sin_angle))
    new_height = int(abs(width * sin_angle) + abs(height * cos_angle))
    
    # 새 이미지 생성
    if len(image.shape) == 3:  # 컬러 이미지
        rotated = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:  # 흑백 이미지
        rotated = np.zeros((new_height, new_width), dtype=image.dtype)
    
    # 회전 변환
    for y in range(new_height):
        for x in range(new_width):
            # 원본 이미지 좌표 계산
            x_orig = (x - new_width // 2) * cos_angle + (y - new_height // 2) * sin_angle + center_x
            y_orig = -(x - new_width // 2) * sin_angle + (y - new_height // 2) * cos_angle + center_y
            
            # 경계 확인
            if 0 <= x_orig < width and 0 <= y_orig < height:
                # 최근접 이웃 보간
                x_int, y_int = int(x_orig), int(y_orig)
                rotated[y, x] = image[y_int, x_int]
    
    return rotated

# 이미지 크기 조절
def resize_image(image, scale_factor):
    """이미지 크기 조절 (최근접 이웃 보간)"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    if len(image.shape) == 3:  # 컬러 이미지
        resized = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:  # 흑백 이미지
        resized = np.zeros((new_height, new_width), dtype=image.dtype)
    
    for y in range(new_height):
        for x in range(new_width):
            # 원본 이미지 좌표
            x_orig = int(x / scale_factor)
            y_orig = int(y / scale_factor)
            
            # 경계 확인
            if 0 <= x_orig < width and 0 <= y_orig < height:
                resized[y, x] = image[y_orig, x_orig]
    
    return resized

# 테스트 이미지 생성
test_image = np.zeros((100, 100), dtype=np.uint8)
test_image[25:75, 25:75] = 255  # 중앙에 흰색 사각형

# 이미지 변환
rotated = rotate_image(test_image, 45)
resized = resize_image(test_image, 2.0)

print("이미지 변환:")
print(f"원본 이미지 크기: {test_image.shape}")
print(f"회전된 이미지 크기: {rotated.shape}")
print(f"크기 조절된 이미지 크기: {resized.shape}")
```

### 이미지 필터

```python
# 컨볼루션 필터
def apply_filter(image, kernel):
    """컨볼루션 필터 적용"""
    if len(image.shape) == 3:  # 컬러 이미지
        height, width, channels = image.shape
        filtered = np.zeros_like(image)
        
        for c in range(channels):
            filtered[:, :, c] = apply_filter_grayscale(image[:, :, c], kernel)
        
        return filtered
    else:  # 흑백 이미지
        return apply_filter_grayscale(image, kernel)

def apply_filter_grayscale(image, kernel):
    """흑백 이미지에 컨볼루션 필터 적용"""
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    
    # 경계 처리를 위한 패딩
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # 결과 이미지
    filtered = np.zeros_like(image)
    
    # 컨볼루션 연산
    for y in range(height):
        for x in range(width):
            region = padded[y:y+k_height, x:x+k_width]
            filtered[y, x] = np.sum(region * kernel)
    
    # 결과를 유효한 범위로 클리핑
    filtered = np.clip(filtered, 0, 255).astype(image.dtype)
    
    return filtered

# 일반적인 필터 커널
# 가우시안 블러
gaussian_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16

# 샤프닝 필터
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# 엣지 검출 (소벨)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# 테스트 이미지 생성
test_image = np.zeros((100, 100), dtype=np.uint8)
# 원 그리기
center_y, center_x = 50, 50
for y in range(100):
    for x in range(100):
        if (y - center_y)**2 + (x - center_x)**2 <= 400:  # 반지름 20
            test_image[y, x] = 255

# 필터 적용
blurred = apply_filter(test_image, gaussian_kernel)
sharpened = apply_filter(test_image, sharpen_kernel)
edges_x = apply_filter(test_image, sobel_x)
edges_y = apply_filter(test_image, sobel_y)
edges = np.sqrt(edges_x.astype(float)**2 + edges_y.astype(float)**2).astype(np.uint8)

print("이미지 필터 적용:")
print(f"원본 이미지 픽셀 값 범위: {np.min(test_image)} - {np.max(test_image)}")
print(f"블러 이미지 픽셀 값 범위: {np.min(blurred)} - {np.max(blurred)}")
print(f"샤프 이미지 픽셀 값 범위: {np.min(sharpened)} - {np.max(sharpened)}")
print(f"엣지 이미지 픽셀 값 범위: {np.min(edges)} - {np.max(edges)}")
```

## 색공간 변환

### RGB와 그레이스케일 변환

```python
# RGB를 그레이스케일로 변환
def rgb_to_grayscale(rgb_image):
    """RGB 이미지를 그레이스케일로 변환"""
    # 가중 평균: 0.299*R + 0.587*G + 0.114*B
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

# 그레이스케일을 RGB로 변환
def grayscale_to_rgb(gray_image):
    """그레이스케일 이미지를 RGB로 변환"""
    return np.stack([gray_image] * 3, axis=-1)

# HSV 변환 (단순 구현)
def rgb_to_hsv(rgb_image):
    """RGB를 HSV로 변환 (단순 구현)"""
    # 정규화
    rgb = rgb_image.astype(float) / 255.0
    
    # 최대값과 최소값
    max_val = np.max(rgb, axis=2)
    min_val = np.min(rgb, axis=2)
    diff = max_val - min_val
    
    # HSV 이미지 초기화
    hsv = np.zeros_like(rgb)
    
    # V (Value)
    hsv[:, :, 2] = max_val
    
    # S (Saturation)
    mask = max_val > 0
    hsv[:, :, 1][mask] = diff[mask] / max_val[mask]
    
    # H (Hue)
    # 간단한 구현 (실제로는 더 복잡함)
    mask = diff > 0
    hsv[:, :, 0][mask] = np.where(
        rgb[:, :, 0][mask] == max_val[mask],
        (rgb[:, :, 1][mask] - rgb[:, :, 2][mask]) / diff[mask],
        np.where(
            rgb[:, :, 1][mask] == max_val[mask],
            2.0 + (rgb[:, :, 2][mask] - rgb[:, :, 0][mask]) / diff[mask],
            4.0 + (rgb[:, :, 0][mask] - rgb[:, :, 1][mask]) / diff[mask]
        )
    )
    hsv[:, :, 0] = (hsv[:, :, 0] / 6.0) % 1.0
    
    return hsv

# 테스트 이미지 생성
rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
# 빨간색 사각형
rgb_image[20:40, 20:40] = [255, 0, 0]
# 초록색 사각형
rgb_image[60:80, 20:40] = [0, 255, 0]
# 파란색 사각형
rgb_image[20:40, 60:80] = [0, 0, 255]
# 흰색 사각형
rgb_image[60:80, 60:80] = [255, 255, 255]

# 색공간 변환
gray_image = rgb_to_grayscale(rgb_image)
hsv_image = rgb_to_hsv(rgb_image)

print("색공간 변환:")
print(f"RGB 이미지 형태: {rgb_image.shape}")
print(f"그레이스케일 이미지 형태: {gray_image.shape}")
print(f"HSV 이미지 형태: {hsv_image.shape}")
print(f"RGB 픽셀 값 범위: {np.min(rgb_image)} - {np.max(rgb_image)}")
print(f"그레이스케일 픽셀 값 범위: {np.min(gray_image)} - {np.max(gray_image)}")
print(f"HSV 픽셀 값 범위: {np.min(hsv_image):.3f} - {np.max(hsv_image):.3f}")
```

## 이미지 분석

### 히스토그램 분석

```python
# 이미지 히스토그램 계산
def histogram(image, bins=256):
    """이미지 히스토그램 계산"""
    if len(image.shape) == 3:  # 컬러 이미지
        # 각 채널별 히스토그램
        histograms = []
        for c in range(image.shape[2]):
            hist, _ = np.histogram(image[:, :, c], bins=bins, range=(0, 256))
            histograms.append(hist)
        return histograms
    else:  # 흑백 이미지
        hist, _ = np.histogram(image, bins=bins, range=(0, 256))
        return hist

# 히스토그램 평활화
def histogram_equalization(image):
    """히스토그램 평활화"""
    if len(image.shape) == 3:  # 컬러 이미지
        # YUV 색공간으로 변환하여 Y 채널만 평활화
        yuv = np.zeros_like(image, dtype=float)
        yuv[:, :, 0] = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        yuv[:, :, 1] = 0.492 * (image[:, :, 2] - yuv[:, :, 0])
        yuv[:, :, 2] = 0.877 * (image[:, :, 0] - yuv[:, :, 0])
        
        # Y 채널 평활화
        y_eq = histogram_equalization_grayscale(yuv[:, :, 0].astype(np.uint8))
        
        # 다시 RGB로 변환 (단순 구현)
        equalized = np.zeros_like(image)
        equalized[:, :, 0] = np.clip(y_eq + 1.13983 * yuv[:, :, 2], 0, 255)
        equalized[:, :, 1] = np.clip(y_eq - 0.39465 * yuv[:, :, 1] - 0.58060 * yuv[:, :, 2], 0, 255)
        equalized[:, :, 2] = np.clip(y_eq + 2.03211 * yuv[:, :, 1], 0, 255)
        
        return equalized.astype(np.uint8)
    else:  # 흑백 이미지
        return histogram_equalization_grayscale(image)

def histogram_equalization_grayscale(image):
    """흑백 이미지 히스토그램 평활화"""
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # 누적 분포 함수
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # 보간
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    return equalized.reshape(image.shape).astype(np.uint8)

# 테스트 이미지 생성
np.random.seed(42)
dark_image = np.random.randint(0, 100, (100, 100), dtype=np.uint8)  # 어두운 이미지

# 히스토그램 분석
hist_dark = histogram(dark_image)
equalized_dark = histogram_equalization(dark_image)
hist_equalized = histogram(equalized_dark)

print("히스토그램 분석:")
print(f"원본 이미지 픽셀 값 범위: {np.min(dark_image)} - {np.max(dark_image)}")
print(f"평활화된 이미지 픽셀 값 범위: {np.min(equalized_dark)} - {np.max(equalized_dark)}")
print(f"원본 히스토그램 최대 빈도: {np.max(hist_dark)}")
print(f"평활화된 히스토그램 최대 빈도: {np.max(hist_equalized)}")
```

### 엣지 검출

```python
# 캐니 엣지 검출 (단순 구현)
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """캐니 엣지 검출 (단순 구현)"""
    # 1. 가우시안 블러
    gaussian_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16
    blurred = apply_filter_grayscale(image, gaussian_kernel)
    
    # 2. 그래디언트 계산 (소벨)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    grad_x = apply_filter_grayscale(blurred, sobel_x)
    grad_y = apply_filter_grayscale(blurred, sobel_y)
    
    # 그래디언트 크기와 방향
    magnitude = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # 3. 비최대값 억제 (단순 구현)
    suppressed = np.zeros_like(magnitude)
    
    for y in range(1, magnitude.shape[0] - 1):
        for x in range(1, magnitude.shape[1] - 1):
            angle = direction[y, x] * 180 / np.pi
            if angle < 0:
                angle += 180
            
            # 방향에 따른 이웃 픽셀
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[y, x-1], magnitude[y, x+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[y-1, x-1], magnitude[y+1, x+1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[y-1, x], magnitude[y+1, x]]
            else:  # 112.5 <= angle < 157.5
                neighbors = [magnitude[y-1, x+1], magnitude[y+1, x-1]]
            
            if magnitude[y, x] >= max(neighbors):
                suppressed[y, x] = magnitude[y, x]
    
    # 4. 이중 임계값 처리
    strong_edges = suppressed > high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    
    # 5. 엣지 연결 (단순 구현)
    edges = np.zeros_like(suppressed)
    edges[strong_edges] = 255
    
    # 약한 엣지 중 강한 엣지와 연결된 것만 유지
    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):
            if weak_edges[y, x]:
                # 8-이웃 확인
                neighborhood = strong_edges[y-1:y+2, x-1:x+2]
                if np.any(neighborhood):
                    edges[y, x] = 255
    
    return edges.astype(np.uint8)

# 테스트 이미지 생성
test_image = np.zeros((100, 100), dtype=np.uint8)
# 사각형 그리기
test_image[20:80, 20:80] = 255
# 원 그리기
center_y, center_x = 50, 50
for y in range(100):
    for x in range(100):
        if 35 < (y - center_y)**2 + (x - center_x)**2 < 45:
            test_image[y, x] = 255

# 엣지 검출
edges = canny_edge_detection(test_image)

print("엣지 검출:")
print(f"원본 이미지 픽셀 값 범위: {np.min(test_image)} - {np.max(test_image)}")
print(f"엣지 이미지 픽셀 값 범위: {np.min(edges)} - {np.max(edges)}")
print(f"엣지 픽셀 비율: {np.sum(edges > 0) / edges.size:.4f}")
```

## 실용적인 이미지 처리 예제

### 노이즈 제거

```python
# 소금-후추 노이즈 추가
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """소금-후추 노이즈 추가"""
    noisy = image.copy()
    
    # 소금 노이즈 (흰색 픽셀)
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy[salt_mask] = 255
    
    # 후추 노이즈 (검은색 픽셀)
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy[pepper_mask] = 0
    
    return noisy

# 중앙값 필터 (노이즈 제거)
def median_filter(image, kernel_size=3):
    """중앙값 필터"""
    if len(image.shape) == 3:  # 컬러 이미지
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = median_filter_grayscale(image[:, :, c], kernel_size)
        return filtered
    else:  # 흑백 이미지
        return median_filter_grayscale(image, kernel_size)

def median_filter_grayscale(image, kernel_size=3):
    """흑백 이미지 중앙값 필터"""
    height, width = image.shape
    pad = kernel_size // 2
    
    # 경계 처리를 위한 패딩
    padded = np.pad(image, pad, mode='reflect')
    
    # 결과 이미지
    filtered = np.zeros_like(image)
    
    # 중앙값 필터 적용
    for y in range(height):
        for x in range(width):
            region = padded[y:y+kernel_size, x:x+kernel_size]
            filtered[y, x] = np.median(region)
    
    return filtered

# 테스트
clean_image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
noisy_image = add_salt_pepper_noise(clean_image, 0.05, 0.05)
denoised_image = median_filter(noisy_image)

print("노이즈 제거:")
print(f"원본 이미지 픽셀 값 범위: {np.min(clean_image)} - {np.max(clean_image)}")
print(f"노이즈 이미지 픽셀 값 범위: {np.min(noisy_image)} - {np.max(noisy_image)}")
print(f"노이즈 제거 이미지 픽셀 값 범위: {np.min(denoised_image)} - {np.max(denoised_image)}")
print(f"노이즈 픽셀 비율: {np.sum((noisy_image == 0) | (noisy_image == 255)) / noisy_image.size:.4f}")
```

### 형태학적 연산

```python
# 이진화
def threshold(image, thresh=128):
    """이진화"""
    return (image > thresh).astype(np.uint8) * 255

# 침식 (Erosion)
def erosion(image, kernel_size=3):
    """침식 연산"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    height, width = image.shape
    pad = kernel_size // 2
    
    padded = np.pad(image, pad, mode='constant')
    eroded = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            region = padded[y:y+kernel_size, x:x+kernel_size]
            if np.all(region == 255):
                eroded[y, x] = 255
    
    return eroded

# 팽창 (Dilation)
def dilation(image, kernel_size=3):
    """팽창 연산"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    height, width = image.shape
    pad = kernel_size // 2
    
    padded = np.pad(image, pad, mode='constant')
    dilated = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            region = padded[y:y+kernel_size, x:x+kernel_size]
            if np.any(region == 255):
                dilated[y, x] = 255
    
    return dilated

# 열기 (Opening)
def opening(image, kernel_size=3):
    """열기 연산 (침식 후 팽창)"""
    return dilation(erosion(image, kernel_size), kernel_size)

# 닫기 (Closing)
def closing(image, kernel_size=3):
    """닫기 연산 (팽창 후 침식)"""
    return erosion(dilation(image, kernel_size), kernel_size)

# 테스트
test_image = np.zeros((100, 100), dtype=np.uint8)
# 큰 사각형
test_image[30:70, 30:70] = 255
# 작은 노이즈 사각형들
test_image[10:15, 10:15] = 255
test_image[85:90, 85:90] = 255
test_image[10:15, 85:90] = 255
test_image[85:90, 10:15] = 255

# 형태학적 연산
opened = opening(test_image, kernel_size=5)
closed = closing(test_image, kernel_size=5)

print("형태학적 연산:")
print(f"원본 이미지 흰색 픽셀 수: {np.sum(test_image == 255)}")
print(f"열기 연산 후 흰색 픽셀 수: {np.sum(opened == 255)}")
print(f"닫기 연산 후 흰색 픽셀 수: {np.sum(closed == 255)}")
```

## 이미지 처리 모범 사례

1. **데이터 타입 관리**: 이미지 처리 시 적절한 데이터 타입 사용 (uint8, float32 등)
2. **경계 처리**: 컨볼루션 연산 시 경계 처리 방법 고려
3. **메모리 효율성**: 대용량 이미지는 블록 단위로 처리
4. **성능 최적화**: 벡터화 연산으로 반복문 최소화
5. **결과 클리핑**: 연산 결과를 유효한 픽셀 값 범위로 클리핑

## 다음 학습 내용

다음으로는 베스트 프랙티스에 대해 알아보겠습니다. [`../07-best-practices/common-pitfalls.md`](../07-best-practices/common-pitfalls.md)를 참조하세요.