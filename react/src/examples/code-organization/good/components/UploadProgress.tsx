import React from 'react';

interface UploadProgressProps {
  progress: number;
}

// ✅ 좋은 예시: 업로드 진행률 컴포넌트를 별도 파일로 분리
const UploadProgress = React.memo(({ progress }: UploadProgressProps) => {
  if (progress <= 0 || progress >= 100) {
    return null;
  }

  return (
    <div className="upload-progress">
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${progress}%` }}
        />
      </div>
      <span className="progress-text">{progress}%</span>
    </div>
  );
});

UploadProgress.displayName = 'UploadProgress';

export default UploadProgress;