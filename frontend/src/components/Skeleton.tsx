import React from 'react';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  lines?: number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'text',
  width,
  height,
  lines = 1
}) => {
  const baseClasses = 'animate-pulse bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 bg-[length:200%_100%]';
  
  const variantClasses = {
    text: 'h-4 rounded-apple-sm',
    circular: 'rounded-full',
    rectangular: '',
    rounded: 'rounded-apple-md'
  };

  const style = {
    width: width || (variant === 'text' ? '100%' : '40px'),
    height: height || (variant === 'text' ? '1rem' : '40px'),
  };

  if (variant === 'text' && lines > 1) {
    return (
      <div className={`space-y-2 ${className}`}>
        {Array.from({ length: lines }, (_, i) => (
          <div
            key={i}
            className={`${baseClasses} ${variantClasses[variant]}`}
            style={{
              ...style,
              width: i === lines - 1 ? '70%' : '100%', // Last line shorter
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
    />
  );
};

// Skeleton components for specific use cases
export const TableSkeleton: React.FC<{ rows?: number; columns?: number }> = ({
  rows = 5,
  columns = 4
}) => (
  <div className="space-y-3">
    {/* Header */}
    <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
      {Array.from({ length: columns }, (_, i) => (
        <Skeleton key={`header-${i}`} variant="text" height="1.5rem" />
      ))}
    </div>
    
    {/* Rows */}
    {Array.from({ length: rows }, (_, rowIndex) => (
      <div key={`row-${rowIndex}`} className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {Array.from({ length: columns }, (_, colIndex) => (
          <Skeleton key={`cell-${rowIndex}-${colIndex}`} variant="text" />
        ))}
      </div>
    ))}
  </div>
);

export const CardSkeleton: React.FC<{ showAvatar?: boolean }> = ({ showAvatar = false }) => (
  <div className="bg-white p-6 rounded-apple-md shadow-apple-subtle">
    {showAvatar && (
      <div className="flex items-center space-x-4 mb-4">
        <Skeleton variant="circular" width={40} height={40} />
        <Skeleton variant="text" width="60%" />
      </div>
    )}
    <Skeleton variant="text" lines={3} />
    <div className="mt-4 flex space-x-2">
      <Skeleton variant="rectangular" width={80} height={32} className="rounded-apple-sm" />
      <Skeleton variant="rectangular" width={80} height={32} className="rounded-apple-sm" />
    </div>
  </div>
);

export const ChartSkeleton: React.FC<{ height?: string }> = ({ height = '300px' }) => (
  <div className="bg-white p-6 rounded-apple-md shadow-apple-subtle">
    <Skeleton variant="text" width="40%" className="mb-4" />
    <div className="relative" style={{ height }}>
      <Skeleton variant="rectangular" width="100%" height="100%" className="rounded-apple-sm" />
      {/* Simulate chart elements */}
      <div className="absolute bottom-4 left-4 right-4 flex justify-between">
        {Array.from({ length: 5 }, (_, i) => (
          <Skeleton key={i} variant="text" width={20} height={12} />
        ))}
      </div>
    </div>
  </div>
);

export default Skeleton;