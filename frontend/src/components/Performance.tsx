import React, { useCallback, useMemo, useState, useEffect, useRef } from 'react';

// Virtual scrolling hook for large datasets
interface VirtualScrollOptions {
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
}

export const useVirtualScroll = <T,>(
  items: T[],
  options: VirtualScrollOptions
) => {
  const { itemHeight, containerHeight, overscan = 5 } = options;
  const [scrollTop, setScrollTop] = useState(0);

  const visibleStart = Math.floor(scrollTop / itemHeight);
  const visibleEnd = Math.min(
    visibleStart + Math.ceil(containerHeight / itemHeight),
    items.length - 1
  );

  const startIndex = Math.max(0, visibleStart - overscan);
  const endIndex = Math.min(items.length - 1, visibleEnd + overscan);

  const visibleItems = useMemo(
    () => items.slice(startIndex, endIndex + 1),
    [items, startIndex, endIndex]
  );

  const totalHeight = items.length * itemHeight;
  const offsetY = startIndex * itemHeight;

  const scrollElementProps = {
    onScroll: useCallback((e: React.UIEvent<HTMLDivElement>) => {
      setScrollTop(e.currentTarget.scrollTop);
    }, []),
    style: { height: containerHeight, overflow: 'auto' }
  };

  return {
    visibleItems,
    totalHeight,
    offsetY,
    startIndex,
    endIndex,
    scrollElementProps
  };
};

// Lazy loading component for images and content
interface LazyLoadProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  rootMargin?: string;
  threshold?: number;
  className?: string;
}

export const LazyLoad: React.FC<LazyLoadProps> = ({
  children,
  fallback = <div className="animate-pulse bg-gray-200 rounded" />,
  rootMargin = '50px',
  threshold = 0.1,
  className = ''
}) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasLoaded) {
          setIsIntersecting(true);
          setHasLoaded(true);
        }
      },
      { rootMargin, threshold }
    );

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [rootMargin, threshold, hasLoaded]);

  return (
    <div ref={elementRef} className={className}>
      {isIntersecting ? children : fallback}
    </div>
  );
};

// Memoized table component for large datasets
interface VirtualTableProps {
  columns: Array<{
    key: string;
    label: string;
    width?: string;
    render?: (value: any, row: any) => React.ReactNode;
  }>;
  data: any[];
  rowHeight?: number;
  maxHeight?: number;
  className?: string;
}

export const VirtualTable: React.FC<VirtualTableProps> = React.memo(({
  columns,
  data,
  rowHeight = 50,
  maxHeight = 400,
  className = ''
}) => {
  const {
    visibleItems,
    totalHeight,
    offsetY,
    scrollElementProps
  } = useVirtualScroll(data, {
    itemHeight: rowHeight,
    containerHeight: maxHeight
  });

  const renderCell = useCallback((column: any, row: any) => {
    const value = row[column.key];
    return column.render ? column.render(value, row) : value;
  }, []);

  return (
    <div className={`border border-border-gray rounded-apple-md overflow-hidden ${className}`}>
      {/* Header */}
      <div className="bg-card-white border-b border-border-gray">
        <div className="grid gap-4 px-4 py-3" style={{
          gridTemplateColumns: columns.map(col => col.width || '1fr').join(' ')
        }}>
          {columns.map((column) => (
            <div key={column.key} className="font-medium text-text-dark text-sm">
              {column.label}
            </div>
          ))}
        </div>
      </div>

      {/* Virtual scrolling body */}
      <div {...scrollElementProps}>
        <div style={{ height: totalHeight, position: 'relative' }}>
          <div style={{ transform: `translateY(${offsetY}px)` }}>
            {visibleItems.map((row, index) => (
              <div
                key={index}
                className="grid gap-4 px-4 py-3 border-b border-border-gray hover:bg-card-white transition-colors"
                style={{
                  gridTemplateColumns: columns.map(col => col.width || '1fr').join(' '),
                  height: rowHeight
                }}
              >
                {columns.map((column) => (
                  <div key={column.key} className="text-sm text-text-dark flex items-center">
                    {renderCell(column, row)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});

VirtualTable.displayName = 'VirtualTable';

// Debounced search hook
export const useDebouncedSearch = (
  initialValue: string = '',
  delay: number = 300
) => {
  const [searchTerm, setSearchTerm] = useState(initialValue);
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState(initialValue);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [searchTerm, delay]);

  return {
    searchTerm,
    debouncedSearchTerm,
    setSearchTerm
  };
};

// Performance monitoring hook
export const usePerformanceMonitor = (componentName: string) => {
  const renderStartTime = useRef<number>();
  const renderCount = useRef(0);

  useEffect(() => {
    renderStartTime.current = performance.now();
    renderCount.current += 1;

    return () => {
      if (renderStartTime.current) {
        const renderTime = performance.now() - renderStartTime.current;
        if (process.env.NODE_ENV === 'development') {
          console.log(
            `${componentName} render #${renderCount.current}: ${renderTime.toFixed(2)}ms`
          );
        }
      }
    };
  });

  const measureOperation = useCallback(<T,>(
    operationName: string,
    operation: () => T
  ): T => {
    const start = performance.now();
    const result = operation();
    const end = performance.now();
    
    if (process.env.NODE_ENV === 'development') {
      console.log(`${componentName} - ${operationName}: ${(end - start).toFixed(2)}ms`);
    }
    
    return result;
  }, [componentName]);

  return { measureOperation };
};

// Optimized pagination hook
export const usePagination = <T,>(
  data: T[],
  itemsPerPage: number = 10
) => {
  const [currentPage, setCurrentPage] = useState(1);

  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return data.slice(startIndex, endIndex);
  }, [data, currentPage, itemsPerPage]);

  const totalPages = Math.ceil(data.length / itemsPerPage);
  const hasNextPage = currentPage < totalPages;
  const hasPreviousPage = currentPage > 1;

  const goToPage = useCallback((page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  }, [totalPages]);

  const nextPage = useCallback(() => {
    if (hasNextPage) {
      setCurrentPage(prev => prev + 1);
    }
  }, [hasNextPage]);

  const previousPage = useCallback(() => {
    if (hasPreviousPage) {
      setCurrentPage(prev => prev - 1);
    }
  }, [hasPreviousPage]);

  const resetPagination = useCallback(() => {
    setCurrentPage(1);
  }, []);

  return {
    currentPage,
    totalPages,
    paginatedData,
    hasNextPage,
    hasPreviousPage,
    goToPage,
    nextPage,
    previousPage,
    resetPagination
  };
};

// Memoized chart component wrapper
export const MemoizedChart = React.memo<{
  data: any;
  type: string;
  options?: any;
  className?: string;
}>(({ data, type, options, className }) => {
  const chartId = useMemo(() => `chart-${Date.now()}-${Math.random()}`, []);
  
  // Memoize expensive chart calculations
  const processedData = useMemo(() => {
    if (!data) return null;
    
    // Add your chart data processing logic here
    return data;
  }, [data]);

  const memoizedOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      },
    },
    ...options
  }), [options]);

  if (!processedData) {
    return (
      <div className={`flex items-center justify-center h-64 bg-card-white rounded-apple-md ${className}`}>
        <div className="text-text-medium">No data available</div>
      </div>
    );
  }

  return (
    <div id={chartId} className={`bg-white p-4 rounded-apple-md shadow-apple-subtle ${className}`}>
      {/* Chart implementation would go here */}
      <div className="h-64 flex items-center justify-center bg-card-white rounded">
        <div className="text-text-medium">Chart: {type}</div>
      </div>
    </div>
  );
});

MemoizedChart.displayName = 'MemoizedChart';

export default {
  useVirtualScroll,
  LazyLoad,
  VirtualTable,
  useDebouncedSearch,
  usePerformanceMonitor,
  usePagination,
  MemoizedChart
};