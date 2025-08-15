import React, { useRef, useEffect } from 'react';

// Utility hooks for accessibility
export const useFocusTrap = (isActive: boolean = true) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          lastElement?.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === lastElement) {
          firstElement?.focus();
          e.preventDefault();
        }
      }
    };

    container.addEventListener('keydown', handleTabKey);
    firstElement?.focus();

    return () => {
      container.removeEventListener('keydown', handleTabKey);
    };
  }, [isActive]);

  return containerRef;
};

// Announcer for screen readers
export const useAnnouncer = () => {
  const announcerRef = useRef<HTMLDivElement>(null);

  const announce = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (announcerRef.current) {
      announcerRef.current.setAttribute('aria-live', priority);
      announcerRef.current.textContent = message;
      
      // Clear after announcement
      setTimeout(() => {
        if (announcerRef.current) {
          announcerRef.current.textContent = '';
        }
      }, 1000);
    }
  };

  const AnnouncerComponent = () => (
    <div
      ref={announcerRef}
      className="sr-only"
      aria-live="polite"
      aria-atomic="true"
    />
  );

  return { announce, AnnouncerComponent };
};

// Focus management hook
export const useFocusManagement = () => {
  const previousFocusRef = useRef<HTMLElement | null>(null);

  const saveFocus = () => {
    previousFocusRef.current = document.activeElement as HTMLElement;
  };

  const restoreFocus = () => {
    if (previousFocusRef.current && typeof previousFocusRef.current.focus === 'function') {
      previousFocusRef.current.focus();
    }
  };

  const focusElement = (selector: string | HTMLElement) => {
    const element = typeof selector === 'string' 
      ? document.querySelector(selector) as HTMLElement
      : selector;
    
    if (element && typeof element.focus === 'function') {
      element.focus();
    }
  };

  return { saveFocus, restoreFocus, focusElement };
};

// Keyboard navigation utilities
export const useKeyboardNavigation = (
  items: Array<{ id: string; element?: HTMLElement }>,
  onSelect?: (id: string) => void
) => {
  const [focusedIndex, setFocusedIndex] = React.useState(0);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setFocusedIndex((prev) => (prev + 1) % items.length);
        break;
      case 'ArrowUp':
        e.preventDefault();
        setFocusedIndex((prev) => (prev - 1 + items.length) % items.length);
        break;
      case 'Home':
        e.preventDefault();
        setFocusedIndex(0);
        break;
      case 'End':
        e.preventDefault();
        setFocusedIndex(items.length - 1);
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (onSelect && items[focusedIndex]) {
          onSelect(items[focusedIndex].id);
        }
        break;
      case 'Escape':
        e.preventDefault();
        // Let parent handle escape
        break;
    }
  };

  useEffect(() => {
    const focusedItem = items[focusedIndex];
    if (focusedItem?.element && typeof focusedItem.element.focus === 'function') {
      focusedItem.element.focus();
    }
  }, [focusedIndex, items]);

  return { focusedIndex, handleKeyDown, setFocusedIndex };
};

// Accessible button component
interface AccessibleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  loadingText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const AccessibleButton: React.FC<AccessibleButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  loadingText = 'Loading...',
  leftIcon,
  rightIcon,
  disabled,
  className = '',
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-apple-sm transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-apple-blue focus:ring-offset-2';
  
  const variantClasses = {
    primary: 'bg-apple-blue text-white hover:bg-blue-600 disabled:bg-gray-300',
    secondary: 'bg-white border border-apple-blue text-apple-blue hover:bg-blue-50 disabled:bg-gray-100',
    ghost: 'bg-transparent text-apple-blue hover:bg-blue-50 disabled:text-gray-400'
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      disabled={disabled || loading}
      aria-disabled={disabled || loading}
      aria-describedby={loading ? 'loading-description' : undefined}
      {...props}
    >
      {loading && (
        <svg
          className="animate-spin -ml-1 mr-2 h-4 w-4"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )}
      
      {!loading && leftIcon && <span className="mr-2">{leftIcon}</span>}
      
      <span>{loading ? loadingText : children}</span>
      
      {!loading && rightIcon && <span className="ml-2">{rightIcon}</span>}
      
      {loading && (
        <span id="loading-description" className="sr-only">
          Loading, please wait
        </span>
      )}
    </button>
  );
};

// Accessible form field component
interface AccessibleFieldProps {
  label: string;
  error?: string;
  hint?: string;
  required?: boolean;
  children: React.ReactElement;
}

export const AccessibleField: React.FC<AccessibleFieldProps> = ({
  label,
  error,
  hint,
  required = false,
  children
}) => {
  const fieldId = React.useId();
  const errorId = `${fieldId}-error`;
  const hintId = `${fieldId}-hint`;

  const clonedChild = React.cloneElement(children, {
    id: fieldId,
    'aria-describedby': [
      hint ? hintId : null,
      error ? errorId : null
    ].filter(Boolean).join(' '),
    'aria-invalid': error ? 'true' : 'false',
    'aria-required': required
  });

  return (
    <div className="space-y-1">
      <label 
        htmlFor={fieldId}
        className="block text-sm font-medium text-text-dark"
      >
        {label}
        {required && <span className="text-error-red ml-1" aria-label="required">*</span>}
      </label>
      
      {hint && (
        <p id={hintId} className="text-sm text-text-medium">
          {hint}
        </p>
      )}
      
      {clonedChild}
      
      {error && (
        <p id={errorId} className="text-sm text-error-red" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

// Skip link component for accessibility
export const SkipLink: React.FC<{ href: string; children: React.ReactNode }> = ({
  href,
  children
}) => (
  <a
    href={href}
    className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-apple-blue text-white px-4 py-2 rounded-apple-sm z-50 focus:outline-none focus:ring-2 focus:ring-white"
  >
    {children}
  </a>
);

export default {
  useFocusTrap,
  useAnnouncer,
  useFocusManagement,
  useKeyboardNavigation,
  AccessibleButton,
  AccessibleField,
  SkipLink
};