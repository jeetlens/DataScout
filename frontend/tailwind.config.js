/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Apple-inspired color palette
        'apple-blue': '#007AFF',
        'apple-blue-light': '#5AC8FA',
        'success-green': '#34C759',
        'warning-orange': '#FF9500',
        'error-red': '#FF3B30',
        'background-white': '#FFFFFF',
        'card-white': '#FAFAFA',
        'border-gray': '#E5E5E7',
        'text-dark': '#1D1D1F',
        'text-medium': '#86868B',
        'text-light': '#F5F5F7',
      },
      fontFamily: {
        'apple': ['-apple-system', 'BlinkMacSystemFont', 'SF Pro Display', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      spacing: {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
        '2xl': '48px',
      },
      borderRadius: {
        'apple-sm': '8px',
        'apple-md': '12px',
        'apple-lg': '16px',
        'apple-xl': '24px',
      },
      boxShadow: {
        'apple-subtle': '0 1px 3px rgba(0, 0, 0, 0.1)',
        'apple-medium': '0 4px 16px rgba(0, 0, 0, 0.1)',
        'apple-large': '0 8px 32px rgba(0, 0, 0, 0.15)',
      },
      animation: {
        'apple-micro': 'ease-out 200ms',
        'apple-standard': 'ease-out 300ms',
        'apple-complex': 'ease-out 500ms',
      },
    },
  },
  plugins: [],
}

