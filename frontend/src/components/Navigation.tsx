import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  DocumentArrowUpIcon, 
  EyeIcon, 
  CogIcon, 
  ChartPieIcon,
  DocumentTextIcon 
} from '@heroicons/react/24/outline';
import { useAppStore } from '../store/appStore';

const Navigation: React.FC = () => {
  const location = useLocation();
  const { currentFile, currentStep } = useAppStore();

  const navItems = [
    {
      path: '/upload',
      label: 'Upload',
      icon: DocumentArrowUpIcon,
      step: 'upload' as const,
      enabled: true,
    },
    {
      path: '/preview',
      label: 'Preview',
      icon: EyeIcon,
      step: 'preview' as const,
      enabled: !!currentFile,
    },
    {
      path: '/configure',
      label: 'Configure',
      icon: CogIcon,
      step: 'configure' as const,
      enabled: !!currentFile,
    },
    {
      path: '/results',
      label: 'Results',
      icon: ChartPieIcon,
      step: 'results' as const,
      enabled: !!currentFile,
    },
    {
      path: '/reports',
      label: 'Reports',
      icon: DocumentTextIcon,
      step: 'reports' as const,
      enabled: !!currentFile,
    },
  ];

  return (
    <nav className="bg-white border-b border-border-gray sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3">
            <div className="w-8 h-8 apple-gradient rounded-apple-sm flex items-center justify-center">
              <ChartBarIcon className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-semibold text-text-dark">
              DataScout
            </h1>
          </Link>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              const isEnabled = item.enabled;
              const Icon = item.icon;

              return (
                <div key={item.path} className="relative">
                  <Link
                    to={item.path}
                    className={`
                      relative px-4 py-2 rounded-apple-sm font-medium text-sm
                      transition-all duration-200 flex items-center space-x-2
                      ${isEnabled 
                        ? isActive
                          ? 'text-apple-blue bg-apple-blue/10'
                          : 'text-text-medium hover:text-apple-blue hover:bg-apple-blue/5'
                        : 'text-text-light cursor-not-allowed'
                      }
                    `}
                    onClick={(e) => !isEnabled && e.preventDefault()}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                    
                    {/* Active indicator */}
                    {isActive && isEnabled && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute inset-0 bg-apple-blue/10 rounded-apple-sm -z-10"
                        initial={false}
                        transition={{
                          type: "spring",
                          stiffness: 500,
                          damping: 30
                        }}
                      />
                    )}
                  </Link>
                </div>
              );
            })}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className="p-2 rounded-apple-sm text-text-medium hover:text-apple-blue hover:bg-apple-blue/5">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Progress Indicator */}
        {currentFile && (
          <div className="pb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-text-dark">
                Analysis Progress
              </span>
              <span className="text-sm text-text-medium">
                {currentFile.name}
              </span>
            </div>
            
            <div className="w-full bg-border-gray/50 rounded-full h-1">
              <motion.div
                className="bg-apple-blue h-1 rounded-full"
                initial={{ width: '20%' }}
                animate={{
                  width: currentStep === 'upload' ? '20%' 
                    : currentStep === 'preview' ? '40%'
                    : currentStep === 'configure' ? '60%'
                    : currentStep === 'results' ? '80%'
                    : '100%'
                }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden border-t border-border-gray">
        <div className="flex overflow-x-auto">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const isEnabled = item.enabled;
            const Icon = item.icon;

            return (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  flex-shrink-0 flex flex-col items-center px-4 py-3 text-xs
                  ${isEnabled 
                    ? isActive
                      ? 'text-apple-blue'
                      : 'text-text-medium'
                    : 'text-text-light'
                  }
                `}
                onClick={(e) => !isEnabled && e.preventDefault()}
              >
                <Icon className="w-5 h-5 mb-1" />
                <span>{item.label}</span>
                {isActive && isEnabled && (
                  <div className="w-4 h-0.5 bg-apple-blue rounded-full mt-1" />
                )}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;