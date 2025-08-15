import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useIsLoading } from '../store/appStore';

const LoadingOverlay: React.FC = () => {
  const isLoading = useIsLoading();

  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-center justify-center"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-white rounded-apple-lg p-8 shadow-apple-large max-w-sm mx-4"
          >
            <div className="flex flex-col items-center space-y-4">
              {/* Apple-style loading spinner */}
              <div className="relative w-12 h-12">
                <div className="absolute inset-0 border-4 border-border-gray rounded-full"></div>
                <div className="absolute inset-0 border-4 border-apple-blue border-t-transparent rounded-full animate-spin"></div>
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold text-text-dark mb-1">
                  Processing...
                </h3>
                <p className="text-text-medium text-sm">
                  Please wait while we analyze your data
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default LoadingOverlay;