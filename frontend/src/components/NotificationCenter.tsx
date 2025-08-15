import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircleIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { useNotifications, useAppStore } from '../store/appStore';

const NotificationCenter: React.FC = () => {
  const notifications = useNotifications();
  const { removeNotification } = useAppStore();

  // Auto-remove notifications after 5 seconds
  useEffect(() => {
    notifications.forEach((notification) => {
      const timer = setTimeout(() => {
        removeNotification(notification.id);
      }, 5000);

      return () => clearTimeout(timer);
    });
  }, [notifications, removeNotification]);

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return CheckCircleIcon;
      case 'error':
        return ExclamationCircleIcon;
      case 'warning':
        return ExclamationTriangleIcon;
      case 'info':
      default:
        return InformationCircleIcon;
    }
  };

  const getNotificationStyles = (type: string) => {
    switch (type) {
      case 'success':
        return 'bg-success-green/10 border-success-green/20 text-success-green';
      case 'error':
        return 'bg-error-red/10 border-error-red/20 text-error-red';
      case 'warning':
        return 'bg-warning-orange/10 border-warning-orange/20 text-warning-orange';
      case 'info':
      default:
        return 'bg-apple-blue/10 border-apple-blue/20 text-apple-blue';
    }
  };

  return (
    <div className="fixed top-20 right-4 z-50 space-y-2 max-w-sm">
      <AnimatePresence>
        {notifications.map((notification) => {
          const Icon = getNotificationIcon(notification.type);
          const styles = getNotificationStyles(notification.type);

          return (
            <motion.div
              key={notification.id}
              initial={{ opacity: 0, x: 100, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 100, scale: 0.9 }}
              className={`
                bg-white rounded-apple-md shadow-apple-medium border-2 p-4
                ${styles}
              `}
            >
              <div className="flex items-start space-x-3">
                <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
                
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-text-dark">
                    {notification.message}
                  </p>
                  <p className="text-xs text-text-medium mt-1">
                    {new Intl.DateTimeFormat('en-US', {
                      hour: 'numeric',
                      minute: 'numeric',
                      second: 'numeric',
                    }).format(notification.timestamp)}
                  </p>
                </div>

                <button
                  onClick={() => removeNotification(notification.id)}
                  className="flex-shrink-0 p-1 rounded-apple-sm hover:bg-black/5 transition-colors"
                >
                  <XMarkIcon className="w-4 h-4 text-text-medium" />
                </button>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
};

export default NotificationCenter;