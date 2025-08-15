import React, { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  CloudArrowUpIcon,
  DocumentIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';
import { useAppStore } from '../store/appStore';
import { apiClient, uploadUtils } from '../api/client';

const FileUpload: React.FC = () => {
  const navigate = useNavigate();
  const { 
    addFile, 
    updateFileStatus, 
    updateFileDataId,
    setCurrentStep, 
    addNotification,
    setLoading 
  } = useAppStore();

  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    // Validate file type
    if (!uploadUtils.isValidFileType(file)) {
      addNotification({
        type: 'error',
        message: 'Invalid file type. Please upload CSV, Excel, JSON, or SQLite files.',
      });
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      addNotification({
        type: 'error',
        message: 'File too large. Please upload files smaller than 100MB.',
      });
      return;
    }

    // Create file entry in store
    const fileData = {
      id: Date.now().toString(),
      name: file.name,
      size: file.size,
      type: file.type || uploadUtils.getFileExtension(file.name),
      uploadedAt: new Date(),
      status: 'uploading' as const,
    };

    addFile(fileData);
    setLoading(true);

    try {
      console.log('Starting file upload...'); // Debug
      
      // Upload file with progress tracking
      const response = await apiClient.uploadFile(file, (progress) => {
        setUploadProgress(progress);
      });

      console.log('Upload completed!'); // Debug
      console.log('Raw response:', response); // Debug
      console.log('Response type:', typeof response); // Debug
      console.log('Response constructor:', response?.constructor?.name); // Debug

      // Update file status and store backend data ID
      updateFileStatus(fileData.id, 'uploaded');
      
      // Try multiple ways to access the data ID
      let dataId = response?.data_id || response?.id || response?.dataId;
      
      // If still not found, try bracket notation
      if (!dataId) {
        dataId = response?.['data_id'] || response?.['id'] || response?.['dataId'];
      }
      
      console.log('=== UPLOAD DEBUG START ===');
      console.log('Raw response:', response);
      console.log('Response data_id:', response.data_id);
      console.log('=== UPLOAD DEBUG END ===');

      // Update file status first
      updateFileStatus(fileData.id, 'uploaded');
      
      // Directly use response.data_id - we know it exists from your screenshot
      if (response && response.data_id) {
        console.log('SUCCESS: Found data_id:', response.data_id);
        updateFileDataId(fileData.id, response.data_id);
        
        // Success notification
        addNotification({
          type: 'success',
          message: `File "${file.name}" uploaded successfully!`,
        });
        
        // Navigate to preview step
        setCurrentStep('preview');
        navigate('/preview');
      } else {
        console.error('FAILED: No data_id found');
        console.error('Response keys:', Object.keys(response || {}));
        addNotification({
          type: 'error',
          message: 'Upload successful but no data ID received. Please try again.',
        });
      }

    } catch (error: any) {
      updateFileStatus(fileData.id, 'error');
      addNotification({
        type: 'error',
        message: error.message || 'Failed to upload file. Please try again.',
      });
    } finally {
      setLoading(false);
      setUploadProgress(null);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-4xl mx-auto"
    >
      {/* Header */}
      <div className="text-center mb-12">
        <motion.h1 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-4xl font-semibold text-text-dark mb-4"
        >
          Upload Your Data
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg text-text-medium max-w-2xl mx-auto"
        >
          Get started by uploading your dataset. We support CSV, Excel, JSON, and SQLite files 
          for comprehensive data analysis and AI-powered insights.
        </motion.p>
      </div>

      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
        className="relative"
      >
        <div
          className={`
            relative border-2 border-dashed rounded-apple-lg p-12 text-center
            transition-all duration-300 cursor-pointer
            ${dragActive 
              ? 'border-apple-blue bg-apple-blue/5 scale-[1.02]' 
              : 'border-border-gray hover:border-apple-blue/50 hover:bg-apple-blue/5'
            }
            ${uploadProgress !== null ? 'pointer-events-none' : ''}
          `}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          {/* Upload Icon */}
          <motion.div
            animate={{ 
              scale: dragActive ? 1.1 : 1,
              rotate: dragActive ? 5 : 0 
            }}
            className="mx-auto mb-6"
          >
            <div className="w-20 h-20 apple-gradient-subtle rounded-full flex items-center justify-center mb-6">
              <CloudArrowUpIcon className="w-10 h-10 text-apple-blue" />
            </div>
          </motion.div>

          {/* Upload Text */}
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-text-dark">
              {dragActive ? 'Drop your file here' : 'Choose a file or drag it here'}
            </h3>
            <p className="text-text-medium">
              Supports CSV, Excel, JSON, and SQLite files up to 100MB
            </p>
          </div>

          {/* Progress Bar */}
          {uploadProgress !== null && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6"
            >
              <div className="w-full bg-border-gray/50 rounded-full h-2 mb-2">
                <motion.div
                  className="bg-apple-blue h-2 rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: `${uploadProgress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <p className="text-sm text-apple-blue font-medium">
                Uploading... {uploadProgress}%
              </p>
            </motion.div>
          )}

          {/* Hidden File Input */}
          <input
            id="file-input"
            type="file"
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            onChange={handleFileSelect}
            accept=".csv,.xlsx,.xls,.json,.sqlite,.db"
            disabled={uploadProgress !== null}
          />
        </div>
      </motion.div>

      {/* Supported Formats */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-12"
      >
        <h3 className="text-lg font-semibold text-text-dark mb-6 text-center">
          Supported File Formats
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { name: 'CSV', desc: 'Comma-separated values', ext: '.csv' },
            { name: 'Excel', desc: 'Microsoft Excel files', ext: '.xlsx, .xls' },
            { name: 'JSON', desc: 'JavaScript Object Notation', ext: '.json' },
            { name: 'SQLite', desc: 'SQLite database files', ext: '.sqlite, .db' },
          ].map((format, index) => (
            <motion.div
              key={format.name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              className="card p-4 text-center"
            >
              <DocumentIcon className="w-8 h-8 text-apple-blue mx-auto mb-2" />
              <h4 className="font-semibold text-text-dark">{format.name}</h4>
              <p className="text-sm text-text-medium mb-1">{format.desc}</p>
              <p className="text-xs text-text-medium font-mono">{format.ext}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Feature Highlights */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="mt-12 grid md:grid-cols-3 gap-6"
      >
        {[
          {
            icon: CheckCircleIcon,
            title: 'Automated Analysis',
            desc: 'Get comprehensive statistical analysis and data insights automatically'
          },
          {
            icon: CheckCircleIcon,
            title: 'AI-Powered Insights',
            desc: 'Leverage AI to discover patterns and generate business recommendations'
          },
          {
            icon: CheckCircleIcon,
            title: 'Professional Reports',
            desc: 'Export beautiful HTML and PDF reports for presentations and sharing'
          }
        ].map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 + index * 0.1 }}
            className="text-center"
          >
            <feature.icon className="w-8 h-8 text-success-green mx-auto mb-3" />
            <h4 className="font-semibold text-text-dark mb-2">{feature.title}</h4>
            <p className="text-sm text-text-medium">{feature.desc}</p>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
};

export default FileUpload;