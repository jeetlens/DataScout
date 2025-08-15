import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  DocumentTextIcon,
  ArrowDownTrayIcon,
  EyeIcon,
  ShareIcon,
  DocumentArrowDownIcon,
  ArchiveBoxIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useAppStore, useCurrentFile, useAnalysisConfig } from '../store/appStore';
import { apiClient } from '../api/client';

const ReportsInterface: React.FC = () => {
  const navigate = useNavigate();
  const currentFile = useCurrentFile();
  const analysisConfig = useAnalysisConfig();
  const { addNotification, setLoading } = useAppStore();

  // Local state
  const [downloadingReport, setDownloadingReport] = useState<string | null>(null);
  const [previewingReport, setPreviewingReport] = useState<string | null>(null);

  const reportTypes = [
    {
      id: 'complete',
      title: 'Complete Analysis Report',
      description: 'Comprehensive report with all analysis results, visualizations, and insights',
      icon: DocumentTextIcon,
      formats: ['HTML', 'PDF'],
      estimatedSize: '2-5 MB',
      features: [
        'Executive Summary',
        'Statistical Analysis',
        'Data Visualizations',
        'AI Insights',
        'Recommendations',
        'Data Quality Assessment'
      ],
      recommended: true,
    },
    {
      id: 'executive',
      title: 'Executive Summary',
      description: 'High-level overview perfect for stakeholders and presentations',
      icon: DocumentArrowDownIcon,
      formats: ['HTML', 'PDF'],
      estimatedSize: '500 KB - 1 MB',
      features: [
        'Key Findings',
        'Executive Summary',
        'Critical Insights',
        'Business Recommendations'
      ],
      recommended: false,
    },
    {
      id: 'ai-insights',
      title: 'AI Insights Report',
      description: 'Focus on AI-generated insights and predictive recommendations',
      icon: ArchiveBoxIcon,
      formats: ['HTML', 'PDF'],
      estimatedSize: '1-2 MB',
      features: [
        'AI Pattern Analysis',
        'Predictive Insights',
        'Data Story',
        'Business Intelligence'
      ],
      recommended: false,
    },
  ];

  const handleDownloadReport = async (reportType: string, format: 'html' | 'pdf') => {
    if (!currentFile || !currentFile.dataId) {
      addNotification({
        type: 'error',
        message: 'No data ID found. Please re-upload and analyze your file.',
      });
      return;
    }

    setDownloadingReport(`${reportType}-${format}`);
    setLoading(true);

    try {
      if (format === 'html') {
        const response = await apiClient.generateHTMLReport(currentFile.dataId, reportType);
        // Create blob and download
        const blob = new Blob([response.html_content], { type: 'text/html' });
        apiClient.downloadFile(blob, `${currentFile.name}_${reportType}_report.html`);
      } else {
        const blob = await apiClient.generatePDFReport(currentFile.dataId, reportType);
        apiClient.downloadFile(blob, `${currentFile.name}_${reportType}_report.pdf`);
      }

      addNotification({
        type: 'success',
        message: `${format.toUpperCase()} report downloaded successfully!`,
      });

    } catch (error: any) {
      addNotification({
        type: 'error',
        message: `Failed to download ${format.toUpperCase()} report: ${error.message}`,
      });
    } finally {
      setDownloadingReport(null);
      setLoading(false);
    }
  };

  const handleDownloadPackage = async () => {
    if (!currentFile || !currentFile.dataId) {
      addNotification({
        type: 'error',
        message: 'No data ID found. Please re-upload and analyze your file.',
      });
      return;
    }

    setDownloadingReport('package');
    setLoading(true);

    try {
      const blob = await apiClient.generateReportPackage(currentFile.dataId);
      apiClient.downloadFile(blob, `${currentFile.name}_complete_package.zip`);

      addNotification({
        type: 'success',
        message: 'Complete report package downloaded successfully!',
      });

    } catch (error: any) {
      addNotification({
        type: 'error',
        message: `Failed to download report package: ${error.message}`,
      });
    } finally {
      setDownloadingReport(null);
      setLoading(false);
    }
  };

  const handlePreviewReport = async (reportType: string) => {
    if (!currentFile || !currentFile.dataId) {
      addNotification({
        type: 'error',
        message: 'No data ID found. Please re-upload and analyze your file.',
      });
      return;
    }

    setPreviewingReport(reportType);

    try {
      const response = await apiClient.generateHTMLReport(currentFile.dataId, reportType);
      // Open in new window
      const newWindow = window.open('', '_blank');
      if (newWindow) {
        newWindow.document.write(response.html_content);
        newWindow.document.close();
      }

      addNotification({
        type: 'success',
        message: 'Report preview opened in new window',
      });

    } catch (error: any) {
      addNotification({
        type: 'error',
        message: `Failed to preview report: ${error.message}`,
      });
    } finally {
      setPreviewingReport(null);
    }
  };

  if (!currentFile) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto text-center"
      >
        <ExclamationTriangleIcon className="w-16 h-16 text-warning-orange mx-auto mb-4" />
        <h2 className="text-2xl font-semibold text-text-dark mb-2">
          No Analysis Available
        </h2>
        <p className="text-text-medium mb-6">
          Please complete an analysis first to generate reports.
        </p>
        <button
          onClick={() => navigate('/upload')}
          className="btn-primary"
        >
          Start New Analysis
        </button>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-6xl mx-auto"
    >
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-semibold text-text-dark mb-2">
              Download Reports
            </h1>
            <p className="text-text-medium">
              Generate and download professional reports from your analysis
            </p>
          </div>
          
          <button
            onClick={() => navigate('/results')}
            className="btn-secondary flex items-center space-x-2"
          >
            <EyeIcon className="w-4 h-4" />
            <span>Back to Results</span>
          </button>
        </div>

        {/* Analysis Info */}
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 apple-gradient-subtle rounded-apple-md flex items-center justify-center">
                <DocumentTextIcon className="w-6 h-6 text-apple-blue" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-text-dark">{currentFile.name}</h3>
                <p className="text-text-medium">
                  Analysis Type: {analysisConfig.analysisType === 'complete' ? 'Complete Analysis' : 
                                 analysisConfig.analysisType === 'executive' ? 'Executive Summary' : 'AI-Focused'}
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2 text-success-green">
              <CheckCircleIcon className="w-5 h-5" />
              <span className="font-medium">Analysis Complete</span>
            </div>
          </div>
        </div>
      </div>

      {/* Complete Package Download */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <div className="card p-6 apple-gradient-subtle border-2 border-apple-blue/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-16 h-16 bg-apple-blue/20 rounded-apple-lg flex items-center justify-center">
                <ArchiveBoxIcon className="w-8 h-8 text-apple-blue" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-text-dark mb-1">
                  Complete Report Package
                </h3>
                <p className="text-text-medium mb-2">
                  Download all reports and visualizations in a single ZIP file
                </p>
                <div className="flex items-center space-x-4 text-sm text-text-medium">
                  <span className="flex items-center space-x-1">
                    <ClockIcon className="w-4 h-4" />
                    <span>Includes HTML & PDF formats</span>
                  </span>
                  <span>5-10 MB</span>
                </div>
              </div>
            </div>
            
            <button
              onClick={handleDownloadPackage}
              disabled={downloadingReport === 'package'}
              className="btn-primary flex items-center space-x-2 disabled:opacity-50"
            >
              <ArrowDownTrayIcon className="w-5 h-5" />
              <span>{downloadingReport === 'package' ? 'Generating...' : 'Download Package'}</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Individual Reports */}
      <div className="space-y-6">
        <h2 className="text-xl font-semibold text-text-dark mb-4">
          Individual Reports
        </h2>
        
        {reportTypes.map((report, index) => (
          <motion.div
            key={report.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + index * 0.1 }}
            className={`card p-6 ${report.recommended ? 'border-2 border-apple-blue/20' : ''}`}
          >
            {report.recommended && (
              <div className="inline-block bg-apple-blue text-white text-xs px-3 py-1 rounded-apple-sm mb-4">
                Recommended
              </div>
            )}
            
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4 flex-1">
                <div className="w-12 h-12 bg-apple-blue/10 rounded-apple-md flex items-center justify-center flex-shrink-0">
                  <report.icon className="w-6 h-6 text-apple-blue" />
                </div>
                
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-text-dark mb-2">
                    {report.title}
                  </h3>
                  <p className="text-text-medium mb-4">
                    {report.description}
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-sm font-medium text-text-dark mb-2">Features:</p>
                      <ul className="text-sm text-text-medium space-y-1">
                        {report.features.map((feature, idx) => (
                          <li key={idx} className="flex items-center space-x-2">
                            <CheckCircleIcon className="w-3 h-3 text-success-green flex-shrink-0" />
                            <span>{feature}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <p className="text-sm font-medium text-text-dark mb-2">Details:</p>
                      <div className="text-sm text-text-medium space-y-1">
                        <p>Formats: {report.formats.join(', ')}</p>
                        <p>Size: {report.estimatedSize}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col space-y-2 ml-4">
                <button
                  onClick={() => handlePreviewReport(report.id)}
                  disabled={previewingReport === report.id}
                  className="btn-ghost flex items-center space-x-2 text-sm disabled:opacity-50"
                >
                  <EyeIcon className="w-4 h-4" />
                  <span>{previewingReport === report.id ? 'Loading...' : 'Preview'}</span>
                </button>
                
                <div className="flex space-x-2">
                  {report.formats.map((format) => (
                    <button
                      key={format}
                      onClick={() => handleDownloadReport(report.id, format.toLowerCase() as 'html' | 'pdf')}
                      disabled={downloadingReport === `${report.id}-${format.toLowerCase()}`}
                      className="btn-secondary text-sm px-3 py-2 disabled:opacity-50"
                    >
                      {downloadingReport === `${report.id}-${format.toLowerCase()}` ? (
                        <ClockIcon className="w-4 h-4 animate-spin" />
                      ) : (
                        <ArrowDownTrayIcon className="w-4 h-4" />
                      )}
                      <span className="ml-1">{format}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Help Section */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="mt-12 card p-6 bg-card-white"
      >
        <h3 className="text-lg font-semibold text-text-dark mb-4">
          Report Formats Guide
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-text-dark mb-2">HTML Reports</h4>
            <ul className="text-sm text-text-medium space-y-1">
              <li>• Interactive charts and visualizations</li>
              <li>• Responsive design for all devices</li>
              <li>• Easy to share via web browsers</li>
              <li>• Smaller file sizes</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-text-dark mb-2">PDF Reports</h4>
            <ul className="text-sm text-text-medium space-y-1">
              <li>• Professional print-ready format</li>
              <li>• Perfect for presentations</li>
              <li>• Consistent formatting across devices</li>
              <li>• Easy to archive and email</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-apple-blue/5 rounded-apple-sm">
          <p className="text-sm text-apple-blue">
            <strong>Tip:</strong> Use the Complete Report Package for the most comprehensive analysis, 
            or download individual reports based on your specific needs.
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ReportsInterface;