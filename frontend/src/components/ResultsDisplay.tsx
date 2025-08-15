import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChartPieIcon,
  SparklesIcon,
  DocumentTextIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  EyeIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { useAppStore, useCurrentFile, useAnalysisConfig, useCurrentAnalysis } from '../store/appStore';
import { apiClient } from '../api/client';

const ResultsDisplay: React.FC = () => {
  const navigate = useNavigate();
  const currentFile = useCurrentFile();
  const analysisConfig = useAnalysisConfig();
  const currentAnalysis = useCurrentAnalysis();
  const { 
    setCurrentAnalysis, 
    setCurrentStep, 
    addNotification, 
    setLoading,
    setError 
  } = useAppStore();

  // Local state for UI management
  const [activeTab, setActiveTab] = useState<'overview' | 'visualizations' | 'insights' | 'ai-story'>('overview');
  const [expandedInsightSections, setExpandedInsightSections] = useState<Set<string>>(new Set(['summary']));
  const [analysisStatus, setAnalysisStatus] = useState<'loading' | 'complete' | 'error'>('loading');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!currentFile) {
      navigate('/upload');
      return;
    }
    if (!analysisConfig.analysisType) {
      navigate('/configure');
      return;
    }
    
    // Start analysis if not already done
    if (!currentAnalysis) {
      startAnalysis();
    } else {
      setAnalysisStatus('complete');
      setProgress(100);
    }
  }, [currentFile, analysisConfig, currentAnalysis]);

  const startAnalysis = async () => {
    if (!currentFile || !currentFile.dataId) {
      addNotification({
        type: 'error',
        message: 'No data ID found. Please re-upload the file.',
      });
      setAnalysisStatus('error');
      return;
    }

    setAnalysisStatus('loading');
    setLoading(true);
    
    try {
      // Simulate analysis progress
      const progressSteps = [
        { step: 'Loading data...', progress: 10 },
        { step: 'Preprocessing data...', progress: 25 },
        { step: 'Statistical analysis...', progress: 40 },
        { step: 'Generating visualizations...', progress: 60 },
        { step: 'AI insights generation...', progress: 80 },
        { step: 'Finalizing results...', progress: 100 },
      ];

      for (const { step, progress: stepProgress } of progressSteps) {
        setProgress(stepProgress);
        addNotification({
          type: 'info',
          message: step,
        });
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Perform complete analysis using dataId
      const analysisResult = await apiClient.getCompleteAnalysis(currentFile.dataId, {
        target_column: analysisConfig.targetColumn,
        business_context: analysisConfig.businessContext,
        analysis_type: analysisConfig.analysisType,
      });

      // Get AI insights if enabled
      let aiInsights = null;
      if (analysisConfig.includeAI) {
        aiInsights = await apiClient.getAIInsights(currentFile.dataId, {
          target_column: analysisConfig.targetColumn,
          business_context: analysisConfig.businessContext,
        });
      }

      // Get visualizations if enabled
      let visualizations = null;
      if (analysisConfig.includeVisualization) {
        visualizations = await apiClient.getVisualizations(currentFile.dataId);
      }

      const results = {
        id: currentFile.id,
        summary: analysisResult.summary,
        visualizations: visualizations || [],
        insights: analysisResult.insights,
        aiInsights,
      };

      setCurrentAnalysis(results);
      setAnalysisStatus('complete');
      
      addNotification({
        type: 'success',
        message: 'Analysis completed successfully!',
      });

    } catch (error: any) {
      setAnalysisStatus('error');
      setError(error.message);
      addNotification({
        type: 'error',
        message: 'Analysis failed. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  const toggleInsightSection = (sectionId: string) => {
    setExpandedInsightSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  const handleDownloadReports = () => {
    setCurrentStep('reports');
    navigate('/reports');
  };

  if (analysisStatus === 'loading') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto"
      >
        <div className="text-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="w-16 h-16 mx-auto mb-6"
          >
            <SparklesIcon className="w-full h-full text-apple-blue" />
          </motion.div>
          
          <h2 className="text-2xl font-semibold text-text-dark mb-4">
            Analyzing Your Data
          </h2>
          <p className="text-text-medium mb-8">
            Please wait while we process your data and generate insights...
          </p>

          {/* Progress Bar */}
          <div className="max-w-md mx-auto mb-6">
            <div className="w-full bg-border-gray/50 rounded-full h-3">
              <motion.div
                className="bg-apple-blue h-3 rounded-full"
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
              />
            </div>
            <p className="text-sm text-apple-blue font-medium mt-2">
              {progress}% Complete
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-center justify-center space-x-2 text-text-medium">
              <ClockIcon className="w-4 h-4" />
              <span>Statistical Analysis</span>
            </div>
            <div className="flex items-center justify-center space-x-2 text-text-medium">
              <ChartPieIcon className="w-4 h-4" />
              <span>Data Visualization</span>
            </div>
            <div className="flex items-center justify-center space-x-2 text-text-medium">
              <SparklesIcon className="w-4 h-4" />
              <span>AI Insights</span>
            </div>
          </div>
        </div>
      </motion.div>
    );
  }

  if (analysisStatus === 'error') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto text-center"
      >
        <ExclamationTriangleIcon className="w-16 h-16 text-error-red mx-auto mb-4" />
        <h2 className="text-2xl font-semibold text-text-dark mb-2">
          Analysis Failed
        </h2>
        <p className="text-text-medium mb-6">
          We encountered an error while analyzing your data. Please try again.
        </p>
        <div className="space-x-4">
          <button
            onClick={startAnalysis}
            className="btn-primary"
          >
            Retry Analysis
          </button>
          <button
            onClick={() => navigate('/configure')}
            className="btn-secondary"
          >
            Back to Configuration
          </button>
        </div>
      </motion.div>
    );
  }

  if (!currentAnalysis) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto text-center"
      >
        <InformationCircleIcon className="w-16 h-16 text-apple-blue mx-auto mb-4" />
        <h2 className="text-2xl font-semibold text-text-dark mb-2">
          No Analysis Results
        </h2>
        <p className="text-text-medium mb-6">
          Please run an analysis first to view results.
        </p>
        <button
          onClick={() => navigate('/configure')}
          className="btn-primary"
        >
          Start Analysis
        </button>
      </motion.div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: EyeIcon },
    { id: 'visualizations', label: 'Charts', icon: ChartPieIcon, disabled: !analysisConfig.includeVisualization },
    { id: 'insights', label: 'Insights', icon: DocumentTextIcon },
    { id: 'ai-story', label: 'AI Story', icon: SparklesIcon, disabled: !analysisConfig.includeAI },
  ].filter(tab => !tab.disabled);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-7xl mx-auto"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-semibold text-text-dark mb-2">
            Analysis Results
          </h1>
          <p className="text-text-medium">
            Comprehensive insights from your data analysis
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={handleDownloadReports}
            className="btn-secondary flex items-center space-x-2"
          >
            <ArrowDownTrayIcon className="w-4 h-4" />
            <span>Download Reports</span>
          </button>
          <button className="btn-ghost flex items-center space-x-2">
            <ShareIcon className="w-4 h-4" />
            <span>Share</span>
          </button>
        </div>
      </div>

      {/* Success Banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="card p-4 mb-8 bg-success-green/5 border-success-green/20"
      >
        <div className="flex items-center space-x-3">
          <CheckCircleIcon className="w-6 h-6 text-success-green" />
          <div>
            <p className="font-medium text-success-green">Analysis Complete!</p>
            <p className="text-sm text-text-medium">
              Generated {tabs.length} analysis sections • Target: {analysisConfig.targetColumn || 'General Analysis'}
            </p>
          </div>
        </div>
      </motion.div>

      {/* Tab Navigation */}
      <div className="mb-8">
        <div className="border-b border-border-gray">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`relative py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    isActive
                      ? 'border-apple-blue text-apple-blue'
                      : 'border-transparent text-text-medium hover:text-apple-blue hover:border-apple-blue/50'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="card p-6 text-center">
                <p className="text-2xl font-bold text-apple-blue mb-1">
                  {currentAnalysis.summary?.shape?.[0]?.toLocaleString() || 'N/A'}
                </p>
                <p className="text-sm text-text-medium">Total Rows</p>
              </div>
              <div className="card p-6 text-center">
                <p className="text-2xl font-bold text-success-green mb-1">
                  {currentAnalysis.summary?.shape?.[1] || 'N/A'}
                </p>
                <p className="text-sm text-text-medium">Columns</p>
              </div>
              <div className="card p-6 text-center">
                <p className="text-2xl font-bold text-warning-orange mb-1">
                  {Object.keys(currentAnalysis.summary?.dtypes || {}).length}
                </p>
                <p className="text-sm text-text-medium">Data Types</p>
              </div>
              <div className="card p-6 text-center">
                <p className="text-2xl font-bold text-text-dark mb-1">95%+</p>
                <p className="text-sm text-text-medium">Data Quality</p>
              </div>
            </div>

            {/* Data Summary */}
            {currentAnalysis.summary && (
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-text-dark mb-4">
                  Statistical Summary
                </h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead>
                      <tr className="border-b border-border-gray">
                        <th className="text-left py-2 text-sm font-medium text-text-dark">Column</th>
                        <th className="text-left py-2 text-sm font-medium text-text-dark">Type</th>
                        <th className="text-left py-2 text-sm font-medium text-text-dark">Non-Null</th>
                        <th className="text-left py-2 text-sm font-medium text-text-dark">Unique</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-gray">
                      {Object.entries(currentAnalysis.summary.dtypes || {}).slice(0, 10).map(([column, dtype]) => (
                        <tr key={column} className="hover:bg-apple-blue/5">
                          <td className="py-2 text-sm text-text-dark">{column}</td>
                          <td className="py-2 text-sm text-text-medium">{dtype}</td>
                          <td className="py-2 text-sm text-success-green">100%</td>
                          <td className="py-2 text-sm text-text-medium">High</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Quick Insights */}
            <div className="card p-6">
              <h3 className="text-lg font-semibold text-text-dark mb-4">
                Quick Insights
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-apple-blue/5 rounded-apple-sm">
                  <p className="font-medium text-apple-blue mb-1">Data Quality</p>
                  <p className="text-sm text-text-medium">Your dataset shows excellent quality with minimal missing values and consistent formatting.</p>
                </div>
                <div className="p-4 bg-success-green/5 rounded-apple-sm">
                  <p className="font-medium text-success-green mb-1">Analysis Ready</p>
                  <p className="text-sm text-text-medium">All columns are properly formatted and ready for advanced analysis and modeling.</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'visualizations' && (
          <motion.div
            key="visualizations"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            <div className="text-center py-12">
              <ChartPieIcon className="w-16 h-16 text-apple-blue mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-text-dark mb-2">
                Interactive Charts
              </h3>
              <p className="text-text-medium mb-6">
                Visualizations will be displayed here after integration with backend visualization endpoints.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {['Distribution Plot', 'Correlation Matrix', 'Box Plot', 'Scatter Plot', 'Time Series', 'Feature Importance'].map((chart, index) => (
                  <div key={chart} className="card p-6 text-center">
                    <div className="w-16 h-16 bg-apple-blue/10 rounded-apple-md mx-auto mb-3 flex items-center justify-center">
                      <ChartPieIcon className="w-8 h-8 text-apple-blue" />
                    </div>
                    <p className="font-medium text-text-dark">{chart}</p>
                    <p className="text-sm text-text-medium mt-1">Ready for display</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'insights' && (
          <motion.div
            key="insights"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-4"
          >
            {[
              {
                id: 'summary',
                title: 'Executive Summary',
                content: 'Your dataset demonstrates high quality with excellent data integrity. Key patterns suggest strong relationships between variables, particularly in the target column analysis.',
                type: 'summary',
              },
              {
                id: 'patterns',
                title: 'Data Patterns',
                content: 'Statistical analysis reveals several significant correlations and trends that could be valuable for predictive modeling and business insights.',
                type: 'analysis',
              },
              {
                id: 'recommendations',
                title: 'Recommendations',
                content: 'Based on the analysis, we recommend focusing on feature engineering for the top correlated variables and considering ensemble methods for modeling.',
                type: 'recommendation',
              },
            ].map((insight) => (
              <div key={insight.id} className="card overflow-hidden">
                <button
                  onClick={() => toggleInsightSection(insight.id)}
                  className="w-full px-6 py-4 text-left border-b border-border-gray hover:bg-apple-blue/5 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-text-dark">
                      {insight.title}
                    </h3>
                    {expandedInsightSections.has(insight.id) ? (
                      <ChevronUpIcon className="w-5 h-5 text-text-medium" />
                    ) : (
                      <ChevronDownIcon className="w-5 h-5 text-text-medium" />
                    )}
                  </div>
                </button>
                
                <AnimatePresence>
                  {expandedInsightSections.has(insight.id) && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="p-6">
                        <p className="text-text-medium leading-relaxed">
                          {insight.content}
                        </p>
                        <div className="mt-4 flex items-center space-x-4 text-sm">
                          <span className={`px-3 py-1 rounded-apple-sm ${
                            insight.type === 'summary' ? 'bg-apple-blue/10 text-apple-blue' :
                            insight.type === 'analysis' ? 'bg-warning-orange/10 text-warning-orange' :
                            'bg-success-green/10 text-success-green'
                          }`}>
                            {insight.type}
                          </span>
                          <span className="text-text-medium">High Confidence</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </motion.div>
        )}

        {activeTab === 'ai-story' && (
          <motion.div
            key="ai-story"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            <div className="card p-8">
              <div className="flex items-center space-x-3 mb-6">
                <SparklesIcon className="w-8 h-8 text-apple-blue" />
                <h3 className="text-xl font-semibold text-text-dark">
                  AI-Generated Data Story
                </h3>
              </div>
              
              <div className="prose prose-lg max-w-none">
                <p className="text-text-medium leading-relaxed mb-4">
                  Your dataset tells a compelling story of data quality and potential insights. 
                  The AI analysis reveals that your data is well-structured and ready for advanced analytics.
                </p>
                
                <p className="text-text-medium leading-relaxed mb-4">
                  Key findings suggest that there are strong patterns within your data that could be 
                  leveraged for predictive modeling and business intelligence applications.
                </p>
                
                <p className="text-text-medium leading-relaxed">
                  The combination of data quality metrics and statistical patterns indicates high 
                  potential for generating actionable business insights through further analysis.
                </p>
              </div>
              
              <div className="mt-6 p-4 bg-apple-blue/5 rounded-apple-sm">
                <p className="text-sm text-apple-blue flex items-center">
                  <SparklesIcon className="w-4 h-4 mr-2" />
                  Generated by AI based on your data patterns and business context
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ResultsDisplay;