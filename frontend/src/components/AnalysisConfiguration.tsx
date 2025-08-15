import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CogIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ChevronRightIcon,
  ChevronLeftIcon,
  SparklesIcon,
  ChartBarIcon,
  DocumentTextIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';
import { useAppStore, useCurrentFile, useDataPreview, useAnalysisConfig } from '../store/appStore';

const AnalysisConfiguration: React.FC = () => {
  const navigate = useNavigate();
  const currentFile = useCurrentFile();
  const dataPreview = useDataPreview();
  const analysisConfig = useAnalysisConfig();
  const { 
    updateAnalysisConfig, 
    setCurrentStep, 
    addNotification,
    setLoading 
  } = useAppStore();

  // Local state for form management
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isTargetDropdownOpen, setIsTargetDropdownOpen] = useState(false);
  const [businessContext, setBusinessContext] = useState(analysisConfig.businessContext || '');
  const [targetColumn, setTargetColumn] = useState(analysisConfig.targetColumn || '');
  const [analysisType, setAnalysisType] = useState(analysisConfig.analysisType);
  const [includeVisualization, setIncludeVisualization] = useState(analysisConfig.includeVisualization);
  const [includeAI, setIncludeAI] = useState(analysisConfig.includeAI);

  const steps = [
    {
      id: 'target',
      title: 'Target Column',
      description: 'Select the column you want to analyze or predict',
      icon: ChartBarIcon,
    },
    {
      id: 'context',
      title: 'Business Context',
      description: 'Provide context to get better AI insights',
      icon: DocumentTextIcon,
    },
    {
      id: 'options',
      title: 'Analysis Options',
      description: 'Configure your analysis settings',
      icon: CogIcon,
    },
    {
      id: 'review',
      title: 'Review & Start',
      description: 'Review your configuration and start analysis',
      icon: CheckCircleIcon,
    },
  ];

  const analysisTypes = [
    {
      id: 'complete',
      title: 'Complete Analysis',
      description: 'Full statistical analysis with visualizations and AI insights',
      features: ['Statistical Summary', 'Data Quality Assessment', 'Visualizations', 'AI Insights', 'Recommendations'],
      recommended: true,
    },
    {
      id: 'executive',
      title: 'Executive Summary',
      description: 'High-level overview perfect for presentations',
      features: ['Key Metrics', 'Executive Summary', 'Key Visualizations', 'Business Insights'],
      recommended: false,
    },
    {
      id: 'ai-focused',
      title: 'AI-Focused',
      description: 'Emphasis on AI-powered insights and predictions',
      features: ['AI Pattern Detection', 'Predictive Insights', 'Anomaly Detection', 'Business Recommendations'],
      recommended: false,
    },
  ];

  useEffect(() => {
    if (!currentFile) {
      navigate('/upload');
      return;
    }
    if (!dataPreview) {
      navigate('/preview');
      return;
    }
  }, [currentFile, dataPreview, navigate]);

  const handleNext = () => {
    if (currentStepIndex < steps.length - 1) {
      setCurrentStepIndex(currentStepIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(currentStepIndex - 1);
    }
  };

  const handleStartAnalysis = () => {
    // Save configuration to store
    updateAnalysisConfig({
      targetColumn: targetColumn || undefined,
      businessContext: businessContext || undefined,
      analysisType,
      includeVisualization,
      includeAI,
    });

    addNotification({
      type: 'success',
      message: 'Analysis configuration saved! Starting analysis...',
    });

    setCurrentStep('results');
    navigate('/results');
  };

  const renderStepContent = () => {
    const step = steps[currentStepIndex];

    switch (step.id) {
      case 'target':
        return (
          <motion.div
            key="target"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <ChartBarIcon className="w-12 h-12 text-apple-blue mx-auto mb-4" />
              <h2 className="text-2xl font-semibold text-text-dark mb-2">
                Select Target Column
              </h2>
              <p className="text-text-medium">
                Choose the main column you want to analyze, or leave blank for general analysis
              </p>
            </div>

            <div className="max-w-md mx-auto">
              <label className="block text-sm font-medium text-text-dark mb-2">
                Target Column (Optional)
              </label>
              
              <div className="relative">
                <button
                  onClick={() => setIsTargetDropdownOpen(!isTargetDropdownOpen)}
                  className="input-field w-full flex items-center justify-between"
                >
                  <span className={targetColumn ? 'text-text-dark' : 'text-text-medium'}>
                    {targetColumn || 'Select a column...'}
                  </span>
                  {isTargetDropdownOpen ? (
                    <ChevronUpIcon className="w-4 h-4 text-text-medium" />
                  ) : (
                    <ChevronDownIcon className="w-4 h-4 text-text-medium" />
                  )}
                </button>

                <AnimatePresence>
                  {isTargetDropdownOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute top-full left-0 right-0 mt-1 bg-white border border-border-gray rounded-apple-sm shadow-apple-medium z-10 max-h-64 overflow-y-auto"
                    >
                      <button
                        onClick={() => {
                          setTargetColumn('');
                          setIsTargetDropdownOpen(false);
                        }}
                        className="w-full px-4 py-3 text-left hover:bg-apple-blue/5 transition-colors border-b border-border-gray"
                      >
                        <span className="text-text-medium italic">No specific target</span>
                      </button>
                      
                      {dataPreview?.columns.map((column) => (
                        <button
                          key={column}
                          onClick={() => {
                            setTargetColumn(column);
                            setIsTargetDropdownOpen(false);
                          }}
                          className={`w-full px-4 py-3 text-left hover:bg-apple-blue/5 transition-colors ${
                            targetColumn === column ? 'bg-apple-blue/10 text-apple-blue' : 'text-text-dark'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span>{column}</span>
                            <span className="text-xs text-text-medium">
                              {dataPreview.dataTypes?.[column]}
                            </span>
                          </div>
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {targetColumn && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 bg-apple-blue/5 rounded-apple-sm"
                >
                  <p className="text-sm text-apple-blue">
                    <strong>Selected:</strong> {targetColumn} ({dataPreview?.dataTypes?.[targetColumn]})
                  </p>
                  <p className="text-xs text-text-medium mt-1">
                    Analysis will focus on patterns and insights related to this column.
                  </p>
                </motion.div>
              )}
            </div>
          </motion.div>
        );

      case 'context':
        return (
          <motion.div
            key="context"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <DocumentTextIcon className="w-12 h-12 text-apple-blue mx-auto mb-4" />
              <h2 className="text-2xl font-semibold text-text-dark mb-2">
                Business Context
              </h2>
              <p className="text-text-medium">
                Help our AI understand your data better with business context
              </p>
            </div>

            <div className="max-w-2xl mx-auto">
              <label className="block text-sm font-medium text-text-dark mb-2">
                What is this data about? (Optional)
              </label>
              
              <textarea
                value={businessContext}
                onChange={(e) => setBusinessContext(e.target.value)}
                placeholder="e.g., This is sales data from our e-commerce platform. We want to understand customer buying patterns and identify opportunities for growth..."
                className="input-field w-full h-32 resize-none"
              />

              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                  "Sales and revenue data for forecasting trends",
                  "Customer data for behavior analysis",
                  "Marketing campaign performance metrics",
                  "Financial data for risk assessment",
                ].map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setBusinessContext(example)}
                    className="p-3 text-left text-sm bg-card-white border border-border-gray rounded-apple-sm hover:border-apple-blue hover:bg-apple-blue/5 transition-colors"
                  >
                    {example}
                  </button>
                ))}
              </div>

              {businessContext && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 bg-success-green/5 rounded-apple-sm"
                >
                  <p className="text-sm text-success-green flex items-center">
                    <SparklesIcon className="w-4 h-4 mr-2" />
                    Great! This context will help generate more relevant insights.
                  </p>
                </motion.div>
              )}
            </div>
          </motion.div>
        );

      case 'options':
        return (
          <motion.div
            key="options"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-8"
          >
            <div className="text-center mb-8">
              <CogIcon className="w-12 h-12 text-apple-blue mx-auto mb-4" />
              <h2 className="text-2xl font-semibold text-text-dark mb-2">
                Analysis Type
              </h2>
              <p className="text-text-medium">
                Choose the type of analysis that best fits your needs
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {analysisTypes.map((type) => (
                <motion.div
                  key={type.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <button
                    onClick={() => setAnalysisType(type.id as any)}
                    className={`relative w-full p-6 rounded-apple-md border-2 text-left transition-all ${
                      analysisType === type.id
                        ? 'border-apple-blue bg-apple-blue/5'
                        : 'border-border-gray bg-white hover:border-apple-blue/50 hover:bg-apple-blue/5'
                    }`}
                  >
                    {type.recommended && (
                      <div className="absolute -top-2 -right-2 bg-apple-blue text-white text-xs px-2 py-1 rounded-apple-sm">
                        Recommended
                      </div>
                    )}
                    
                    <h3 className="text-lg font-semibold text-text-dark mb-2">
                      {type.title}
                    </h3>
                    <p className="text-sm text-text-medium mb-4">
                      {type.description}
                    </p>
                    
                    <div className="space-y-2">
                      {type.features.map((feature, index) => (
                        <div key={index} className="flex items-center text-xs text-text-medium">
                          <CheckCircleIcon className="w-3 h-3 text-success-green mr-2 flex-shrink-0" />
                          {feature}
                        </div>
                      ))}
                    </div>
                  </button>
                </motion.div>
              ))}
            </div>

            <div className="max-w-md mx-auto space-y-4">
              <div className="flex items-center justify-between p-4 bg-card-white rounded-apple-sm border border-border-gray">
                <div>
                  <p className="font-medium text-text-dark">Include Visualizations</p>
                  <p className="text-sm text-text-medium">Generate charts and graphs</p>
                </div>
                <button
                  onClick={() => setIncludeVisualization(!includeVisualization)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    includeVisualization ? 'bg-apple-blue' : 'bg-border-gray'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full top-0.5 transition-transform ${
                      includeVisualization ? 'translate-x-6' : 'translate-x-0.5'
                    }`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between p-4 bg-card-white rounded-apple-sm border border-border-gray">
                <div>
                  <p className="font-medium text-text-dark">AI-Powered Insights</p>
                  <p className="text-sm text-text-medium">Get intelligent recommendations</p>
                </div>
                <button
                  onClick={() => setIncludeAI(!includeAI)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    includeAI ? 'bg-apple-blue' : 'bg-border-gray'
                  }`}
                >
                  <div
                    className={`absolute w-5 h-5 bg-white rounded-full top-0.5 transition-transform ${
                      includeAI ? 'translate-x-6' : 'translate-x-0.5'
                    }`}
                  />
                </button>
              </div>
            </div>
          </motion.div>
        );

      case 'review':
        return (
          <motion.div
            key="review"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="text-center mb-8">
              <CheckCircleIcon className="w-12 h-12 text-apple-blue mx-auto mb-4" />
              <h2 className="text-2xl font-semibold text-text-dark mb-2">
                Review Configuration
              </h2>
              <p className="text-text-medium">
                Confirm your analysis settings before we begin
              </p>
            </div>

            <div className="max-w-2xl mx-auto space-y-4">
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-text-dark mb-4">Analysis Summary</h3>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-text-medium">Dataset:</span>
                    <span className="font-medium text-text-dark">{currentFile?.name}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-text-medium">Target Column:</span>
                    <span className="font-medium text-text-dark">
                      {targetColumn || 'General analysis'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-text-medium">Analysis Type:</span>
                    <span className="font-medium text-text-dark">
                      {analysisTypes.find(t => t.id === analysisType)?.title}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-text-medium">Business Context:</span>
                    <span className="font-medium text-text-dark">
                      {businessContext ? 'Provided' : 'Not provided'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-text-medium">Visualizations:</span>
                    <span className={`font-medium ${includeVisualization ? 'text-success-green' : 'text-text-medium'}`}>
                      {includeVisualization ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-text-medium">AI Insights:</span>
                    <span className={`font-medium ${includeAI ? 'text-success-green' : 'text-text-medium'}`}>
                      {includeAI ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-center">
                <button
                  onClick={handleStartAnalysis}
                  className="btn-primary text-lg px-8 py-4 flex items-center space-x-2 mx-auto"
                >
                  <SparklesIcon className="w-5 h-5" />
                  <span>Start Analysis</span>
                  <ArrowRightIcon className="w-5 h-5" />
                </button>
                <p className="text-sm text-text-medium mt-2">
                  This may take a few minutes depending on your data size
                </p>
              </div>
            </div>
          </motion.div>
        );

      default:
        return null;
    }
  };

  if (!currentFile || !dataPreview) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto text-center"
      >
        <InformationCircleIcon className="w-16 h-16 text-apple-blue mx-auto mb-4" />
        <h2 className="text-2xl font-semibold text-text-dark mb-2">
          Configuration Not Available
        </h2>
        <p className="text-text-medium mb-6">
          Please upload and preview your data first.
        </p>
        <button
          onClick={() => navigate('/upload')}
          className="btn-primary"
        >
          Go to Upload
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
      {/* Progress Indicator */}
      <div className="mb-12">
        <div className="flex items-center justify-between mb-6">
          {steps.map((step, index) => {
            const StepIcon = step.icon;
            const isActive = index === currentStepIndex;
            const isCompleted = index < currentStepIndex;
            
            return (
              <div key={step.id} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center mb-2 transition-all ${
                      isActive
                        ? 'bg-apple-blue text-white'
                        : isCompleted
                        ? 'bg-success-green text-white'
                        : 'bg-border-gray text-text-medium'
                    }`}
                  >
                    <StepIcon className="w-6 h-6" />
                  </div>
                  <div className="text-center">
                    <p className={`text-sm font-medium ${isActive ? 'text-apple-blue' : 'text-text-medium'}`}>
                      {step.title}
                    </p>
                    <p className="text-xs text-text-medium hidden md:block">
                      {step.description}
                    </p>
                  </div>
                </div>
                
                {index < steps.length - 1 && (
                  <div
                    className={`flex-1 h-0.5 mx-4 transition-colors ${
                      isCompleted ? 'bg-success-green' : 'bg-border-gray'
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Step Content */}
      <div className="min-h-[500px]">
        <AnimatePresence mode="wait">
          {renderStepContent()}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between mt-12 pt-6 border-t border-border-gray">
        <button
          onClick={handlePrevious}
          disabled={currentStepIndex === 0}
          className="btn-secondary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeftIcon className="w-4 h-4" />
          <span>Previous</span>
        </button>

        <div className="text-sm text-text-medium">
          Step {currentStepIndex + 1} of {steps.length}
        </div>

        {currentStepIndex < steps.length - 1 ? (
          <button
            onClick={handleNext}
            className="btn-primary flex items-center space-x-2"
          >
            <span>Next</span>
            <ChevronRightIcon className="w-4 h-4" />
          </button>
        ) : (
          <div></div>
        )}
      </div>
    </motion.div>
  );
};

export default AnalysisConfiguration;