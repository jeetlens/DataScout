import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  EyeIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  MagnifyingGlassIcon,
  AdjustmentsHorizontalIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';
import { useAppStore, useCurrentFile, useDataPreview } from '../store/appStore';
import { apiClient } from '../api/client';

const DataPreview: React.FC = () => {
  const navigate = useNavigate();
  const currentFile = useCurrentFile();
  const dataPreview = useDataPreview();
  const { 
    setDataPreview, 
    setCurrentStep, 
    addNotification, 
    setLoading,
    setError 
  } = useAppStore();

  // Local state for pagination and filtering
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  
  const rowsPerPage = 10;

  // Load data info when component mounts
  useEffect(() => {
    if (currentFile && !dataPreview) {
      loadDataInfo();
    }
  }, [currentFile]);

  const loadDataInfo = async () => {
    if (!currentFile || !currentFile.dataId) {
      addNotification({
        type: 'error',
        message: 'No data ID found. Please re-upload the file.',
      });
      return;
    }

    setLoading(true);
    try {
      const dataInfo = await apiClient.getDataInfo(currentFile.dataId);
      
      setDataPreview({
        columns: dataInfo.basic_info.column_names,
        rows: dataInfo.sample_data.sample_data || [],
        totalRows: dataInfo.basic_info.rows,
        dataTypes: dataInfo.basic_info.data_types,
      });

      addNotification({
        type: 'success',
        message: `Data loaded successfully! ${dataInfo.basic_info.rows} rows, ${dataInfo.basic_info.columns} columns`,
      });

    } catch (error: any) {
      setError(error.message);
      addNotification({
        type: 'error',
        message: 'Failed to load data preview. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleColumnToggle = (column: string) => {
    setSelectedColumns(prev => 
      prev.includes(column)
        ? prev.filter(c => c !== column)
        : [...prev, column]
    );
  };

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const handleContinue = () => {
    setCurrentStep('configure');
    navigate('/configure');
  };

  if (!currentFile) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto"
      >
        <div className="text-center">
          <ExclamationTriangleIcon className="w-16 h-16 text-warning-orange mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-text-dark mb-2">
            No File Selected
          </h2>
          <p className="text-text-medium mb-6">
            Please upload a file first to preview your data.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="btn-primary"
          >
            Upload File
          </button>
        </div>
      </motion.div>
    );
  }

  // Filter and paginate data
  const filteredColumns = dataPreview?.columns.filter(column => 
    column.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  const displayColumns = selectedColumns.length > 0 ? selectedColumns : filteredColumns;
  const startIndex = (currentPage - 1) * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const currentRows = dataPreview?.rows.slice(startIndex, endIndex) || [];
  const totalPages = Math.ceil((dataPreview?.rows.length || 0) / rowsPerPage);

  const getDataTypeColor = (dataType: string) => {
    if (dataType.includes('int') || dataType.includes('float')) return 'text-apple-blue';
    if (dataType.includes('object') || dataType.includes('string')) return 'text-success-green';
    if (dataType.includes('datetime')) return 'text-warning-orange';
    if (dataType.includes('bool')) return 'text-error-red';
    return 'text-text-medium';
  };

  const getDataTypeIcon = (dataType: string) => {
    if (dataType.includes('int') || dataType.includes('float')) return '123';
    if (dataType.includes('object') || dataType.includes('string')) return 'Aa';
    if (dataType.includes('datetime')) return '📅';
    if (dataType.includes('bool')) return '✓';
    return '?';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-7xl mx-auto"
    >
      {/* Header */}
      <div className="mb-8">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-6"
        >
          <div>
            <h1 className="text-3xl font-semibold text-text-dark mb-2">
              Data Preview
            </h1>
            <p className="text-text-medium">
              Review your data structure and quality before analysis
            </p>
          </div>
          
          <button
            onClick={handleContinue}
            className="btn-primary flex items-center space-x-2"
          >
            <span>Continue to Analysis</span>
            <ArrowRightIcon className="w-4 h-4" />
          </button>
        </motion.div>

        {/* File Info Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="card p-6 mb-6"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 apple-gradient-subtle rounded-apple-md flex items-center justify-center">
                <EyeIcon className="w-6 h-6 text-apple-blue" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-text-dark">{currentFile.name}</h3>
                <p className="text-text-medium">
                  {dataPreview?.totalRows.toLocaleString()} rows × {dataPreview?.columns.length} columns
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm text-text-medium">File Size</p>
                <p className="font-medium text-text-dark">
                  {(currentFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <div className="w-2 h-2 bg-success-green rounded-full"></div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-6"
      >
        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="w-5 h-5 text-text-medium absolute left-3 top-1/2 transform -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search columns..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input-field pl-10 pr-4 py-2 w-full lg:w-80"
          />
        </div>

        {/* Column Filter */}
        <div className="flex items-center space-x-2">
          <AdjustmentsHorizontalIcon className="w-5 h-5 text-text-medium" />
          <span className="text-sm text-text-medium">
            Showing {displayColumns.length} of {dataPreview?.columns.length || 0} columns
          </span>
        </div>
      </motion.div>

      {/* Data Quality Summary */}
      {dataPreview && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6"
        >
          <div className="card p-4">
            <div className="flex items-center space-x-3">
              <CheckCircleIcon className="w-6 h-6 text-success-green" />
              <div>
                <p className="text-sm text-text-medium">Data Quality</p>
                <p className="font-semibold text-success-green">Excellent</p>
              </div>
            </div>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center space-x-3">
              <InformationCircleIcon className="w-6 h-6 text-apple-blue" />
              <div>
                <p className="text-sm text-text-medium">Data Types</p>
                <p className="font-semibold text-text-dark">
                  {Object.values(dataPreview.dataTypes || {}).filter((type, index, arr) => 
                    arr.indexOf(type) === index
                  ).length} unique types
                </p>
              </div>
            </div>
          </div>
          
          <div className="card p-4">
            <div className="flex items-center space-x-3">
              <ExclamationTriangleIcon className="w-6 h-6 text-warning-orange" />
              <div>
                <p className="text-sm text-text-medium">Missing Values</p>
                <p className="font-semibold text-text-dark">0 columns</p>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Data Table */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card overflow-hidden"
      >
        {/* Table Header */}
        <div className="px-6 py-4 border-b border-border-gray bg-card-white">
          <h3 className="text-lg font-semibold text-text-dark">Data Sample</h3>
          <p className="text-sm text-text-medium">
            Showing rows {startIndex + 1}-{Math.min(endIndex, dataPreview?.rows.length || 0)} of {dataPreview?.rows.length || 0}
          </p>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-card-white border-b border-border-gray">
              <tr>
                {displayColumns.map((column) => (
                  <th
                    key={column}
                    className="px-6 py-3 text-left cursor-pointer hover:bg-apple-blue/5 transition-colors"
                    onClick={() => handleSort(column)}
                  >
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold text-text-dark">{column}</span>
                      <div className="flex flex-col items-center">
                        <span className="text-xs font-mono px-2 py-1 rounded bg-apple-blue/10 text-apple-blue">
                          {getDataTypeIcon(dataPreview?.dataTypes?.[column] || '')}
                        </span>
                      </div>
                    </div>
                    <div className="mt-1">
                      <span className={`text-xs font-medium ${getDataTypeColor(dataPreview?.dataTypes?.[column] || '')}`}>
                        {dataPreview?.dataTypes?.[column]}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border-gray">
              {currentRows.map((row, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.05 * index }}
                  className="hover:bg-apple-blue/5 transition-colors"
                >
                  {displayColumns.map((column) => (
                    <td key={column} className="px-6 py-4 text-sm text-text-dark">
                      <div className="max-w-xs truncate">
                        {row[column] !== null && row[column] !== undefined 
                          ? String(row[column]) 
                          : <span className="text-text-light italic">null</span>
                        }
                      </div>
                    </td>
                  ))}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="px-6 py-4 border-t border-border-gray bg-card-white">
            <div className="flex items-center justify-between">
              <div className="text-sm text-text-medium">
                Page {currentPage} of {totalPages}
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className="p-2 rounded-apple-sm hover:bg-apple-blue/10 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeftIcon className="w-4 h-4 text-text-medium" />
                </button>
                
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const page = i + 1;
                  return (
                    <button
                      key={page}
                      onClick={() => setCurrentPage(page)}
                      className={`px-3 py-1 rounded-apple-sm text-sm font-medium transition-colors ${
                        page === currentPage
                          ? 'bg-apple-blue text-white'
                          : 'text-text-medium hover:text-apple-blue hover:bg-apple-blue/10'
                      }`}
                    >
                      {page}
                    </button>
                  );
                })}
                
                <button
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                  className="p-2 rounded-apple-sm hover:bg-apple-blue/10 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRightIcon className="w-4 h-4 text-text-medium" />
                </button>
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* Column Details Sidebar (if needed) */}
      {selectedColumns.length > 0 && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed right-4 top-1/2 transform -translate-y-1/2 w-64 card p-4 max-h-96 overflow-y-auto"
        >
          <h4 className="font-semibold text-text-dark mb-3">Selected Columns</h4>
          <div className="space-y-2">
            {selectedColumns.map((column) => (
              <div key={column} className="flex items-center justify-between text-sm">
                <span className="text-text-dark">{column}</span>
                <button
                  onClick={() => handleColumnToggle(column)}
                  className="text-error-red hover:text-error-red/80"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default DataPreview;