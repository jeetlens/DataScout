// API configuration
const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

// Error types
export class APIError extends Error {
  public status: number;
  public response?: any;
  
  constructor(
    message: string,
    status: number,
    response?: any
  ) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.response = response;
  }
}

// Request types
export interface UploadResponse {
  data_id: string;
  status: string;
  message: string;
  data_info: {
    rows: number;
    columns: number;
    column_names: string[];
    data_types: Record<string, string>;
  };
  sample_data: any[];
  validation: any;
}

export interface DataInfo {
  data_id: string;
  basic_info: {
    rows: number;
    columns: number;
    column_names: string[];
    data_types: Record<string, string>;
    memory_usage_mb: number;
  };
  sample_data: {
    sample_data: any[];
    total_rows: number;
    total_columns: number;
    column_names: string[];
  };
  validation: any;
}

export interface AnalysisConfig {
  target_column?: string;
  business_context?: string;
  analysis_type?: 'complete' | 'executive' | 'ai-focused';
}

// Helper function to handle API responses
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP error! status: ${response.status}`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorData.message || errorMessage;
    } catch {
      // If we can't parse the error, use the status text
      errorMessage = response.statusText || errorMessage;
    }
    throw new APIError(errorMessage, response.status);
  }

  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    const result = await response.json();
    console.log('API Response:', result); // Debug log
    return result;
  }
  
  // For non-JSON responses (like file downloads)
  return response as any;
}

// Progress tracking for uploads
export function createProgressTracker(onProgress: (progress: number) => void) {
  return (event: ProgressEvent) => {
    if (event.lengthComputable) {
      const progress = Math.round((event.loaded * 100) / event.total);
      onProgress(progress);
    }
  };
}

// API client class
export class DataScoutAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // File upload with progress tracking
  async uploadFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<UploadResponse> {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();

      if (onProgress) {
        xhr.upload.addEventListener('progress', createProgressTracker(onProgress));
      }

      xhr.addEventListener('load', async () => {
        try {
          console.log('XHR Status:', xhr.status);
          console.log('XHR Response Text:', xhr.responseText);
          console.log('XHR Content Type:', xhr.getResponseHeader('content-type'));
          
          // Parse JSON directly from responseText
          if (xhr.status >= 200 && xhr.status < 300) {
            const result = JSON.parse(xhr.responseText);
            console.log('Parsed result:', result);
            resolve(result);
          } else {
            reject(new APIError(`HTTP error! status: ${xhr.status}`, xhr.status));
          }
        } catch (error) {
          console.error('Upload response parsing error:', error);
          reject(error);
        }
      });

      xhr.addEventListener('error', () => {
        reject(new APIError('Network error during upload', 0));
      });

      xhr.open('POST', `${this.baseURL}/upload`);
      xhr.send(formData);
    });
  }

  // Get data information
  async getDataInfo(dataId: string): Promise<DataInfo> {
    const response = await fetch(`${this.baseURL}/data/${dataId}/info`);
    return handleResponse<DataInfo>(response);
  }

  // Get data list
  async getDataList(): Promise<DataInfo[]> {
    const response = await fetch(`${this.baseURL}/data/list`);
    return handleResponse<DataInfo[]>(response);
  }

  // Delete data
  async deleteData(dataId: string): Promise<{ message: string }> {
    const response = await fetch(`${this.baseURL}/data/${dataId}`, {
      method: 'DELETE',
    });
    return handleResponse<{ message: string }>(response);
  }

  // Preprocessing
  async preprocessData(dataId: string, config: any = {}): Promise<any> {
    const response = await fetch(`${this.baseURL}/data/${dataId}/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse(response);
  }

  // Summary
  async getSummary(dataId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/data/${dataId}/summary`);
    return handleResponse(response);
  }

  // Feature analysis
  async getFeatures(dataId: string, targetColumn?: string): Promise<any> {
    const url = targetColumn 
      ? `${this.baseURL}/data/${dataId}/features?target_column=${encodeURIComponent(targetColumn)}`
      : `${this.baseURL}/data/${dataId}/features`;
    
    const response = await fetch(url);
    return handleResponse(response);
  }

  // Visualizations
  async getVisualizations(dataId: string, plotTypes?: string[]): Promise<any> {
    let url = `${this.baseURL}/data/${dataId}/visualizations`;
    if (plotTypes && plotTypes.length > 0) {
      url += `?plot_types=${plotTypes.join(',')}`;
    }
    
    const response = await fetch(url);
    return handleResponse(response);
  }

  // Specific visualization
  async getVisualization(dataId: string, plotType: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/data/${dataId}/visualize/${plotType}`);
    return handleResponse(response);
  }

  // AI Insights
  async getAIInsights(dataId: string, config: AnalysisConfig = {}): Promise<any> {
    let url = `${this.baseURL}/data/${dataId}/ai-insights`;
    
    // Add query parameters for GET request
    const params = new URLSearchParams();
    if (config.target_column) params.append('target_column', config.target_column);
    if (config.business_context) params.append('business_context', config.business_context);
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url);
    return handleResponse(response);
  }

  // AI Summary
  async getAISummary(dataId: string, config: AnalysisConfig = {}): Promise<any> {
    let url = `${this.baseURL}/data/${dataId}/ai-summary`;
    
    // Add query parameters for GET request
    const params = new URLSearchParams();
    if (config.business_context) params.append('business_context', config.business_context);
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url);
    return handleResponse(response);
  }

  // AI Story
  async getAIStory(dataId: string, config: AnalysisConfig = {}): Promise<any> {
    let url = `${this.baseURL}/data/${dataId}/ai-story`;
    
    // Add query parameters for GET request
    const params = new URLSearchParams();
    if (config.business_context) params.append('business_context', config.business_context);
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url);
    return handleResponse(response);
  }

  // Complete Analysis
  async getCompleteAnalysis(dataId: string, config: AnalysisConfig = {}): Promise<any> {
    let url = `${this.baseURL}/data/${dataId}/complete-analysis`;
    
    // Add query parameters for GET request
    const params = new URLSearchParams();
    if (config.target_column) params.append('target_column', config.target_column);
    if (config.business_context) params.append('business_context', config.business_context);
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url);
    return handleResponse(response);
  }

  // Report Generation
  async generateHTMLReport(dataId: string, reportType: string = 'complete'): Promise<any> {
    const response = await fetch(`${this.baseURL}/data/${dataId}/reports/html?report_type=${reportType}`);
    return handleResponse(response);
  }

  async generatePDFReport(dataId: string, reportType: string = 'complete'): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/data/${dataId}/reports/pdf?report_type=${reportType}`);
    if (!response.ok) {
      throw new APIError(`PDF generation failed: ${response.statusText}`, response.status);
    }
    return response.blob();
  }

  async generateReportPackage(dataId: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/data/${dataId}/reports/package`);
    if (!response.ok) {
      throw new APIError(`Report package generation failed: ${response.statusText}`, response.status);
    }
    return response.blob();
  }

  // System endpoints
  async getHealth(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${this.baseURL}/health`);
    return handleResponse(response);
  }

  async getAICapabilities(): Promise<any> {
    const response = await fetch(`${this.baseURL}/../ai/capabilities`);
    return handleResponse(response);
  }

  // Utility method to download files
  downloadFile(blob: Blob, filename: string) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }
}

// Create singleton instance
export const apiClient = new DataScoutAPI();

// Utility functions for common operations
export const uploadUtils = {
  // Validate file type
  isValidFileType(file: File): boolean {
    const validTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json',
      'application/x-sqlite3',
    ];
    return validTypes.includes(file.type) || 
           file.name.toLowerCase().endsWith('.csv') ||
           file.name.toLowerCase().endsWith('.xlsx') ||
           file.name.toLowerCase().endsWith('.xls') ||
           file.name.toLowerCase().endsWith('.json') ||
           file.name.toLowerCase().endsWith('.sqlite') ||
           file.name.toLowerCase().endsWith('.db');
  },

  // Format file size
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  // Get file extension
  getFileExtension(filename: string): string {
    return filename.split('.').pop()?.toLowerCase() || '';
  },
};