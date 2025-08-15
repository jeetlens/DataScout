import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// Types for our application state
export interface DataFile {
  id: string; // Frontend file ID (timestamp)
  dataId?: string; // Backend data ID (from upload response)
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
  status: 'uploading' | 'uploaded' | 'processing' | 'analyzed' | 'error';
}

export interface AnalysisResults {
  id: string;
  summary: any;
  visualizations: any[];
  insights: any;
  aiInsights?: any;
  reports?: {
    html?: string;
    pdf?: string;
  };
}

export interface AppState {
  // Current workflow state
  currentStep: 'upload' | 'preview' | 'configure' | 'results' | 'reports';
  
  // File management
  currentFile: DataFile | null;
  files: DataFile[];
  
  // Data preview
  dataPreview: {
    columns: string[];
    rows: any[];
    totalRows: number;
    dataTypes: Record<string, string>;
  } | null;
  
  // Analysis configuration
  analysisConfig: {
    targetColumn?: string;
    businessContext?: string;
    analysisType: 'complete' | 'executive' | 'ai-focused';
    includeVisualization: boolean;
    includeAI: boolean;
  };
  
  // Results
  currentAnalysis: AnalysisResults | null;
  
  // UI state
  isLoading: boolean;
  error: string | null;
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    timestamp: Date;
  }>;
}

export interface AppActions {
  // Navigation
  setCurrentStep: (step: AppState['currentStep']) => void;
  
  // File management
  setCurrentFile: (file: DataFile | null) => void;
  addFile: (file: DataFile) => void;
  updateFileStatus: (id: string, status: DataFile['status']) => void;
  updateFileDataId: (id: string, dataId: string) => void;
  removeFile: (id: string) => void;
  
  // Data preview
  setDataPreview: (preview: AppState['dataPreview']) => void;
  
  // Analysis configuration
  updateAnalysisConfig: (config: Partial<AppState['analysisConfig']>) => void;
  resetAnalysisConfig: () => void;
  
  // Results
  setCurrentAnalysis: (analysis: AnalysisResults | null) => void;
  
  // UI state
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addNotification: (notification: Omit<AppState['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  
  // Reset
  resetState: () => void;
}

const initialState: AppState = {
  currentStep: 'upload',
  currentFile: null,
  files: [],
  dataPreview: null,
  analysisConfig: {
    analysisType: 'complete',
    includeVisualization: true,
    includeAI: true,
  },
  currentAnalysis: null,
  isLoading: false,
  error: null,
  notifications: [],
};

export const useAppStore = create<AppState & AppActions>()(
  devtools(
    (set, get) => ({
      ...initialState,
      
      // Navigation
      setCurrentStep: (step) => set({ currentStep: step }),
      
      // File management
      setCurrentFile: (file) => set({ currentFile: file }),
      
      addFile: (file) => set((state) => ({
        files: [...state.files, file],
        currentFile: file,
      })),
      
      updateFileStatus: (id, status) => set((state) => ({
        files: state.files.map(file => 
          file.id === id ? { ...file, status } : file
        ),
        currentFile: state.currentFile?.id === id 
          ? { ...state.currentFile, status }
          : state.currentFile,
      })),
      
      updateFileDataId: (id, dataId) => set((state) => ({
        files: state.files.map(file => 
          file.id === id ? { ...file, dataId } : file
        ),
        currentFile: state.currentFile?.id === id 
          ? { ...state.currentFile, dataId }
          : state.currentFile,
      })),
      
      removeFile: (id) => set((state) => ({
        files: state.files.filter(file => file.id !== id),
        currentFile: state.currentFile?.id === id ? null : state.currentFile,
      })),
      
      // Data preview
      setDataPreview: (preview) => set({ dataPreview: preview }),
      
      // Analysis configuration
      updateAnalysisConfig: (config) => set((state) => ({
        analysisConfig: { ...state.analysisConfig, ...config },
      })),
      
      resetAnalysisConfig: () => set({
        analysisConfig: initialState.analysisConfig,
      }),
      
      // Results
      setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
      
      // UI state
      setLoading: (loading) => set({ isLoading: loading }),
      setError: (error) => set({ error }),
      
      addNotification: (notification) => set((state) => ({
        notifications: [
          ...state.notifications,
          {
            ...notification,
            id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
            timestamp: new Date(),
          },
        ],
      })),
      
      removeNotification: (id) => set((state) => ({
        notifications: state.notifications.filter(n => n.id !== id),
      })),
      
      clearNotifications: () => set({ notifications: [] }),
      
      // Reset
      resetState: () => set(initialState),
    }),
    {
      name: 'datascout-store',
    }
  )
);

// Selectors for easier state access
export const useCurrentFile = () => useAppStore((state) => state.currentFile);
export const useCurrentStep = () => useAppStore((state) => state.currentStep);
export const useDataPreview = () => useAppStore((state) => state.dataPreview);
export const useAnalysisConfig = () => useAppStore((state) => state.analysisConfig);
export const useCurrentAnalysis = () => useAppStore((state) => state.currentAnalysis);
export const useIsLoading = () => useAppStore((state) => state.isLoading);
export const useError = () => useAppStore((state) => state.error);
export const useNotifications = () => useAppStore((state) => state.notifications);