import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';

// Import components (we'll create these next)
import Navigation from './components/Navigation';
import FileUpload from './components/FileUpload';
import DataPreview from './components/DataPreview';
import AnalysisConfiguration from './components/AnalysisConfiguration';
import ResultsDisplay from './components/ResultsDisplay';
import ReportsInterface from './components/ReportsInterface';
import NotificationCenter from './components/NotificationCenter';
import LoadingOverlay from './components/LoadingOverlay';

// Main App component
function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background-white">
        {/* Navigation */}
        <Navigation />
        
        {/* Main Content */}
        <main className="container mx-auto px-4 py-8">
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<Navigate to="/upload" replace />} />
              <Route path="/upload" element={<FileUpload />} />
              <Route path="/preview" element={<DataPreview />} />
              <Route path="/configure" element={<AnalysisConfiguration />} />
              <Route path="/results" element={<ResultsDisplay />} />
              <Route path="/reports" element={<ReportsInterface />} />
            </Routes>
          </AnimatePresence>
        </main>

        {/* Global UI Elements */}
        <NotificationCenter />
        <LoadingOverlay />
      </div>
    </Router>
  );
}

export default App;