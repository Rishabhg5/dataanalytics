import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import DataUpload from './pages/DataUpload';
import DatasetOverview from './pages/DatasetOverview';
import DataPreparation from './pages/DataPreparation';
import Analytics from './pages/Analytics';
import Insights from './pages/Insights';
import Reports from './pages/Reports';
import { Toaster } from '@/components/ui/sonner';
import '@/App.css';

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="upload" element={<DataUpload />} />
            
            {/* Dataset-centric routes */}
            <Route path="dataset/:datasetId" element={<DatasetOverview />}>
              <Route path="overview" element={<div />} />
              <Route path="preparation" element={<DataPreparation />} />
              <Route path="analytics" element={<Analytics />} />
              <Route path="insights" element={<Insights />} />
              <Route path="reports" element={<Reports />} />
            </Route>
            
            {/* Legacy routes - redirect to home */}
            <Route path="preparation" element={<Navigate to="/" replace />} />
            <Route path="analytics" element={<Navigate to="/" replace />} />
            <Route path="insights" element={<Navigate to="/" replace />} />
            <Route path="reports" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" />
    </>
  );
}

export default App;