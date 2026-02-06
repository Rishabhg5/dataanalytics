import React from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Layout from './components/Layout';
import Home from './pages/Home';
import Login from './pages/Login';
import DataUpload from './pages/DataUpload';
import DatasetOverview from './pages/DatasetOverview';
import DataPreparation from './pages/DataPreparation';
import Analytics from './pages/Analytics';
import Insights from './pages/Insights';
import AIInsights from './pages/AIInsights';
import Reports from './pages/Reports';
import UserManagement from './pages/UserManagement';
import AuditLogs from './pages/AuditLogs';
import { Toaster } from './components/ui/sonner';
//import DataComparison from './pages/DataComparison';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <HashRouter>
        <Routes>
          {/* Auth route - no layout */}
          <Route path="/login" element={<Login />} />
          
          {/* Main app routes with layout */}
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="upload" element={<DataUpload />} />
            
            {/* Direct navigation routes */}
            <Route path="preparation" element={<DataPreparation />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="insights" element={<Insights />} />
            <Route path="ai-insights" element={<AIInsights />} />
            <Route path="reports" element={<Reports />} />
            {/* <Route path="comparison" element={<DataComparison />} /> */}
            
            {/* Admin routes */}
            <Route path="users" element={<UserManagement />} />
            <Route path="audit-logs" element={<AuditLogs />} />
            
            {/* Dataset-centric routes */}
            <Route path="dataset/:datasetId" element={<DatasetOverview />}>
              <Route path="overview" element={<div />} />
              <Route path="preparation" element={<DataPreparation />} />
              <Route path="analytics" element={<Analytics />} />
              <Route path="insights" element={<Insights />} />
              <Route path="ai-insights" element={<AIInsights />} />
              <Route path="reports" element={<Reports />} />
            </Route>
          </Route>
        </Routes>
      </HashRouter>
      <Toaster position="top-right" />
    </AuthProvider>
  );
}

export default App;
