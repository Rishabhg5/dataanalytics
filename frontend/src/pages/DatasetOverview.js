import React, { useState, useEffect } from 'react';
import { useParams, Link, Outlet, useLocation } from 'react-router-dom';
import axios from 'axios';
import { Database, Wrench, BarChart3, Lightbulb, FileText, ChevronRight, Calendar, Layers } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function DatasetOverview() {
  const { datasetId } = useParams();
  const location = useLocation();
  const [dataset, setDataset] = useState(null);
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (datasetId) {
      fetchDataset();
      fetchOverview();
    }
  }, [datasetId]);

  const fetchDataset = async () => {
    try {
      const response = await axios.get(`${API}/datasets/${datasetId}`);
      setDataset(response.data.dataset);
    } catch (error) {
      console.error('Error fetching dataset:', error);
      toast.error('Failed to load dataset');
    } finally {
      setLoading(false);
    }
  };

  const fetchOverview = async () => {
    try {
      const response = await axios.get(`${API}/datasets/${datasetId}/overview`);
      if (response.data) {
        setOverview(response.data);
      }
    } catch (error) {
      console.error('Error fetching overview:', error);
    }
  };

  const tabs = [
    { path: `/dataset/${datasetId}/overview`, label: 'Overview', icon: Database },
    { path: `/dataset/${datasetId}/preparation`, label: 'Data Prep', icon: Wrench },
    { path: `/dataset/${datasetId}/analytics`, label: 'Analytics', icon: BarChart3 },
    { path: `/dataset/${datasetId}/insights`, label: 'Insights', icon: Lightbulb },
    { path: `/dataset/${datasetId}/reports`, label: 'Reports', icon: FileText },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-slate-600">Loading dataset...</p>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-slate-600">Dataset not found</p>
      </div>
    );
  }

  // Check if we're on the overview page specifically
  const isOverviewPage = location.pathname === `/dataset/${datasetId}/overview`;

  return (
    <div className="max-w-7xl mx-auto">
      {/* Dataset Header */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 mb-2">{dataset.title}</h1>
            <p className="text-slate-600">{dataset.name}</p>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <Calendar className="w-4 h-4" />
            {new Date(dataset.uploaded_at).toLocaleDateString()}
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-50 rounded-lg p-4">
            <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Total Rows</p>
            <p className="text-2xl font-bold text-slate-900">{dataset.rows.toLocaleString()}</p>
          </div>
          <div className="bg-slate-50 rounded-lg p-4">
            <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Columns</p>
            <p className="text-2xl font-bold text-slate-900">{dataset.columns}</p>
          </div>
          <div className="bg-slate-50 rounded-lg p-4">
            <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Data Quality</p>
            <p className="text-2xl font-bold text-green-600">Good</p>
          </div>
          <div className="bg-slate-50 rounded-lg p-4">
            <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Status</p>
            <p className="text-2xl font-bold text-indigo-600">Ready</p>
          </div>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm mb-6 overflow-x-auto">
        <div className="flex">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = location.pathname === tab.path;
            return (
              <Link
                key={tab.path}
                to={tab.path}
                data-testid={`tab-${tab.label.toLowerCase().replace(' ', '-')}`}
                className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-colors whitespace-nowrap ${
                  isActive
                    ? 'border-indigo-600 text-indigo-600 bg-indigo-50'
                    : 'border-transparent text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <Icon className="w-5 h-5" strokeWidth={1.5} />
                <span className="font-medium">{tab.label}</span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Overview Content (only on overview page) */}
      {isOverviewPage && (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold text-slate-900 mb-4">Dataset Information</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-sm font-medium text-slate-700 mb-3">Columns</h3>
                <div className="flex flex-wrap gap-2">
                  {dataset.column_names.map((col, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-slate-100 border border-slate-200 rounded-lg text-sm text-slate-700"
                    >
                      {col}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-slate-700 mb-3">Column Types</h3>
                <div className="space-y-2">
                  {Object.entries(dataset.column_types).slice(0, 5).map(([col, type]) => (
                    <div key={col} className="flex justify-between text-sm">
                      <span className="text-slate-600">{col}:</span>
                      <span className="font-medium text-slate-900">{type}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {tabs.slice(1).map((tab) => {
              const Icon = tab.icon;
              return (
                <Link
                  key={tab.path}
                  to={tab.path}
                  className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 hover:border-indigo-200 hover:shadow-md transition-all"
                >
                  <Icon className="w-8 h-8 text-indigo-600 mb-4" strokeWidth={1.5} />
                  <h3 className="font-semibold text-slate-900 mb-2">{tab.label}</h3>
                  <p className="text-sm text-slate-600">
                    {tab.label === 'Data Prep' && 'Clean and prepare your data'}
                    {tab.label === 'Analytics' && 'Explore with visualizations'}
                    {tab.label === 'Insights' && 'Discover trends & forecasts'}
                    {tab.label === 'Reports' && 'Generate PDF reports'}
                  </p>
                  <div className="mt-4 flex items-center text-sm text-indigo-600 font-medium">
                    Open <ChevronRight className="w-4 h-4 ml-1" />
                  </div>
                </Link>
              );
            })}
          </div>

          {/* Stored Overview */}
          {overview && (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold text-slate-900 mb-4">Analysis Summary</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {overview.statistics && (
                  <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                    <Layers className="w-6 h-6 text-green-600 mb-2" />
                    <p className="text-sm font-medium text-green-900">Statistics Generated</p>
                    <p className="text-xs text-green-700 mt-1">Descriptive analysis complete</p>
                  </div>
                )}
                {overview.trends && (
                  <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <BarChart3 className="w-6 h-6 text-blue-600 mb-2" />
                    <p className="text-sm font-medium text-blue-900">Trends Analyzed</p>
                    <p className="text-xs text-blue-700 mt-1">Pattern detection complete</p>
                  </div>
                )}
                {overview.forecast && (
                  <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                    <Lightbulb className="w-6 h-6 text-purple-600 mb-2" />
                    <p className="text-sm font-medium text-purple-900">Forecast Generated</p>
                    <p className="text-xs text-purple-700 mt-1">Predictions available</p>
                  </div>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-4">
                Last updated: {new Date(overview.last_updated).toLocaleString()}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Nested routes content */}
      {!isOverviewPage && <Outlet />}
    </div>
  );
}