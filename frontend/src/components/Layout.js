import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation, useParams } from 'react-router-dom';
import { LayoutDashboard, Upload, Database, Search, ChevronDown, ChevronRight, Wrench, BarChart3, Lightbulb, FileText } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Layout() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedDatasets, setExpandedDatasets] = useState(true);

  useEffect(() => {
    fetchDatasets();
    // Poll for updates every 3 seconds
    const interval = setInterval(() => {
      fetchDatasets(searchQuery);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchDatasets = async (search = '') => {
    try {
      const url = search ? `${API}/datasets?search=${encodeURIComponent(search)}` : `${API}/datasets`;
      const response = await axios.get(url);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const handleSearch = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    fetchDatasets(query);
  };

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } fixed inset-y-0 left-0 z-50 w-72 bg-white border-r border-slate-200 transition-transform duration-200 lg:translate-x-0 lg:static flex flex-col`}
      >
        <div className="p-6 border-b border-slate-200">
          <Link to="/" className="block">
            <h1 className="text-2xl font-bold text-slate-900">E1 Analytics</h1>
            <p className="text-sm text-slate-600 mt-1">Data Intelligence Platform</p>
          </Link>
        </div>
        
        {/* Main Navigation */}
        <nav className="p-4 border-b border-slate-200">
          <Link
            to="/"
            data-testid="nav-home"
            className={`sidebar-link flex items-center gap-3 px-4 py-3 rounded-lg mb-2 ${
              location.pathname === '/' ? 'active' : 'text-slate-700'
            }`}
          >
            <LayoutDashboard className="w-5 h-5" strokeWidth={1.5} />
            <span className="font-medium">Home</span>
          </Link>
          <Link
            to="/upload"
            data-testid="nav-upload"
            className={`sidebar-link flex items-center gap-3 px-4 py-3 rounded-lg mb-2 ${
              location.pathname === '/upload' ? 'active' : 'text-slate-700'
            }`}
          >
            <Upload className="w-5 h-5" strokeWidth={1.5} />
            <span className="font-medium">Upload Data</span>
          </Link>
        </nav>

        {/* Analysis Navigation */}
        <nav className="p-4 border-b border-slate-200">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-4">Analysis</p>
          <Link
            to="/preparation"
            data-testid="nav-data-prep"
            className={`sidebar-link flex items-center gap-3 px-4 py-3 rounded-lg mb-2 ${
              location.pathname === '/preparation' || location.pathname.includes('/preparation') ? 'active' : 'text-slate-700'
            }`}
          >
            <Wrench className="w-5 h-5" strokeWidth={1.5} />
            <span className="font-medium">Data Prep</span>
          </Link>
          <Link
            to="/analytics"
            data-testid="nav-analytics"
            className={`sidebar-link flex items-center gap-3 px-4 py-3 rounded-lg mb-2 ${
              location.pathname === '/analytics' || location.pathname.includes('/analytics') ? 'active' : 'text-slate-700'
            }`}
          >
            <BarChart3 className="w-5 h-5" strokeWidth={1.5} />
            <span className="font-medium">Analytics</span>
          </Link>
          <Link
            to="/insights"
            data-testid="nav-insights"
            className={`sidebar-link flex items-center gap-3 px-4 py-3 rounded-lg mb-2 ${
              location.pathname === '/insights' || location.pathname.includes('/insights') ? 'active' : 'text-slate-700'
            }`}
          >
            <Lightbulb className="w-5 h-5" strokeWidth={1.5} />
            <span className="font-medium">Insights</span>
          </Link>
          <Link
            to="/reports"
            data-testid="nav-reports"
            className={`sidebar-link flex items-center gap-3 px-4 py-3 rounded-lg mb-2 ${
              location.pathname === '/reports' || location.pathname.includes('/reports') ? 'active' : 'text-slate-700'
            }`}
          >
            <FileText className="w-5 h-5" strokeWidth={1.5} />
            <span className="font-medium">Reports</span>
          </Link>
        </nav>

        {/* Datasets Section */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="mb-3">
            <button
              onClick={() => setExpandedDatasets(!expandedDatasets)}
              className="flex items-center gap-2 text-sm font-semibold text-slate-700 uppercase tracking-wider mb-2 w-full hover:text-slate-900"
            >
              {expandedDatasets ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
              <Database className="w-4 h-4" />
              Datasets ({datasets.length})
            </button>
            
            {expandedDatasets && (
              <div className="mb-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search datasets..."
                    value={searchQuery}
                    onChange={handleSearch}
                    className="w-full h-9 pl-9 pr-3 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
              </div>
            )}
          </div>

          {expandedDatasets && (
            <div className="space-y-1">
              {datasets.length === 0 ? (
                <p className="text-sm text-slate-500 px-4 py-2">No datasets found</p>
              ) : (
                datasets.map((dataset) => (
                  <Link
                    key={dataset.id}
                    to={`/dataset/${dataset.id}/overview`}
                    data-testid={`dataset-${dataset.id}`}
                    className={`block px-4 py-2.5 rounded-lg hover:bg-slate-100 transition-colors ${
                      location.pathname.includes(`/dataset/${dataset.id}`) ? 'bg-indigo-50 text-indigo-700 border-l-4 border-indigo-600' : 'text-slate-700'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <Database className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{dataset.title}</p>
                        <p className="text-xs text-slate-500 truncate">{dataset.rows.toLocaleString()} rows</p>
                      </div>
                    </div>
                  </Link>
                ))
              )}
            </div>
          )}
        </div>
        
        <div className="p-6 border-t border-slate-200">
          <div className="bg-slate-100 rounded-lg p-4">
            <p className="text-xs font-medium text-slate-700 uppercase tracking-wider mb-1">Version</p>
            <p className="text-sm text-slate-900">v1.0.0</p>
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <button
              data-testid="mobile-menu-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <LayoutDashboard className="w-6 h-6" />
            </button>
            <div className="flex-1" />
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}