import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { 
  LayoutDashboard, Upload, Database, Search, ChevronDown, ChevronRight, 
  Wrench, BarChart3, Lightbulb, FileText, Users, ScrollText, LogOut, 
  LogIn, Brain, Shield, Menu, X
} from 'lucide-react';
import axios from 'axios';
import logo from '../logo.png';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout, isAdmin, isManager } = useAuth();
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedDatasets, setExpandedDatasets] = useState(false);

  useEffect(() => {
    fetchDatasets();
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

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const ROLE_COLORS = {
    admin: 'bg-purple-100 text-purple-800',
    manager: 'bg-blue-100 text-blue-800',
    analyst: 'bg-green-100 text-green-800',
    viewer: 'bg-slate-100 text-slate-800'
  };

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } fixed inset-y-0 left-0 z-50 w-72 bg-white border-r border-slate-200 transition-transform duration-200 lg:translate-x-0 lg:static flex flex-col`}
      >
        {/* Header - Fixed */}
<div className="p-5 border-b border-slate-200">
  <Link to="/" className="flex items-center gap-3 group">
    {/* Logo */}
    <img
      src={logo}
      alt="E1 Analytics Logo"
      className="w-10 h-10 rounded-lg object-contain"
    />

    {/* Text Container */}
    <div className="flex flex-col">
      <h1 className="text-xl font-bold text-slate-900 leading-tight">
        Data Analytics
      </h1>
      <p className="text-xs text-slate-600">
        Data Intelligence Platform
      </p>
    </div>
  </Link>
</div>
        
        {/* Scrollable Navigation Container */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {/* Main Navigation */}
          <nav className="px-2.5 py-2.5">
            <Link
              to="/"
              data-testid="nav-home"
              className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                location.pathname === '/' ? 'active' : 'text-slate-700'
              }`}
            >
              <LayoutDashboard className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-xs">Home</span>
            </Link>
            <Link
              to="/upload"
              data-testid="nav-upload"
              className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                location.pathname === '/upload' ? 'active' : 'text-slate-700'
              }`}
            >
              <Upload className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-xs">Upload Data</span>
            </Link>
          </nav>

          {/* Analysis Navigation */}
          <nav className="px-2.5 py-2.5 border-t border-slate-200">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5 px-2.5">Analysis</p>
            <Link
              to="/preparation"
              data-testid="nav-data-prep"
              className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                location.pathname === '/preparation' || location.pathname.includes('/preparation') ? 'active' : 'text-slate-700'
              }`}
            >
              <Wrench className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-xs">Data Prep</span>
            </Link>
            <Link
              to="/analytics"
              data-testid="nav-analytics"
              className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                location.pathname === '/analytics' || location.pathname.includes('/analytics') ? 'active' : 'text-slate-700'
              }`}
            >
              <BarChart3 className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-xs">Analytics</span>
            </Link>
            <Link
              to="/insights"
              data-testid="nav-insights"
              className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                location.pathname === '/insights' || location.pathname.includes('/insights') ? 'active' : 'text-slate-700'
              }`}
            >
              <Lightbulb className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-xs">Insights</span>
            </Link>
            <Link
              to="/ai-insights"
              data-testid="nav-ai-insights"
              className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                location.pathname === '/ai-insights' || location.pathname.includes('/ai-insights') ? 'active' : 'text-slate-700'
              }`}
            >
              <Brain className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-xs">AI Insights</span>
              <span className="ml-auto px-1.5 py-0.5 bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-xs rounded-full flex-shrink-0 font-semibold">NEW</span>
            </Link>
            <Link
              to="/reports"
              data-testid="nav-reports"
              className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                location.pathname === '/reports' || location.pathname.includes('/reports') ? 'active' : 'text-slate-700'
              }`}
            >
              <FileText className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-xs">Reports</span>
            </Link>
          </nav>

          {/* Admin Navigation - only shown to admin/manager */}
          {(isAdmin() || isManager()) && (
            <nav className="px-2.5 py-2.5 border-t border-slate-200">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5 px-2.5">Administration</p>
              <Link
                to="/users"
                data-testid="nav-users"
                className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                  location.pathname === '/users' ? 'active' : 'text-slate-700'
                }`}
              >
                <Users className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                <span className="font-medium text-xs">User Management</span>
              </Link>
              <Link
                to="/audit-logs"
                data-testid="nav-audit"
                className={`sidebar-link flex items-center gap-2 px-2.5 py-1.5 rounded-lg mb-0.5 transition-all ${
                  location.pathname === '/audit-logs' ? 'active' : 'text-slate-700'
                }`}
              >
                <ScrollText className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                <span className="font-medium text-xs">Audit Logs</span>
              </Link>
            </nav>
          )}

          {/* Datasets Section */}
          <div className="px-2.5 py-2.5 border-t border-slate-200">
            <div className="mb-1.5">
              <button
                onClick={() => setExpandedDatasets(!expandedDatasets)}
                className="flex items-center gap-1.5 text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 w-full hover:text-slate-900 transition-colors px-2.5"
              >
                {expandedDatasets ? <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 flex-shrink-0" />}
                <Database className="w-3.5 h-3.5 flex-shrink-0" />
                <span>My Datasets</span>
                <span className="ml-auto text-xs font-semibold text-slate-500">({datasets.length})</span>
              </button>
              
              {expandedDatasets && (
                <div className="mb-2">
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-2 text-slate-400 pointer-events-none" />
                    <input
                      type="text"
                      placeholder="Search datasets..."
                      value={searchQuery}
                      onChange={handleSearch}
                      className="w-full h-7 pl-8 pr-2.5 rounded-lg border border-slate-300 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:border-transparent transition-all"
                    />
                  </div>
                </div>
              )}
            </div>

            {expandedDatasets && (
              <div className="space-y-1">
                {datasets.length === 0 ? (
                  <div className="px-2 py-6 text-center">
                    <Database className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                    <p className="text-xs text-slate-500 mb-2">No datasets found</p>
                    <Link
                      to="/upload"
                      className="inline-block text-xs text-indigo-600 hover:text-indigo-700 font-medium transition-colors"
                    >
                      Upload your first dataset
                    </Link>
                  </div>
                ) : (
                  datasets.map((dataset) => (
                    <Link
                      key={dataset.id}
                      to={`/dataset/${dataset.id}/overview`}
                      data-testid={`dataset-${dataset.id}`}
                      className={`block px-2 py-2 rounded-lg hover:bg-slate-100 transition-all ${ 
                        location.pathname.includes(`/dataset/${dataset.id}`) ? 'bg-indigo-50 border-l-3 border-indigo-600' : 'border-l-3 border-transparent'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <div className="p-1 bg-slate-100 rounded-lg flex-shrink-0">
                          <Database className="w-3.5 h-3.5 text-slate-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-slate-900 truncate mb-0.5">{dataset.title || dataset.name}</p>
                          <div className="flex items-center gap-2 text-xs text-slate-500">
                            <span className="flex items-center gap-1">
                              <span className="w-1.5 h-1.5 bg-green-500 rounded-full flex-shrink-0"></span>
                              {dataset.rows.toLocaleString()}
                            </span>
                            <span>•</span>
                            <span>{dataset.columns} cols</span>
                          </div>
                        </div>
                      </div>
                    </Link>
                  ))
                )}
              </div>
            )}
          </div>
        </div>

        {/* User Section - Fixed at Bottom */}
        <div className="flex-shrink-0 px-3 py-2.5 border-t border-slate-200 bg-slate-50">
          {user ? (
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center flex-shrink-0">
                <span className="text-indigo-700 font-semibold text-xs">
                  {user.name?.charAt(0).toUpperCase() || 'U'}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-slate-900 truncate">{user.name}</p>
                <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-xs font-medium ${ROLE_COLORS[user.role]}`}>
                  <Shield className="w-2.5 h-2.5" />
                  {user.role}
                </span>
              </div>
              <button
                onClick={handleLogout}
                className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-200 rounded-lg transition-all flex-shrink-0"
                title="Logout"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className="flex items-center gap-2 px-3 py-2 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 transition-colors"
            >
              <LogIn className="w-4 h-4" />
              <span className="font-medium text-xs">Sign In</span>
            </Link>
          )}
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
        <header className="bg-white border-b border-slate-200 px-6 py-3.5">
          <div className="flex items-center justify-between">
            <button
              data-testid="mobile-menu-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-slate-100 transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <div className="flex-1" />
            {!user && (
              <Link
                to="/login"
                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-xs font-medium"
              >
                <LogIn className="w-3.5 h-3.5" />
                Sign In
              </Link>
            )}
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