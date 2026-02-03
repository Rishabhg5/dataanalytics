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

  // Adjusted role colors for the dark theme footer
  const ROLE_COLORS = {
    admin: 'bg-purple-500/20 text-purple-200 border border-purple-500/30',
    manager: 'bg-blue-500/20 text-blue-200 border border-blue-500/30',
    analyst: 'bg-green-500/20 text-green-200 border border-green-500/30',
    viewer: 'bg-slate-500/20 text-slate-200 border border-slate-500/30'
  };

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } fixed inset-y-0 left-0 z-50 w-72 bg-[#002e5b] border-r border-slate-700 transition-transform duration-200 lg:translate-x-0 lg:static flex flex-col`}
      >
        {/* Header - Fixed */}
        <div className="p-5 border-b border-white/10">
          <Link to="/" className="flex items-center gap-3 group">
            {/* Logo */}
            <div className="p-1 bg-white rounded-lg">
                <img
                src={logo}
                alt="E1 Analytics Logo"
                className="w-8 h-8 object-contain"
                />
            </div>

            {/* Text Container */}
            <div className="flex flex-col">
              <h1 className="text-xl font-bold text-white leading-tight">
                Data Analytics
              </h1>
              <p className="text-xs text-slate-300">
                Data Intelligence Platform
              </p>
            </div>
          </Link>
        </div>
        
        {/* Scrollable Navigation Container */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {/* Main Navigation */}
          <nav className="px-2.5 py-4">
            <Link
              to="/"
              data-testid="nav-home"
              className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                location.pathname === '/' 
                ? 'bg-indigo-500 text-white shadow-md' 
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
              }`}
            >
              <LayoutDashboard className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-sm">Home</span>
            </Link>
            <Link
              to="/upload"
              data-testid="nav-upload"
              className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                location.pathname === '/upload' 
                ? 'bg-indigo-500 text-white shadow-md' 
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
              }`}
            >
              <Upload className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-sm">Upload Data</span>
            </Link>
          </nav>

          {/* Analysis Navigation */}
          <nav className="px-2.5 py-4 border-t border-white/10">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-3">Analysis</p>
            <Link
              to="/preparation"
              data-testid="nav-data-prep"
              className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                location.pathname.includes('/preparation') 
                ? 'bg-indigo-500 text-white shadow-md' 
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
              }`}
            >
              <Wrench className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-sm">Data Prep</span>
            </Link>
            <Link
              to="/analytics"
              data-testid="nav-analytics"
              className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                location.pathname.includes('/analytics') 
                ? 'bg-indigo-500 text-white shadow-md' 
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
              }`}
            >
              <BarChart3 className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-sm">Analytics</span>
            </Link>
            <Link
              to="/insights"
              data-testid="nav-insights"
              className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                location.pathname.includes('/insights') 
                ? 'bg-indigo-500 text-white shadow-md' 
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
              }`}
            >
              <Lightbulb className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-sm">Insights</span>
            </Link>
            <Link
              to="/ai-insights"
              data-testid="nav-ai-insights"
              className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                location.pathname.includes('/ai-insights') 
                ? 'bg-indigo-500 text-white shadow-md' 
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
              }`}
            >
              <Brain className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-sm">AI Insights</span>
              <span className="ml-auto px-2 py-0.5 bg-gradient-to-r from-pink-500 to-purple-500 text-white text-[10px] rounded-full flex-shrink-0 font-bold">NEW</span>
            </Link>
            <Link
              to="/reports"
              data-testid="nav-reports"
              className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                location.pathname.includes('/reports') 
                ? 'bg-indigo-500 text-white shadow-md' 
                : 'text-slate-300 hover:bg-white/10 hover:text-white'
              }`}
            >
              <FileText className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
              <span className="font-medium text-sm">Reports</span>
            </Link>
          </nav>

          {/* Admin Navigation */}
          {(isAdmin() || isManager()) && (
            <nav className="px-2.5 py-4 border-t border-white/10">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-3">Administration</p>
              <Link
                to="/users"
                data-testid="nav-users"
                className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                  location.pathname === '/users' 
                  ? 'bg-indigo-500 text-white shadow-md' 
                  : 'text-slate-300 hover:bg-white/10 hover:text-white'
                }`}
              >
                <Users className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
                <span className="font-medium text-sm">User Management</span>
              </Link>
              <Link
                to="/audit-logs"
                data-testid="nav-audit"
                className={`sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all ${
                  location.pathname === '/audit-logs' 
                  ? 'bg-indigo-500 text-white shadow-md' 
                  : 'text-slate-300 hover:bg-white/10 hover:text-white'
                }`}
              >
                <ScrollText className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
                <span className="font-medium text-sm">Audit Logs</span>
              </Link>
            </nav>
          )}

          {/* Datasets Section */}
          <div className="px-2.5 py-4 border-t border-white/10">
            <div className="mb-2">
              <button
                onClick={() => setExpandedDatasets(!expandedDatasets)}
                className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 w-full hover:text-white transition-colors px-3"
              >
                {expandedDatasets ? <ChevronDown className="w-4 h-4 flex-shrink-0" /> : <ChevronRight className="w-4 h-4 flex-shrink-0" />}
                <Database className="w-4 h-4 flex-shrink-0" />
                <span>My Datasets</span>
                <span className="ml-auto text-xs font-bold bg-white/10 px-2 py-0.5 rounded text-white">({datasets.length})</span>
              </button>
              
              {expandedDatasets && (
                <div className="mb-3 px-1">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400 pointer-events-none" />
                    <input
                      type="text"
                      placeholder="Search datasets..."
                      value={searchQuery}
                      onChange={handleSearch}
                      className="w-full h-9 pl-9 pr-3 rounded-lg border border-slate-600 bg-[#00254a] text-white placeholder-slate-400 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                    />
                  </div>
                </div>
              )}
            </div>

            {expandedDatasets && (
              <div className="space-y-1">
                {datasets.length === 0 ? (
                  <div className="px-2 py-6 text-center border border-dashed border-slate-700 rounded-lg bg-white/5">
                    <Database className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                    <p className="text-xs text-slate-400 mb-2">No datasets found</p>
                    <Link
                      to="/upload"
                      className="inline-block text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
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
                      className={`block px-3 py-2.5 rounded-lg transition-all ${ 
                        location.pathname.includes(`/dataset/${dataset.id}`) 
                        ? 'bg-indigo-600/20 border border-indigo-500/30' 
                        : 'hover:bg-white/10 border border-transparent'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="p-1.5 bg-white/10 rounded-md flex-shrink-0">
                          <Database className="w-3.5 h-3.5 text-white" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-white truncate mb-0.5">{dataset.title || dataset.name}</p>
                          <div className="flex items-center gap-2 text-[10px] text-slate-400">
                            <span className="flex items-center gap-1">
                              <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full flex-shrink-0"></span>
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
        <div className="flex-shrink-0 px-4 py-4 border-t border-white/10 bg-[#00254a]">
          {user ? (
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-indigo-500 flex items-center justify-center flex-shrink-0 shadow-lg">
                <span className="text-white font-bold text-sm">
                  {user.name?.charAt(0).toUpperCase() || 'U'}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">{user.name}</p>
                <div className={`inline-flex items-center gap-1 mt-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${ROLE_COLORS[user.role]}`}>
                  <Shield className="w-2.5 h-2.5" />
                  {user.role}
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="p-2 text-slate-400 hover:text-white hover:bg-white/10 rounded-lg transition-all flex-shrink-0"
                title="Logout"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className="flex items-center gap-2 px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 transition-colors w-full justify-center shadow-lg"
            >
              <LogIn className="w-4 h-4" />
              <span className="font-medium text-sm">Sign In</span>
            </Link>
          )}
        </div>
      </aside>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden backdrop-blur-sm"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden bg-slate-50">
        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-4 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}