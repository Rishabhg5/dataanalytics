import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { Database, TrendingUp, AlertTriangle, BarChart3, Upload, ArrowRight, Search, Wrench } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Home() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState(''); // Search state

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async (search = '') => {
    try {
      const url = search ? `${API}/datasets?search=${encodeURIComponent(search)}` : `${API}/datasets`;
      const response = await axios.get(url);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
      toast.error('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    fetchDatasets(query);
  };

  const stats = [
    {
      label: 'Total Datasets',
      value: datasets.length,
      icon: Database,
      color: 'text-indigo-600',
      bg: 'bg-indigo-50',
    },
    {
      label: 'Total Records',
      value: datasets.reduce((sum, ds) => sum + ds.rows, 0),
      icon: BarChart3,
      color: 'text-blue-600',
      bg: 'bg-blue-50',
    },
    {
      label: 'Avg Columns',
      value: datasets.length > 0 ? Math.round(datasets.reduce((sum, ds) => sum + ds.columns, 0) / datasets.length) : 0,
      icon: TrendingUp,
      color: 'text-green-600',
      bg: 'bg-green-50',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Hero Section */}
<div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm relative">
  <div className="p-8 lg:p-12">
    {/* Top Right Actions Container */}
    <div className="flex flex-col sm:flex-row items-center gap-3 absolute top-6 right-6 max-w-md w-full sm:w-auto">
      {/* Integrated Search Bar */}
      <div className="relative w-full sm:w-64">
        <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
        <input
          type="text"
          placeholder="Search datasets..."
          value={searchQuery}
          onChange={handleSearch}
          className="w-full h-9 pl-10 pr-4 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:border-transparent transition-all bg-slate-50/50 p-4"
        />
      </div>

      {datasets.length === 0 && (
        <Link
          to="/upload"
          data-testid="get-started-btn"
          className="inline-flex items-center gap-2 bg-indigo-600 text-white hover:bg-indigo-700 h-9 px-4 rounded-lg text-sm font-medium transition-all active:scale-95 whitespace-nowrap shadow-sm"
        >
          <Upload className="w-4 h-4" />
          Upload
        </Link>
      )}
    </div>

    {/* Content */}
    <div className="max-w-2xl">
      <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
        Welcome to Data Analytics
      </h1>
      <p className="text-lg text-slate-600 leading-relaxed">
        Your intelligent data analytics platform for advanced insights, predictive analytics, and beautiful visualizations.
      </p>
    </div>
  </div>
</div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div
              key={index}
              data-testid={`stat-card-${index}`}
              className="stat-card bg-white border border-slate-200 rounded-xl shadow-sm p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-lg ${stat.bg}`}>
                  <Icon className={`w-6 h-6 ${stat.color}`} strokeWidth={1.5} />
                </div>
              </div>
              <p className="text-3xl font-bold text-slate-900 mb-1">{stat.value.toLocaleString()}</p>
              <p className="text-sm text-slate-600">{stat.label}</p>
            </div>
          );
        })}
      </div>

      {/* Recent Datasets */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-semibold text-slate-900">
            {searchQuery ? 'Search Results' : 'Recent Datasets'}
          </h2>
          <Link
            to="/upload"
            className="text-sm font-medium text-indigo-600 hover:text-indigo-700 flex items-center gap-1"
          >
            Upload More
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        {datasets.length === 0 ? (
          <div className="text-center py-12">
            <Database className="w-12 h-12 text-slate-300 mx-auto mb-3" />
            <p className="text-slate-500">No datasets found matching your search.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {datasets.slice(0, 5).map((dataset) => (
              <Link
                key={dataset.id}
                to={`/analytics?dataset=${dataset.id}`}
                className="block p-4 rounded-lg border border-slate-100 bg-slate-50/50 hover:bg-white hover:border-indigo-200 hover:shadow-md transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="p-2 bg-white rounded-lg border border-slate-200">
                      <Database className="w-5 h-5 text-indigo-600" />
                    </div>
                    <div>
                      <h3 className="font-medium text-slate-900 mb-0.5">{dataset.name}</h3>
                      <p className="text-xs text-slate-500">
                        {dataset.rows.toLocaleString()} rows • {dataset.columns} columns
                      </p>
                    </div>
                  </div>
                  <ArrowRight className="w-5 h-5 text-slate-300" />
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 pb-8">
        {[
          { title: 'Data Ingestion', desc: 'Upload CSV and Excel files', icon: Upload, link: '/upload' },
          { title: 'Data Cleaning', desc: 'Handle missing values & duplicates', icon: Wrench, link: '/preparation' },
          { title: 'Analytics', desc: 'Descriptive & predictive insights', icon: BarChart3, link: '/analytics' },
          { title: 'Insights', desc: 'Trends, anomalies & forecasts', icon: AlertTriangle, link: '/insights' },
        ].map((feature, idx) => {
          const Icon = feature.icon;
          return (
            <Link
              key={idx}
              to={feature.link}
              className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 hover:border-indigo-200 hover:shadow-md transition-all group"
            >
              <Icon className="w-8 h-8 text-indigo-600 mb-4 transition-transform group-hover:-translate-y-1" strokeWidth={1.5} />
              <h3 className="font-semibold text-slate-900 mb-2">{feature.title}</h3>
              <p className="text-sm text-slate-600">{feature.desc}</p>
            </Link>
          );
        })}
      </div>
    </div>
  );
}