import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { Database, TrendingUp, AlertTriangle, BarChart3, Upload, ArrowRight } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Home() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
      toast.error('Failed to load datasets');
    } finally {
      setLoading(false);
    }
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
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="p-8 lg:p-12">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
            Welcome to E1 Analytics
          </h1>
          <p className="text-lg text-slate-600 leading-relaxed max-w-2xl mb-8">
            Your intelligent data analytics platform for advanced insights, predictive analytics, and beautiful visualizations.
          </p>
          {datasets.length === 0 && (
            <Link
              to="/upload"
              data-testid="get-started-btn"
              className="inline-flex items-center gap-2 bg-indigo-600 text-white hover:bg-indigo-700 h-12 px-8 rounded-lg font-medium transition-all active:scale-95"
            >
              <Upload className="w-5 h-5" />
              Get Started - Upload Data
              <ArrowRight className="w-5 h-5" />
            </Link>
          )}
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
      {datasets.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-slate-900">Recent Datasets</h2>
            <Link
              to="/upload"
              data-testid="upload-more-btn"
              className="text-sm font-medium text-indigo-600 hover:text-indigo-700 flex items-center gap-1"
            >
              Upload More
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="space-y-3">
            {datasets.slice(0, 5).map((dataset) => (
              <Link
                key={dataset.id}
                to={`/analytics?dataset=${dataset.id}`}
                data-testid={`dataset-item-${dataset.id}`}
                className="block p-4 rounded-lg border border-slate-200 hover:border-indigo-200 hover:shadow-md transition-all"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-slate-900 mb-1">{dataset.name}</h3>
                    <p className="text-sm text-slate-600">
                      {dataset.rows.toLocaleString()} rows × {dataset.columns} columns
                    </p>
                  </div>
                  <ArrowRight className="w-5 h-5 text-slate-400" />
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          {
            title: 'Data Ingestion',
            desc: 'Upload CSV and Excel files',
            icon: Upload,
            link: '/upload',
          },
          {
            title: 'Data Cleaning',
            desc: 'Handle missing values & duplicates',
            icon: Wrench,
            link: '/preparation',
          },
          {
            title: 'Analytics',
            desc: 'Descriptive & predictive insights',
            icon: BarChart3,
            link: '/analytics',
          },
          {
            title: 'Insights',
            desc: 'Trends, anomalies & forecasts',
            icon: AlertTriangle,
            link: '/insights',
          },
        ].map((feature, idx) => {
          const Icon = feature.icon;
          return (
            <Link
              key={idx}
              to={feature.link}
              data-testid={`feature-card-${idx}`}
              className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 hover:border-indigo-200 hover:shadow-md transition-all"
            >
              <Icon className="w-8 h-8 text-indigo-600 mb-4" strokeWidth={1.5} />
              <h3 className="font-semibold text-slate-900 mb-2">{feature.title}</h3>
              <p className="text-sm text-slate-600">{feature.desc}</p>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

const Wrench = ({ className, strokeWidth }) => (
  <svg className={className} strokeWidth={strokeWidth} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M11.42 15.17L17.25 21A2.652 2.652 0 0021 17.25l-5.877-5.877M11.42 15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 11-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.743-.14a4.5 4.5 0 004.486-6.336l-3.276 3.277a3.004 3.004 0 01-2.25-2.25l3.276-3.276a4.5 4.5 0 00-6.336 4.486c.091 1.076-.071 2.264-.904 2.95l-.102.085m-1.745 1.437L5.909 7.5H4.5L2.25 3.75l1.5-1.5L7.5 4.5v1.409l4.26 4.26m-1.745 1.437l1.745-1.437m6.615 8.206L15.75 15.75M4.867 19.125h.008v.008h-.008v-.008z" />
  </svg>
);