import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import { ScrollText, Filter, AlertTriangle, User, Database, FileText, Settings } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ACTION_COLORS = {
  login: 'bg-blue-100 text-blue-800',
  register: 'bg-green-100 text-green-800',
  upload_dataset: 'bg-indigo-100 text-indigo-800',
  delete_dataset: 'bg-red-100 text-red-800',
  clean_dataset: 'bg-amber-100 text-amber-800',
  generate_pdf_report: 'bg-purple-100 text-purple-800',
  ai_analyze_data: 'bg-cyan-100 text-cyan-800',
  prescriptive_analytics: 'bg-emerald-100 text-emerald-800',
  update_role: 'bg-orange-100 text-orange-800',
  toggle_status: 'bg-rose-100 text-rose-800',
  ml_prediction: 'bg-violet-100 text-violet-800',
  ml_clustering: 'bg-teal-100 text-teal-800'
};

const RESOURCE_ICONS = {
  auth: User,
  dataset: Database,
  user: User,
  dashboard: FileText,
  default: Settings
};

export default function AuditLogs() {
  const { isAdmin, isManager } = useAuth();
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    action: '',
    resource_type: ''
  });

  useEffect(() => {
    if (isAdmin() || isManager()) {
      fetchLogs();
    }
  }, [filters]);

  const fetchLogs = async () => {
    try {
      const params = new URLSearchParams();
      params.append('limit', '200');
      if (filters.action) params.append('action', filters.action);
      if (filters.resource_type) params.append('resource_type', filters.resource_type);
      
      const response = await axios.get(`${API}/audit/logs?${params.toString()}`);
      setLogs(response.data);
    } catch (error) {
      console.error('Error fetching audit logs:', error);
      toast.error('Failed to load audit logs');
    } finally {
      setLoading(false);
    }
  };

  if (!isAdmin() && !isManager()) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-8 text-center">
          <AlertTriangle className="w-12 h-12 text-amber-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-amber-900 mb-2">Access Restricted</h2>
          <p className="text-amber-700">You need admin or manager privileges to view audit logs.</p>
        </div>
      </div>
    );
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const uniqueActions = [...new Set(logs.map(log => log.action))].filter(Boolean);
  const uniqueResourceTypes = [...new Set(logs.map(log => log.resource_type))].filter(Boolean);

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="MMD__heading font-bold text-slate-900 tracking-tight ">
          Audit Logs
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Track all system activities and user actions for compliance and security monitoring.
        </p>
      </div>

      {/* Filters */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Filter className="w-5 h-5 text-slate-600" />
          <h3 className="font-semibold text-slate-900">Filters</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Action Type</label>
            <select
              value={filters.action}
              onChange={(e) => setFilters({ ...filters, action: e.target.value })}
              className="w-full h-10 px-3 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">All Actions</option>
              {uniqueActions.map(action => (
                <option key={action} value={action}>{action.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Resource Type</label>
            <select
              value={filters.resource_type}
              onChange={(e) => setFilters({ ...filters, resource_type: e.target.value })}
              className="w-full h-10 px-3 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">All Resources</option>
              {uniqueResourceTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={() => setFilters({ action: '', resource_type: '' })}
              className="h-10 px-4 bg-slate-100 text-slate-700 rounded-lg text-sm hover:bg-slate-200"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-sm text-slate-600">Total Logs</p>
          <p className="text-2xl font-bold text-slate-900">{logs.length}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-sm text-slate-600">Unique Users</p>
          <p className="text-2xl font-bold text-slate-900">
            {new Set(logs.map(l => l.user_id)).size}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-sm text-slate-600">Actions Today</p>
          <p className="text-2xl font-bold text-slate-900">
            {logs.filter(l => new Date(l.timestamp).toDateString() === new Date().toDateString()).length}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-sm text-slate-600">AI/ML Operations</p>
          <p className="text-2xl font-bold text-indigo-600">
            {logs.filter(l => l.action?.includes('ai_') || l.action?.includes('ml_')).length}
          </p>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <p className="text-slate-600">Loading audit logs...</p>
        </div>
      ) : logs.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center">
          <ScrollText className="w-16 h-16 text-slate-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-900 mb-2">No audit logs yet</h3>
          <p className="text-slate-600">System activities will appear here once recorded.</p>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50">
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Timestamp</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">User</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Action</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Resource</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {logs.map((log) => {
                  const ResourceIcon = RESOURCE_ICONS[log.resource_type] || RESOURCE_ICONS.default;
                  
                  return (
                    <tr key={log.id} className="hover:bg-slate-50">
                      <td className="px-6 py-3 text-sm text-slate-600 whitespace-nowrap">
                        {formatDate(log.timestamp)}
                      </td>
                      <td className="px-6 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
                            <User className="w-4 h-4 text-slate-600" />
                          </div>
                          <span className="text-sm text-slate-900">{log.user_email || 'Anonymous'}</span>
                        </div>
                      </td>
                      <td className="px-6 py-3">
                        <span className={`inline-flex px-2.5 py-1 rounded-full text-xs font-medium ${ACTION_COLORS[log.action] || 'bg-slate-100 text-slate-800'}`}>
                          {log.action?.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="px-6 py-3">
                        <div className="flex items-center gap-2">
                          <ResourceIcon className="w-4 h-4 text-slate-500" />
                          <div>
                            <span className="text-sm text-slate-900 capitalize">{log.resource_type}</span>
                            {log.resource_id && (
                              <p className="text-xs text-slate-500 font-mono">{log.resource_id.substring(0, 8)}...</p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-3 text-sm text-slate-600">
                        {log.details ? (
                          <code className="text-xs bg-slate-100 px-2 py-1 rounded">
                            {JSON.stringify(log.details).substring(0, 50)}...
                          </code>
                        ) : '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
