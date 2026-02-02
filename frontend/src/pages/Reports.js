import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { FileDown, Trash2, Calendar } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Reports() {
  const { datasetId } = useParams();
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (datasetId) {
      fetchDataset();
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

  const handleGeneratePDF = async () => {
    try {
      toast.info('Generating comprehensive PDF report...');
      
      const response = await axios.get(`${API}/reports/${datasetId}/pdf`, {
        responseType: 'blob',
      });
      
      // Download PDF
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analytics_report_${dataset.name.replace(/\.[^/.]+$/, '')}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      toast.success('PDF report generated successfully');
    } catch (error) {
      console.error('PDF generation error:', error);
      toast.error('Failed to generate PDF report');
    }
  };

  const handleExportCSV = async () => {
    try {
      const response = await axios.get(`${API}/datasets/${datasetId}?limit=10000`);
      const data = response.data.data;
      
      if (data.length === 0) {
        toast.error('No data to export');
        return;
      }

      // Convert to CSV
      const headers = Object.keys(data[0]);
      const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(header => {
          const value = row[header];
          return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
        }).join(','))
      ].join('\n');

      // Download
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${dataset.name.replace(/\.[^/.]+$/, '')}_export.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      toast.success('Dataset exported successfully');
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export dataset');
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <p className="text-slate-600">Loading reports...</p>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="text-center py-12">
        <p className="text-slate-600">Dataset not found</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          Reports & Export
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Generate comprehensive PDF reports and export data for {dataset.title || dataset.name}.
        </p>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <h2 className="text-xl font-semibold text-slate-900 mb-2">Dataset Information</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div>
              <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Name</p>
              <p className="text-sm font-medium text-slate-900">{dataset.name}</p>
            </div>
            <div>
              <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Rows</p>
              <p className="text-sm font-medium text-slate-900">{dataset.rows.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Columns</p>
              <p className="text-sm font-medium text-slate-900">{dataset.columns}</p>
            </div>
            <div>
              <p className="text-xs font-medium text-slate-600 uppercase tracking-wider mb-1">Uploaded</p>
              <p className="text-sm font-medium text-slate-900">{formatDate(dataset.uploaded_at)}</p>
            </div>
          </div>
        </div>

        <div className="p-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Available Reports</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* PDF Report Card */}
            <div className="border border-slate-200 rounded-xl p-6 hover:shadow-md transition-all">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-purple-50 rounded-lg">
                  <FileDown className="w-8 h-8 text-purple-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-slate-900 mb-2">Comprehensive Analytics Report</h4>
                  <p className="text-sm text-slate-600 mb-4">
                    PDF report with executive summary, KPIs, visualizations, trends, anomalies, forecasts, and actionable recommendations.
                  </p>
                  <button
                    data-testid="pdf-report-btn"
                    onClick={handleGeneratePDF}
                    className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium flex items-center justify-center gap-2"
                  >
                    <FileDown className="w-4 h-4" />
                    Generate PDF Report
                  </button>
                </div>
              </div>
            </div>

            {/* CSV Export Card */}
            <div className="border border-slate-200 rounded-xl p-6 hover:shadow-md transition-all">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-indigo-50 rounded-lg">
                  <FileDown className="w-8 h-8 text-indigo-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-slate-900 mb-2">CSV Data Export</h4>
                  <p className="text-sm text-slate-600 mb-4">
                    Export your cleaned and processed data as CSV file for use in other tools and applications.
                  </p>
                  <button
                    data-testid="csv-export-btn"
                    onClick={handleExportCSV}
                    className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium flex items-center justify-center gap-2"
                  >
                    <FileDown className="w-4 h-4" />
                    Export as CSV
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Report Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">Executive Summary</h4>
          <p className="text-sm text-slate-600">Actionable overview with key findings and strategic implications for decision makers.</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">Advanced Analytics</h4>
          <p className="text-sm text-slate-600">Descriptive statistics, trend analysis, anomaly detection, and predictive forecasting.</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">Actionable Insights</h4>
          <p className="text-sm text-slate-600">Prioritized recommendations with implementation steps and expected business impact.</p>
        </div>
      </div>
    </div>
  );
}

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Reports() {
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

  const handleExportCSV = async (datasetId, datasetName) => {
    try {
      const response = await axios.get(`${API}/datasets/${datasetId}?limit=10000`);
      const data = response.data.data;
      
      if (data.length === 0) {
        toast.error('No data to export');
        return;
      }

      // Convert to CSV
      const headers = Object.keys(data[0]);
      const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(header => {
          const value = row[header];
          return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
        }).join(','))
      ].join('\n');

      // Download
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetName.replace(/\.[^/.]+$/, '')}_export.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      toast.success('Dataset exported successfully');
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export dataset');
    }
  };

  const handleGeneratePDF = async (datasetId, datasetName) => {
    try {
      toast.info('Generating comprehensive PDF report...');
      
      const response = await axios.get(`${API}/reports/${datasetId}/pdf`, {
        responseType: 'blob',
      });
      
      // Download PDF
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analytics_report_${datasetName.replace(/\.[^/.]+$/, '')}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      toast.success('PDF report generated successfully');
    } catch (error) {
      console.error('PDF generation error:', error);
      toast.error('Failed to generate PDF report');
    }
  };

  const handleDelete = async (datasetId) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;
    
    try {
      await axios.delete(`${API}/datasets/${datasetId}`);
      toast.success('Dataset deleted successfully');
      fetchDatasets();
    } catch (error) {
      console.error('Delete error:', error);
      toast.error('Failed to delete dataset');
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Progress Indicator */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-sm">Upload</span>
            <ChevronRight className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-sm">Prepare</span>
            <ChevronRight className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-sm">Analytics</span>
            <ChevronRight className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-sm">Insights</span>
            <ChevronRight className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2 text-indigo-600 font-medium">
            <span className="text-sm">Reports</span>
          </div>
        </div>
      </div>

      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          Reports & Export
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Manage your datasets and export data for external use.
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <p className="text-slate-600">Loading reports...</p>
        </div>
      ) : datasets.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-12 text-center">
          <FileDown className="w-16 h-16 text-slate-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-900 mb-2">No datasets yet</h3>
          <p className="text-slate-600">Upload your first dataset to get started</p>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50">
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">
                    Dataset Name
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">
                    Rows
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">
                    Columns
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">
                    Uploaded
                  </th>
                  <th className="px-6 py-4 text-right text-xs font-medium text-slate-700 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {datasets.map((dataset) => (
                  <tr key={dataset.id} data-testid={`report-row-${dataset.id}`} className="hover:bg-slate-50">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <FileDown className="w-5 h-5 text-slate-400" />
                        <span className="font-medium text-slate-900">{dataset.name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      {dataset.rows.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      {dataset.columns}
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        {formatDate(dataset.uploaded_at)}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          data-testid={`pdf-btn-${dataset.id}`}
                          onClick={() => handleGeneratePDF(dataset.id, dataset.name)}
                          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium flex items-center gap-2"
                        >
                          <FileDown className="w-4 h-4" />
                          PDF Report
                        </button>
                        <button
                          data-testid={`export-btn-${dataset.id}`}
                          onClick={() => handleExportCSV(dataset.id, dataset.name)}
                          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium flex items-center gap-2"
                        >
                          <FileDown className="w-4 h-4" />
                          CSV
                        </button>
                        <button
                          data-testid={`delete-btn-${dataset.id}`}
                          onClick={() => handleDelete(dataset.id)}
                          className="px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors text-sm font-medium flex items-center gap-2"
                        >
                          <Trash2 className="w-4 h-4" />
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Export Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">PDF Analytics Report</h4>
          <p className="text-sm text-slate-600">Generate comprehensive PDF reports with descriptive statistics, trend analysis, anomaly detection, forecasting, and prescriptive recommendations.</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">CSV Export</h4>
          <p className="text-sm text-slate-600">Export your cleaned and processed data as CSV files for use in other tools.</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">Data Management</h4>
          <p className="text-sm text-slate-600">Delete datasets you no longer need to free up space and keep your workspace organized.</p>
        </div>
      </div>
    </div>
  );
}