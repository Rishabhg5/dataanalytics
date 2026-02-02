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
