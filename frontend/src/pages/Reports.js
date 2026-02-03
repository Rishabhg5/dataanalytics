import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { FileDown, Trash2, Calendar, ChevronRight, Clock, History, Download, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Reports() {
  const { datasetId } = useParams();
  const [datasets, setDatasets] = useState([]);
  const [reportHistory, setReportHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generatingPdf, setGeneratingPdf] = useState(null);

  useEffect(() => {
    fetchDatasets();
    fetchReportHistory();
  }, [datasetId]);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      if (datasetId) {
        const filtered = response.data.filter(ds => ds.id === datasetId);
        setDatasets(filtered);
      } else {
        setDatasets(response.data);
      }
    } catch (error) {
      console.error('Error fetching datasets:', error);
      toast.error('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const fetchReportHistory = async () => {
    try {
      const endpoint = datasetId 
        ? `${API}/reports/history/${datasetId}`
        : `${API}/reports/all`;
      const response = await axios.get(endpoint);
      setReportHistory(response.data || []);
    } catch (error) {
      console.error('Error fetching report history:', error);
    }
  };

  const handleGeneratePDF = async (dsId, datasetName) => {
    setGeneratingPdf(dsId);
    try {
      toast.info('Generating comprehensive PDF report...');
      
      // Use fetch API for better blob handling
      const response = await fetch(`${API}/reports/${dsId}/pdf`);
      
      if (!response.ok) {
        throw new Error('Failed to generate PDF');
      }
      
      const blob = await response.blob();
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const filename = `analytics_report_${datasetName.replace(/\.[^/.]+$/, '')}_${timestamp}.pdf`;
      
      // Download PDF
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      toast.success('PDF report generated and saved to history');
      fetchReportHistory(); // Refresh history
    } catch (error) {
      console.error('PDF generation error:', error);
      toast.error('Failed to generate PDF report');
    } finally {
      setGeneratingPdf(null);
    }
  };

  const handleDownloadReport = async (reportId) => {
    try {
      const response = await fetch(`${API}/reports/download/${reportId}`);
      
      if (!response.ok) {
        throw new Error('Report file not found');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${reportId}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      toast.success('Report downloaded');
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download report - file may no longer exist');
    }
  };

  const handleExportCSV = async (dsId, datasetName) => {
    try {
      const response = await axios.get(`${API}/datasets/${dsId}?limit=10000`);
      const data = response.data.data;
      
      if (data.length === 0) {
        toast.error('No data to export');
        return;
      }

      const headers = Object.keys(data[0]);
      const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(header => {
          const value = row[header];
          return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
        }).join(','))
      ].join('\n');

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

  const handleDelete = async (dsId) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;
    
    try {
      await axios.delete(`${API}/datasets/${dsId}`);
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

  // If we're in a dataset context, show a different layout
  const isDatasetContext = !!datasetId;

  return (
    <div className="max-w-7xl mx-auto">
      {/* Progress Indicator */}
      

      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          {isDatasetContext ? 'Dataset Reports' : 'Reports & Export'}
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          {isDatasetContext 
            ? `Generate and track PDF reports for ${datasets[0]?.title || 'this dataset'}.`
            : 'Generate comprehensive PDF reports and track report history over time.'}
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-slate-600">Loading...</p>
        </div>
      ) : isDatasetContext ? (
        /* Dataset-specific view - shows generate button and history together */
        <>
          {datasets.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">{datasets[0].title || datasets[0].name}</h3>
                  <p className="text-sm text-slate-600">{datasets[0].rows.toLocaleString()} rows • {datasets[0].columns} columns</p>
                </div>
                <div className="flex gap-3">
                  <button
                    data-testid="generate-pdf-btn"
                    onClick={() => handleGeneratePDF(datasets[0].id, datasets[0].name)}
                    disabled={generatingPdf === datasets[0].id}
                    className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium flex items-center gap-2 disabled:opacity-50"
                  >
                    {generatingPdf === datasets[0].id ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <FileDown className="w-5 h-5" />
                        Generate PDF Report
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => handleExportCSV(datasets[0].id, datasets[0].name)}
                    className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium flex items-center gap-2"
                  >
                    <FileDown className="w-5 h-5" />
                    Export CSV
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Report History for this dataset */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-200 bg-slate-50">
              <div className="flex items-center gap-2">
                <History className="w-5 h-5 text-slate-600" />
                <h3 className="font-semibold text-slate-900">Report History ({reportHistory.length})</h3>
              </div>
              <p className="text-sm text-slate-600 mt-1">Track all generated reports for this dataset</p>
            </div>
            
            {reportHistory.length === 0 ? (
              <div className="p-12 text-center">
                <History className="w-16 h-16 text-slate-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-slate-900 mb-2">No reports generated yet</h3>
                <p className="text-slate-600">Click "Generate PDF Report" above to create your first report.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Report</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Generated</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Generated By</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Charts</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-slate-700 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {reportHistory.map((report) => (
                      <tr key={report.id} className="hover:bg-slate-50">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <FileDown className="w-5 h-5 text-purple-500" />
                            <div>
                              <span className="font-medium text-slate-900 block">{report.title}</span>
                              <span className="text-xs text-slate-500 font-mono">{report.id.substring(0, 8)}...</span>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-slate-600">
                          <div className="flex items-center gap-2">
                            <Clock className="w-4 h-4" />
                            {formatDate(report.generated_at)}
                          </div>
                        </td>
                        <td className="px-6 py-4 text-slate-600">
                          {report.generated_by_email || 'Anonymous'}
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-full text-xs font-medium">
                            {report.charts_included?.length || 0} charts
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center justify-end">
                            <button
                              onClick={() => handleDownloadReport(report.id)}
                              className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium"
                            >
                              <Download className="w-4 h-4" />
                              Download
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      ) : (
        /* Global reports view - shows all datasets */
        <>
          {datasets.length === 0 ? (
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
                            <div>
                              <span className="font-medium text-slate-900 block">{dataset.title || dataset.name}</span>
                              <span className="text-xs text-slate-500">{dataset.name}</span>
                            </div>
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
                              disabled={generatingPdf === dataset.id}
                              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium flex items-center gap-2 disabled:opacity-50"
                            >
                              {generatingPdf === dataset.id ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                              ) : (
                                <FileDown className="w-4 h-4" />
                              )}
                              Generate PDF
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

          {/* All Reports History */}
          {reportHistory.length > 0 && (
            <div className="mt-8 bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-200 bg-slate-50">
                <div className="flex items-center gap-2">
                  <History className="w-5 h-5 text-slate-600" />
                  <h3 className="font-semibold text-slate-900">All Report History ({reportHistory.length})</h3>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Report</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Dataset</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Generated</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-slate-700 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {reportHistory.slice(0, 10).map((report) => (
                      <tr key={report.id} className="hover:bg-slate-50">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <FileDown className="w-5 h-5 text-purple-500" />
                            <span className="font-medium text-slate-900">{report.title}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-slate-600">{report.dataset_name}</td>
                        <td className="px-6 py-4 text-slate-600">{formatDate(report.generated_at)}</td>
                        <td className="px-6 py-4 text-right">
                          <button
                            onClick={() => handleDownloadReport(report.id)}
                            className="px-3 py-1.5 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-sm"
                          >
                            Download
                          </button>
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
              <p className="text-sm text-slate-600">Generate comprehensive PDF reports with statistics, trend analysis, anomaly detection, forecasting, and recommendations. Each report is saved with timestamp.</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-6">
              <h4 className="font-semibold text-slate-900 mb-2">Report History</h4>
              <p className="text-sm text-slate-600">Track all generated reports over time. View when each report was created and by whom. Download past reports anytime.</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-6">
              <h4 className="font-semibold text-slate-900 mb-2">CSV Export</h4>
              <p className="text-sm text-slate-600">Export your cleaned and processed data as CSV files for use in other tools like Excel or Google Sheets.</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
