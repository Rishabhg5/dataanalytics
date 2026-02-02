import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { Upload, FileText, CheckCircle2, Database, Globe, FileJson } from 'lucide-react';
import { toast } from 'sonner';
import { useNavigate } from 'react-router-dom';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function DataUpload() {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState(null);
  const [uploadMode, setUploadMode] = useState('file'); // file, api, mysql
  const [datasetTitle, setDatasetTitle] = useState('');
  const [apiUrl, setApiUrl] = useState('');
  const [mysqlConfig, setMysqlConfig] = useState({
    host: '',
    database: '',
    user: '',
    password: '',
    query: '',
    port: 3306
  });
  const navigate = useNavigate();

  const handleFileUpload = async (file) => {
    if (!file) return;

    const allowedTypes = ['.csv', '.xlsx', '.xls', '.json', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      toast.error('Please upload CSV, Excel, JSON, or TXT file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    
    const title = datasetTitle || file.name.split('.')[0];

    try {
      const response = await axios.post(`${API}/datasets/upload?title=${encodeURIComponent(title)}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      setUploadedDataset(response.data);
      toast.success('Dataset uploaded successfully!');
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(error.response?.data?.detail || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const handleApiUpload = async () => {
    if (!apiUrl) {
      toast.error('Please enter an API URL');
      return;
    }

    setUploading(true);
    try {
      const response = await axios.post(`${API}/datasets/upload-from-api?api_url=${encodeURIComponent(apiUrl)}`);
      setUploadedDataset(response.data);
      toast.success('Data imported from API successfully!');
    } catch (error) {
      console.error('API upload error:', error);
      toast.error(error.response?.data?.detail || 'Failed to import from API');
    } finally {
      setUploading(false);
    }
  };

  const handleMySQLUpload = async () => {
    if (!mysqlConfig.host || !mysqlConfig.database || !mysqlConfig.user || !mysqlConfig.query) {
      toast.error('Please fill in all MySQL connection details');
      return;
    }

    setUploading(true);
    try {
      const response = await axios.post(`${API}/datasets/upload-from-mysql`, mysqlConfig);
      setUploadedDataset(response.data);
      toast.success('Data imported from MySQL successfully!');
    } catch (error) {
      console.error('MySQL upload error:', error);
      toast.error(error.response?.data?.detail || 'Failed to import from MySQL');
    } finally {
      setUploading(false);
    }
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileUpload(files[0]);
    }
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFileUpload(files[0]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          Upload Your Data
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Import data from multiple sources: files, APIs, or databases. We'll automatically detect schema and data types.
        </p>
      </div>

      {!uploadedDataset ? (
        <>
          {/* Source Selection */}
          <div className="flex gap-3 mb-6">
            <button
              onClick={() => setUploadMode('file')}
              data-testid="mode-file"
              className={`flex-1 flex items-center justify-center gap-2 h-12 px-6 rounded-lg font-medium transition-all ${
                uploadMode === 'file' ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-50'
              }`}
            >
              <Upload className="w-5 h-5" />
              File Upload
            </button>
            <button
              onClick={() => setUploadMode('api')}
              data-testid="mode-api"
              className={`flex-1 flex items-center justify-center gap-2 h-12 px-6 rounded-lg font-medium transition-all ${
                uploadMode === 'api' ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-50'
              }`}
            >
              <Globe className="w-5 h-5" />
              API Import
            </button>
            <button
              onClick={() => setUploadMode('mysql')}
              data-testid="mode-mysql"
              className={`flex-1 flex items-center justify-center gap-2 h-12 px-6 rounded-lg font-medium transition-all ${
                uploadMode === 'mysql' ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-50'
              }`}
            >
              <Database className="w-5 h-5" />
              MySQL Database
            </button>
          </div>

          {/* File Upload Mode */}
          {uploadMode === 'file' && (
            <>
              {/* Dataset Title Input */}
              <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Dataset Title (Optional)
                </label>
                <input
                  type="text"
                  value={datasetTitle}
                  onChange={(e) => setDatasetTitle(e.target.value)}
                  placeholder="e.g., Sales Q1 2024, Customer Database"
                  className="w-full h-11 rounded-lg border border-slate-300 px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                />
                <p className="text-xs text-slate-500 mt-2">Give your dataset a meaningful title for easy identification</p>
              </div>
              
              <div
              data-testid="upload-dropzone"
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              className={`upload-zone border-2 border-dashed rounded-xl p-12 text-center ${
                dragOver ? 'drag-over' : 'border-slate-300'
              } ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="p-6 bg-indigo-50 rounded-full">
                  <Upload className="w-12 h-12 text-indigo-600" strokeWidth={1.5} />
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold text-slate-900 mb-2">
                    {uploading ? 'Uploading...' : 'Drop your file here'}
                  </h3>
                  <p className="text-slate-600 mb-4">
                    or click to browse from your computer
                  </p>
                  <p className="text-sm text-slate-500">
                    Supports CSV, XLS, XLSX, JSON, and TXT files
                  </p>
                </div>

                <input
                  type="file"
                  data-testid="file-input"
                  accept=".csv,.xlsx,.xls,.json,.txt"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                  disabled={uploading}
                />
                <label
                  htmlFor="file-upload"
                  className="bg-indigo-600 text-white hover:bg-indigo-700 h-11 px-8 rounded-lg font-medium transition-all active:scale-95 cursor-pointer inline-flex items-center"
                >
                  Select File
                </label>
              </div>
            </div>
          )}
          {/* API Import Mode */}
          {uploadMode === 'api' && (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-8">
              <div className="flex items-center gap-3 mb-6">
                <Globe className="w-8 h-8 text-indigo-600" />
                <div>
                  <h3 className="text-xl font-semibold text-slate-900">Import from API</h3>
                  <p className="text-sm text-slate-600">Enter a REST API endpoint that returns JSON data</p>
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">API URL</label>
                  <input
                    type="url"
                    data-testid="api-url-input"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    placeholder="https://api.example.com/data"
                    className="w-full h-11 rounded-lg border border-slate-300 px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
                
                <button
                  onClick={handleApiUpload}
                  disabled={uploading}
                  data-testid="api-import-btn"
                  className="w-full bg-indigo-600 text-white hover:bg-indigo-700 h-12 px-6 rounded-lg font-medium transition-all active:scale-95 disabled:opacity-50"
                >
                  {uploading ? 'Importing...' : 'Import from API'}
                </button>
              </div>
            </div>
          )}

          {/* MySQL Import Mode */}
          {uploadMode === 'mysql' && (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-8">
              <div className="flex items-center gap-3 mb-6">
                <Database className="w-8 h-8 text-indigo-600" />
                <div>
                  <h3 className="text-xl font-semibold text-slate-900">Import from MySQL</h3>
                  <p className="text-sm text-slate-600">Connect to your MySQL database and run a query</p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Host</label>
                  <input
                    type="text"
                    value={mysqlConfig.host}
                    onChange={(e) => setMysqlConfig({...mysqlConfig, host: e.target.value})}
                    placeholder="localhost"
                    className="w-full h-11 rounded-lg border border-slate-300 px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Port</label>
                  <input
                    type="number"
                    value={mysqlConfig.port}
                    onChange={(e) => setMysqlConfig({...mysqlConfig, port: parseInt(e.target.value)})}
                    placeholder="3306"
                    className="w-full h-11 rounded-lg border border-slate-300 px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Database</label>
                  <input
                    type="text"
                    value={mysqlConfig.database}
                    onChange={(e) => setMysqlConfig({...mysqlConfig, database: e.target.value})}
                    placeholder="my_database"
                    className="w-full h-11 rounded-lg border border-slate-300 px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">User</label>
                  <input
                    type="text"
                    value={mysqlConfig.user}
                    onChange={(e) => setMysqlConfig({...mysqlConfig, user: e.target.value})}
                    placeholder="username"
                    className="w-full h-11 rounded-lg border border-slate-300 px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
                  <input
                    type="password"
                    value={mysqlConfig.password}
                    onChange={(e) => setMysqlConfig({...mysqlConfig, password: e.target.value})}
                    placeholder="••••••••"
                    className="w-full h-11 rounded-lg border border-slate-300 px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-slate-700 mb-2">SQL Query</label>
                  <textarea
                    value={mysqlConfig.query}
                    onChange={(e) => setMysqlConfig({...mysqlConfig, query: e.target.value})}
                    placeholder="SELECT * FROM table_name LIMIT 1000"
                    rows={3}
                    className="w-full rounded-lg border border-slate-300 px-4 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  />
                </div>
              </div>
              
              <button
                onClick={handleMySQLUpload}
                disabled={uploading}
                data-testid="mysql-import-btn"
                className="w-full bg-indigo-600 text-white hover:bg-indigo-700 h-12 px-6 rounded-lg font-medium transition-all active:scale-95 disabled:opacity-50"
              >
                {uploading ? 'Importing...' : 'Import from MySQL'}
              </button>
            </div>
          )}
        </>
      ) : (
        <div data-testid="upload-success" className="bg-white border border-slate-200 rounded-xl shadow-sm p-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="p-3 bg-green-50 rounded-full">
              <CheckCircle2 className="w-8 h-8 text-green-600" />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-slate-900">Upload Successful!</h3>
              <p className="text-slate-600">Your dataset is ready for analysis</p>
            </div>
          </div>

          <div className="bg-slate-50 rounded-lg p-6 mb-6">
            <div className="flex items-start gap-3 mb-4">
              <FileText className="w-5 h-5 text-slate-600 mt-1" />
              <div className="flex-1">
                <h4 className="font-semibold text-slate-900 mb-1">{uploadedDataset.name}</h4>
                <div className="grid grid-cols-2 gap-4 text-sm text-slate-600">
                  <div>
                    <span className="font-medium">Rows:</span> {uploadedDataset.rows.toLocaleString()}
                  </div>
                  <div>
                    <span className="font-medium">Columns:</span> {uploadedDataset.columns}
                  </div>
                </div>
              </div>
            </div>

            <div className="border-t border-slate-200 pt-4">
              <p className="text-xs font-medium text-slate-700 uppercase tracking-wider mb-2">Columns Detected</p>
              <div className="flex flex-wrap gap-2">
                {uploadedDataset.column_names.map((col, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-white border border-slate-200 rounded-lg text-sm text-slate-700"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              data-testid="view-data-btn"
              onClick={() => navigate(`/dataset/${uploadedDataset.id}/overview`)}
              className="flex-1 bg-indigo-600 text-white hover:bg-indigo-700 h-11 px-6 rounded-lg font-medium transition-all active:scale-95"
            >
              View Dataset Overview
            </button>
            <button
              data-testid="upload-another-btn"
              onClick={() => {
                setUploadedDataset(null);
                setDatasetTitle('');
              }}
              className="flex-1 border border-slate-300 text-slate-700 hover:bg-slate-50 h-11 px-6 rounded-lg font-medium transition-all"
            >
              Upload Another
            </button>
          </div>
        </div>
      )}

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">Multiple Sources</h4>
          <p className="text-sm text-slate-600">Import from files, REST APIs, or MySQL databases with automatic schema detection</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">Large Files</h4>
          <p className="text-sm text-slate-600">Handle datasets with millions of rows efficiently with optimized processing</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h4 className="font-semibold text-slate-900 mb-2">Secure Storage</h4>
          <p className="text-sm text-slate-600">Your data is securely stored in MongoDB and ready for instant analysis</p>
        </div>
      </div>
    </div>
  );
}