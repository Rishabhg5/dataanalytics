import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { Upload, FileText, CheckCircle2 } from 'lucide-react';
import { toast } from 'sonner';
import { useNavigate } from 'react-router-dom';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function DataUpload() {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState(null);
  const navigate = useNavigate();

  const handleFileUpload = async (file) => {
    if (!file) return;

    const allowedTypes = ['.csv', '.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      toast.error('Please upload a CSV or Excel file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/datasets/upload`, formData, {
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
          Upload CSV or Excel files to start analyzing your data. We'll automatically detect schema and data types.
        </p>
      </div>

      {!uploadedDataset ? (
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
                Supports CSV, XLS, and XLSX files
              </p>
            </div>

            <input
              type="file"
              data-testid="file-input"
              accept=".csv,.xlsx,.xls"
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
              onClick={() => navigate(`/preparation?dataset=${uploadedDataset.id}`)}
              className="flex-1 bg-indigo-600 text-white hover:bg-indigo-700 h-11 px-6 rounded-lg font-medium transition-all active:scale-95"
            >
              Prepare Data
            </button>
            <button
              data-testid="upload-another-btn"
              onClick={() => setUploadedDataset(null)}
              className="flex-1 border border-slate-300 text-slate-700 hover:bg-slate-50 h-11 px-6 rounded-lg font-medium transition-all"
            >
              Upload Another
            </button>
          </div>
        </div>
      )}

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        {[
          {
            title: 'Auto-Detection',
            desc: 'We automatically detect column types and data structure',
          },
          {
            title: 'Large Files',
            desc: 'Handle datasets with millions of rows efficiently',
          },
          {
            title: 'Secure Storage',
            desc: 'Your data is securely stored and ready for analysis',
          },
        ].map((item, idx) => (
          <div key={idx} className="bg-white border border-slate-200 rounded-xl p-6">
            <h4 className="font-semibold text-slate-900 mb-2">{item.title}</h4>
            <p className="text-sm text-slate-600">{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}