import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import axios from 'axios';
import { Trash2, CheckCircle2, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function DataPreparation() {
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('dataset');
  
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(datasetId || '');
  const [datasetData, setDatasetData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      fetchDatasetData(selectedDataset);
    }
  }, [selectedDataset]);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
      toast.error('Failed to load datasets');
    }
  };

  const fetchDatasetData = async (id) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/datasets/${id}?limit=100`);
      setDatasetData(response.data);
    } catch (error) {
      console.error('Error fetching dataset:', error);
      toast.error('Failed to load dataset data');
    } finally {
      setLoading(false);
    }
  };

  const handleCleanOperation = async (operation, column = null, parameters = null) => {
    if (!selectedDataset) return;

    try {
      const response = await axios.post(`${API}/datasets/${selectedDataset}/clean`, {
        dataset_id: selectedDataset,
        operation,
        column,
        parameters,
      });
      
      toast.success(`${operation.replace('_', ' ')} completed successfully`);
      fetchDatasetData(selectedDataset);
    } catch (error) {
      console.error('Cleaning error:', error);
      toast.error(error.response?.data?.detail || 'Operation failed');
    }
  };

  const calculateMissingPercentage = (column) => {
    if (!datasetData) return 0;
    const missingCount = datasetData.data.filter(row => row[column] === null || row[column] === undefined || row[column] === '').length;
    return ((missingCount / datasetData.data.length) * 100).toFixed(1);
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          Data Preparation
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Clean and prepare your data for analysis. Handle missing values, remove duplicates, and convert data types.
        </p>
      </div>

      {/* Dataset Selector */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
        <label className="block text-sm font-medium text-slate-700 mb-2">Select Dataset</label>
        <select
          data-testid="dataset-selector"
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="w-full h-11 rounded-lg border border-slate-300 bg-white px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
        >
          <option value="">Choose a dataset...</option>
          {datasets.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.name} ({ds.rows} rows × {ds.columns} columns)
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="text-center py-12">
          <p className="text-slate-600">Loading dataset...</p>
        </div>
      )}

      {!loading && datasetData && (
        <>
          {/* Quick Actions */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Quick Actions</h3>
            <div className="flex flex-wrap gap-3">
              <button
                data-testid="remove-duplicates-btn"
                onClick={() => handleCleanOperation('remove_duplicates')}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium"
              >
                Remove Duplicates
              </button>
              <button
                data-testid="drop-all-missing-btn"
                onClick={() => handleCleanOperation('drop_missing')}
                className="px-4 py-2 border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium"
              >
                Drop All Rows with Missing Values
              </button>
            </div>
          </div>

          {/* Column-level Operations */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Column Operations</h3>
            <div className="space-y-4">
              {datasetData.dataset.column_names.map((column) => {
                const missingPct = calculateMissingPercentage(column);
                const columnType = datasetData.dataset.column_types[column];
                
                return (
                  <div
                    key={column}
                    data-testid={`column-row-${column}`}
                    className="border border-slate-200 rounded-lg p-4"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <h4 className="font-semibold text-slate-900">{column}</h4>
                          <span className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs rounded">
                            {columnType}
                          </span>
                        </div>
                        {missingPct > 0 && (
                          <div className="flex items-center gap-2 text-sm text-amber-600">
                            <AlertCircle className="w-4 h-4" />
                            <span>{missingPct}% missing values</span>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex gap-2">
                        {missingPct > 0 && (
                          <>
                            <button
                              data-testid={`fill-mean-${column}`}
                              onClick={() => handleCleanOperation('fill_missing', column, { method: 'mean' })}
                              className="px-3 py-1.5 text-sm bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors"
                            >
                              Fill Mean
                            </button>
                            <button
                              data-testid={`drop-missing-${column}`}
                              onClick={() => handleCleanOperation('drop_missing', column)}
                              className="px-3 py-1.5 text-sm bg-red-50 text-red-700 rounded-lg hover:bg-red-100 transition-colors"
                            >
                              Drop Nulls
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Data Preview */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">
              Data Preview ({datasetData.total_rows} rows total, showing first 100)
            </h3>
            <div className="overflow-x-auto">
              <table className="data-table w-full">
                <thead>
                  <tr className="border-b border-slate-200">
                    {datasetData.dataset.column_names.map((col) => (
                      <th key={col} className="px-4 py-3 text-left text-slate-700 whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetData.data.slice(0, 10).map((row, idx) => (
                    <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                      {datasetData.dataset.column_names.map((col) => (
                        <td key={col} className="px-4 py-3 text-slate-900 whitespace-nowrap">
                          {row[col] !== null && row[col] !== undefined ? String(row[col]) : (
                            <span className="text-slate-400 italic">null</span>
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {!loading && !datasetData && selectedDataset && (
        <div className="text-center py-12">
          <p className="text-slate-600">No data found</p>
        </div>
      )}
    </div>
  );
}