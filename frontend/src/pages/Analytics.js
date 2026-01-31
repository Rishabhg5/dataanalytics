import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import axios from 'axios';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const COLORS = ['#4F46E5', '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export default function Analytics() {
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('dataset');
  
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(datasetId || '');
  const [datasetData, setDatasetData] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      fetchDatasetData(selectedDataset);
      fetchStatistics(selectedDataset);
    }
  }, [selectedDataset]);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const fetchDatasetData = async (id) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/datasets/${id}?limit=1000`);
      setDatasetData(response.data);
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to load dataset');
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async (id) => {
    try {
      const response = await axios.post(`${API}/analytics/descriptive`, {
        dataset_id: id,
        analysis_type: 'descriptive',
      });
      setStatistics(response.data.statistics);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    }
  };

  const numericColumns = datasetData?.dataset.column_names.filter(
    col => datasetData.dataset.column_types[col].includes('int') || 
           datasetData.dataset.column_types[col].includes('float')
  ) || [];

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          Analytics Dashboard
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Explore your data with interactive visualizations and descriptive statistics.
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
              {ds.name} ({ds.rows} rows)
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="text-center py-12">
          <p className="text-slate-600">Loading analytics...</p>
        </div>
      )}

      {!loading && datasetData && statistics && (
        <>
          {/* Statistics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            {Object.keys(statistics).slice(0, 4).map((col, idx) => {
              const stat = statistics[col];
              return (
                <div
                  key={col}
                  data-testid={`stat-card-${idx}`}
                  className="bg-white border border-slate-200 rounded-xl shadow-sm p-6"
                >
                  <div className="flex items-center gap-2 mb-3">
                    <Activity className="w-5 h-5 text-indigo-600" strokeWidth={1.5} />
                    <h3 className="font-semibold text-slate-900 truncate">{col}</h3>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Mean:</span>
                      <span className="font-medium text-slate-900">{stat.mean.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Median:</span>
                      <span className="font-medium text-slate-900">{stat.median.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Std Dev:</span>
                      <span className="font-medium text-slate-900">{stat.std.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Bar Chart */}
            {numericColumns.length > 0 && (
              <div data-testid="bar-chart" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Distribution Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={datasetData.data.slice(0, 20)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey={datasetData.dataset.column_names[0]} stroke="#64748B" />
                    <YAxis stroke="#64748B" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}
                    />
                    <Legend />
                    {numericColumns.slice(0, 2).map((col, idx) => (
                      <Bar key={col} dataKey={col} fill={COLORS[idx]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Line Chart */}
            {numericColumns.length > 0 && (
              <div data-testid="line-chart" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Trend Analysis</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={datasetData.data.slice(0, 50)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey={datasetData.dataset.column_names[0]} stroke="#64748B" />
                    <YAxis stroke="#64748B" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}
                    />
                    <Legend />
                    {numericColumns.slice(0, 3).map((col, idx) => (
                      <Line key={col} type="monotone" dataKey={col} stroke={COLORS[idx]} strokeWidth={2} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Area Chart */}
            {numericColumns.length > 0 && (
              <div data-testid="area-chart" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Area Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={datasetData.data.slice(0, 50)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey={datasetData.dataset.column_names[0]} stroke="#64748B" />
                    <YAxis stroke="#64748B" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}
                    />
                    <Legend />
                    {numericColumns.slice(0, 2).map((col, idx) => (
                      <Area key={col} type="monotone" dataKey={col} fill={COLORS[idx]} stroke={COLORS[idx]} fillOpacity={0.6} />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Statistics Table */}
            <div data-testid="stats-table" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">Detailed Statistics</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="px-3 py-2 text-left text-slate-700 font-medium">Metric</th>
                      {Object.keys(statistics).slice(0, 3).map(col => (
                        <th key={col} className="px-3 py-2 text-right text-slate-700 font-medium">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {['mean', 'median', 'std', 'min', 'max'].map(metric => (
                      <tr key={metric} className="border-b border-slate-100">
                        <td className="px-3 py-2 text-slate-600 capitalize">{metric}</td>
                        {Object.keys(statistics).slice(0, 3).map(col => (
                          <td key={col} className="px-3 py-2 text-right text-slate-900 font-medium">
                            {statistics[col][metric].toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {!loading && !datasetData && selectedDataset && (
        <div className="text-center py-12">
          <p className="text-slate-600">No data available</p>
        </div>
      )}
    </div>
  );
}