import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, AreaChart, Area, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, ChevronRight, Activity } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const COLORS = ['#4F46E5', '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6'];

const CHART_TYPES = [
  { value: 'bar', label: 'Bar Chart' },
  { value: 'line', label: 'Line Chart' },
  { value: 'area', label: 'Area Chart' },
  { value: 'pie', label: 'Pie Chart' },
  { value: 'scatter', label: 'Scatter Plot' },
];

export default function Analytics() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = searchParams.get('dataset');
  
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(datasetId || '');
  const [datasetData, setDatasetData] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Chart customization state
  const [selectedChartType, setSelectedChartType] = useState('bar');
  const [xAxisColumn, setXAxisColumn] = useState('');
  const [yAxisColumns, setYAxisColumns] = useState([]);

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
      
      // Auto-select columns
      if (response.data.dataset.column_names.length > 0) {
        setXAxisColumn(response.data.dataset.column_names[0]);
        const numericCols = response.data.dataset.column_names.filter(
          col => response.data.dataset.column_types[col].includes('int') || 
                 response.data.dataset.column_types[col].includes('float')
        );
        if (numericCols.length > 0) {
          setYAxisColumns([numericCols[0]]);
        }
      }
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

  const allColumns = datasetData?.dataset.column_names || [];

  const handleYAxisToggle = (column) => {
    if (yAxisColumns.includes(column)) {
      setYAxisColumns(yAxisColumns.filter(col => col !== column));
    } else {
      setYAxisColumns([...yAxisColumns, column]);
    }
  };

  const renderChart = () => {
    if (!datasetData || yAxisColumns.length === 0) return null;

    const chartData = datasetData.data.slice(0, 50);

    switch (selectedChartType) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey={xAxisColumn} stroke="#64748B" angle={-45} textAnchor="end" height={80} />
              <YAxis stroke="#64748B" />
              <Tooltip contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }} />
              <Legend />
              {yAxisColumns.map((col, idx) => (
                <Bar key={col} dataKey={col} fill={COLORS[idx % COLORS.length]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'line':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey={xAxisColumn} stroke="#64748B" />
              <YAxis stroke="#64748B" />
              <Tooltip contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }} />
              <Legend />
              {yAxisColumns.map((col, idx) => (
                <Line key={col} type="monotone" dataKey={col} stroke={COLORS[idx % COLORS.length]} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey={xAxisColumn} stroke="#64748B" />
              <YAxis stroke="#64748B" />
              <Tooltip contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }} />
              <Legend />
              {yAxisColumns.map((col, idx) => (
                <Area key={col} type="monotone" dataKey={col} fill={COLORS[idx % COLORS.length]} stroke={COLORS[idx % COLORS.length]} fillOpacity={0.6} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'pie':
        if (yAxisColumns.length === 0) return null;
        const pieData = chartData.slice(0, 10).map(item => ({
          name: String(item[xAxisColumn]),
          value: Number(item[yAxisColumns[0]]) || 0,
        }));
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'scatter':
        if (yAxisColumns.length < 2) {
          return <p className="text-center text-slate-600 py-8">Scatter plot requires at least 2 numeric columns</p>;
        }
        return (
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis type="number" dataKey={yAxisColumns[0]} name={yAxisColumns[0]} stroke="#64748B" />
              <YAxis type="number" dataKey={yAxisColumns[1]} name={yAxisColumns[1]} stroke="#64748B" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              <Scatter name="Data Points" data={chartData} fill="#4F46E5" />
            </ScatterChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  const handleProceedToInsights = () => {
    if (selectedDataset) {
      navigate(`/insights?dataset=${selectedDataset}`);
    } else {
      toast.error('Please select a dataset first');
    }
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
          <div className="flex items-center gap-2 text-indigo-600 font-medium">
            <span className="text-sm">Analytics</span>
            <ChevronRight className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-sm">Insights</span>
            <ChevronRight className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-sm">Reports</span>
          </div>
        </div>
      </div>

      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          Analytics Dashboard
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Explore your data with customizable visualizations and descriptive statistics.
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

          {/* Chart Customization */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Customize Your Visualization</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Chart Type */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Chart Type</label>
                <select
                  data-testid="chart-type-selector"
                  value={selectedChartType}
                  onChange={(e) => setSelectedChartType(e.target.value)}
                  className="w-full h-11 rounded-lg border border-slate-300 bg-white px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                >
                  {CHART_TYPES.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>

              {/* X-Axis */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">X-Axis Column</label>
                <select
                  data-testid="x-axis-selector"
                  value={xAxisColumn}
                  onChange={(e) => setXAxisColumn(e.target.value)}
                  className="w-full h-11 rounded-lg border border-slate-300 bg-white px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                >
                  {allColumns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>

              {/* Y-Axis */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Y-Axis Columns (Numeric)</label>
                <div className="border border-slate-300 rounded-lg p-3 max-h-32 overflow-y-auto bg-white">
                  {numericColumns.map(col => (
                    <label key={col} className="flex items-center gap-2 mb-2 cursor-pointer hover:bg-slate-50 p-1 rounded">
                      <input
                        type="checkbox"
                        checked={yAxisColumns.includes(col)}
                        onChange={() => handleYAxisToggle(col)}
                        className="w-4 h-4 text-indigo-600 border-slate-300 rounded focus:ring-indigo-600"
                      />
                      <span className="text-sm text-slate-900">{col}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Dynamic Chart */}
          <div data-testid="custom-chart" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">
              {CHART_TYPES.find(t => t.value === selectedChartType)?.label || 'Chart'}
            </h3>
            {renderChart()}
          </div>

          {/* Detailed Statistics Table */}
          <div data-testid="stats-table" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Detailed Statistics</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="px-3 py-2 text-left text-slate-700 font-medium">Metric</th>
                    {Object.keys(statistics).slice(0, 5).map(col => (
                      <th key={col} className="px-3 py-2 text-right text-slate-700 font-medium">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {['mean', 'median', 'std', 'min', 'max', 'q25', 'q75'].map(metric => (
                    <tr key={metric} className="border-b border-slate-100">
                      <td className="px-3 py-2 text-slate-600 capitalize">{metric === 'std' ? 'Std Dev' : metric}</td>
                      {Object.keys(statistics).slice(0, 5).map(col => (
                        <td key={col} className="px-3 py-2 text-right text-slate-900 font-medium">
                          {statistics[col][metric]?.toFixed(2) || 'N/A'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Next Step Button */}
          <div className="flex justify-end">
            <button
              data-testid="proceed-to-insights-btn"
              onClick={handleProceedToInsights}
              className="flex items-center gap-2 bg-indigo-600 text-white hover:bg-indigo-700 h-12 px-8 rounded-lg font-medium transition-all active:scale-95"
            >
              Proceed to Insights
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        </>
      )}
    </div>
  );
}