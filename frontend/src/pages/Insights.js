import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, Activity, ChevronRight, Brain, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Insights() {
  const [searchParams] = useSearchParams();
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const datasetIdParam = datasetId || searchParams.get('dataset');
  
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(datasetIdParam || '');
  const [datasetData, setDatasetData] = useState(null);
  const [trends, setTrends] = useState(null);
  const [anomalies, setAnomalies] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      fetchDatasetData(selectedDataset);
    }
  }, [selectedDataset]);

  useEffect(() => {
    if (selectedColumn && selectedDataset) {
      analyzeTrends();
      detectAnomalies();
      generateForecast();
    }
  }, [selectedColumn]);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const fetchDatasetData = async (id) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/datasets/${id}?limit=1000`);
      setDatasetData(response.data);
      
      // Auto-select first numeric column
      const numericCol = response.data.dataset.column_names.find(col => 
        response.data.dataset.column_types[col].includes('int') || 
        response.data.dataset.column_types[col].includes('float')
      );
      if (numericCol) setSelectedColumn(numericCol);
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to load dataset');
    } finally {
      setLoading(false);
    }
  };

  const analyzeTrends = async () => {
    if (!selectedDataset || !selectedColumn) return;
    
    try {
      const response = await axios.post(`${API}/analytics/trends`, {
        dataset_id: selectedDataset,
        analysis_type: 'trends',
        parameters: { column: selectedColumn },
      });
      setTrends(response.data);
    } catch (error) {
      console.error('Trend analysis error:', error);
      toast.error('Failed to analyze trends');
    }
  };

  const detectAnomalies = async () => {
    if (!selectedDataset || !selectedColumn) return;
    
    try {
      const response = await axios.post(`${API}/analytics/anomalies`, {
        dataset_id: selectedDataset,
        analysis_type: 'anomalies',
        columns: [selectedColumn],
        parameters: { contamination: 0.1 },
      });
      setAnomalies(response.data);
    } catch (error) {
      console.error('Anomaly detection error:', error);
      toast.error('Failed to detect anomalies');
    }
  };

  const generateForecast = async () => {
    if (!selectedDataset || !selectedColumn) return;
    
    try {
      const response = await axios.post(`${API}/analytics/forecast`, {
        dataset_id: selectedDataset,
        analysis_type: 'forecast',
        parameters: { column: selectedColumn, periods: 10 },
      });
      setForecast(response.data);
    } catch (error) {
      console.error('Forecast error:', error);
      toast.error('Failed to generate forecast');
    }
  };

  const numericColumns = datasetData?.dataset.column_names.filter(
    col => datasetData.dataset.column_types[col].includes('int') || 
           datasetData.dataset.column_types[col].includes('float')
  ) || [];

  const getForecastData = () => {
    if (!forecast) return [];
    
    const historicalData = forecast.historical.map((val, idx) => ({
      index: idx,
      actual: val,
      forecast: null,
    }));
    
    const forecastData = forecast.forecast.map((val, idx) => ({
      index: forecast.historical.length + idx,
      actual: null,
      forecast: val,
    }));
    
    return [...historicalData, ...forecastData];
  };

  const handleProceedToReports = () => {
    if (selectedDataset) {
      navigate(`/dataset/${selectedDataset}/reports`);
    } else {
      toast.error('Please select a dataset first');
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Progress Indicator */}
      
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          Advanced Insights
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Discover trends, detect anomalies, and forecast future values with predictive analytics.
        </p>
      </div>

      {/* Selectors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
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
                {ds.name}
              </option>
            ))}
          </select>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
          <label className="block text-sm font-medium text-slate-700 mb-2">Select Column to Analyze</label>
          <select
            data-testid="column-selector"
            value={selectedColumn}
            onChange={(e) => setSelectedColumn(e.target.value)}
            disabled={!selectedDataset}
            className="w-full h-11 rounded-lg border border-slate-300 bg-white px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600 disabled:opacity-50"
          >
            <option value="">Choose a column...</option>
            {numericColumns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div className="text-center py-12">
          <p className="text-slate-600">Loading insights...</p>
        </div>
      )}

      {!loading && selectedColumn && (
        <>
          {/* Insights Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {trends && (
              <div data-testid="trend-card" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3 mb-4">
                  {trends.trend === 'increasing' ? (
                    <TrendingUp className="w-8 h-8 text-green-600" strokeWidth={1.5} />
                  ) : (
                    <TrendingDown className="w-8 h-8 text-red-600" strokeWidth={1.5} />
                  )}
                  <h3 className="text-lg font-semibold text-slate-900">Trend Analysis</h3>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Direction:</span>
                    <span className={`font-medium capitalize ${trends.trend === 'increasing' ? 'text-green-600' : 'text-red-600'}`}>
                      {trends.trend}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Slope:</span>
                    <span className="font-medium text-slate-900">{trends.slope.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">R²:</span>
                    <span className="font-medium text-slate-900">{trends.r_squared.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            )}

            {anomalies && (
              <div data-testid="anomaly-card" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3 mb-4">
                  <AlertTriangle className="w-8 h-8 text-amber-600" strokeWidth={1.5} />
                  <h3 className="text-lg font-semibold text-slate-900">Anomaly Detection</h3>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Total Anomalies:</span>
                    <span className="font-medium text-slate-900">{anomalies.total_anomalies}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Percentage:</span>
                    <span className="font-medium text-amber-600">{anomalies.anomaly_percentage.toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            )}

            {forecast && (
              <div data-testid="forecast-card" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Activity className="w-8 h-8 text-indigo-600" strokeWidth={1.5} />
                  <h3 className="text-lg font-semibold text-slate-900">Forecast</h3>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Periods:</span>
                    <span className="font-medium text-slate-900">{forecast.forecast_periods}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600">Avg Forecast:</span>
                    <span className="font-medium text-slate-900">
                      {(forecast.forecast.reduce((a, b) => a + b, 0) / forecast.forecast.length).toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Trend Line Chart */}
            {trends && datasetData && (
              <div data-testid="trend-chart" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Trend Visualization</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={datasetData.data.slice(0, 100).map((row, idx) => ({
                      index: idx,
                      actual: row[selectedColumn],
                      trend: trends.trend_line[idx],
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="index" stroke="#64748B" />
                    <YAxis stroke="#64748B" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="actual" stroke="#4F46E5" strokeWidth={2} name="Actual" />
                    <Line type="monotone" dataKey="trend" stroke="#10B981" strokeWidth={2} strokeDasharray="5 5" name="Trend" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Forecast Chart */}
            {forecast && (
              <div data-testid="forecast-chart" className="bg-white border border-slate-200 rounded-xl shadow-sm p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Forecast Prediction</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={getForecastData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="index" stroke="#64748B" />
                    <YAxis stroke="#64748B" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}
                    />
                    <Legend />
                    <Area type="monotone" dataKey="actual" stroke="#4F46E5" fill="#4F46E5" fillOpacity={0.6} name="Historical" />
                    <Area type="monotone" dataKey="forecast" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} name="Forecast" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </>
      )}

      {!loading && !selectedColumn && selectedDataset && (
        <div className="text-center py-12 bg-white border border-slate-200 rounded-xl">
          <p className="text-slate-600">Select a numeric column to view insights</p>
        </div>
      )}

      {/* Next Step Buttons */}
      {selectedColumn && (
        <div className="flex flex-col md:flex-row gap-4 justify-end mt-6">
          <Link
            to={`/ai-insights?dataset=${selectedDataset}`}
            data-testid="proceed-to-ai-insights-btn"
            className="flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:from-purple-700 hover:to-indigo-700 h-12 px-8 rounded-lg font-medium transition-all active:scale-95"
          >
            <Brain className="w-5 h-5" />
            AI-Powered Analysis
            <Sparkles className="w-4 h-4" />
          </Link>
 <button
  data-testid="proceed-to-reports-btn"
  onClick={() => window.location.href = `/dataset/${datasetIdParam}/reports`}  // Add the path directly here
  className="flex items-center justify-center gap-2 bg-indigo-600 text-white hover:bg-indigo-700 h-12 px-8 rounded-lg font-medium transition-all active:scale-95"
>
  Proceed to Reports
  <ChevronRight className="w-5 h-5" /> {/* Chevron icon */}
</button>
        </div>
      )}
    </div>
  );
}