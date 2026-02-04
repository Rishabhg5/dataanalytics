import React, { useState, useEffect } from 'react';
import { useSearchParams, useParams } from 'react-router-dom';
import axios from 'axios';
import { Brain, Sparkles, Target, TrendingUp, AlertTriangle, Zap, ChevronRight, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function AIInsights() {
  const [searchParams] = useSearchParams();
  const { datasetId } = useParams();
  const datasetIdParam = datasetId || searchParams.get('dataset');

  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(datasetIdParam || '');
  const [loading, setLoading] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [prescriptiveAnalysis, setPrescriptiveAnalysis] = useState(null);
  const [mlPrediction, setMlPrediction] = useState(null);
  const [clustering, setClustering] = useState(null);
  const [businessContext, setBusinessContext] = useState('');
  const [optimizationGoals, setOptimizationGoals] = useState(['increase revenue', 'reduce costs']);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const runAIAnalysis = async () => {
    if (!selectedDataset) {
      toast.error('Please select a dataset first');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post(`${API}/ai/analyze-data`, {
        dataset_id: selectedDataset,
        insight_type: 'data_analysis'
      });
      setAiAnalysis(response.data);
      toast.success('AI analysis complete');
    } catch (error) {
      console.error('AI analysis error:', error);
      toast.error('Failed to run AI analysis');
    } finally {
      setLoading(false);
    }
  };

  const runPrescriptiveAnalysis = async () => {
    if (!selectedDataset) {
      toast.error('Please select a dataset first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/ai/prescriptive`, {
        dataset_id: selectedDataset,
        business_context: businessContext || 'general business analytics',
        optimization_goals: optimizationGoals
      });
      setPrescriptiveAnalysis(response.data);
      toast.success('Prescriptive analysis complete');
    } catch (error) {
      console.error('Prescriptive analysis error:', error);
      toast.error('Failed to run prescriptive analysis');
    } finally {
      setLoading(false);
    }
  };

  const runMLPrediction = async () => {
    if (!selectedDataset) {
      toast.error('Please select a dataset first');
      return;
    }

    setLoading(true);
    try {
      const dataset = datasets.find(d => d.id === selectedDataset);
      const numericCols = Object.entries(dataset?.column_types || {})
        .filter(([, type]) => type.includes('int') || type.includes('float'))
        .map(([col]) => col);
      
      if (numericCols.length < 2) {
        toast.error('Need at least 2 numeric columns for prediction');
        setLoading(false);
        return;
      }

      const response = await axios.post(`${API}/ml/predict`, {
        dataset_id: selectedDataset,
        analysis_type: 'prediction',
        parameters: { target_column: numericCols[0] }
      });
      setMlPrediction(response.data);
      toast.success('ML prediction complete');
    } catch (error) {
      console.error('ML prediction error:', error);
      toast.error('Failed to run ML prediction');
    } finally {
      setLoading(false);
    }
  };

  const runClustering = async () => {
    if (!selectedDataset) {
      toast.error('Please select a dataset first');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/ml/cluster`, {
        dataset_id: selectedDataset,
        analysis_type: 'clustering',
        parameters: { n_clusters: 3 }
      });
      setClustering(response.data);
      toast.success('Clustering analysis complete');
    } catch (error) {
      console.error('Clustering error:', error);
      toast.error('Failed to run clustering analysis');
    } finally {
      setLoading(false);
    }
  };

  const runAllAnalyses = async () => {
    await Promise.all([
      runAIAnalysis(),
      runPrescriptiveAnalysis(),
      runClustering()
    ]);
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
          <div className="flex items-center gap-2 text-indigo-600 font-medium">
            <span className="text-sm">AI Insights</span>
            <ChevronRight className="w-4 h-4" />
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-sm">Reports</span>
          </div>
        </div>
      </div>

      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="MMD__heading font-bold text-slate-900 tracking-tight" >
              AI-Powered Insights
            </h1>
            <p className="text-lg text-slate-600 leading-relaxed mt-2">
              Leverage machine learning and AI to uncover deep patterns, predictions, and prescriptive recommendations.
            </p>
          </div>
        </div>
      </div>

      {/* Dataset Selector */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Select Dataset</label>
            <select
              data-testid="ai-dataset-selector"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full h-11 rounded-lg border border-slate-300 bg-white px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
            >
              <option value="">Choose a dataset...</option>
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.title || ds.name} ({ds.rows} rows)
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Business Context (Optional)</label>
            <input
              type="text"
              value={businessContext}
              onChange={(e) => setBusinessContext(e.target.value)}
              placeholder="e.g., E-commerce sales, Customer retention..."
              className="w-full h-11 rounded-lg border border-slate-300 bg-white px-4 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
            />
          </div>
        </div>
        
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            onClick={runAllAnalyses}
            disabled={!selectedDataset || loading}
            data-testid="run-all-btn"
            className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700 h-11 px-6 rounded-lg font-medium transition-all disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
            Run All AI Analyses
          </button>
          <button
            onClick={runAIAnalysis}
            disabled={!selectedDataset || loading}
            className="flex items-center gap-2 bg-cyan-600 text-white hover:bg-cyan-700 h-11 px-6 rounded-lg font-medium transition-all disabled:opacity-50"
          >
            <Brain className="w-4 h-4" />
            Data Analysis
          </button>
          <button
            onClick={runPrescriptiveAnalysis}
            disabled={!selectedDataset || loading}
            className="flex items-center gap-2 bg-emerald-600 text-white hover:bg-emerald-700 h-11 px-6 rounded-lg font-medium transition-all disabled:opacity-50"
          >
            <Target className="w-4 h-4" />
            Prescriptive
          </button>
          <button
            onClick={runMLPrediction}
            disabled={!selectedDataset || loading}
            className="flex items-center gap-2 bg-violet-600 text-white hover:bg-violet-700 h-11 px-6 rounded-lg font-medium transition-all disabled:opacity-50"
          >
            <TrendingUp className="w-4 h-4" />
            ML Prediction
          </button>
          <button
            onClick={runClustering}
            disabled={!selectedDataset || loading}
            className="flex items-center gap-2 bg-amber-600 text-white hover:bg-amber-700 h-11 px-6 rounded-lg font-medium transition-all disabled:opacity-50"
          >
            <Sparkles className="w-4 h-4" />
            Clustering
          </button>
        </div>
      </div>

      {/* AI Analysis Results */}
      {aiAnalysis && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Brain className="w-6 h-6 text-cyan-600" />
            <h3 className="text-xl font-semibold text-slate-900">AI Data Analysis</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-slate-50 rounded-lg p-4">
              <p className="text-sm text-slate-600">Total Records</p>
              <p className="text-2xl font-bold text-slate-900">{aiAnalysis.data_quality?.total_records?.toLocaleString()}</p>
            </div>
            <div className="bg-slate-50 rounded-lg p-4">
              <p className="text-sm text-slate-600">Data Completeness</p>
              <p className="text-2xl font-bold text-green-600">
                {(100 - (aiAnalysis.data_quality?.missing_percentage || 0)).toFixed(1)}%
              </p>
            </div>
            <div className="bg-slate-50 rounded-lg p-4">
              <p className="text-sm text-slate-600">Duplicates</p>
              <p className="text-2xl font-bold text-amber-600">{aiAnalysis.data_quality?.duplicate_rows || 0}</p>
            </div>
          </div>

          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 rounded-lg p-4 border border-cyan-100">
            <div className="flex items-start gap-2">
              <Sparkles className="w-5 h-5 text-cyan-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-cyan-900 mb-1">AI-Generated Insight</p>
                <p className="text-cyan-800 whitespace-pre-wrap">{aiAnalysis.ai_insight}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Prescriptive Analysis Results */}
      {prescriptiveAnalysis && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-6 h-6 text-emerald-600" />
            <h3 className="text-xl font-semibold text-slate-900">Prescriptive Analytics - "What Should We Do?"</h3>
          </div>

          {/* Risk Factors */}
          {prescriptiveAnalysis.risk_factors?.length > 0 && (
            <div className="mb-6">
              <h4 className="font-medium text-slate-900 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-600" />
                Risk Factors ({prescriptiveAnalysis.risk_factors.length})
              </h4>
              <div className="space-y-3">
                {prescriptiveAnalysis.risk_factors.map((risk, idx) => (
                  <div key={idx} className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-medium text-amber-900">{risk.metric || risk.current_status}</p>
                        <p className="text-sm text-amber-700 mt-1">{risk.recommendation}</p>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        risk.risk_level === 'HIGH' ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'
                      }`}>
                        {risk.risk_level}
                      </span>
                    </div>
                    {risk.action && (
                      <p className="mt-2 text-sm text-amber-600">→ {risk.action}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Optimization Opportunities */}
          {prescriptiveAnalysis.optimization_opportunities?.length > 0 && (
            <div className="mb-6">
              <h4 className="font-medium text-slate-900 mb-3 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-600" />
                Optimization Opportunities ({prescriptiveAnalysis.optimization_opportunities.length})
              </h4>
              <div className="space-y-3">
                {prescriptiveAnalysis.optimization_opportunities.map((opp, idx) => (
                  <div key={idx} className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <p className="font-medium text-green-900">{opp.insight || opp.recommendation}</p>
                    <p className="text-sm text-green-700 mt-1">{opp.action}</p>
                    {opp.projected_benefit && (
                      <p className="mt-2 text-sm text-green-600">Expected: {opp.projected_benefit}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI Strategic Recommendations */}
          {prescriptiveAnalysis.ai_strategic_recommendations && (
            <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg p-4 border border-emerald-100">
              <div className="flex items-start gap-2">
                <Sparkles className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-emerald-900 mb-2">AI Strategic Recommendations</p>
                  <p className="text-emerald-800 whitespace-pre-wrap">{prescriptiveAnalysis.ai_strategic_recommendations}</p>
                </div>
              </div>
            </div>
          )}

          {/* Action Items */}
          {prescriptiveAnalysis.action_items?.length > 0 && (
            <div className="mt-6">
              <h4 className="font-medium text-slate-900 mb-3">Priority Action Items</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="py-2 px-3 text-left font-medium text-slate-700">Priority</th>
                      <th className="py-2 px-3 text-left font-medium text-slate-700">Action</th>
                      <th className="py-2 px-3 text-left font-medium text-slate-700">Owner</th>
                      <th className="py-2 px-3 text-left font-medium text-slate-700">Timeline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prescriptiveAnalysis.action_items.map((item, idx) => (
                      <tr key={idx} className="border-b border-slate-100">
                        <td className="py-2 px-3">
                          <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs font-medium">
                            P{item.priority}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-slate-900">{item.action}</td>
                        <td className="py-2 px-3 text-slate-600">{item.owner}</td>
                        <td className="py-2 px-3 text-slate-600">{item.timeline}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ML Prediction Results */}
      {mlPrediction && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="w-6 h-6 text-violet-600" />
            <h3 className="text-xl font-semibold text-slate-900">ML Prediction Model</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-violet-50 rounded-lg p-4">
              <p className="text-sm text-violet-600">Model Type</p>
              <p className="text-lg font-semibold text-violet-900">{mlPrediction.model_type}</p>
            </div>
            <div className="bg-violet-50 rounded-lg p-4">
              <p className="text-sm text-violet-600">R² Score</p>
              <p className="text-2xl font-bold text-violet-900">{mlPrediction.performance?.r2_score?.toFixed(3)}</p>
            </div>
            <div className="bg-violet-50 rounded-lg p-4">
              <p className="text-sm text-violet-600">Mean Absolute Error</p>
              <p className="text-2xl font-bold text-violet-900">{mlPrediction.performance?.mean_absolute_error?.toFixed(2)}</p>
            </div>
          </div>

          {/* Feature Importance */}
          <h4 className="font-medium text-slate-900 mb-3">Feature Importance</h4>
          <div className="space-y-2 mb-4">
            {Object.entries(mlPrediction.feature_importance || {}).slice(0, 5).map(([feature, importance]) => (
              <div key={feature} className="flex items-center gap-3">
                <span className="text-sm text-slate-600 w-32 truncate">{feature}</span>
                <div className="flex-1 bg-slate-100 rounded-full h-2">
                  <div 
                    className="bg-violet-600 h-2 rounded-full" 
                    style={{ width: `${(importance * 100).toFixed(0)}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-slate-900 w-12">{(importance * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>

          {mlPrediction.ai_insight && (
            <div className="bg-gradient-to-r from-violet-50 to-purple-50 rounded-lg p-4 border border-violet-100">
              <div className="flex items-start gap-2">
                <Sparkles className="w-5 h-5 text-violet-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-violet-900 mb-1">Model Insight</p>
                  <p className="text-violet-800">{mlPrediction.ai_insight}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Clustering Results */}
      {clustering && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Sparkles className="w-6 h-6 text-amber-600" />
            <h3 className="text-xl font-semibold text-slate-900">Customer Segmentation (K-Means Clustering)</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {Object.entries(clustering.cluster_stats || {}).map(([cluster, stats]) => (
              <div key={cluster} className="bg-amber-50 rounded-lg p-4">
                <p className="text-sm text-amber-600 capitalize">{cluster.replace('_', ' ')}</p>
                <p className="text-2xl font-bold text-amber-900">{stats.size} records</p>
                <p className="text-sm text-amber-700">{stats.percentage?.toFixed(1)}% of total</p>
              </div>
            ))}
          </div>

          {clustering.ai_insight && (
            <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-4 border border-amber-100">
              <div className="flex items-start gap-2">
                <Sparkles className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium text-amber-900 mb-1">Segmentation Insight</p>
                  <p className="text-amber-800">{clustering.ai_insight}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {!aiAnalysis && !prescriptiveAnalysis && !mlPrediction && !clustering && (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center">
          <Brain className="w-16 h-16 text-slate-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-900 mb-2">No Analysis Results Yet</h3>
          <p className="text-slate-600 mb-4">Select a dataset and run AI analyses to see insights.</p>
        </div>
      )}
    </div>
  );
}
