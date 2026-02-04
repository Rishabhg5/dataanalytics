import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { 
  BarChart2, GitCommit, AlertTriangle, CheckCircle, Activity, 
  Table, Layers, ArrowRight, TrendingUp 
} from 'lucide-react';

export default function DataComparison() {
  const { token } = useAuth();
  const [datasets, setDatasets] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/api/datasets', {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => res.json())
    .then(data => setDatasets(data));
  }, [token]);

  const toggleSelection = (id) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(x => x !== id));
    } else if (selectedIds.length < 2) { // Allow max 2 for deep comparison simplicity
      setSelectedIds([...selectedIds, id]);
    }
  };

  const runComparison = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/comparison/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ dataset_ids: selectedIds })
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 pb-20">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Deep Data Comparison</h1>
        <p className="text-slate-600">Select 2 datasets to analyze statistical drift, schema changes, and distribution shifts.</p>
      </div>

      {/* Dataset Selector */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {datasets.map(ds => (
          <div 
            key={ds.id}
            onClick={() => toggleSelection(ds.id)}
            className={`cursor-pointer p-4 rounded-xl border transition-all ${
              selectedIds.includes(ds.id) 
                ? 'bg-indigo-50 border-indigo-500 ring-2 ring-indigo-500' 
                : 'bg-white border-slate-200 hover:border-indigo-300'
            }`}
          >
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-semibold text-slate-900">{ds.title || ds.name}</h3>
                <p className="text-xs text-slate-500 mt-1">{ds.rows.toLocaleString()} rows • {ds.columns} cols</p>
              </div>
              {selectedIds.includes(ds.id) && <CheckCircle className="w-5 h-5 text-indigo-600" />}
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={runComparison}
        disabled={selectedIds.length !== 2 || loading}
        className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-indigo-700 transition-colors flex items-center gap-2"
      >
        {loading ? <Activity className="w-4 h-4 animate-spin" /> : <GitCommit className="w-4 h-4" />}
        Compare 2 Datasets
      </button>

      {/* RESULTS SECTION */}
      {result && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          
          {/* 1. Data Health & Quality Table */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="p-4 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
              <Table className="w-5 h-5 text-slate-500" />
              <h3 className="font-semibold text-slate-900">Data Quality & Health</h3>
            </div>
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-slate-500">
                <tr>
                  <th className="p-4 font-medium">Metric</th>
                  {result.quality_comparison.map(q => (
                    <th key={q.dataset} className="p-4 font-medium text-slate-900">{q.dataset}</th>
                  ))}
                  <th className="p-4 font-medium">Difference</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                <tr>
                  <td className="p-4 text-slate-500">Total Rows</td>
                  {result.quality_comparison.map(q => <td key={q.dataset} className="p-4 font-mono">{q.total_rows.toLocaleString()}</td>)}
                  <td className="p-4 text-slate-400">
                    {Math.abs(result.quality_comparison[0].total_rows - result.quality_comparison[1].total_rows).toLocaleString()}
                  </td>
                </tr>
                <tr>
                  <td className="p-4 text-slate-500">Missing Values</td>
                  {result.quality_comparison.map(q => (
                    <td key={q.dataset} className="p-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${q.missing_percent > 5 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'}`}>
                        {q.missing_percent}%
                      </span>
                    </td>
                  ))}
                  <td className="p-4 text-slate-400">Δ {Math.abs(result.quality_comparison[0].missing_percent - result.quality_comparison[1].missing_percent).toFixed(2)}%</td>
                </tr>
                <tr>
                  <td className="p-4 text-slate-500">Duplicate Rows</td>
                  {result.quality_comparison.map(q => <td key={q.dataset} className="p-4 font-mono">{q.duplicate_rows}</td>)}
                  <td className="p-4"></td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* 2. Numeric Drift Analysis */}
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-indigo-600" />
              Statistical Drift Analysis
            </h3>
            
            <div className="grid grid-cols-1 gap-6">
              {result.numeric_comparison.map((col) => (
                <div key={col.column} className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h4 className="text-lg font-bold text-slate-900">{col.column}</h4>
                      <p className="text-sm text-slate-500 mt-1">Comparing distributions</p>
                    </div>
                    
                    {/* Drift Badge */}
                    <div className={`px-3 py-1.5 rounded-full text-xs font-bold flex items-center gap-1.5 ${
                      col.drift_analysis.status === "High Drift Detected" 
                        ? 'bg-red-100 text-red-700 border border-red-200' 
                        : col.drift_analysis.status === "Moderate Drift"
                        ? 'bg-amber-100 text-amber-700 border border-amber-200'
                        : 'bg-green-100 text-green-700 border border-green-200'
                    }`}>
                      {col.drift_analysis.status === "Stable" ? <CheckCircle className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                      {col.drift_analysis.status}
                      <span className="font-normal opacity-70 ml-1">(p={col.drift_analysis.p_value.toFixed(3)})</span>
                    </div>
                  </div>

                  {/* Histogram Visualization */}
                  <div className="mb-6">
                    <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Distribution Shape</h5>
                    <div className="h-40 flex items-end gap-2 border-b border-slate-200 pb-2">
                      {col.histogram.labels.map((label, idx) => (
                        <div key={idx} className="flex-1 flex gap-1 h-full items-end justify-center group relative">
                          {/* Dataset 1 Bar */}
                          <div 
                            className="w-full bg-indigo-500/80 hover:bg-indigo-600 transition-all rounded-t-sm"
                            style={{ height: `${Object.values(col.histogram.datasets)[0][idx]}%` }}
                          />
                          {/* Dataset 2 Bar */}
                          <div 
                            className="w-full bg-emerald-500/80 hover:bg-emerald-600 transition-all rounded-t-sm"
                            style={{ height: `${Object.values(col.histogram.datasets)[1][idx]}%` }}
                          />
                          
                          {/* Tooltip */}
                          <div className="absolute bottom-full mb-2 hidden group-hover:block bg-slate-800 text-white text-xs p-2 rounded z-10 whitespace-nowrap">
                            Range: {label}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="flex justify-between mt-2 text-xs text-slate-400 px-1">
                      <span>Low Value</span>
                      <span>High Value</span>
                    </div>
                    {/* Legend */}
                    <div className="flex justify-center gap-4 mt-3">
                      <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                        <div className="w-3 h-3 bg-indigo-500 rounded-sm"></div>
                        {Object.keys(col.histogram.datasets)[0]}
                      </div>
                      <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                        <div className="w-3 h-3 bg-emerald-500 rounded-sm"></div>
                        {Object.keys(col.histogram.datasets)[1]}
                      </div>
                    </div>
                  </div>

                  {/* Statistical Summary Table */}
                  <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-lg text-sm">
                     {Object.entries(col.stats).map(([dsName, stats]) => (
                       <div key={dsName}>
                          <h6 className="font-semibold text-slate-700 mb-2 truncate" title={dsName}>{dsName}</h6>
                          <div className="space-y-1 text-slate-600">
                            <div className="flex justify-between"><span>Mean:</span> <span className="font-mono">{stats.mean.toFixed(2)}</span></div>
                            <div className="flex justify-between"><span>Std Dev:</span> <span className="font-mono">{stats.std.toFixed(2)}</span></div>
                            <div className="flex justify-between"><span>Zeros:</span> <span className="font-mono">{stats.zeros}</span></div>
                          </div>
                       </div>
                     ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* 3. Schema Differences */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <div className="flex items-center gap-2 mb-4">
              <Layers className="w-5 h-5 text-indigo-600" />
              <h3 className="text-lg font-bold text-slate-900">Schema Differences</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {Object.entries(result.schema_diff).map(([ds, cols]) => (
                <div key={ds}>
                  <h4 className="font-medium text-slate-900 mb-3 flex items-center gap-2">
                    <ArrowRight className="w-4 h-4 text-indigo-400" />
                    Unique to: <span className="text-indigo-600">{ds}</span>
                  </h4>
                  {cols.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {cols.map(c => (
                        <span key={c} className="px-2 py-1 bg-slate-100 text-slate-700 text-xs rounded border border-slate-200 font-mono">
                          {c}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-400 italic">No unique columns (Schema matches)</p>
                  )}
                </div>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}