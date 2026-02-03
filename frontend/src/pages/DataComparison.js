import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { BarChart, Activity, GitCommit, AlertTriangle, CheckCircle } from 'lucide-react';

export default function DataComparison() {
  const { token } = useAuth();
  const [datasets, setDatasets] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch available datasets on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/datasets', {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => res.json())
    .then(data => setDatasets(data))
    .catch(err => console.error(err));
  }, [token]);

  const toggleSelection = (id) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(x => x !== id));
    } else {
      if (selectedIds.length >= 3) return; // Limit to 3 for UI clarity
      setSelectedIds([...selectedIds, id]);
    }
  };

  const runComparison = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/comparison/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ dataset_ids: selectedIds })
      });
      const data = await res.json();
      setComparisonResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <header>
        <h1 className="text-3xl font-bold text-white mb-2">Data Comparison</h1>
        <p className="text-slate-400">Select up to 3 datasets to analyze differences and drift.</p>
      </header>

      {/* 1. Dataset Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {datasets.map(ds => (
          <div 
            key={ds.id}
            onClick={() => toggleSelection(ds.id)}
            className={`cursor-pointer p-4 rounded-xl border transition-all ${
              selectedIds.includes(ds.id) 
                ? 'bg-indigo-600/20 border-indigo-500 ring-1 ring-indigo-500' 
                : 'bg-white/5 border-white/10 hover:bg-white/10'
            }`}
          >
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-semibold text-white">{ds.title || ds.name}</h3>
                <p className="text-xs text-slate-400 mt-1">{ds.rows} rows • {ds.columns} cols</p>
              </div>
              {selectedIds.includes(ds.id) && <CheckCircle className="w-5 h-5 text-indigo-400" />}
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={runComparison}
        disabled={selectedIds.length < 2 || loading}
        className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-indigo-700 transition-colors flex items-center gap-2"
      >
        {loading ? <Activity className="w-4 h-4 animate-spin" /> : <GitCommit className="w-4 h-4" />}
        Compare Selected Data
      </button>

      {/* 2. Results View */}
      {comparisonResult && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-800/50 p-4 rounded-xl border border-white/10">
              <h4 className="text-sm text-slate-400">Common Columns</h4>
              <p className="text-2xl font-bold text-white">{comparisonResult.summary.common_columns_count}</p>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-xl border border-white/10">
              <h4 className="text-sm text-slate-400">Combined Rows</h4>
              <p className="text-2xl font-bold text-white">{comparisonResult.summary.total_rows_combined}</p>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-xl border border-white/10">
              <h4 className="text-sm text-slate-400">Schema Match</h4>
              <p className="text-xl font-bold text-emerald-400">
                {Object.values(comparisonResult.schema_diff.unique_columns_per_dataset).every(x => x.length === 0) 
                  ? "Perfect Match" 
                  : "Partial Match"}
              </p>
            </div>
          </div>

          {/* Numeric Comparison */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <BarChart className="w-5 h-5 text-indigo-400" />
              Numeric Drift Analysis
            </h2>
            <div className="grid gap-6">
              {comparisonResult.numeric_comparison.map(col => (
                <div key={col.column} className="bg-white/5 rounded-xl border border-white/10 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="font-semibold text-lg text-white">{col.column}</h3>
                    {col.statistical_test?.significant && (
                      <span className="flex items-center gap-1 text-amber-400 text-sm bg-amber-400/10 px-2 py-1 rounded">
                        <AlertTriangle className="w-3 h-3" /> Significant Drift
                      </span>
                    )}
                  </div>
                  
                  {/* Comparison Bar */}
                  <div className="space-y-3">
                    {col.metrics.map(metric => (
                      <div key={metric.dataset} className="flex items-center gap-4">
                        <span className="w-32 text-sm text-slate-400 truncate">{metric.dataset}</span>
                        <div className="flex-1 h-8 bg-slate-700/50 rounded-md overflow-hidden relative">
                          {/* Visual bar representing value relative to max in group */}
                          <div 
                            className="h-full bg-indigo-500/50 absolute top-0 left-0 transition-all duration-1000"
                            style={{ 
                              width: `${(metric.mean / Math.max(...col.metrics.map(m => m.mean))) * 100}%` 
                            }}
                          />
                          <div className="absolute inset-0 flex items-center justify-between px-3 text-xs">
                            <span className="text-white z-10 font-medium">Mean: {metric.mean.toFixed(2)}</span>
                            <span className="text-slate-300 z-10">Std: {metric.std.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Schema Differences (if any) */}
          <div className="bg-slate-800/50 rounded-xl border border-white/10 p-6">
             <h2 className="text-xl font-bold text-white mb-4">Schema Differences</h2>
             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(comparisonResult.schema_diff.unique_columns_per_dataset).map(([ds, cols]) => (
                  <div key={ds}>
                    <h4 className="font-medium text-indigo-300 mb-2">{ds} (Unique Columns)</h4>
                    {cols.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {cols.map(c => (
                          <span key={c} className="text-xs bg-slate-700 text-slate-200 px-2 py-1 rounded border border-white/10">
                            {c}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-sm text-slate-500 italic">No unique columns</span>
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