'use client'

import { useState, useEffect } from 'react'
import { api, Metrics, HealthStatus, PredictionStats } from '@/lib/api'

export default function MonitoringPage() {
    const [health, setHealth] = useState<HealthStatus | null>(null)
    const [metrics, setMetrics] = useState<Metrics | null>(null)
    const [predStats, setPredStats] = useState<PredictionStats | null>(null)
    const [alerts, setAlerts] = useState<any[]>([])
    const [loading, setLoading] = useState(true)
    const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

    const fetchData = async () => {
        try {
            const [healthData, metricsData, statsData, alertsData] = await Promise.all([
                api.getHealth(),
                api.getMetrics(),
                api.getPredictionStats(),
                api.getAlerts()
            ])
            setHealth(healthData)
            setMetrics(metricsData)
            setPredStats(statsData)
            setAlerts(alertsData.alerts || [])
            setLastUpdate(new Date())
        } catch (err) {
            console.error('Failed to fetch monitoring data:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchData()
        const interval = setInterval(fetchData, 5000) // Refresh every 5 seconds
        return () => clearInterval(interval)
    }, [])

    if (loading) {
        return (
            <div className="space-y-8">
                <header>
                    <h1 className="text-3xl font-bold gradient-text">System Monitoring</h1>
                    <p className="text-gray-400 mt-2">Loading metrics...</p>
                </header>
                <div className="glass-card p-6 animate-pulse">
                    <div className="h-32 bg-white/10 rounded"></div>
                </div>
            </div>
        )
    }

    const highRiskPct = predStats ?
        (predStats.high_risk_count / Math.max(predStats.total_predictions, 1) * 100) : 0
    const lowRiskPct = predStats ?
        (predStats.low_risk_count / Math.max(predStats.total_predictions, 1) * 100) : 0
    const moderatePct = 100 - highRiskPct - lowRiskPct

    return (
        <div className="space-y-8">
            <header className="flex justify-between items-start">
                <div>
                    <h1 className="text-3xl font-bold gradient-text">System Monitoring</h1>
                    <p className="text-gray-400 mt-2">Real-time performance metrics and system health</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-gray-500">Auto-updating every 5s</p>
                    <p className="text-sm text-gray-400">{lastUpdate.toLocaleTimeString()}</p>
                    <span className="inline-flex items-center gap-1 text-xs text-green-400 mt-1">
                        <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                        Live
                    </span>
                </div>
            </header>

            {/* Health Status */}
            <div className="glass-card p-6">
                <div className="flex items-center gap-4 mb-4">
                    <span className={`w-4 h-4 rounded-full animate-pulse ${health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                        }`}></span>
                    <h2 className="text-xl font-semibold">
                        System Health: <span className={health?.status === 'healthy' ? 'text-green-400' : 'text-red-400'}>
                            {health?.status?.toUpperCase() || 'Unknown'}
                        </span>
                    </h2>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-white/5 rounded-lg">
                        <p className="text-gray-400 text-sm">Uptime</p>
                        <p className="text-2xl font-bold text-green-400">{health?.uptime_hours.toFixed(1) || 0}h</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                        <p className="text-gray-400 text-sm">Avg Latency</p>
                        <p className="text-2xl font-bold">{metrics?.avg_latency_ms.toFixed(1) || 0}ms</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                        <p className="text-gray-400 text-sm">P95 Latency</p>
                        <p className="text-2xl font-bold">{predStats?.p95_latency_ms.toFixed(1) || 0}ms</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                        <p className="text-gray-400 text-sm">Error Rate</p>
                        <p className={`text-2xl font-bold ${(metrics?.error_rate || 0) < 0.01 ? 'text-green-400' : 'text-red-400'}`}>
                            {((metrics?.error_rate || 0) * 100).toFixed(2)}%
                        </p>
                    </div>
                </div>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="glass-card p-6">
                    <h2 className="text-xl font-semibold mb-6">Prediction Metrics</h2>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                            <span className="text-gray-400">Total Predictions</span>
                            <span className="text-2xl font-bold">{metrics?.total_predictions.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                            <span className="text-gray-400">Predictions (Last Hour)</span>
                            <span className="text-xl font-bold">{metrics?.predictions_last_hour.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                            <span className="text-gray-400">High Risk Rate</span>
                            <span className={`text-xl font-bold ${(metrics?.high_risk_rate || 0) > 0.3 ? 'text-yellow-400' : 'text-green-400'}`}>
                                {((metrics?.high_risk_rate || 0) * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                            <span className="text-gray-400">Error Count (Window)</span>
                            <span className="text-xl font-bold">{predStats?.error_count || 0}</span>
                        </div>
                    </div>
                </div>

                <div className="glass-card p-6">
                    <h2 className="text-xl font-semibold mb-6">Risk Distribution</h2>
                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between mb-2">
                                <span className="text-gray-400">Low Risk</span>
                                <span className="text-green-400">{lowRiskPct.toFixed(1)}%</span>
                            </div>
                            <div className="h-4 bg-white/10 rounded overflow-hidden">
                                <div className="h-full bg-green-500 transition-all duration-500" style={{ width: `${lowRiskPct}%` }}></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between mb-2">
                                <span className="text-gray-400">Moderate Risk</span>
                                <span className="text-yellow-400">{moderatePct.toFixed(1)}%</span>
                            </div>
                            <div className="h-4 bg-white/10 rounded overflow-hidden">
                                <div className="h-full bg-yellow-500 transition-all duration-500" style={{ width: `${moderatePct}%` }}></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between mb-2">
                                <span className="text-gray-400">High Risk</span>
                                <span className="text-red-400">{highRiskPct.toFixed(1)}%</span>
                            </div>
                            <div className="h-4 bg-white/10 rounded overflow-hidden">
                                <div className="h-full bg-red-500 transition-all duration-500" style={{ width: `${highRiskPct}%` }}></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Alerts */}
            <div className="glass-card p-6">
                <h2 className="text-xl font-semibold mb-4">Active Alerts</h2>
                {alerts.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                        <span className="text-3xl">✓</span>
                        <p className="mt-2">No active alerts - System operating normally</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {alerts.map((alert: any, i) => (
                            <div key={i} className={`flex items-center gap-4 p-4 rounded-lg border ${alert.level === 'warning' ? 'bg-yellow-500/10 border-yellow-500/30' :
                                    alert.level === 'error' ? 'bg-red-500/10 border-red-500/30' :
                                        'bg-blue-500/10 border-blue-500/30'
                                }`}>
                                <span>{alert.level === 'warning' ? '⚠️' : alert.level === 'error' ? '🚨' : 'ℹ️'}</span>
                                <div>
                                    <p className="font-medium">{alert.message}</p>
                                    <p className="text-sm text-gray-400">{new Date(alert.timestamp).toLocaleString()}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Capacity */}
            <div className="glass-card p-6">
                <h2 className="text-xl font-semibold mb-4">System Capacity</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                        <p className="text-gray-400 mb-2">Daily Capacity</p>
                        <p className="text-3xl font-bold">1M+</p>
                        <p className="text-green-400 text-sm">predictions/day</p>
                    </div>
                    <div>
                        <p className="text-gray-400 mb-2">Current Load</p>
                        <p className="text-3xl font-bold">{((metrics?.total_predictions || 0) / 1000000 * 100).toFixed(2)}%</p>
                        <p className="text-green-400 text-sm">of capacity</p>
                    </div>
                    <div>
                        <p className="text-gray-400 mb-2">Model Version</p>
                        <p className="text-3xl font-bold">{metrics?.model_version || '1.0.0'}</p>
                        <p className="text-gray-400 text-sm">HealthcareRiskModel</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
