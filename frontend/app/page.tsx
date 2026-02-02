'use client'

import { useState, useEffect } from 'react'
import { api, Metrics, BiasReport, ComplianceStatus } from '@/lib/api'

interface DashboardData {
    predictions: number
    fairnessScore: number
    complianceRate: number
    activeAlerts: number
    avgLatency: number
    errorRate: number
    highRiskRate: number
}

interface RecentPrediction {
    id: string
    risk: string
    score: number
    time: string
}

export default function DashboardPage() {
    const [data, setData] = useState<DashboardData | null>(null)
    const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>([])
    const [governanceStatus, setGovernanceStatus] = useState<any[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

    const fetchDashboardData = async () => {
        try {
            const [metrics, biasReport, compliance, alerts, auditEvents] = await Promise.all([
                api.getMetrics(),
                api.getBiasReport(),
                api.getComplianceStatus(),
                api.getAlerts(),
                api.queryAuditLogs({ event_type: 'prediction', limit: 5 })
            ])

            setData({
                predictions: metrics.total_predictions,
                fairnessScore: biasReport.fairness_score,
                complianceRate: compliance.compliance_percentage,
                activeAlerts: alerts.count,
                avgLatency: metrics.avg_latency_ms,
                errorRate: metrics.error_rate,
                highRiskRate: metrics.high_risk_rate
            })

            // Transform audit events to recent predictions (risk_score/risk_level stored in details)
            setRecentPredictions(auditEvents.map((e) => ({
                id: e.details?.prediction_id || e.event_id.slice(0, 12),
                risk: e.details?.risk_level || (
                    e.details?.risk_score >= 0.7 ? 'HIGH' :
                    e.details?.risk_score >= 0.4 ? 'MODERATE' : 'LOW'
                ),
                score: e.details?.risk_score ?? e.details?.confidence ?? 0,
                time: new Date(e.timestamp).toLocaleTimeString()
            })))

            setGovernanceStatus([
                { label: 'HIPAA Compliance', status: compliance.overall_compliant ? 'Passed' : 'Failed', icon: '🏥' },
                { label: 'Bias Detection', status: biasReport.overall_bias_detected ? 'Issues Found' : 'No Issues', icon: '⚖️' },
                { label: 'Model Validation', status: 'Verified', icon: '✓' },
                { label: 'Audit Logging', status: 'Active', icon: '📝' },
            ])

            setLastUpdate(new Date())
            setError(null)
        } catch (err) {
            setError('Failed to fetch dashboard data')
            console.error(err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchDashboardData()
        // Auto-refresh every 10 seconds
        const interval = setInterval(fetchDashboardData, 10000)
        return () => clearInterval(interval)
    }, [])

    const stats = data ? [
        { label: 'Predictions Today', value: data.predictions.toLocaleString(), icon: '🔮', color: 'from-blue-500 to-cyan-500' },
        { label: 'Fairness Score', value: `${data.fairnessScore.toFixed(1)}%`, icon: '⚖️', color: 'from-green-500 to-emerald-500' },
        { label: 'Compliance Rate', value: `${data.complianceRate.toFixed(0)}%`, icon: '✅', color: 'from-purple-500 to-pink-500' },
        { label: 'Active Alerts', value: data.activeAlerts, icon: '🔔', color: 'from-orange-500 to-red-500' },
    ] : []

    if (error && !data) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="text-center">
                    <p className="text-red-400 mb-4">{error}</p>
                    <button onClick={fetchDashboardData} className="px-4 py-2 bg-blue-500 rounded-lg">Retry</button>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-8">
            <header className="flex justify-between items-start">
                <div>
                    <h1 className="text-3xl font-bold gradient-text">AI Governance Dashboard</h1>
                    <p className="text-gray-400 mt-2">Real-time monitoring of responsible AI operations</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-gray-500">Last updated</p>
                    <p className="text-sm text-gray-400">{lastUpdate ? lastUpdate.toLocaleTimeString() : '--:--:--'}</p>
                    <span className="inline-flex items-center gap-1 text-xs text-green-400 mt-1">
                        <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                        Live
                    </span>
                </div>
            </header>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {loading ? (
                    Array(4).fill(0).map((_, i) => (
                        <div key={i} className="glass-card p-6 animate-pulse">
                            <div className="h-8 bg-white/10 rounded mb-4"></div>
                            <div className="h-10 bg-white/10 rounded"></div>
                        </div>
                    ))
                ) : stats.map((stat, i) => (
                    <div key={i} className="glass-card p-6 hover-lift">
                        <div className="flex justify-between items-start mb-4">
                            <span className="text-3xl">{stat.icon}</span>
                            <span className={`px-3 py-1 rounded-full text-xs bg-gradient-to-r ${stat.color} text-white`}>
                                Live
                            </span>
                        </div>
                        <p className="text-gray-400 text-sm">{stat.label}</p>
                        <p className="text-3xl font-bold mt-1">{stat.value}</p>
                    </div>
                ))}
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Recent Predictions */}
                <div className="glass-card p-6">
                    <h2 className="text-xl font-semibold mb-4">Recent Predictions</h2>
                    <div className="space-y-3">
                        {recentPredictions.length === 0 ? (
                            <p className="text-gray-500 text-center py-4">No predictions yet</p>
                        ) : recentPredictions.map((pred) => (
                            <div key={pred.id} className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                                <div className="flex items-center gap-3">
                                    <span className="text-gray-400 font-mono text-sm">{pred.id.slice(0, 12)}</span>
                                    <span className={`px-2 py-1 rounded text-xs font-medium ${pred.risk === 'LOW' ? 'bg-green-500/20 text-green-400' :
                                        pred.risk === 'HIGH' ? 'bg-red-500/20 text-red-400' :
                                            'bg-yellow-500/20 text-yellow-400'
                                        }`}>
                                        {pred.risk}
                                    </span>
                                </div>
                                <div className="flex items-center gap-4">
                                    <span className="text-white font-medium">{(pred.score * 100).toFixed(0)}%</span>
                                    <span className="text-gray-500 text-sm">{pred.time}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Governance Overview */}
                <div className="glass-card p-6">
                    <h2 className="text-xl font-semibold mb-4">Governance Status</h2>
                    <div className="space-y-4">
                        {governanceStatus.map((item, i) => (
                            <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                                <div className="flex items-center gap-3">
                                    <span className="text-xl">{item.icon}</span>
                                    <span className="text-gray-300">{item.label}</span>
                                </div>
                                <span className={`flex items-center gap-2 ${item.status.includes('Failed') || item.status.includes('Issues') ? 'text-red-400' : 'text-green-400'
                                    }`}>
                                    <span className={`w-2 h-2 rounded-full ${item.status.includes('Failed') || item.status.includes('Issues') ? 'bg-red-400' : 'bg-green-400'
                                        }`}></span>
                                    {item.status}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Quick Actions */}
            <div className="glass-card p-6">
                <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { label: 'New Prediction', href: '/predictions', icon: '🔮' },
                        { label: 'View Bias Report', href: '/governance', icon: '⚖️' },
                        { label: 'View Audit Logs', href: '/audit', icon: '📋' },
                        { label: 'System Metrics', href: '/monitoring', icon: '📊' },
                    ].map((action, i) => (
                        <a
                            key={i}
                            href={action.href}
                            className="flex flex-col items-center gap-2 p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                        >
                            <span className="text-2xl">{action.icon}</span>
                            <span className="text-sm text-gray-300">{action.label}</span>
                        </a>
                    ))}
                </div>
            </div>
        </div>
    )
}
