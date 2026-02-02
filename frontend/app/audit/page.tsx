'use client'

import { useState, useEffect } from 'react'
import { api, AuditEvent } from '@/lib/api'

export default function AuditPage() {
    const [filter, setFilter] = useState('all')
    const [searchQuery, setSearchQuery] = useState('')
    const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([])
    const [stats, setStats] = useState<any>(null)
    const [loading, setLoading] = useState(true)
    const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

    const fetchData = async () => {
        try {
            const [events, auditStats] = await Promise.all([
                api.queryAuditLogs({ limit: 100 }),
                api.getAuditStats()
            ])
            setAuditEvents(events)
            setStats(auditStats)
            setLastUpdate(new Date())
        } catch (err) {
            console.error('Failed to fetch audit data:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchData()
        const interval = setInterval(fetchData, 15000) // Refresh every 15 seconds
        return () => clearInterval(interval)
    }, [])

    const filteredEvents = auditEvents.filter(e =>
        (filter === 'all' || e.event_type === filter) &&
        (searchQuery === '' ||
            e.event_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
            e.action.toLowerCase().includes(searchQuery.toLowerCase()))
    )

    const eventTypes = ['all', 'prediction', 'bias_check', 'compliance_check', 'model_load']

    const getEventTypeColor = (type: string) => {
        switch (type) {
            case 'prediction': return 'bg-blue-500/20 text-blue-400'
            case 'bias_check': return 'bg-purple-500/20 text-purple-400'
            case 'compliance_check': return 'bg-green-500/20 text-green-400'
            case 'model_load': return 'bg-orange-500/20 text-orange-400'
            default: return 'bg-gray-500/20 text-gray-400'
        }
    }

    if (loading) {
        return (
            <div className="space-y-8">
                <header>
                    <h1 className="text-3xl font-bold gradient-text">Audit Trail</h1>
                    <p className="text-gray-400 mt-2">Loading audit logs...</p>
                </header>
                <div className="glass-card p-6 animate-pulse">
                    <div className="h-64 bg-white/10 rounded"></div>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-8">
            <header className="flex justify-between items-start">
                <div>
                    <h1 className="text-3xl font-bold gradient-text">Audit Trail</h1>
                    <p className="text-gray-400 mt-2">Immutable log of all governance events with 7-year retention</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-gray-500">Last updated</p>
                    <p className="text-sm text-gray-400">{lastUpdate.toLocaleTimeString()}</p>
                    <span className="inline-flex items-center gap-1 text-xs text-green-400 mt-1">
                        <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                        Live
                    </span>
                </div>
            </header>

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card p-4 text-center">
                    <p className="text-2xl font-bold">{stats?.total_events?.toLocaleString() || auditEvents.length}</p>
                    <p className="text-gray-400 text-sm">Total Events</p>
                </div>
                <div className="glass-card p-4 text-center">
                    <p className="text-2xl font-bold">
                        {stats?.prediction_count?.toLocaleString() ||
                         stats?.events_by_type?.prediction?.toLocaleString() ||
                         auditEvents.filter(e => e.event_type === 'prediction').length}
                    </p>
                    <p className="text-gray-400 text-sm">Predictions Logged</p>
                </div>
                <div className="glass-card p-4 text-center">
                    <p className={`text-2xl font-bold ${stats?.chain_intact !== false ? 'text-green-400' : 'text-red-400'}`}>
                        {stats?.chain_intact !== false ? '\u2713' : '\u2717'}
                    </p>
                    <p className="text-gray-400 text-sm">Chain Integrity</p>
                </div>
                <div className="glass-card p-4 text-center">
                    <p className="text-2xl font-bold">{stats?.retention_days ? `${Math.round(stats.retention_days / 365)}yr` : '7yr'}</p>
                    <p className="text-gray-400 text-sm">Retention Period</p>
                </div>
            </div>

            {/* Filters */}
            <div className="flex flex-wrap gap-4 items-center">
                <input
                    type="text"
                    placeholder="Search events..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:border-blue-500 w-64"
                />
                <div className="flex gap-2">
                    {eventTypes.map((type) => (
                        <button
                            key={type}
                            onClick={() => setFilter(type)}
                            className={`px-3 py-1 rounded-full text-sm transition-colors ${filter === type
                                    ? 'bg-blue-500 text-white'
                                    : 'bg-white/10 text-gray-400 hover:bg-white/20'
                                }`}
                        >
                            {type.replace('_', ' ')}
                        </button>
                    ))}
                </div>
                <button
                    onClick={fetchData}
                    className="ml-auto px-4 py-1 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors"
                >
                    Refresh
                </button>
            </div>

            {/* Events Table */}
            <div className="glass-card overflow-hidden">
                <table className="w-full">
                    <thead>
                        <tr className="text-left text-gray-400 bg-white/5">
                            <th className="px-6 py-4">Event ID</th>
                            <th className="px-6 py-4">Type</th>
                            <th className="px-6 py-4">Action</th>
                            <th className="px-6 py-4">Timestamp</th>
                            <th className="px-6 py-4">Model Version</th>
                            <th className="px-6 py-4">Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredEvents.length === 0 ? (
                            <tr>
                                <td colSpan={6} className="px-6 py-8 text-center text-gray-500">
                                    No audit events found
                                </td>
                            </tr>
                        ) : filteredEvents.slice(0, 50).map((event) => (
                            <tr key={event.event_id} className="border-t border-white/5 hover:bg-white/5">
                                <td className="px-6 py-4 font-mono text-sm text-blue-400">{event.event_id.slice(0, 12)}</td>
                                <td className="px-6 py-4">
                                    <span className={`px-2 py-1 rounded text-xs ${getEventTypeColor(event.event_type)}`}>
                                        {event.event_type}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-gray-300">{event.action}</td>
                                <td className="px-6 py-4 text-gray-400 text-sm">{new Date(event.timestamp).toLocaleString()}</td>
                                <td className="px-6 py-4 font-mono text-sm">{event.model_version}</td>
                                <td className="px-6 py-4 text-sm text-gray-400 max-w-xs truncate">
                                    {JSON.stringify(event.details).slice(0, 50)}...
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
