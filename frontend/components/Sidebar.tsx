'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState, useEffect } from 'react'

const navItems = [
    { href: '/', label: 'Dashboard', icon: '\u{1F4CA}' },
    { href: '/predictions', label: 'Predictions', icon: '\u{1F52E}' },
    { href: '/governance', label: 'Governance', icon: '\u2696\uFE0F' },
    { href: '/audit', label: 'Audit Trail', icon: '\u{1F4CB}' },
    { href: '/monitoring', label: 'Monitoring', icon: '\u{1F4C8}' },
]

export default function Sidebar() {
    const pathname = usePathname()
    const [healthStatus, setHealthStatus] = useState<string>('checking')
    const [modelVersion, setModelVersion] = useState<string>('1.0.0')

    useEffect(() => {
        const checkHealth = async () => {
            try {
                const res = await fetch('/api/v1/monitoring/health')
                if (res.ok) {
                    const data = await res.json()
                    setHealthStatus(data.status || 'healthy')
                    setModelVersion(data.version || '1.0.0')
                } else {
                    setHealthStatus('unhealthy')
                }
            } catch {
                setHealthStatus('offline')
            }
        }

        checkHealth()
        const interval = setInterval(checkHealth, 15000)
        return () => clearInterval(interval)
    }, [])

    const statusColor = healthStatus === 'healthy' ? 'bg-green-500' :
        healthStatus === 'checking' ? 'bg-yellow-500' : 'bg-red-500'
    const statusTextColor = healthStatus === 'healthy' ? 'text-green-400' :
        healthStatus === 'checking' ? 'text-yellow-400' : 'text-red-400'
    const statusLabel = healthStatus === 'healthy' ? 'System Healthy' :
        healthStatus === 'checking' ? 'Connecting...' :
        healthStatus === 'offline' ? 'Backend Offline' : 'System Unhealthy'

    return (
        <aside className="fixed left-0 top-0 h-screen w-64 glass-card border-r border-white/10 p-6 flex flex-col">
            <div className="mb-8">
                <h1 className="text-xl font-bold gradient-text">AI Governance</h1>
                <p className="text-sm text-gray-400 mt-1">Healthcare System</p>
            </div>

            <nav className="flex-1 space-y-2">
                {navItems.map((item) => {
                    const isActive = pathname === item.href
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${isActive
                                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                                    : 'text-gray-400 hover:bg-white/5 hover:text-white'
                                }`}
                        >
                            <span className="text-xl">{item.icon}</span>
                            <span className="font-medium">{item.label}</span>
                        </Link>
                    )
                })}
            </nav>

            <div className="glass-card p-4 mt-auto">
                <div className="flex items-center gap-2 mb-2">
                    <span className={`w-2 h-2 ${statusColor} rounded-full animate-pulse`}></span>
                    <span className={`text-sm ${statusTextColor}`}>{statusLabel}</span>
                </div>
                <p className="text-xs text-gray-500">v{modelVersion} &bull; PyTorch + SHAP</p>
            </div>
        </aside>
    )
}
