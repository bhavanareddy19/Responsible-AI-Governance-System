'use client'

import { useState, useEffect } from 'react'
import { api, BiasReport, ComplianceStatus, ComplianceCheck } from '@/lib/api'

export default function GovernancePage() {
    const [biasReport, setBiasReport] = useState<BiasReport | null>(null)
    const [compliance, setCompliance] = useState<ComplianceStatus | null>(null)
    const [loading, setLoading] = useState(true)
    const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
    const [error, setError] = useState<string | null>(null)

    const fetchData = async () => {
        try {
            const [bias, comp] = await Promise.all([
                api.getBiasReport(),
                api.getComplianceStatus()
            ])
            setBiasReport(bias)
            setCompliance(comp)
            setLastUpdate(new Date())
            setError(null)
        } catch (err) {
            setError('Failed to fetch governance data. Ensure backend is running.')
            console.error('Failed to fetch governance data:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchData()
        const interval = setInterval(fetchData, 30000)
        return () => clearInterval(interval)
    }, [])

    const getStandardLabel = (standard: string): string => {
        switch (standard) {
            case 'hipaa': return 'HIPAA'
            case 'fda_aiml': return 'FDA AI/ML'
            case 'gdpr': return 'GDPR'
            case 'iso_42001': return 'ISO 42001'
            default: return standard.toUpperCase()
        }
    }

    if (loading) {
        return (
            <div className="space-y-8">
                <header>
                    <h1 className="text-3xl font-bold gradient-text">AI Governance</h1>
                    <p className="text-gray-400 mt-2">Loading governance data...</p>
                </header>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="glass-card p-6 animate-pulse">
                            <div className="h-20 bg-white/10 rounded"></div>
                        </div>
                    ))}
                </div>
            </div>
        )
    }

    if (error && !biasReport && !compliance) {
        return (
            <div className="space-y-8">
                <header>
                    <h1 className="text-3xl font-bold gradient-text">AI Governance</h1>
                </header>
                <div className="glass-card p-6 text-center">
                    <p className="text-red-400 mb-4">{error}</p>
                    <button onClick={fetchData} className="px-4 py-2 bg-blue-500 rounded-lg">Retry</button>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-8">
            <header className="flex justify-between items-start">
                <div>
                    <h1 className="text-3xl font-bold gradient-text">AI Governance</h1>
                    <p className="text-gray-400 mt-2">Bias detection, fairness metrics, and regulatory compliance</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-gray-500">Last updated</p>
                    <p className="text-sm text-gray-400">{lastUpdate.toLocaleTimeString()}</p>
                    <button
                        onClick={fetchData}
                        className="text-xs text-blue-400 hover:text-blue-300 mt-1"
                    >
                        Refresh
                    </button>
                </div>
            </header>

            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="glass-card p-6 text-center">
                    <p className="text-gray-400 text-sm">Fairness Score</p>
                    <p className={`text-4xl font-bold my-2 ${(biasReport?.fairness_score || 0) >= 90 ? 'text-green-400' :
                            (biasReport?.fairness_score || 0) >= 70 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                        {biasReport?.fairness_score.toFixed(1) || 0}%
                    </p>
                    <p className={`text-sm ${(biasReport?.fairness_score || 0) >= 90 ? 'text-green-400' : 'text-yellow-400'
                        }`}>
                        {(biasReport?.fairness_score || 0) >= 90 ? 'Above threshold' : 'Review recommended'}
                    </p>
                </div>
                <div className="glass-card p-6 text-center">
                    <p className="text-gray-400 text-sm">Compliance Status</p>
                    <p className={`text-4xl font-bold my-2 ${compliance?.overall_compliant ? 'text-green-400' : 'text-red-400'}`}>
                        {compliance?.checks_passed || 0}/{(compliance?.checks_passed || 0) + (compliance?.checks_failed || 0)}
                    </p>
                    <p className={`text-sm ${compliance?.overall_compliant ? 'text-green-400' : 'text-red-400'}`}>
                        {compliance?.overall_compliant ? 'All checks passed' : `${compliance?.checks_failed} checks failed`}
                    </p>
                </div>
                <div className="glass-card p-6 text-center">
                    <p className="text-gray-400 text-sm">Risk Category</p>
                    <p className={`text-4xl font-bold my-2 ${compliance?.risk_category === 'low' ? 'text-green-400' :
                            compliance?.risk_category === 'medium' ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                        {compliance?.risk_category?.toUpperCase() || 'N/A'}
                    </p>
                    <p className="text-gray-400 text-sm">
                        {compliance?.risk_category === 'low' ? 'No remediation needed' : 'Action may be required'}
                    </p>
                </div>
            </div>

            {/* Bias Detection */}
            <div className="glass-card p-6">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-semibold">Bias Detection Results</h2>
                    <span className={`px-3 py-1 rounded-full text-xs ${biasReport?.overall_bias_detected ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                        }`}>
                        {biasReport?.overall_bias_detected ? 'Bias Detected' : 'No Bias Detected'}
                    </span>
                </div>

                <div className="mb-4">
                    <p className="text-gray-400 text-sm mb-2">Attributes Analyzed:</p>
                    <div className="flex gap-2">
                        {biasReport?.attributes_analyzed.map((attr, i) => (
                            <span key={i} className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
                                {attr}
                            </span>
                        ))}
                    </div>
                </div>

                <div className="mb-4">
                    <p className="text-gray-400 text-sm">
                        Total Samples Analyzed: <span className="text-white font-medium">{biasReport?.total_samples.toLocaleString()}</span>
                    </p>
                </div>

                {biasReport?.bias_summary && biasReport.bias_summary.length > 0 && (
                    <div className="mt-4 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                        <h3 className="text-sm font-medium text-yellow-400 mb-2">Bias Summary</h3>
                        <div className="space-y-2">
                            {biasReport.bias_summary.map((item: any, i) => (
                                <div key={i} className="text-sm text-gray-300">
                                    <span className="text-yellow-400">{item.attribute}</span> ({item.group}): {item.issue}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {biasReport?.recommendations && biasReport.recommendations.length > 0 && (
                    <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                        <h3 className="text-sm font-medium text-blue-400 mb-2">Recommendations</h3>
                        <ul className="space-y-1 text-sm text-gray-300">
                            {biasReport.recommendations.map((rec, i) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="text-blue-400">&#x2022;</span>
                                    {rec}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>

            {/* Compliance Checks - Dynamic from API */}
            <div className="glass-card p-6">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-semibold">Regulatory Compliance</h2>
                    <span className="text-sm text-gray-400">
                        {compliance?.compliance_percentage.toFixed(0)}% compliant
                    </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {(compliance?.all_checks || []).map((check) => (
                        <div key={check.check_id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                            <div>
                                <p className="font-medium">{check.requirement}</p>
                                <p className="text-sm text-gray-400">{getStandardLabel(check.standard)}</p>
                                <p className="text-xs text-gray-500 mt-1">{check.details}</p>
                            </div>
                            <span className={`flex items-center gap-2 ${check.passed ? 'text-green-400' : 'text-red-400'}`}>
                                <span className={`w-2 h-2 rounded-full ${check.passed ? 'bg-green-400' : 'bg-red-400'}`}></span>
                                {check.passed ? 'Passed' : 'Failed'}
                            </span>
                        </div>
                    ))}
                    {(!compliance?.all_checks || compliance.all_checks.length === 0) && (
                        <p className="text-gray-500 col-span-2 text-center py-4">No compliance check data available</p>
                    )}
                </div>

                {/* Failed checks remediation */}
                {compliance?.failed_checks && compliance.failed_checks.length > 0 && (
                    <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                        <h3 className="text-sm font-medium text-red-400 mb-2">Required Remediation</h3>
                        <ul className="space-y-1 text-sm text-gray-300">
                            {compliance.failed_checks.map((check, i) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="text-red-400">&#x2022;</span>
                                    <span><strong>{check.requirement}:</strong> {check.remediation}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    )
}
