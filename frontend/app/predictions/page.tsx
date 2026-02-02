'use client'

import { useState } from 'react'
import { api } from '@/lib/api'

interface ExplanationData {
    top_risk_factors: Array<{ feature: string; value: number; contribution: number }>
    protective_factors: Array<{ feature: string; value: number; contribution: number }>
    clinical_rationale: string
    recommendations: string[]
}

interface PredictionResult {
    prediction_id: string
    risk_score: number
    risk_level: string
    confidence: string
    model_version: string
    timestamp: string
    explanation?: ExplanationData
}

export default function PredictionsPage() {
    const [formData, setFormData] = useState({
        age: 55,
        systolic_bp: 130,
        diastolic_bp: 85,
        heart_rate: 78,
        bmi: 27,
        diabetes: 0,
        hypertension: 1,
        heart_disease: 0,
        creatinine: 1.1,
        hemoglobin: 13.5,
        glucose: 105,
        previous_admissions_30d: 0,
        oxygen_saturation: 97,
        respiratory_rate: 16,
        temperature: 98.6,
        pain_score: 0,
        gender_male: 1,
        gender_female: 0
    })

    const [result, setResult] = useState<PredictionResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError(null)

        try {
            // Call real backend API for prediction
            const prediction = await api.makePrediction(formData)

            // Fetch SHAP explanation for the prediction
            let explanation: ExplanationData | undefined
            try {
                const explainData = await api.getExplanation(prediction.prediction_id)
                explanation = {
                    top_risk_factors: explainData.top_risk_factors || [],
                    protective_factors: explainData.protective_factors || [],
                    clinical_rationale: explainData.clinical_rationale || '',
                    recommendations: explainData.recommendations || []
                }
            } catch (explainErr) {
                console.warn('Failed to fetch explanation:', explainErr)
            }

            setResult({
                prediction_id: prediction.prediction_id,
                risk_score: prediction.risk_score,
                risk_level: prediction.risk_level,
                confidence: prediction.confidence,
                model_version: prediction.model_version,
                timestamp: prediction.timestamp,
                explanation
            })
        } catch (err: any) {
            setError(err.message || 'Failed to generate prediction. Ensure backend is running.')
            console.error('Prediction error:', err)
        } finally {
            setLoading(false)
        }
    }

    const inputFields = [
        { name: 'age', label: 'Age', type: 'number', min: 18, max: 120 },
        { name: 'systolic_bp', label: 'Systolic BP (mmHg)', type: 'number', min: 80, max: 250 },
        { name: 'diastolic_bp', label: 'Diastolic BP (mmHg)', type: 'number', min: 40, max: 150 },
        { name: 'heart_rate', label: 'Heart Rate (bpm)', type: 'number', min: 30, max: 200 },
        { name: 'oxygen_saturation', label: 'O2 Saturation (%)', type: 'number', min: 70, max: 100 },
        { name: 'bmi', label: 'BMI', type: 'number', step: 0.1, min: 10, max: 60 },
        { name: 'creatinine', label: 'Creatinine (mg/dL)', type: 'number', step: 0.1, min: 0.1, max: 15 },
        { name: 'hemoglobin', label: 'Hemoglobin (g/dL)', type: 'number', step: 0.1, min: 5, max: 20 },
        { name: 'glucose', label: 'Glucose (mg/dL)', type: 'number', min: 30, max: 600 },
        { name: 'previous_admissions_30d', label: 'Admissions (30d)', type: 'number', min: 0, max: 10 },
    ]

    const checkboxFields = [
        { name: 'diabetes', label: 'Diabetes' },
        { name: 'hypertension', label: 'Hypertension' },
        { name: 'heart_disease', label: 'Heart Disease' },
    ]

    const genderOptions = [
        { label: 'Male', male: 1, female: 0 },
        { label: 'Female', male: 0, female: 1 },
    ]

    return (
        <div className="space-y-8">
            <header>
                <h1 className="text-3xl font-bold gradient-text">Patient Risk Prediction</h1>
                <p className="text-gray-400 mt-2">Enter patient data to generate risk assessment with SHAP explanations</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Input Form */}
                <div className="glass-card p-6">
                    <h2 className="text-xl font-semibold mb-6">Patient Data</h2>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid grid-cols-2 gap-4">
                            {inputFields.map((field) => (
                                <div key={field.name}>
                                    <label className="block text-sm text-gray-400 mb-1">{field.label}</label>
                                    <input
                                        type={field.type}
                                        step={field.step || 1}
                                        min={field.min}
                                        max={field.max}
                                        value={formData[field.name as keyof typeof formData]}
                                        onChange={(e) => setFormData({ ...formData, [field.name]: parseFloat(e.target.value) || 0 })}
                                        className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:border-blue-500"
                                    />
                                </div>
                            ))}
                        </div>

                        {/* Gender Selection */}
                        <div>
                            <label className="block text-sm text-gray-400 mb-2">Gender</label>
                            <div className="flex gap-4">
                                {genderOptions.map((option) => (
                                    <label key={option.label} className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="radio"
                                            name="gender"
                                            checked={formData.gender_male === option.male}
                                            onChange={() => setFormData({
                                                ...formData,
                                                gender_male: option.male,
                                                gender_female: option.female
                                            })}
                                            className="w-4 h-4"
                                        />
                                        <span className="text-sm text-gray-300">{option.label}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        <div className="flex gap-6">
                            {checkboxFields.map((field) => (
                                <label key={field.name} className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={formData[field.name as keyof typeof formData] === 1}
                                        onChange={(e) => setFormData({ ...formData, [field.name]: e.target.checked ? 1 : 0 })}
                                        className="w-4 h-4 rounded bg-white/10 border-white/20"
                                    />
                                    <span className="text-sm text-gray-300">{field.label}</span>
                                </label>
                            ))}
                        </div>

                        {error && (
                            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
                                {error}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
                        >
                            {loading ? 'Analyzing with PyTorch Model...' : 'Generate Prediction'}
                        </button>
                    </form>
                </div>

                {/* Results */}
                <div className="glass-card p-6">
                    <h2 className="text-xl font-semibold mb-6">Prediction Result</h2>

                    {!result ? (
                        <div className="text-center py-12 text-gray-500">
                            <p className="text-4xl mb-4">&#x1F52E;</p>
                            <p>Enter patient data and click Generate Prediction</p>
                            <p className="text-xs mt-2 text-gray-600">Powered by PyTorch neural network with SHAP explainability</p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {/* Risk Score */}
                            <div className={`p-6 rounded-xl text-center ${result.risk_level === 'HIGH' ? 'risk-high' :
                                    result.risk_level === 'MODERATE' ? 'risk-moderate' : 'risk-low'
                                }`}>
                                <p className="text-sm opacity-80">Risk Score</p>
                                <p className="text-5xl font-bold my-2">{(result.risk_score * 100).toFixed(1)}%</p>
                                <p className="text-lg font-medium">{result.risk_level} RISK</p>
                                <p className="text-xs opacity-70 mt-1">Confidence: {result.confidence} | Model: {result.model_version}</p>
                            </div>

                            {/* SHAP Explanation */}
                            {result.explanation && (
                                <>
                                    <div>
                                        <h3 className="text-sm font-medium text-gray-400 mb-3">
                                            Risk-Increasing Factors (SHAP)
                                        </h3>
                                        <div className="space-y-2">
                                            {result.explanation.top_risk_factors.length === 0 ? (
                                                <p className="text-gray-500 text-sm">No significant risk-increasing factors</p>
                                            ) : result.explanation.top_risk_factors.map((f, i) => (
                                                <div key={i} className="flex justify-between items-center p-2 bg-red-500/10 rounded">
                                                    <span className="text-gray-300 text-sm">{f.feature}: {f.value.toFixed(2)}</span>
                                                    <span className="text-red-400 text-sm font-medium">+{(f.contribution * 100).toFixed(1)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div>
                                        <h3 className="text-sm font-medium text-gray-400 mb-3">
                                            Protective Factors (SHAP)
                                        </h3>
                                        <div className="space-y-2">
                                            {result.explanation.protective_factors.length === 0 ? (
                                                <p className="text-gray-500 text-sm">No significant protective factors</p>
                                            ) : result.explanation.protective_factors.map((f, i) => (
                                                <div key={i} className="flex justify-between items-center p-2 bg-green-500/10 rounded">
                                                    <span className="text-gray-300 text-sm">{f.feature}: {f.value.toFixed(2)}</span>
                                                    <span className="text-green-400 text-sm font-medium">{(f.contribution * 100).toFixed(1)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="p-4 bg-white/5 rounded-lg">
                                        <h3 className="text-sm font-medium text-gray-400 mb-2">Clinical Rationale</h3>
                                        <p className="text-gray-300 text-sm whitespace-pre-line">{result.explanation.clinical_rationale}</p>
                                    </div>

                                    {result.explanation.recommendations && result.explanation.recommendations.length > 0 && (
                                        <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                                            <h3 className="text-sm font-medium text-blue-400 mb-2">Clinical Recommendations</h3>
                                            <ul className="space-y-1">
                                                {result.explanation.recommendations.map((rec, i) => (
                                                    <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                                                        <span className="text-blue-400">&#x2022;</span>
                                                        {rec}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </>
                            )}

                            <div className="text-center space-y-1">
                                <p className="text-xs text-gray-500">
                                    Prediction ID: {result.prediction_id}
                                </p>
                                <p className="text-xs text-gray-600">
                                    Logged to immutable audit trail | {new Date(result.timestamp).toLocaleString()}
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
