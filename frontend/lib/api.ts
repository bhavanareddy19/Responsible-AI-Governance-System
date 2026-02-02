/** API client for frontend communication. */

const API_BASE = '/api/v1';

export interface PredictionResult {
    prediction_id: string;
    risk_score: number;
    risk_level: string;
    confidence: string;
    model_version: string;
    timestamp: string;
}

export interface BiasReport {
    generated_at: string;
    total_samples: number;
    overall_bias_detected: boolean;
    attributes_analyzed: string[];
    fairness_score: number;
    bias_summary: any[];
    recommendations: string[];
}

export interface ComplianceCheck {
    check_id: string;
    requirement: string;
    standard: string;
    passed: boolean;
    description: string;
    details: string;
    remediation: string | null;
}

export interface ComplianceStatus {
    report_id: string;
    timestamp: string;
    overall_compliant: boolean;
    compliance_percentage: number;
    risk_category: string;
    checks_passed: number;
    checks_failed: number;
    failed_checks: any[];
    all_checks: ComplianceCheck[];
}

export interface HealthStatus {
    status: string;
    version: string;
    uptime_hours: number;
    model_loaded: boolean;
    database_connected: boolean;
}

export interface Metrics {
    total_predictions: number;
    predictions_last_hour: number;
    avg_latency_ms: number;
    error_rate: number;
    high_risk_rate: number;
    model_version: string;
}

export interface AuditEvent {
    event_id: string;
    event_type: string;
    timestamp: string;
    action: string;
    model_version: string;
    details: Record<string, any>;
}

export interface PredictionStats {
    window_start: string;
    window_end: string;
    total_predictions: number;
    avg_latency_ms: number;
    p95_latency_ms: number;
    error_count: number;
    high_risk_count: number;
    low_risk_count: number;
}

class APIClient {
    private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return response.json();
    }

    // Predictions
    async makePrediction(patientData: Record<string, number>): Promise<PredictionResult> {
        return this.fetch('/predictions', {
            method: 'POST',
            body: JSON.stringify(patientData),
        });
    }

    async getExplanation(predictionId: string): Promise<any> {
        return this.fetch(`/predictions/${predictionId}/explain`);
    }

    // Governance
    async getBiasReport(): Promise<BiasReport> {
        return this.fetch('/governance/bias-report');
    }

    async getComplianceStatus(): Promise<ComplianceStatus> {
        return this.fetch('/governance/compliance-status');
    }

    async queryAuditLogs(params?: { event_type?: string; limit?: number }): Promise<AuditEvent[]> {
        return this.fetch('/governance/audit-query', {
            method: 'POST',
            body: JSON.stringify(params || { limit: 100 }),
        });
    }

    async getAuditStats(): Promise<any> {
        return this.fetch('/governance/audit-stats');
    }

    // Monitoring
    async getHealth(): Promise<HealthStatus> {
        return this.fetch('/monitoring/health');
    }

    async getMetrics(): Promise<Metrics> {
        return this.fetch('/monitoring/metrics');
    }

    async getPredictionStats(): Promise<PredictionStats> {
        return this.fetch('/monitoring/predictions/stats');
    }

    async getAlerts(): Promise<{ alerts: any[]; count: number }> {
        return this.fetch('/monitoring/alerts');
    }
}

export const api = new APIClient();
