import time
from typing import Dict, Any


class MetricsCollector:
    """In-memory telemetry and performance metrics collector."""

    def __init__(self):
        self.total_queries = 0
        self.total_ingestions = 0
        self.total_errors = 0
        self.total_tokens_used = 0
        self.latency_records = []

    def record_query(self, latency_seconds: float, token_count: int, success: bool = True):
        self.total_queries += 1
        self.total_tokens_used += token_count
        self.latency_records.append(latency_seconds)
        if not success:
            self.total_errors += 1

    def record_ingestion(self, document_count: int):
        self.total_ingestions += document_count

    def get_metrics_summary(self) -> Dict[str, Any]:
        avg_latency = (sum(self.latency_records) / len(self.latency_records)) if self.latency_records else 0.0
        return {
            "total_queries": self.total_queries,
            "total_ingestions": self.total_ingestions,
            "total_errors": self.total_errors,
            "total_tokens_used": self.total_tokens_used,
            "avg_latency_seconds": round(avg_latency, 4),
            "latency_history_count": len(self.latency_records)
        }


metrics_collector = MetricsCollector()
