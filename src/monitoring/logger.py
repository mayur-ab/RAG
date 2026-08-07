from typing import Dict, Any, List

from config.logging_config import logger


class QueryTelemetryLogger:
    """Logs detailed execution trace for user queries."""

    @staticmethod
    def log_query_execution(
        query: str,
        retrieved_count: int,
        reranked_count: int,
        latency_breakdown: Dict[str, float],
        token_usage: Dict[str, Any],
        model: str,
        user_roles: List[str],
    ):
        payload = {
            "query": query,
            "retrieved_count": retrieved_count,
            "reranked_count": reranked_count,
            "latency_breakdown": latency_breakdown,
            "token_usage": token_usage,
            "model": model,
            "user_roles": user_roles,
        }
        logger.info(f"Query executed: '{query[:50]}...'", extra={"payload": payload})
