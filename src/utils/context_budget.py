from typing import Any, Dict, Optional

from config.logging_config import logger
from config.settings import settings

CONTEXT_WARN_THRESHOLD = 0.80
CONTEXT_INFO_THRESHOLD = 0.60


def get_context_window() -> int:
    if settings.LLM_PROVIDER == "ollama":
        return settings.OLLAMA_NUM_CTX
    return settings.MAX_CONTEXT_TOKENS


def build_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    model: str = "",
    log: bool = True,
) -> Dict[str, Any]:
    """Build token_usage dict with context window stats and optional logging."""
    ctx = get_context_window()
    used_pct: Optional[float] = None
    if ctx and prompt_tokens > 0:
        used_pct = round((prompt_tokens / ctx) * 100, 1)

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "context_window": ctx,
        "context_used_percent": used_pct,
    }

    if log and prompt_tokens > 0 and ctx:
        log_context_usage(prompt_tokens, completion_tokens, model=model, context_window=ctx)

    return usage


def log_context_usage(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    model: str = "unknown",
    context_window: Optional[int] = None,
) -> None:
    ctx = context_window or get_context_window()
    if not ctx or prompt_tokens <= 0:
        return

    prompt_ratio = prompt_tokens / ctx
    total_ratio = (prompt_tokens + completion_tokens) / ctx

    if prompt_ratio >= CONTEXT_WARN_THRESHOLD or total_ratio >= 0.95:
        logger.warning(
            "Context window pressure: prompt=%s/%s (%.1f%%), total=%s/%s, model=%s. "
            "Consider reducing retrieval chunks or chat compact size.",
            prompt_tokens,
            ctx,
            prompt_ratio * 100,
            prompt_tokens + completion_tokens,
            ctx,
            model,
        )
    elif prompt_ratio >= CONTEXT_INFO_THRESHOLD:
        logger.info(
            "Context usage: prompt=%s/%s (%.1f%%), model=%s",
            prompt_tokens,
            ctx,
            prompt_ratio * 100,
            model,
        )
