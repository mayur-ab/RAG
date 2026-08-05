import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from config.settings import settings


def check_ollama() -> bool:
    print("=== Ollama Health Check ===\n")
    base = settings.OLLAMA_BASE_URL.rstrip("/")

    try:
        response = httpx.get(f"{base}/api/tags", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        print(f"FAIL: Cannot reach Ollama at {base}")
        print(f"      {exc}")
        print("\nStart Ollama, then run: ollama serve")
        return False

    models = [m.get("name", "") for m in response.json().get("models", [])]
    print(f"OK: Ollama running at {base}")
    print(f"    Installed models: {', '.join(models) or '(none)'}\n")

    required = {
        "embedding": settings.OLLAMA_EMBEDDING_MODEL,
        "llm": settings.OLLAMA_LLM_MODEL,
    }
    all_ok = True

    for role, model in required.items():
        matched = any(model in name for name in models)
        status = "OK" if matched else "MISSING"
        print(f"  [{status}] {role}: {model}")
        if not matched:
            all_ok = False
            print(f"         Run: ollama pull {model}")

    if all_ok:
        print("\n=== All required Ollama models are available ===")
    else:
        print("\n=== Pull missing models before ingesting or querying ===")

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if check_ollama() else 1)
