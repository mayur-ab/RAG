"""Bulk-ingest every supported file in Docs/ and Uploaded-Docs/ (recursive)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from collections import Counter
from pathlib import Path
from config.settings import settings
from src.service import RAGPipelineService

SUPPORTED = {".txt", ".md", ".pdf", ".doc", ".docx", ".html", ".htm", ".csv", ".xlsx", ".xls"}


def collect_files(*folders: str) -> list[Path]:
    seen: set[str] = set()
    files: list[Path] = []
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def scan_all_extensions(*folders: str) -> Counter:
    counts: Counter = Counter()
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                counts[path.suffix.lower() or "(no ext)"] += 1
    return counts


def chunking_for(path: Path) -> str:
    if path.suffix.lower() in {".pdf", ".doc", ".csv", ".xlsx", ".xls"}:
        return "recursive"
    return settings.CHUNKING_STRATEGY


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest Docs/ and Uploaded-Docs/")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest all files even if already indexed (replaces existing chunks)",
    )
    args = parser.parse_args()

    all_ext = scan_all_extensions(settings.DOCS_DIR, settings.UPLOADED_DOCS_DIR)
    docs = collect_files(settings.DOCS_DIR, settings.UPLOADED_DOCS_DIR)

    print("=== Folder scan ===")
    for ext, count in all_ext.most_common():
        status = "supported" if ext in SUPPORTED else "SKIPPED"
        print(f"  {ext:8} {count:4}  [{status}]")
    print(f"  {'—'*22}")
    print(f"  Total on disk: {sum(all_ext.values())}  |  To ingest: {len(docs)}\n")

    if not docs:
        print(f"No supported files found under {settings.DOCS_DIR} or {settings.UPLOADED_DOCS_DIR}")
        sys.exit(1)

    print(f"=== Bulk ingest: {len(docs)} file(s) ===")
    print(f"    Embeddings: {settings.EMBEDDING_PROVIDER}")
    print(f"    Default chunking: {settings.CHUNKING_STRATEGY} (.pdf/.doc use recursive)")
    if args.force:
        print("    Mode: FORCE (re-index everything)\n")
    else:
        print("    Mode: RESUME (skip unchanged files — safe to stop and restart)\n")

    service = RAGPipelineService()
    ok, skipped, failed = 0, 0, 0
    failed_files: list[str] = []
    t0 = time.time()

    for i, path in enumerate(docs, 1):
        source = str(path).replace("\\", "/")
        strategy = chunking_for(path)
        print(f"[{i}/{len(docs)}] {path.name} ({strategy})...", end=" ", flush=True)
        try:
            res = service.ingest_source(
                source=source,
                chunking_strategy=strategy,
                skip_if_unchanged=not args.force,
                force=args.force,
            )
            if res.get("skipped"):
                print(f"SKIP — {res['total_chunks']} chunks already indexed")
                skipped += 1
            else:
                print(f"OK — {res['total_chunks']} chunks")
                ok += 1
        except Exception as exc:
            print(f"FAILED — {exc}")
            failed += 1
            failed_files.append(path.name)

    elapsed = round(time.time() - t0, 1)
    stats = service.vector_store.get_stats()
    print(f"\n=== Done in {elapsed}s ===")
    print(f"Ingested: {ok}  Skipped: {skipped}  Failed: {failed}")
    print(f"Total vectors in Chroma: {stats.get('total_vectors', 0)}")
    print(f"BM25 chunks indexed: {len(service._all_chunk_documents)}")

    skipped = sum(c for ext, c in all_ext.items() if ext not in SUPPORTED)
    if skipped:
        print(f"\nNote: {skipped} file(s) skipped (unsupported types like .xls).")

    if failed_files:
        print("\nFailed files:")
        for name in failed_files[:20]:
            print(f"  - {name}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
