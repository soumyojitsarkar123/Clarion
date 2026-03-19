#!/usr/bin/env python
"""
Lightweight CLI helper for uploading and analyzing documents.

Usage:
    python experiments/doc_cli.py upload document.pdf
    python experiments/doc_cli.py analyze document.pdf
    python experiments/doc_cli.py wait document_id
    python experiments/doc_cli.py batch "dir/*.pdf"
    python experiments/doc_cli.py stats

Works on Windows. Uses only standard library + requests.
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install with: pip install requests")
    sys.exit(1)

DEFAULT_API_BASE = "http://localhost:8000"


def upload_document(file_path: str, api_base: str = DEFAULT_API_BASE) -> str:
    """Upload a single document and return document_id."""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    url = f"{api_base}/upload"

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = requests.post(url, files=files, timeout=60)

        response.raise_for_status()
        data = response.json()

        doc_id = data.get("document_id")
        print(f"Uploaded: {file_path} -> {doc_id}")
        return doc_id
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to backend at {api_base}")
        print("Make sure the backend is running: python main.py")
        sys.exit(1)


def analyze_document(document_id: str, api_base: str = DEFAULT_API_BASE) -> dict:
    """Trigger analysis on a document."""
    url = f"{api_base}/analyze/{document_id}"

    try:
        response = requests.post(url, json={}, timeout=300)
        if response.status_code != 200:
            return {"error": response.text, "status_code": response.status_code}
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to backend at {api_base}")
        print("Make sure the backend is running: python main.py")
        sys.exit(1)


def get_status(document_id: str, api_base: str = DEFAULT_API_BASE) -> dict:
    """Get document processing status."""
    url = f"{api_base}/status/{document_id}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return response.json()


def wait_for_completion(
    document_id: str, api_base: str = DEFAULT_API_BASE, poll_interval: int = 5
) -> dict:
    """Poll status until processing completes."""
    print(f"Waiting for {document_id} to complete...", end=" ", flush=True)

    while True:
        status = get_status(document_id, api_base)
        state = status.get("state", "unknown")

        if state in ("completed", "failed", "error"):
            print(f"Done! State: {state}")
            return status

        print(".", end="", flush=True)
        time.sleep(poll_interval)

    return status


def upload_and_analyze(
    file_path: str, api_base: str = DEFAULT_API_BASE, wait: bool = False
) -> str:
    """Upload document, trigger analysis, optionally wait for completion."""
    doc_id = upload_document(file_path, api_base)

    print(f"Triggering analysis...")
    result = analyze_document(doc_id, api_base)

    if "error" in result:
        if "no text content" in result["error"].lower():
            print(f"[!] Warning: Document has no extractable text (may be scanned PDF)")
            print(
                f"    Document uploaded. Check status: python doc_cli.py status {doc_id}"
            )
        else:
            print(f"[!] Error: {result['error']}")
    else:
        if wait:
            status = wait_for_completion(doc_id, api_base)
            print(f"Final status: {status.get('state')}")

    return doc_id


def batch_upload(
    file_paths: list, api_base: str = DEFAULT_API_BASE, wait: bool = False
) -> list:
    """Upload multiple documents."""
    doc_ids = []

    for fp in file_paths:
        doc_id = upload_and_analyze(fp, api_base, wait=wait)
        doc_ids.append(doc_id)

    return doc_ids


def check_dataset_stats(api_base: str = DEFAULT_API_BASE) -> dict:
    """Check relation dataset statistics."""
    url = f"{api_base}/dataset/relations/stats"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to backend at {api_base}")
        print("Make sure the backend is running: python main.py")
        sys.exit(1)


def print_stats(stats: dict) -> None:
    """Pretty print dataset statistics."""
    print("\n=== Relation Dataset Statistics ===")
    print(f"Total records:       {stats['total_records']}")
    print(f"Labeled records:     {stats['labeled_records']}")
    print(f"Unlabeled records:   {stats['unlabeled_records']}")
    print(f"Avg LLM confidence:  {stats.get('average_llm_confidence', 'N/A')}")
    print(f"Avg co-occurrence:   {stats.get('average_cooccurrence', 'N/A')}")

    if "relation_types" in stats:
        print("\nRelation types:")
        for rel_type, count in stats["relation_types"].items():
            print(f"  - {rel_type}: {count}")


def task_upload(args) -> int:
    """Upload a single document."""
    doc_id = upload_and_analyze(args.file, args.api_base, wait=args.wait)
    print(f"\nDocument ID: {doc_id}")
    return 0


def task_analyze(args) -> int:
    """Analyze existing document by ID."""
    result = analyze_document(args.document_id, args.api_base)
    print(f"Analysis triggered for: {args.document_id}")
    print(f"Result: {result}")
    return 0


def task_wait(args) -> int:
    """Wait for document to complete processing."""
    status = wait_for_completion(args.document_id, args.api_base, args.interval)
    print(f"\nFinal status: {status}")
    return 0


def task_status(args) -> int:
    """Get document status."""
    status = get_status(args.document_id, args.api_base)
    print(f"Document: {args.document_id}")
    print(f"State: {status.get('state')}")
    print(f"Status: {status.get('status')}")
    return 0


def task_batch(args) -> int:
    """Batch upload multiple files."""
    import fnmatch

    if "*" in args.pattern or "?" in args.pattern:
        files = glob.glob(args.pattern)
    else:
        files = args.pattern.split(",")

    files = [f.strip() for f in files if f.strip()]

    if not files:
        print("ERROR: No files found")
        return 1

    print(f"Found {len(files)} files to upload")

    doc_ids = batch_upload(files, args.api_base, wait=args.wait)

    print(f"\nUploaded {len(doc_ids)} documents:")
    for fid in doc_ids:
        print(f"  - {fid}")

    return 0


def task_stats(args) -> int:
    """Show dataset statistics."""
    stats = check_dataset_stats(args.api_base)
    print_stats(stats)
    return 0


def task_check(args) -> int:
    """Upload and check dataset growth."""
    if not args.file:
        print("ERROR: --file required for check task")
        return 1

    before = check_dataset_stats(args.api_base)
    before_total = before["total_records"]

    print(f"Before: {before_total} records")

    doc_id = upload_and_analyze(args.file, args.api_base, wait=args.wait)

    after = check_dataset_stats(args.api_base)
    after_total = after["total_records"]

    print(f"\nAfter: {after_total} records")
    print(f"Growth: +{after_total - before_total} records")
    print(f"Document ID: {doc_id}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Document upload and analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Backend API base URL (default: {DEFAULT_API_BASE})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # upload command
    upload_parser = subparsers.add_parser(
        "upload", help="Upload and analyze a document"
    )
    upload_parser.add_argument("file", help="Document file to upload")
    upload_parser.add_argument(
        "--wait", "-w", action="store_true", help="Wait for processing to complete"
    )

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze existing document by ID"
    )
    analyze_parser.add_argument("document_id", help="Document ID to analyze")

    # wait command
    wait_parser = subparsers.add_parser("wait", help="Wait for document to complete")
    wait_parser.add_argument("document_id", help="Document ID to wait for")
    wait_parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=5,
        help="Poll interval in seconds (default: 5)",
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Get document status")
    status_parser.add_argument("document_id", help="Document ID to check")

    # batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Batch upload multiple documents"
    )
    batch_parser.add_argument(
        "pattern", help="File pattern (e.g., 'docs/*.pdf' or 'a.pdf,b.pdf')"
    )
    batch_parser.add_argument(
        "--wait", "-w", action="store_true", help="Wait for each to complete"
    )

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")

    # check command
    check_parser = subparsers.add_parser(
        "check", help="Upload and check dataset growth"
    )
    check_parser.add_argument("file", help="Document file to upload")
    check_parser.add_argument(
        "--wait", "-w", action="store_true", help="Wait for processing to complete"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    tasks = {
        "upload": task_upload,
        "analyze": task_analyze,
        "wait": task_wait,
        "status": task_status,
        "batch": task_batch,
        "stats": task_stats,
        "check": task_check,
    }

    return tasks[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
