"""CLI entry point for Local Vuln Research System."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="vulnresearch",
        description="Local vulnerability research pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="One-time setup: download model instructions, init DB")
    sub.add_parser("benchmark", help="Run model benchmark to determine optimal config")
    sub.add_parser("eval", help="Run evaluation harness against known-vuln corpora")
    sub.add_parser("update-cve", help="Refresh CVE/OSV/GHSA/EPSS/KEV knowledge base")

    audit = sub.add_parser("audit", help="Audit a target repository")
    audit.add_argument("repo_path", help="Path to target repository")
    audit.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    estimate = sub.add_parser("estimate", help="Estimate resources needed for a target")
    estimate.add_argument("repo_path", help="Path to target repository")

    args = parser.parse_args()

    if args.command == "setup":
        _cmd_setup()
    elif args.command == "benchmark":
        _cmd_benchmark()
    elif args.command == "eval":
        _cmd_eval()
    elif args.command == "update-cve":
        _cmd_update_cve()
    elif args.command == "audit":
        _cmd_audit(args.repo_path, args.resume)
    elif args.command == "estimate":
        _cmd_estimate(args.repo_path)


def _cmd_setup():
    from src.config import config

    model_name = config.get("model.name", "the code model")
    model_file = config.get("model.file", "models/model.gguf")
    download_url = config.get("model.download_url", "https://huggingface.co/bartowski")

    print("=== Local Vuln Research System Setup ===\n")
    print("[1] Model download:")
    print(f"    Download the GGUF model ({model_name}) from HuggingFace:")
    print(f"    {download_url}")
    print(f"    Place it at: {model_file}")
    print()
    print("    See config.yaml (model.file / model.draft_file) to switch models.")
    print()
    print("[2] Install Python dependencies:")
    print("    pip install -r requirements.txt")
    print()
    print("[3] Start llama-server:")
    print("    python start_server.py")
    print()
    print("[4] Run benchmark (optimize config):")
    print("    python run_benchmark.py")
    print()
    print("[5] Download CVE database:")
    print("    python -m src.main update-cve")
    print()
    print("[6] Run evaluation (calibrate thresholds):")
    print("    python -m src.main eval")
    print()


def _cmd_benchmark():
    from src.benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    runner.run()


def _cmd_eval():
    from src.eval.harness import EvalHarness
    harness = EvalHarness()
    harness.run()


def _cmd_update_cve():
    from src.knowledge.downloader import CVEDownloader
    from src.knowledge.importer import CVEImporter

    print("=== CVE Database Update ===\n")

    print("[1/3] Downloading from NVD, EPSS, KEV...")
    downloader = CVEDownloader()
    downloader.download_all()

    print("\n[2/3] Importing into SQLite database...")
    importer = CVEImporter()
    importer.import_all()

    print("\n[3/3] Generating embeddings for semantic search...")
    try:
        from src.knowledge.embeddings import EmbeddingIndex
        embedder = EmbeddingIndex()
        embedder.generate_all()
    except Exception as e:
        print(f"    [!] Embedding generation skipped: {e}")
        print(f"    [!] Fix with: pip install tf-keras sentence-transformers")
        print(f"    [!] FTS5 keyword search works without embeddings.")

    print("\n[+] CVE database update complete.")
    from src.knowledge.cve_db import CVEDatabase
    db = CVEDatabase()
    stats = db.stats()
    print(f"    Total CVEs: {stats['total_cves']:,}")
    print(f"    CISA KEV:   {stats['kev_members']:,}")
    print(f"    With EPSS:  {stats['with_epss']:,}")
    print(f"    Critical:   {stats['critical']:,}")
    print(f"    High:       {stats['high']:,}")
    db.close()


def _cmd_audit(repo_path: str, resume: bool):
    from pathlib import Path
    from src.orchestrator import Orchestrator

    target = Path(repo_path).resolve()
    if not target.exists():
        print(f"[!] Repository not found: {target}")
        sys.exit(1)

    orch = Orchestrator(target, resume=resume)
    code = orch.run()

    if code == 0:
        checkpoint_dir = orch.checkpoint_dir
        if checkpoint_dir:
            report = checkpoint_dir / "report.md"
            if report.exists():
                print(f"\n[+] Report: {report}")
            print(f"\n[+] Checkpoint data: {checkpoint_dir}")
    sys.exit(code)


def _cmd_estimate(repo_path: str):
    from pathlib import Path
    from src.analysis.scaling import LargeCodebaseAdapter, ChunkedFileProcessor

    target = Path(repo_path).resolve()
    if not target.exists():
        print(f"[!] Repository not found: {target}")
        sys.exit(1)

    print(f"\n=== Scan Estimate: {target} ===\n")

    adapter = LargeCodebaseAdapter()
    processor = ChunkedFileProcessor(adapter.config)
    files = processor.list_files(target)

    total_size = sum(
        f.stat().st_size for f in files
    ) if files else 0
    size_mb = total_size / (1024 * 1024) if files else 0.0
    file_count = len(files)

    config = adapter.get_adaptive_config(size_mb, file_count)
    resources = adapter.estimate_resources(file_count, config)

    print(f"Files: {file_count}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Recommended config: {resources['recommended_config']}")
    print(f"\nEstimated analysis scope:")
    print(f"  Functions: ~{resources['estimated_functions']:,}")
    print(f"  Sources: ~{resources['estimated_sources']:,}")
    print(f"  Sinks: ~{resources['estimated_sinks']:,}")
    print(f"  Paths: ~{resources['estimated_paths']:,}")
    print(f"  LLM analysis: ~{resources['estimated_llm_minutes']:.0f} minutes")
    print(f"\nConfig:")
    print(f"  Max LLM paths: {'unlimited' if config.max_llm_paths == 0 else config.max_llm_paths}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Files per chunk: {config.max_files_per_chunk}")


if __name__ == "__main__":
    main()
