from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.corpus_policy import CorpusDecision, evaluate_corpus_source


DROP_HEARTBEAT_KEYS = {"commands", "executed"}
MAX_HEARTBEAT_STRING = 1200
MAX_HEARTBEAT_LIST = 20


@dataclass
class OrganismState:
    processed: list[str] = field(default_factory=list)
    batches: list[dict[str, object]] = field(default_factory=list)
    cycles: int = 0
    last_heartbeat_utc: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(path: Path) -> OrganismState:
    if not path.exists():
        return OrganismState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OrganismState(
        processed=list(payload.get("processed", [])),
        batches=list(payload.get("batches", [])),
        cycles=int(payload.get("cycles", 0)),
        last_heartbeat_utc=str(payload.get("last_heartbeat_utc", "")),
    )


def save_state(path: Path, state: OrganismState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str], *, cwd: Path = ROOT, timeout: int | None = None) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def load_corpus_factory_config(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def default_workspace(config: dict[str, object]) -> Path:
    raw = str(config.get("local_workspace") or (Path.home() / "Desktop" / "F51-Corpus-Factory"))
    return Path(raw)


def safe_batch_name(prefix: str, cycle: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw = f"{prefix}_{stamp}_c{cycle:06d}"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def infer_domain(path: Path) -> str:
    text = "/".join(path.parts).lower()
    if "math" in text or "algebra" in text or "geometry" in text:
        return "mathematics"
    if "physics" in text:
        return "physics"
    if "chem" in text:
        return "chemistry"
    if "bio" in text or "medical" in text or "medicine" in text:
        return "biology"
    if "code" in text or "python" in text or "computer" in text or "cs" in text:
        return "computer_science"
    if "finance" in text or "econom" in text:
        return "austrian_economics"
    return "mathematics"


def discover_candidates(
    source_dir: Path,
    processed: set[str],
    *,
    max_files: int,
    max_bytes: int,
    min_file_bytes: int,
    default_license: str,
    require_policy_approved: bool,
    max_scan_files: int,
    policy_sample_bytes: int,
) -> list[Path]:
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".tex"}
    )
    selected: list[Path] = []
    total_bytes = 0
    source_root = source_dir.resolve()
    inspected = 0
    for path in files:
        inspected += 1
        if inspected > max_scan_files:
            break
        rel = path.resolve().relative_to(source_root).as_posix()
        if rel in processed:
            continue
        size = path.stat().st_size
        if size < min_file_bytes:
            continue
        if require_policy_approved:
            sample = path.read_bytes()[:policy_sample_bytes]
            text = sample.decode("utf-8", errors="replace")
            result = evaluate_corpus_source(
                text=text,
                source_url=f"file://{path.resolve().as_posix()}",
                license_label=default_license,
                domain_hint=infer_domain(path.relative_to(source_root)),
            )
            if result.decision != CorpusDecision.APPROVE:
                continue
        if selected and total_bytes + size > max_bytes:
            break
        selected.append(path)
        total_bytes += size
        if len(selected) >= max_files:
            break
    return selected


def stage_batch(
    *,
    source_dir: Path,
    files: list[Path],
    batch_raw_dir: Path,
    metadata_path: Path,
    default_license: str,
) -> list[str]:
    batch_raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    source_root = source_dir.resolve()
    staged_relatives: list[str] = []
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            relative = path.resolve().relative_to(source_root).as_posix()
            destination = batch_raw_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            staged_relatives.append(relative)
            record = {
                "path": relative,
                "source_url": f"file://{path.resolve().as_posix()}",
                "license": default_license,
                "domain_hint": infer_domain(path.relative_to(source_root)),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return staged_relatives


def parse_json_command(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"command did not emit JSON: {stdout[:500]}")
    return json.loads(stdout[start : end + 1])


def compact_payload(value: object, *, max_string: int = MAX_HEARTBEAT_STRING) -> object:
    if isinstance(value, dict):
        compacted: dict[str, object] = {}
        for key, item in value.items():
            if key in DROP_HEARTBEAT_KEYS:
                compacted[f"{key}_count"] = len(item) if isinstance(item, list) else 1
                continue
            compacted[key] = compact_payload(item, max_string=max_string)
        return compacted
    if isinstance(value, list):
        items = [compact_payload(item, max_string=max_string) for item in value[:MAX_HEARTBEAT_LIST]]
        if len(value) > MAX_HEARTBEAT_LIST:
            items.append({"truncated_items": len(value) - MAX_HEARTBEAT_LIST})
        return items
    if isinstance(value, str) and len(value) > max_string:
        keep = max(0, max_string - 80)
        return f"[truncated {len(value) - keep} chars]\n{value[-keep:]}"
    return value


def command_tail(stdout: str, stderr: str, *, limit: int = 1200) -> dict[str, str]:
    return {
        "stdout_tail": str(compact_payload(stdout, max_string=limit)),
        "stderr_tail": str(compact_payload(stderr, max_string=limit)),
    }


def add_command_result(
    heartbeat: dict[str, object],
    prefix: str,
    rc: int,
    stdout: str,
    stderr: str,
    *,
    limit: int = 1200,
) -> None:
    heartbeat[f"{prefix}_returncode"] = rc
    heartbeat[f"{prefix}_output"] = command_tail(stdout, stderr, limit=limit)


def remote_tokenize_busy(config: dict[str, object]) -> bool:
    target = str(config["cloud"]["master"]["ssh"])
    command = target.split() + ["pgrep -f 'tokenize_v2|corpus_cloud_tokenize|tokenize_remote' >/dev/null"]
    rc, _, _ = run_command(command, timeout=20)
    return rc == 0


def gpu_probe() -> dict[str, object]:
    code = (
        "import json, torch; "
        "ok=torch.cuda.is_available(); "
        "payload={'cuda_available': ok, 'device': torch.cuda.get_device_name(0) if ok else None, "
        "'memory_gb': round(torch.cuda.get_device_properties(0).total_memory/1e9, 2) if ok else 0}; "
        "print(json.dumps(payload, sort_keys=True))"
    )
    rc, stdout, stderr = run_command([sys.executable, "-c", code], timeout=30)
    if rc != 0:
        return {"cuda_available": False, "error": stderr.strip() or stdout.strip()}
    return json.loads(stdout.strip())


def write_heartbeat(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compacted = compact_payload(payload)
    path.write_text(json.dumps(compacted, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_pid_file(workspace: Path) -> Path:
    return workspace / "logs" / "nucleo_dataset_organism.pid"


def write_pid_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        probe = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        return probe.returncode == 0
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_json_if_exists(path: Path) -> object:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def status_report(workspace: Path, pid_file: Path) -> dict[str, object]:
    pid: int | None = None
    if pid_file.exists():
        raw_pid = pid_file.read_text(encoding="utf-8").strip()
        if raw_pid.isdigit():
            pid = int(raw_pid)
    heartbeat_path = workspace / "state" / "nucleo_dataset_organism_heartbeat.json"
    state_path = workspace / "state" / "nucleo_dataset_organism_state.json"
    state = read_json_if_exists(state_path) or {}
    heartbeat = read_json_if_exists(heartbeat_path) or {}
    batches = state.get("batches", []) if isinstance(state, dict) else []
    heartbeat_summary: dict[str, object] = {}
    command_returncodes: dict[str, object] = {}
    if isinstance(heartbeat, dict):
        for key in (
            "utc",
            "status",
            "batch",
            "deferred_batch",
            "selected_files",
            "staged_files",
            "hub_tokenizer_busy",
            "gpu",
        ):
            if key in heartbeat:
                heartbeat_summary[key] = heartbeat[key]
        for key, value in heartbeat.items():
            if key.endswith("_returncode"):
                command_returncodes[key] = value
    return {
        "ok": True,
        "workspace": str(workspace),
        "pid_file": str(pid_file),
        "pid": pid,
        "running": process_exists(pid) if pid is not None else False,
        "cycles": state.get("cycles") if isinstance(state, dict) else None,
        "processed_files": len(state.get("processed", [])) if isinstance(state, dict) else None,
        "batches": len(batches) if isinstance(batches, list) else None,
        "last_batch": batches[-1] if isinstance(batches, list) and batches else None,
        "heartbeat_path": str(heartbeat_path),
        "heartbeat": heartbeat_summary,
        "command_returncodes": command_returncodes,
    }


def run_remote_token_pipeline(
    *,
    batch_name: str,
    args: argparse.Namespace,
    heartbeat: dict[str, object],
) -> bool:
    del batch_name, args
    heartbeat["status"] = "retired_cloud_token_pipeline"
    heartbeat["detail"] = "remote token executables are archived and non-operational"
    return False


def run_local_token_pipeline(
    *,
    batch_name: str,
    args: argparse.Namespace,
    heartbeat: dict[str, object],
) -> bool:
    del batch_name, args
    heartbeat["status"] = "retired_cloud_token_pipeline"
    heartbeat["detail"] = "hub token executables are archived and non-operational"
    return False


def run_best_token_pipeline(
    *,
    batch_name: str,
    args: argparse.Namespace,
    config: dict[str, object],
    heartbeat: dict[str, object],
) -> bool:
    busy = remote_tokenize_busy(config)
    heartbeat["hub_tokenizer_busy"] = busy
    if busy and args.local_tokenize_when_hub_busy:
        return run_local_token_pipeline(batch_name=batch_name, args=args, heartbeat=heartbeat)
    if busy:
        heartbeat["status"] = "ingested_deferred_tokenize_hub_busy"
        return True
    return run_remote_token_pipeline(batch_name=batch_name, args=args, heartbeat=heartbeat)


def one_cycle(args: argparse.Namespace, config: dict[str, object], state: OrganismState) -> dict[str, object]:
    workspace = Path(args.workspace) if args.workspace else default_workspace(config)
    source_dir = Path(args.source_dir).resolve()
    state_path = workspace / "state" / "nucleo_dataset_organism_state.json"
    heartbeat_path = workspace / "state" / "nucleo_dataset_organism_heartbeat.json"
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    batch_name = safe_batch_name(args.batch_prefix, state.cycles + 1)
    batch_dir = workspace / "batches" / batch_name
    raw_dir = batch_dir / "raw"
    pack_dir = batch_dir / "pack"
    metadata_path = batch_dir / "metadata.jsonl"

    processed = set(state.processed)
    files = discover_candidates(
        source_dir,
        processed,
        max_files=args.max_files_per_cycle,
        max_bytes=args.max_bytes_per_cycle,
        min_file_bytes=args.min_file_bytes,
        default_license=args.default_license,
        require_policy_approved=args.require_policy_approved,
        max_scan_files=args.max_scan_files,
        policy_sample_bytes=args.policy_sample_bytes,
    )

    heartbeat = {
        "utc": utc_now(),
        "cycle": state.cycles + 1,
        "batch": batch_name,
        "apply": args.apply,
        "source_dir": str(source_dir),
        "workspace": str(workspace),
        "gpu": gpu_probe(),
        "selected_files": len(files),
    }

    if args.apply:
        for batch in state.batches:
            if batch.get("status") == "ingested_deferred_tokenize_hub_busy":
                heartbeat["batch"] = batch["batch"]
                heartbeat["selected_files"] = 0
                heartbeat["deferred_batch"] = batch["batch"]
                if run_best_token_pipeline(
                    batch_name=str(batch["batch"]),
                    args=args,
                    config=config,
                    heartbeat=heartbeat,
                ):
                    batch["status"] = heartbeat["status"]
                    batch["tokenized_utc"] = heartbeat["utc"]
                state.cycles += 1
                state.last_heartbeat_utc = str(heartbeat["utc"])
                save_state(state_path, state)
                write_heartbeat(heartbeat_path, heartbeat)
                return heartbeat

    if not files:
        heartbeat["status"] = "idle_no_new_files"
        state.cycles += 1
        state.last_heartbeat_utc = str(heartbeat["utc"])
        save_state(state_path, state)
        write_heartbeat(heartbeat_path, heartbeat)
        return heartbeat

    staged_relatives = stage_batch(
        source_dir=source_dir,
        files=files,
        batch_raw_dir=raw_dir,
        metadata_path=metadata_path,
        default_license=args.default_license,
    )
    heartbeat["staged_files"] = len(staged_relatives)

    build_cmd = [
        sys.executable,
        "src/tools/build_corpus_pack.py",
        "--input",
        str(raw_dir),
        "--output",
        str(pack_dir),
        "--metadata",
        str(metadata_path),
        "--default-license",
        args.default_license,
        "--source-url-base",
        "file://local-f51-corpus",
    ]
    rc, stdout, stderr = run_command(build_cmd, timeout=args.command_timeout_seconds)
    add_command_result(heartbeat, "build", rc, stdout, stderr)
    if rc != 0:
        heartbeat["status"] = "build_failed"
        write_heartbeat(heartbeat_path, heartbeat)
        return heartbeat
    build_report = parse_json_command(stdout)
    heartbeat["build_report"] = build_report
    pack_path = Path(str(build_report["pack_path"]))
    if int(build_report.get("approved", 0)) <= 0:
        heartbeat["status"] = "no_approved_documents"
        write_heartbeat(heartbeat_path, heartbeat)
        return heartbeat

    verify_cmd = [sys.executable, "src/tools/verify_corpus_pack.py", str(pack_path)]
    rc, stdout, stderr = run_command(verify_cmd, timeout=args.command_timeout_seconds)
    add_command_result(heartbeat, "verify", rc, stdout, stderr)
    if rc != 0:
        heartbeat["status"] = "verify_failed"
        write_heartbeat(heartbeat_path, heartbeat)
        return heartbeat

    upload_cmd = [
        sys.executable,
        "src/scripts/upload_corpus_pack.py",
        str(pack_path),
        "--batch-name",
        batch_name,
    ]
    if not args.apply:
        upload_cmd.append("--dry-run")
    rc, stdout, stderr = run_command(upload_cmd, timeout=args.command_timeout_seconds)
    add_command_result(heartbeat, "upload", rc, stdout, stderr)
    if rc != 0:
        heartbeat["status"] = "upload_failed"
        write_heartbeat(heartbeat_path, heartbeat)
        return heartbeat
    upload_report = parse_json_command(stdout)
    heartbeat["upload_report"] = compact_payload(upload_report)

    if args.apply:
        remote_pack = str(upload_report["plan"]["remote_pack_path"])
        ingest_cmd = [
            sys.executable,
            "src/scripts/ingest_remote_corpus_pack.py",
            "--remote-pack",
            remote_pack,
            "--batch-name",
            batch_name,
        ]
        rc, stdout, stderr = run_command(ingest_cmd, timeout=args.command_timeout_seconds)
        add_command_result(heartbeat, "ingest", rc, stdout, stderr)
        if rc != 0:
            heartbeat["status"] = "ingest_failed"
            write_heartbeat(heartbeat_path, heartbeat)
            return heartbeat

        if not run_best_token_pipeline(batch_name=batch_name, args=args, config=config, heartbeat=heartbeat):
            write_heartbeat(heartbeat_path, heartbeat)
            return heartbeat
    else:
        heartbeat["status"] = "dry_run_ready"

    if args.apply:
        state.processed.extend(staged_relatives)
    state.batches.append(
        {
            "batch": batch_name,
            "utc": heartbeat["utc"],
            "status": heartbeat["status"],
            "staged_files": len(staged_relatives),
            "approved": build_report.get("approved", 0),
            "pack_path": str(pack_path),
            "apply": args.apply,
        }
    )
    state.cycles += 1
    state.last_heartbeat_utc = str(heartbeat["utc"])
    save_state(state_path, state)
    write_heartbeat(heartbeat_path, heartbeat)
    return heartbeat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F51 NUCLEO local dataset organism.")
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "corpus_factory.yaml"))
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--source-dir", default=str(ROOT / "data" / "corpus"))
    parser.add_argument("--batch-prefix", default="world_seed")
    parser.add_argument("--default-license", default="cc-by")
    parser.add_argument(
        "--require-policy-approved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preselect only files that already pass corpus policy.",
    )
    parser.add_argument("--max-files-per-cycle", type=int, default=100)
    parser.add_argument("--max-bytes-per-cycle", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--min-file-bytes", type=int, default=200)
    parser.add_argument("--max-scan-files", type=int, default=500)
    parser.add_argument("--policy-sample-bytes", type=int, default=200_000)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--command-timeout-seconds", type=int, default=900)
    parser.add_argument("--remote-tokenize-timeout-seconds", type=int, default=7200)
    parser.add_argument("--local-tokenize-timeout-seconds", type=int, default=7200)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true", help="Print compact runtime status and exit.")
    parser.add_argument("--pid-file", default=None, help="PID file path. Defaults to workspace logs.")
    parser.add_argument("--apply", action="store_true", help="Actually upload and ingest on NUCLEO.")
    parser.add_argument(
        "--local-tokenize-when-hub-busy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tokenize locally with the synced 80k tokenizer when the Hub tokenizer is busy.",
    )
    parser.add_argument(
        "--defer-tokenize-when-busy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not add corpus-factory tokenization while the Hub is already tokenizing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_corpus_factory_config(Path(args.config))
    workspace = Path(args.workspace) if args.workspace else default_workspace(config)
    state_path = workspace / "state" / "nucleo_dataset_organism_state.json"
    pid_file = Path(args.pid_file) if args.pid_file else default_pid_file(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if args.status:
        print(json.dumps(compact_payload(status_report(workspace, pid_file)), indent=2, sort_keys=True))
        return 0
    if not args.once:
        write_pid_file(pid_file)

    while True:
        state = load_state(state_path)
        heartbeat = one_cycle(args, config, state)
        print(json.dumps(compact_payload(heartbeat), indent=2, sort_keys=True), flush=True)
        if args.once:
            return 0 if str(heartbeat.get("status", "")).endswith(("ready", "indexed", "busy")) else 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
