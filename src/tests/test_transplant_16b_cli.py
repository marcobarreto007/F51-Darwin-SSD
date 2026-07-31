from __future__ import annotations

import json
import sys
from pathlib import Path

from f51_darwin.transplant_16b import cli
from f51_darwin.transplant_16b.contracts import (
    SourceFile,
    SourceIdentity,
)
from f51_darwin.transplant_16b.sources import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _fake_sources(tmp_path: Path) -> SourceIdentity:
    files = {}
    for name in (
        "config",
        "weights",
        "tokenizer",
        "tokenizer_config",
        "special",
        "organ",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        files[name] = SourceFile(
            path=str(path),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
    return SourceIdentity(
        smol_snapshot="test-snapshot",
        smol_config=files["config"],
        smol_weights=files["weights"],
        tokenizer_json=files["tokenizer"],
        tokenizer_config=files["tokenizer_config"],
        special_tokens=files["special"],
        organ_checkpoint=files["organ"],
    )


def test_plan_writes_hash_bound_atomic_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sources = _fake_sources(tmp_path)
    monkeypatch.setattr(cli, "discover_source_identity", lambda *_: sources)
    runtime = tmp_path / "runtime"
    checkpoints = tmp_path / "checkpoints"
    tokenizer = tmp_path / "tokenizer-target"

    rc = cli.main(
        [
            "plan",
            "--config",
            str(ROOT / "src/configs/darwin_x_1.6b_smol_transplant.yaml"),
            "--runtime-root",
            str(runtime),
            "--checkpoint-root",
            str(checkpoints),
            "--tokenizer-root",
            str(tokenizer),
        ]
    )

    assert rc == 0
    payload = json.loads((runtime / "plan.json").read_text("utf-8"))
    assert payload["schema"] == "darwin-smol-transplant-plan-v1"
    assert (
        payload["target"]["model_name"]
        == "F51-Darwin-X-1.6B-Smol-Transplant-V1"
    )
    assert len(payload["plan_id"]) == 64
    assert payload["layer_groups"] == [
        [0],
        [1, 2],
        [3],
        [4, 5],
        [6],
        [7, 8],
        [9],
        [10, 11],
        [12],
        [13, 14],
        [15],
        [16, 17],
        [18],
        [19, 20],
        [21],
        [22, 23],
    ]


def test_tiny_build_does_not_import_calibration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sources = _fake_sources(tmp_path)
    monkeypatch.setattr(cli, "discover_source_identity", lambda *_: sources)
    runtime = tmp_path / "runtime"
    assert (
        cli.main(
            [
                "plan",
                "--config",
                str(ROOT / "src/configs/darwin_x_1.6b_smol_transplant.yaml"),
                "--runtime-root",
                str(runtime),
                "--checkpoint-root",
                str(tmp_path / "checkpoints"),
                "--tokenizer-root",
                str(tmp_path / "tokenizer-target"),
            ]
        )
        == 0
    )
    sys.modules.pop("f51_darwin.transplant_16b.calibration", None)

    assert (
        cli.main(
            [
                "build",
                "--plan",
                str(runtime / "plan.json"),
                "--tiny-test-mode",
            ]
        )
        == 0
    )
    assert "f51_darwin.transplant_16b.calibration" not in sys.modules
    assert (runtime / "tiny-build-ok.json").is_file()
