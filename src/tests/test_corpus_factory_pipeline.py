import json
import tarfile
from pathlib import Path

from f51_darwin.corpus_factory_pipeline import build_corpus_pack, normalize_training_text
from f51_darwin.corpus_pack_verifier import verify_corpus_pack


def test_normalize_training_text_removes_noise() -> None:
    raw = "  theorem proof  \r\n\r\n\r\nalgebra\x00  "
    assert normalize_training_text(raw) == "theorem proof\n\nalgebra\n"


def test_build_corpus_pack_exports_only_approved_items(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    (input_dir / "math.txt").write_text(
        (
            "This theorem has a proof using algebra, calculus, statistics, "
            "probability, and a differential equation. "
        )
        * 8,
        encoding="utf-8",
    )
    (input_dir / "activism.txt").write_text(
        (
            "Activists demand that everyone must dismantle the system and join "
            "the movement. Organizers mobilize pressure and praxis. "
        )
        * 8,
        encoding="utf-8",
    )
    (input_dir / "nc_science.txt").write_text(
        (
            "The clinical randomized trial reports diagnosis, biomarkers, "
            "epidemiology, pathophysiology, and statistics. "
        )
        * 8,
        encoding="utf-8",
    )

    metadata = input_dir / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "path": "math.txt",
                        "source_url": "https://www.gutenberg.org/example",
                        "license": "PD-US",
                    }
                ),
                json.dumps(
                    {
                        "path": "activism.txt",
                        "source_url": "https://example.org/editorial",
                        "license": "CC-BY",
                    }
                ),
                json.dumps(
                    {
                        "path": "nc_science.txt",
                        "source_url": "https://example.edu/course",
                        "license": "CC-BY-NC",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_corpus_pack(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_path=metadata,
    )

    assert summary.scanned == 3
    assert summary.approved == 1
    assert summary.quarantine == 2
    assert summary.rejected == 0

    manifest = [
        json.loads(line)
        for line in Path(summary.manifest_path).read_text(encoding="utf-8").splitlines()
    ]
    by_path = {record["relative_path"]: record for record in manifest}
    assert by_path["math.txt"]["decision"] == "approve"
    assert by_path["math.txt"]["approved_path"].startswith("approved/")
    assert by_path["activism.txt"]["decision"] == "quarantine"
    assert by_path["nc_science.txt"]["decision"] == "quarantine"
    assert "metadata.jsonl" not in by_path

    approved_files = list((output_dir / "approved").glob("*.txt"))
    assert len(approved_files) == 1

    assert summary.pack_path is not None
    with tarfile.open(summary.pack_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "manifest.jsonl" in names
    assert any(name.startswith("approved/math_") for name in names)
    assert not any("activism" in name for name in names)

    report = verify_corpus_pack(Path(summary.pack_path))
    assert report.ok
    assert report.manifest_records == 3
    assert report.approved_records == 1
    assert report.packed_approved_files == 1


def test_build_corpus_pack_rejects_duplicate_normalized_hash(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    text = (
        "This theorem has a proof using algebra, calculus, statistics, "
        "probability, and a differential equation. "
    ) * 8
    (input_dir / "first.txt").write_text(text, encoding="utf-8")
    (input_dir / "second.txt").write_text("\n\n" + text + "  \n", encoding="utf-8")

    summary = build_corpus_pack(
        input_dir=input_dir,
        output_dir=output_dir,
        default_license="PD-US",
        source_url_base="https://www.gutenberg.org",
    )

    assert summary.scanned == 2
    assert summary.approved == 1
    assert summary.rejected == 1
    manifest = [
        json.loads(line)
        for line in Path(summary.manifest_path).read_text(encoding="utf-8").splitlines()
    ]
    rejected = [record for record in manifest if record["decision"] == "reject"]
    assert rejected[0]["reason"] == "duplicate normalized hash inside batch"


def test_verify_corpus_pack_detects_tampered_approved_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    tampered = tmp_path / "tampered.tar.gz"
    input_dir.mkdir()
    (input_dir / "math.txt").write_text(
        (
            "This theorem has a proof using algebra, calculus, statistics, "
            "probability, and a differential equation. "
        )
        * 8,
        encoding="utf-8",
    )
    summary = build_corpus_pack(
        input_dir=input_dir,
        output_dir=output_dir,
        default_license="PD-US",
        source_url_base="https://www.gutenberg.org",
    )
    assert summary.pack_path is not None

    with tarfile.open(summary.pack_path, "r:gz") as source, tarfile.open(tampered, "w:gz") as target:
        for member in source.getmembers():
            extracted = source.extractfile(member) if member.isfile() else None
            if member.name.startswith("approved/") and extracted is not None:
                payload = b"tampered text\n"
                info = tarfile.TarInfo(member.name)
                info.size = len(payload)
                target.addfile(info, fileobj=__import__("io").BytesIO(payload))
            else:
                target.addfile(member, fileobj=extracted)

    report = verify_corpus_pack(tampered)
    assert not report.ok
    assert any("sha256 mismatch" in error for error in report.errors)
