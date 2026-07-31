from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


GENESIS_HEAD = "0" * 64
ALLOWED_STATUSES = frozenset(
    {"complete", "folded", "classified_nonweight"}
)

# A classe de proveniencia de um registro NAO e inferida do nome do metodo.
# Um nome e uma alegacao; "exact_copy" pode fabricar valores tanto quanto
# "projection" pode apenas realocar. A classe e derivada da EVIDENCIA que o
# registro carrega:
#
#   provenance_sha256 presente -> heranca, e o digest do mapa `kept` prova
#       de onde cada escalar veio. Deriva tem de ser zero: bytes identicos.
#   provenance_sha256 ausente  -> fabricacao, os valores nao existiam no
#       doador. O hash quebra e deve quebrar, entao a unica evidencia
#       possivel e a medida de quanto se afastou.
#
# Isso torna impossivel ganhar credito de proveniencia batizando bem o
# metodo. Credito exige o certificado.


@dataclass(frozen=True)
class DriftRecord:
    """Quanto o alvo se afastou do doador, e sob qual metrica.

    SHA responde SE o peso foi herdado; nao responde QUANTO ele mudou --
    e descontinuo, um bit muda e o digest fica descorrelacionado, entao
    nao possui estrutura metrica nenhuma. Registro de proveniencia sem
    par metrico engana em uma direcao previsivel: apos qualquer edicao
    todo hash quebra e o ledger grita "tudo mudou", sem distinguir um
    escalar reescrito de uma camada reinicializada.
    """

    metric: str
    value: float
    measured_on: str = ""

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("drift metric must be named")
        if not math.isfinite(self.value):
            raise ValueError("drift value must be finite")
        if self.value < 0.0:
            raise ValueError("drift value must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DriftRecord":
        return cls(
            metric=str(raw["metric"]),
            value=float(raw["value"]),
            measured_on=str(raw.get("measured_on") or ""),
        )


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CoverageRecord:
    source_tensor: str
    target_tensors: tuple[str, ...]
    method: str
    status: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    provenance_sha256: str | None = None
    drift: DriftRecord | None = None

    def __post_init__(self) -> None:
        if not self.source_tensor:
            raise ValueError("source_tensor must not be empty")
        if not self.method:
            raise ValueError("coverage method must not be empty")
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported coverage status: {self.status}")
        if len(set(self.target_tensors)) != len(self.target_tensors):
            raise ValueError("target_tensors contains duplicates")
        if self.status != "classified_nonweight" and not self.target_tensors:
            raise ValueError("weight coverage must name at least one target")

        if self.provenance_sha256 is not None:
            digest = self.provenance_sha256
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(
                    "provenance_sha256 must be a lowercase hex SHA-256 digest"
                )

        # A regra de acoplamento: todo registro de peso tem de carregar
        # exatamente uma das duas evidencias possiveis, e elas se excluem.
        if self.status != "classified_nonweight":
            if self.provenance_sha256 is not None:
                if self.drift is not None and self.drift.value != 0.0:
                    raise ValueError(
                        f"record names a bijection, so the target is "
                        f"byte-identical to the donor and drift must be "
                        f"exactly 0.0; got {self.drift.value} under "
                        f"{self.drift.metric!r}"
                    )
            elif self.drift is None:
                raise ValueError(
                    f"weight coverage by {self.method!r} carries neither a "
                    f"provenance_sha256 (claiming inheritance) nor a measured "
                    f"drift (admitting manufacture); one is required"
                )

    @property
    def inherited(self) -> bool:
        """O alvo e byte-identico a um rearranjo do doador.

        Derivado da evidencia, nao do nome do metodo.
        """
        return self.provenance_sha256 is not None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CoverageRecord":
        raw_drift = raw.get("drift")
        provenance = raw.get("provenance_sha256")
        return cls(
            source_tensor=str(raw["source_tensor"]),
            target_tensors=tuple(
                str(target) for target in raw["target_tensors"]
            ),
            method=str(raw["method"]),
            status=str(raw["status"]),
            metrics=dict(raw.get("metrics") or {}),
            provenance_sha256=str(provenance) if provenance else None,
            drift=DriftRecord.from_mapping(raw_drift) if raw_drift else None,
        )


class CoverageLedger:
    def __init__(
        self,
        path: Path,
        *,
        plan_id: str,
        head: str,
        records: tuple[CoverageRecord, ...],
        next_sequence: int,
    ) -> None:
        self.path = path
        self.plan_id = plan_id
        self.head = head
        self.records = records
        self._next_sequence = next_sequence

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        plan_id: str,
    ) -> "CoverageLedger":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            header = {
                "sequence": 0,
                "plan_id": plan_id,
                "previous": GENESIS_HEAD,
                "record": {
                    "kind": "header",
                    "schema": "darwin-smol-coverage-v1",
                },
            }
            line = {**header, "digest": _digest(header)}
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(_canonical(line) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

        previous = GENESIS_HEAD
        records: list[CoverageRecord] = []
        expected_sequence = 0
        with target.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid coverage ledger JSON at line {line_number}"
                    ) from exc
                declared_digest = str(payload.pop("digest", ""))
                actual_digest = _digest(payload)
                if declared_digest != actual_digest:
                    raise ValueError(
                        "coverage ledger digest mismatch at "
                        f"line {line_number}"
                    )
                if int(payload.get("sequence", -1)) != expected_sequence:
                    raise ValueError(
                        "coverage ledger sequence mismatch at "
                        f"line {line_number}"
                    )
                if str(payload.get("previous")) != previous:
                    raise ValueError(
                        "coverage ledger previous-head mismatch at "
                        f"line {line_number}"
                    )
                if str(payload.get("plan_id")) != plan_id:
                    raise ValueError(
                        "coverage ledger plan identity mismatch at "
                        f"line {line_number}"
                    )
                record = payload.get("record")
                if expected_sequence == 0:
                    if not isinstance(record, dict) or record.get("kind") != "header":
                        raise ValueError("coverage ledger header is missing")
                else:
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"coverage ledger record missing at line {line_number}"
                        )
                    records.append(CoverageRecord.from_mapping(record))
                previous = declared_digest
                expected_sequence += 1

        if expected_sequence == 0:
            raise ValueError("coverage ledger is empty")
        return cls(
            target,
            plan_id=plan_id,
            head=previous,
            records=tuple(records),
            next_sequence=expected_sequence,
        )

    def append(self, record: CoverageRecord) -> str:
        if any(
            existing.source_tensor == record.source_tensor
            for existing in self.records
        ):
            raise ValueError(
                f"source tensor already classified: {record.source_tensor}"
            )
        payload = {
            "sequence": self._next_sequence,
            "plan_id": self.plan_id,
            "previous": self.head,
            "record": asdict(record),
        }
        digest = _digest(payload)
        line = {**payload, "digest": digest}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(line) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records = (*self.records, record)
        self.head = digest
        self._next_sequence += 1
        return digest

    def assert_complete(
        self,
        donor_tensors: set[str],
        target_language_tensors: set[str],
    ) -> None:
        covered_donors = {record.source_tensor for record in self.records}
        if covered_donors != donor_tensors:
            missing = sorted(donor_tensors - covered_donors)
            extra = sorted(covered_donors - donor_tensors)
            raise ValueError(
                f"donor coverage mismatch: missing={missing}; extra={extra}"
            )
        covered_targets = {
            target
            for record in self.records
            for target in record.target_tensors
        }
        if covered_targets != target_language_tensors:
            missing = sorted(target_language_tensors - covered_targets)
            extra = sorted(covered_targets - target_language_tensors)
            raise ValueError(
                f"target coverage mismatch: missing={missing}; extra={extra}"
            )
        random_targets = sorted(
            target
            for record in self.records
            if record.method.lower().startswith("random")
            for target in record.target_tensors
        )
        if random_targets:
            raise ValueError(
                f"random target tensors are forbidden: {random_targets}"
            )
