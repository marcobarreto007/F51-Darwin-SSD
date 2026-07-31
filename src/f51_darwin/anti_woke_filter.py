from __future__ import annotations

import json
from pathlib import Path

from f51_darwin.corpus_policy import CorpusDecision, evaluate_corpus_source


class AntiWokeFilter:
    """Compatibility wrapper for the F51 corpus policy.

    "Woke zero" is implemented as political-noise quarantine, not as blind
    blocking of protected-class or medical terms. Scientific text that mentions
    race, sex, gender, demographics, or epidemiology can still pass when the
    domain and license are strong.
    """

    def __init__(self, *, default_license: str = "unknown") -> None:
        self.default_license = default_license
        self.checked = 0
        self.blocked = 0
        self.quarantined = 0

    def should_block(self, text: str, source_url: str = "", file_path: str = "") -> bool:
        self.checked += 1
        source = source_url or file_path or "unknown://local"
        result = evaluate_corpus_source(
            text=text,
            source_url=source,
            license_label=self.default_license,
        )
        if result.decision == CorpusDecision.REJECT:
            self.blocked += 1
            return True
        if result.decision == CorpusDecision.QUARANTINE:
            self.quarantined += 1
        return False

    def report(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "blocked": self.blocked,
            "quarantined": self.quarantined,
            "blocked_rate": round(self.blocked / max(self.checked, 1), 4),
            "status": "F51 corpus policy active",
        }


if __name__ == "__main__":
    import sys

    filt = AntiWokeFilter()
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/data/corpus")
    for path in list(corpus.glob("*.txt"))[:100]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:50_000]
        except OSError:
            continue
        if filt.should_block(text, file_path=str(path)):
            print(f"BLOCKED: {path.name}")
    print(json.dumps(filt.report(), indent=2, sort_keys=True))
