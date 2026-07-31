#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Sequence

from f51_darwin.transplant_16b.exact_assembly import (
    build_exact_brain,
    publish_exact_candidate,
    write_exact_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build function-preserving Smol brain inside Darwin organs."
    )
    parser.add_argument("action", choices=("plan", "build", "publish"))
    args = parser.parse_args(argv)
    if args.action == "plan":
        plan = write_exact_plan()
        print(f"EXACT_PLAN_OK plan_id={plan['plan_id']}")
        return 0
    if args.action == "publish":
        candidate = publish_exact_candidate()
        print(
            "EXACT_CANDIDATE_PUBLISHED "
            f"checkpoint_sha256={candidate['checkpoint_sha256']}"
        )
        return 0
    report = build_exact_brain()
    print(
        "EXACT_BUILD_OK "
        f"checkpoint={report['checkpoint']} "
        f"top1={report['parity']['top1_agreement']:.6f} "
        f"kl={report['parity']['mean_kl']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
