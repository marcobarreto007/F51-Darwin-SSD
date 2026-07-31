#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Sequence

from f51_darwin.transplant_16b.dense_assembly import (
    build_dense_brain,
    publish_dense_candidate,
    write_dense_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a structurally dense Smol brain inside Darwin organs."
        )
    )
    parser.add_argument("action", choices=("plan", "build", "publish"))
    args = parser.parse_args(argv)
    if args.action == "plan":
        plan = write_dense_plan()
        print(f"DENSE_PLAN_OK plan_id={plan['plan_id']}")
        return 0
    if args.action == "publish":
        candidate = publish_dense_candidate()
        print(
            "DENSE_CANDIDATE_PUBLISHED "
            f"checkpoint_sha256={candidate['checkpoint_sha256']}"
        )
        return 0
    report = build_dense_brain()
    print(
        "DENSE_BUILD_OK "
        f"checkpoint={report['checkpoint']} "
        f"top1={report['parity']['top1_agreement']:.6f} "
        f"kl={report['parity']['mean_kl']:.6f} "
        f"moe_modules={report['moe_modules']} "
        f"dense_ffn_modules={report['dense_ffn_modules']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
