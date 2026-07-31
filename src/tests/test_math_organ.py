from __future__ import annotations

from f51_darwin.math_organ import LocalArithmeticOrgan, solve_arithmetic_expression


def test_local_arithmetic_organ_solves_supported_arithmetic() -> None:
    organ = LocalArithmeticOrgan()

    assert organ.solve("7+8").answer == "15"
    assert organ.solve("8*3").answer == "24"
    assert organ.solve("14-6").answer == "8"
    assert solve_arithmetic_expression("9 + 4") == "13"


def test_local_arithmetic_organ_rejects_unsupported_or_unsafe_input() -> None:
    organ = LocalArithmeticOrgan()

    assert organ.solve("6-14").error == "negative_result"
    assert organ.solve("__import__('os')").error == "unsupported_expression"
    assert organ.solve("10/2").error == "unsupported_expression"
    assert organ.solve("(7+8)").error == "unsupported_expression"
    assert solve_arithmetic_expression("6-14") is None
