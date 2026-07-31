from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MathOrganResult:
    expression: str
    success: bool
    answer: str | None = None
    error: str = ""
    organ: str = "local_arithmetic"


class LocalArithmeticOrgan:
    """Small deterministic organ for safe, offline arithmetic verification."""

    _pattern = re.compile(r"\s*(\d+)\s*([+\-*])\s*(\d+)\s*")

    def solve(self, expression: str) -> MathOrganResult:
        match = self._pattern.fullmatch(expression)
        if not match:
            return MathOrganResult(expression=expression, success=False, error="unsupported_expression")

        left = int(match.group(1))
        operator = match.group(2)
        right = int(match.group(3))
        if operator == "+":
            return MathOrganResult(expression=expression, success=True, answer=str(left + right))
        if operator == "-":
            result = left - right
            if result < 0:
                return MathOrganResult(expression=expression, success=False, error="negative_result")
            return MathOrganResult(expression=expression, success=True, answer=str(result))
        if operator == "*":
            return MathOrganResult(expression=expression, success=True, answer=str(left * right))

        return MathOrganResult(expression=expression, success=False, error="unsupported_operator")


def solve_arithmetic_expression(expression: str) -> str | None:
    result = LocalArithmeticOrgan().solve(expression)
    return result.answer if result.success else None
