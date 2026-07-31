"""F51 Wolfram Bridge — The organism's mathematical cortex extension.

Connects F51 Neural Organism to Wolfram Alpha API for:
    - Symbolic computation (derivatives, integrals, limits)
    - Equation solving (algebraic, differential)
    - Factual queries (population, physics constants, geography)
    - Mathematical properties (domain, range, parity)
    - Number theory, combinatorics, statistics

Doctrine:
    Wolfram is NOT the brain. Wolfram is the calculator.
    The organism THINKS. Wolfram COMPUTES.
    Every query is logged. Every result is cached.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import parse, request


@dataclass
class WolframConfig:
    app_id: str = ""
    api_base: str = "http://api.wolframalpha.com"
    timeout_sec: float = 15.0
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    log_queries: bool = True


@dataclass
class WolframResult:
    query: str
    success: bool
    result_text: str = ""
    pods: list[dict[str, str]] = field(default_factory=list)
    timing_sec: float = 0.0
    error: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "success": self.success,
            "result_text": self.result_text,
            "pods": self.pods,
            "timing_sec": self.timing_sec,
            "error": self.error,
            "cached": self.cached,
        }


class WolframBridge:
    """Connects the neural organism to Wolfram Alpha for mathematical computation."""

    def __init__(self, config: WolframConfig) -> None:
        self.config = config
        self._cache: dict[str, WolframResult] = {}
        self._query_log: list[dict[str, Any]] = []

    def query(self, input_text: str, *, language: str = "en") -> WolframResult:
        """Execute a Wolfram Alpha query."""
        if not self.config.app_id:
            return WolframResult(query=input_text, success=False, error="No App ID configured")

        # Check cache
        cache_key = f"{input_text}:{language}"
        if self.config.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            return WolframResult(
                query=input_text,
                success=cached.success,
                result_text=cached.result_text,
                pods=cached.pods,
                timing_sec=0.0,
                cached=True,
            )

        # Build URL
        params = parse.urlencode({
            "appid": self.config.app_id,
            "input": input_text,
            "format": "plaintext",
            "output": "json",
        })
        url = f"{self.config.api_base}/v2/query?{params}"

        result = WolframResult(query=input_text, success=False)
        start = time.perf_counter()

        for attempt in range(self.config.max_retries):
            try:
                req = request.Request(url)
                with request.urlopen(req, timeout=self.config.timeout_sec) as response:
                    data = json.loads(response.read().decode("utf-8"))

                qr = data.get("queryresult", {})
                result.success = qr.get("success", False)
                result.timing_sec = round(time.perf_counter() - start, 3)

                if result.success:
                    result.pods = self._extract_pods(qr)
                    result.result_text = self._extract_primary_text(qr)

                if self.config.log_queries:
                    self._query_log.append({
                        "query": input_text,
                        "success": result.success,
                        "timing": result.timing_sec,
                        "attempt": attempt + 1,
                    })

                break

            except Exception as exc:
                if attempt == self.config.max_retries - 1:
                    result.error = str(exc)
                else:
                    time.sleep(0.5 * (attempt + 1))

        # Cache successful results
        if self.config.cache_enabled and result.success:
            self._cache[cache_key] = result

        return result

    def _extract_pods(self, query_result: dict) -> list[dict[str, str]]:
        pods: list[dict[str, str]] = []
        for pod in query_result.get("pods", []):
            pod_data = {"title": pod.get("title", "")}
            for subpod in pod.get("subpods", []):
                text = subpod.get("plaintext", "").strip()
                if text:
                    pod_data["plaintext"] = text
                    break
            if "plaintext" in pod_data:
                pods.append(pod_data)
        return pods

    def _extract_primary_text(self, query_result: dict) -> str:
        for pod in query_result.get("pods", []):
            if pod.get("primary"):
                for subpod in pod.get("subpods", []):
                    text = subpod.get("plaintext", "").strip()
                    if text:
                        return text
        # Fallback: first pod with text
        for pod in query_result.get("pods", []):
            for subpod in pod.get("subpods", []):
                text = subpod.get("plaintext", "").strip()
                if text:
                    return text
        return ""

    def compute(
        self,
        expression: str,
        *,
        operation: str = "evaluate",
    ) -> WolframResult:
        """High-level compute interface for the organism."""
        prefixes = {
            "derivative": "derivative of ",
            "integral": "integral of ",
            "solve": "solve ",
            "limit": "limit of ",
            "simplify": "simplify ",
            "factor": "factor ",
            "expand": "expand ",
            "evaluate": "",
        }
        prefix = prefixes.get(operation, "")
        return self.query(f"{prefix}{expression}")

    def math_pipeline(
        self,
        problem: str,
    ) -> dict[str, Any]:
        """Complete math pipeline: parse → compute → explain.

        This is the interface the Darwin-SSD cortex will call.
        """
        result = self.query(problem)
        return {
            "input": problem,
            "computed": result.success,
            "result": result.result_text,
            "details": result.pods,
            "timing": result.timing_sec,
            "cached": result.cached,
            "error": result.error if not result.success else "",
        }

    def health_check(self) -> dict[str, Any]:
        """Verify Wolfram API connectivity."""
        result = self.query("2+2")
        return {
            "connected": result.success,
            "response_time": result.timing_sec,
            "result": result.result_text,
            "app_id_configured": bool(self.config.app_id),
            "queries_logged": len(self._query_log),
        }

    def save_query_log(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self._query_log, indent=2) + "\n", encoding="utf-8")
        return output
