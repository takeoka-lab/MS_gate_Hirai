"""Safeguarded root finding for model-specific hXX drive calibration."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np


def find_hxx_zero(
    evaluator: Callable[[float], Mapping[str, float] | None],
    seed_omegas: Iterable[float],
    *,
    lower_bound: float,
    upper_bound: float,
    tolerance_rad: float = 2e-4,
    max_iterations: int = 3,
) -> dict:
    """Find ``h_XX(omega)=0`` using cached seeds and safeguarded secants.

    ``evaluator`` may return ``None`` when a requested propagator is pending.
    The returned best point is still useful for resumable notebook stages.
    """

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    tolerance_rad = float(tolerance_rad)
    max_iterations = int(max_iterations)
    if not 0.0 < lower_bound < upper_bound:
        raise ValueError("require 0 < lower_bound < upper_bound")
    if tolerance_rad <= 0.0 or max_iterations < 0:
        raise ValueError("tolerance_rad must be positive and iterations nonnegative")

    evaluated = {}

    def evaluate(omega):
        omega = float(omega)
        key = round(omega, 9)
        if key not in evaluated:
            result = evaluator(omega)
            if result is not None:
                evaluated[key] = dict(result)
        return evaluated.get(key)

    for omega in sorted({float(value) for value in seed_omegas}):
        if lower_bound <= omega <= upper_bound:
            evaluate(omega)

    def sorted_points():
        return sorted(
            (
                float(result["omega_per_second"]),
                float(result["h_XX_rad_per_gate"]),
                result,
            )
            for result in evaluated.values()
        )

    def best_point():
        points = sorted_points()
        return min(points, key=lambda item: abs(item[1])) if points else None

    def find_bracket():
        candidates = []
        points = sorted_points()
        for left, right in zip(points[:-1], points[1:]):
            if left[1] == 0.0 or right[1] == 0.0 or left[1] * right[1] < 0.0:
                candidates.append((left, right))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda pair: abs(pair[0][1]) + abs(pair[1][1]),
        )

    best = best_point()
    if best is not None and abs(best[1]) <= tolerance_rad:
        return {
            "best": best[2],
            "converged": True,
            "iterations": 0,
            "bracket": None,
            "evaluated_points": len(evaluated),
            "pending": False,
        }

    bracket = find_bracket()
    if bracket is None:
        evaluate(lower_bound)
        evaluate(upper_bound)
        best = best_point()
        if best is not None and abs(best[1]) <= tolerance_rad:
            return {
                "best": best[2],
                "converged": True,
                "iterations": 0,
                "bracket": None,
                "evaluated_points": len(evaluated),
                "pending": False,
            }
        bracket = find_bracket()

    if bracket is None:
        return {
            "best": None if best is None else best[2],
            "converged": False,
            "iterations": 0,
            "bracket": None,
            "evaluated_points": len(evaluated),
            "pending": True,
        }

    left, right = bracket
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        omega_left, h_left, _ = left
        omega_right, h_right, _ = right
        denominator = h_right - h_left
        if denominator == 0.0:
            candidate = 0.5 * (omega_left + omega_right)
        else:
            candidate = omega_left - h_left * (
                omega_right - omega_left
            ) / denominator
        span = omega_right - omega_left
        if not (omega_left + 1e-6 * span < candidate < omega_right - 1e-6 * span):
            candidate = 0.5 * (omega_left + omega_right)
        result = evaluate(candidate)
        if result is None:
            break
        h_candidate = float(result["h_XX_rad_per_gate"])
        point = (float(result["omega_per_second"]), h_candidate, result)
        best = best_point()
        if abs(h_candidate) <= tolerance_rad:
            return {
                "best": result,
                "converged": True,
                "iterations": iteration,
                "bracket": (omega_left, omega_right),
                "evaluated_points": len(evaluated),
                "pending": False,
            }
        if h_left * h_candidate <= 0.0:
            right = point
        else:
            left = point

    best = best_point()
    return {
        "best": None if best is None else best[2],
        "converged": bool(best is not None and abs(best[1]) <= tolerance_rad),
        "iterations": iteration,
        "bracket": (left[0], right[0]),
        "evaluated_points": len(evaluated),
        "pending": bool(best is None or abs(best[1]) > tolerance_rad),
    }
