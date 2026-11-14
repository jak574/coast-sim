"""Measures of a run's science, computed from per-step collection records."""

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from ..common.numerics import MIN_SAMPLES
from ..ditl.plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype

if TYPE_CHECKING:
    from ..ditl.telemetry import Housekeeping
    from ..targets import PlanEntry

Collection = tuple[int, float, float]
"""One step's science: obsid, step start (Unix seconds) and seconds collected."""


def dynamic_science_value(
    records: Iterable["Housekeeping"],
    entries: Iterable["PlanEntry"],
    step_size: float,
) -> float:
    """Value actual science using each executed entry's frozen dynamic merit.

    Dispatch and the planners record the selection value on ``entry.merit``.
    Match by observation and collection interval, so revisits can have different
    values. Only telemetry's collected seconds earn value; calibration earns
    none. Missing or ambiguous provenance raises instead of substituting base
    merits or assigning a value from a different visit.
    """
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    by_obsid: dict[int, list[PlanEntry]] = {}
    for entry in entries:
        if entry_obstype(entry) in PLAN_SCIENCE_OBSTYPES and not entry.calibration:
            by_obsid.setdefault(int(entry.obsid), []).append(entry)
    contributions = []
    for record in records:
        seconds = float(record.collection_seconds or 0.0)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Collection seconds must be finite and nonnegative")
        if seconds == 0 or record.calibration:
            continue
        begin = record.timestamp.timestamp()
        candidates = [] if record.obsid is None else by_obsid.get(int(record.obsid), [])
        values = {
            float(entry.merit)
            for entry in candidates
            if entry.collection_seconds_between(begin, begin + step_size) > 0
        }
        if len(values) != 1 or not all(math.isfinite(value) for value in values):
            raise ValueError(
                "Science collection has no unambiguous executed dynamic merit: "
                f"obsid={record.obsid}, timestamp={record.timestamp.isoformat()}"
            )
        contributions.append(seconds * values.pop())
    return math.fsum(contributions)


def visit_starts(
    records: Iterable[Collection], step_size: float
) -> dict[int, list[float]]:
    """Return when each visit to each target began.

    A visit is a run of consecutive steps collecting science for the same
    obsid; a step without collection for it ends the visit.
    """
    starts: dict[int, list[float]] = {}
    last_step: dict[int, float] = {}
    for obsid, utime, seconds in sorted(records, key=lambda r: r[1]):
        if seconds <= 0:
            continue
        previous = last_step.get(obsid)
        if previous is None or utime - previous > step_size * 1.5:
            starts.setdefault(obsid, []).append(utime)
        last_step[obsid] = utime
    return starts


def cadence_error(
    starts: Mapping[int, list[float]], cadence: Mapping[int, float]
) -> tuple[float | None, int]:
    """Return how far revisits missed their cadence, and how many targets were revisited.

    For each target with a cadence and at least two visits, the error is the
    mean gap between visits minus the cadence, as a fraction of the cadence,
    taken absolutely. The result is the mean over those targets, or None if
    none was revisited.
    """
    errors = []
    for obsid, interval in cadence.items():
        visits = sorted(starts.get(obsid, []))
        if len(visits) < MIN_SAMPLES or interval <= 0:
            continue
        mean_gap = (visits[-1] - visits[0]) / (len(visits) - 1)
        errors.append(abs(mean_gap - interval) / interval)
    if not errors:
        return None, 0
    return sum(errors) / len(errors), len(errors)


def program_shares(
    collected: Mapping[int, float], program: Mapping[int, str]
) -> dict[str, float]:
    """Return each program's fraction of the science collected."""
    totals: dict[str, float] = {}
    for obsid, seconds in collected.items():
        if seconds <= 0:
            continue
        name = program.get(obsid)
        if name is not None:
            totals[name] = totals.get(name, 0.0) + seconds
    total = sum(totals.values())
    if total <= 0:
        return {}
    return {name: seconds / total for name, seconds in sorted(totals.items())}
