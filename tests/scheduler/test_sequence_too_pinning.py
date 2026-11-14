"""Waiting ToOs follow every step of the visit which authorized an interrupt."""

from datetime import timedelta

import pytest

from conops.config import ObservationTiming
from conops.ditl import ReplanReason, RollingHorizonDITL
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Pointing, SequenceStep, TOORequest

from .planning_scenario import BEGIN, MIN, T0, make_config, make_target


@pytest.mark.parametrize(
    "kind", [PriorityScheduler, LocalSearchScheduler, CpSatScheduler]
)
@pytest.mark.parametrize("deferred_first", [False, True])
@pytest.mark.parametrize("calibration", [False, True])
def test_waiting_too_follows_whole_interrupting_visit(
    kind, deferred_first, calibration
):
    config = make_config(3)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming(
        setup_seconds=30, cleanup_seconds=85, handoff_seconds=40
    )
    target = make_target(
        config, 1, 105, 10, merit=10, minutes=60, snapshot=60, ss_min=10
    )
    steps = [
        SequenceStep(name="same" if calibration else "first", duration_seconds=20 * MIN)
    ]
    if calibration:
        steps.append(
            SequenceStep(name="calibration", duration_seconds=5 * MIN, science=False)
        )
    steps.append(
        SequenceStep(
            name="same" if calibration else "second", duration_seconds=20 * MIN
        )
    )
    sequence = ObservationSequence(steps=steps)
    request = Pointing(
        config=config,
        obsid=1000001,
        name="interrupting",
        ra=115,
        dec=10,
        merit=400,
        sequence=sequence,
        visits=1,
    )
    interrupting = TOORequest(
        request=request,
        submit_time=T0 + 20 * MIN,
        deadline=T0 + 30 * MIN,
        interrupt=True,
    )
    waiting = TOORequest(
        config=config,
        obsid=1000002,
        name="waiting",
        ra=115,
        dec=10,
        merit=500,
        exptime=10 * MIN,
        submit_time=T0 + 20 * MIN,
        deadline=T0 + 120 * MIN,
        interrupt=False,
    )
    options = (
        {}
        if kind is PriorityScheduler
        else dict(time_limit=0, max_iterations=0, seed=2029)
    )
    if kind is CpSatScheduler:
        options.update(
            chunk=timedelta(hours=6),
            solver_time_limit=1200,
            solver_deterministic_time=20,
            workers=1,
        )
    run = RollingHorizonDITL(
        config,
        [target],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=3),
        replan_interval=timedelta(hours=4),
        commit_lead_time=timedelta(0),
        scheduler=kind,
        scheduler_options=options,
    )
    run.step_size = 60
    for too in (waiting, interrupting) if deferred_first else (interrupting, waiting):
        run.submit_too(too)
    assert run.calc()
    entries = run.plan.entries
    selected = [e for e in entries if int(e.obsid) == 1000001]
    assert [e.step for e in selected] == [s.name for s in steps]
    indices = [entries.index(e) for e in selected]
    assert indices == list(range(indices[0], indices[0] + len(steps)))
    deferred = [e for e in entries if int(e.obsid) == 1000002]
    assert len(deferred) == 1
    assert float(deferred[0].begin) >= float(selected[-1].end)
    science = sum(
        h.collection_seconds
        for h in run.telemetry.housekeeping
        if h.obsid == 1000001 and not h.calibration
    )
    assert science == pytest.approx(40 * MIN)
    calibration_seconds = sum(
        h.collection_seconds
        for h in run.telemetry.housekeeping
        if h.obsid == 1000001 and h.calibration
    )
    assert calibration_seconds == pytest.approx(5 * MIN if calibration else 0)
    rapid = [r for r in run.replans if r.reason is ReplanReason.RAPID]
    assert [(r.trigger_obsid, r.interrupted_obsid) for r in rapid] == [(1000001, 1)]
    assert all(too.executed for too in run.too_register)
    assert run.validate_plan_matches_execution() == []
