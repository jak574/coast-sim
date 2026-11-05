"""A sequence's exact ingress must still meet its parent's start window."""

from datetime import timedelta

import pytest

from conops import QueueDITL, create_ditl
from conops.config import ObservationTiming
from conops.ditl.queue_ditl import _ObservationRequest
from conops.targets import (
    ObservationSequence,
    Pointing,
    ScheduledActivity,
    SequenceStep,
    VisitWindow,
)

from .planning_scenario import BEGIN, T0, make_config


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("first_science", [False, True])
def test_first_sequence_step_keeps_the_parent_admission_deadline(
    windowed: bool, first_science: bool
) -> None:
    config = make_config(2)
    parent = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="first", duration_seconds=60, science=first_science),
                SequenceStep(name="second", duration_seconds=60),
            ]
        ),
        **(
            {"visit_windows": [VisitWindow(start=T0, deadline=T0 + 30)]}
            if windowed
            else {"earliest_start": T0, "deadline": T0 + 30}
        ),
    )
    if windowed:
        parent.visit = 0
    queue = create_ditl(config, [parent], BEGIN, BEGIN + timedelta(hours=2))
    assert isinstance(queue, QueueDITL)
    step = queue._begin_sequence(parent, T0)
    queue.ppt = step
    assert queue._ppt_deadline(step) == T0 + 30
    slew = queue._new_ppt_slew(step, T0)
    slew.slewstart, slew.slewend, slew.slewtime = T0, T0 + 60, 60
    assert (
        queue._validate_ppt_slew(step, slew, T0, _ObservationRequest(T0, 60, None))
        is None
    )
    assert any("after its deadline" in e.description for e in queue.log.events)
    # Once admitted, later steps may begin after the parent's latest start.
    assert queue._sequence is not None
    queue._sequence.index = 1
    queue.ppt = parent.step_pointing(1)
    assert queue._ppt_deadline(queue.ppt) is None


@pytest.mark.parametrize("latest_start, fits", [(119, False), (120, True)])
@pytest.mark.parametrize("science_in_eclipse", [False, True])
def test_exact_sequence_admission_reserves_every_step_before_a_mandatory_hold(
    latest_start: int, fits: bool, science_in_eclipse: bool
) -> None:
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.constraint.science_in_eclipse = science_in_eclipse
    config.payload.observation_timing = ObservationTiming()
    config.spacecraft_bus.attitude_control.settle_time = 0
    parent = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        roll_deg=0,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="first", duration_seconds=60),
                SequenceStep(name="second", duration_seconds=60),
            ]
        ),
    )
    queue = create_ditl(
        config,
        [parent],
        BEGIN,
        BEGIN + timedelta(hours=2),
        activities=[
            ScheduledActivity(
                name="Required hold",
                obsid=998001,
                begin=T0 + 30,
                slide_seconds=latest_start - 30,
                duration_seconds=60,
            )
        ],
    )
    assert isinstance(queue, QueueDITL)
    queue.step_size = 60
    queue.ustart, queue.uend = T0, T0 + 7200
    step = queue._begin_sequence(parent, T0)
    step.begin, step.end = T0, T0 + 60
    step.collection_begin, step.collection_end = T0, T0 + 60
    queue.ppt = step
    slew = queue._new_ppt_slew(step, T0)
    slew.slewstart = slew.slewend = T0
    slew.endra, slew.enddec, slew.endroll = 105, 10, 0
    assert queue._sequence_fits_window(slew) is fits
