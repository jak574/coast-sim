"""Observation groups: named sets of targets observed under one window, one
budget and, where wanted, together.

A category is an observation-ID range with a program's settings; a group is
a list of targets, such as the targets of one redshift bin or one field,
that share a window in which their science may be collected, a cap on the
science they collect together, and a preference for being observed one after
another. Dispatch, the planners and the long-range allocator honour all three.

A group can also be consecutive: its targets are then observed in the order
listed, with no other science between the first and the last of them.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from math import isinf

from pydantic import Field, field_serializer, field_validator, model_validator

from ..common.common import unixtime_from
from ._base import ConfigModel


class ObservationGroup(ConfigModel):
    """A set of targets observed under one window and one budget."""

    name: str = Field(description="The group's name")
    members: list[int] = Field(
        min_length=1, description="The observation IDs of the group's targets"
    )
    window_start: float | None = Field(
        default=None,
        description=(
            "When the group's science may begin, in Unix seconds, a datetime or an "
            "ISO string; None for no start. Narrows each member's own earliest start"
        ),
    )
    window_deadline: float | None = Field(
        default=None,
        description=(
            "By when the group's science must begin collecting, in Unix seconds, a "
            "datetime or an ISO string; None for no deadline. Narrows each member's "
            "own deadline"
        ),
    )
    time_budget_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "The most science the group's targets may collect together over a run, "
            "in seconds, counting what they had collected before it; None for no cap"
        ),
    )
    prefer_contiguous: bool = Field(
        default=True,
        description=(
            "Whether the group's targets are preferred one after another: dispatch "
            "adds targets.group_contiguity_weight to a member's merit once another "
            "member has just been observed, and the priority-first placement goes "
            "on with the group once it has placed one of its targets"
        ),
    )

    consecutive: bool = Field(
        default=False,
        description=(
            "Whether the group's targets are observed in the order of members, "
            "each one completed before the next begins, with no science of any "
            "other target between the group's first observation and its last. "
            "Ground contacts, charging, thruster dumps and other activities may "
            "fall between two of its observations. The group ends at a member "
            "that can no longer be observed, and the members after it are then "
            "not observed. A group with no window of its own does not begin "
            "where a request with visit windows or a deadline would lose its "
            "window while the group runs"
        ),
    )

    @field_validator("window_start", "window_deadline", mode="before")
    @classmethod
    def _as_unix(cls, value: float | int | str | datetime | None) -> float | None:
        if value is None:
            return None
        return unixtime_from(value)

    @field_serializer("window_start", "window_deadline", when_used="json")
    def _serialize_time(self, value: float | None) -> str | None:
        if value is None:
            return None
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    @model_validator(mode="after")
    def _check(self) -> ObservationGroup:
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"group {self.name!r} lists a member twice")
        if (
            self.window_start is not None
            and self.window_deadline is not None
            and self.window_deadline <= self.window_start
        ):
            raise ValueError(f"group {self.name!r}'s window must end after it starts")
        return self

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """The window's start and deadline, each None where unset."""
        return self.window_start, self.window_deadline

    @property
    def has_window(self) -> bool:
        """Whether the group has a start or a deadline of its own."""
        return self.window_start is not None or self.window_deadline is not None


def group_start_wait(
    start: float, needed: float, windows: Iterable[tuple[float, float]]
) -> float | None:
    """Until when a consecutive group with no window of its own waits to
    begin, where beginning at ``start`` would close a time-bound request's
    window; None where it may begin at ``start``.

    The group needs ``needed`` seconds of science and takes no other
    science between its first observation and its last, so a request whose
    collection has to begin inside that time cannot be observed in it. Each
    of ``windows`` is one such request's chance still to be taken, as
    ``(precedes_from, closes)``: its collection has to begin by ``closes``,
    and a group that begins at ``precedes_from`` or later leaves room for
    the request before it. The group waits where a window closes before
    the group would end and the request cannot go before it. A group with
    a window of its own is not held this way: its window says when it runs.

    The time returned is the earliest at which the first such window no
    longer holds the group: when the request can go first. A request that
    begins by ``closes`` may still be collecting after it, so the group
    does not begin between the two either. Where no time is known from
    which the request can go first (``precedes_from`` is infinite, as in
    dispatch, which selects the request itself once its window is open),
    the group waits until just after the window has closed.
    """
    wait: float | None = None
    for precedes_from, closes in windows:
        if start >= precedes_from or closes >= start + needed:
            continue
        if isinf(precedes_from):
            if start > closes:
                continue
            until = closes + 1.0
        else:
            until = precedes_from
        wait = until if wait is None else min(wait, until)
    return wait


class ObservationGroups(ConfigModel):
    """The observation groups of a mission, each target in at most one."""

    groups: list[ObservationGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_group_each(self) -> ObservationGroups:
        seen: dict[int, str] = {}
        for group in self.groups:
            for obsid in group.members:
                if obsid in seen:
                    raise ValueError(
                        f"observation {obsid} is in groups {seen[obsid]!r} and "
                        f"{group.name!r}; a target belongs to one group"
                    )
                seen[obsid] = group.name
        return self

    def group_of(self, obsid: int) -> ObservationGroup | None:
        """The group ``obsid`` belongs to, or None."""
        for group in self.groups:
            if obsid in group.members:
                return group
        return None

    def position(self, obsid: int) -> int | None:
        """Where ``obsid`` comes in its consecutive group's order, or None
        where it is in no consecutive group."""
        group = self.group_of(obsid)
        if group is None or not group.consecutive:
            return None
        return group.members.index(obsid)

    @property
    def consecutive(self) -> list[ObservationGroup]:
        """The groups whose targets are observed in order with no other
        science between them."""
        return [group for group in self.groups if group.consecutive]

    def __bool__(self) -> bool:
        return bool(self.groups)
