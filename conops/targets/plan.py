from __future__ import annotations

from collections.abc import Iterator

from .plan_entry import PlanEntry


class TargetList:
    """List of potential targets for Spacecraft to observe"""

    targets: list[PlanEntry]

    def __init__(self) -> None:
        self.targets = list()

    def __getitem__(self, number: int) -> PlanEntry:
        return self.targets[number]

    def __iter__(self) -> Iterator[PlanEntry]:
        return iter(self.targets)

    def add_target(self, plan_entry: PlanEntry) -> None:
        self.targets.append(plan_entry)

    def __len__(self) -> int:
        return len(self.targets)


class Plan:
    """Simple Plan class"""

    entries: list[PlanEntry]

    def __init__(self) -> None:
        self.entries = list()

    def __getitem__(self, number: int) -> PlanEntry:
        return self.entries[number]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PlanEntry]:
        return iter(self.entries)

    def which_ppt(self, utime: float) -> PlanEntry | None:
        """At a given utime, which PPT is the current one?"""
        for ppt in self.entries:
            if ppt.begin <= utime < ppt.end:
                return ppt
        return None

    def extend(self, ppt: list[PlanEntry]) -> None:
        self.entries.extend(ppt)

    def append(self, ppt: PlanEntry) -> None:
        self.entries.append(ppt)
