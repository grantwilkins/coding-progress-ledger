"""State warmness map (Workstream K2).

A `WarmnessMap` tracks which sites currently hold a materialized copy
of each state object. K3's `reconstitution_cost` short_circuits to a
zero ResourceCost when `warmness.is_warm(state, dst_site)`. K4's fluid
simulator mutates the warmness map deterministically as it advances
time: a successful reconstitution adds the (state, dst_site) entry; an
LRU eviction under KV_memory pressure removes one.

Per A3 audit: warmness is keyed by (state_id, site) -- per-(state, site)
dedup, NOT per-(component, site). This is the structural difference
between L1 and L2 that the H5b finding made load_bearing. K3 must use
this map to enforce L1 semantics in its cost model.

Hard rule (per CLAUDE.md): only K4 mutates a warmness map. Everything
else reads it (`is_warm`, `fraction_warm`, `entries`).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .manifest import ServingGroupManifest


@dataclass(frozen=True)
class WarmnessEntry:
    """A single (state_id, site) cache entry.

    `mode` records HOW the state was materialized at this site (e.g.,
    "kv_transfer", "context_replay"). `age_s` is how long the entry has
    existed; K4 uses it for LRU eviction under KV_memory pressure.
    `eviction_pressure` is set by K4 when the destination is approaching
    its KV capacity; readers can use it as a hint that the entry may
    not survive much longer.
    """
    state_id: str
    site: str
    mode: str
    age_s: float = 0.0
    eviction_pressure: float = 0.0  # 0..1, set by K4


@dataclass(frozen=True)
class WarmnessMap:
    """Read_only view of which (state, site) pairs are currently warm.

    Construction:
        WarmnessMap.empty()                          -- all cold
        WarmnessMap.from_dict({sid: ["phoenix"]})    -- bulk from JSON_style data
        WarmnessMap.from_episode_seed(state_warmness)-- from MobilityEpisode.state_warmness

    Mutation (K4 only): use `with_added` / `with_evicted` to produce a
    new map with the change applied. The frozen=True dataclass + dict
    copy semantics mean mutations are explicit and testable.
    """
    entries: dict[tuple[str, str], WarmnessEntry] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "WarmnessMap":
        return cls(entries={})

    @classmethod
    def from_dict(cls, raw: dict[str, list[str] | tuple[str, ...]],
                  default_mode: str = "warm_reuse") -> "WarmnessMap":
        """Bulk_construct from a `state_id -> sites` mapping (the shape
        used in MobilityEpisode.state_warmness JSON)."""
        entries: dict[tuple[str, str], WarmnessEntry] = {}
        for state_id, sites in raw.items():
            for site in sites:
                entries[(state_id, site)] = WarmnessEntry(
                    state_id=state_id, site=site, mode=default_mode,
                )
        return cls(entries=entries)

    @classmethod
    def from_episode_seed(cls, state_warmness: dict[str, tuple[str, ...]]) -> "WarmnessMap":
        return cls.from_dict({k: list(v) for k, v in state_warmness.items()})

    # -------- read_only API --------

    def is_warm(self, state_id: str, site: str) -> bool:
        return (state_id, site) in self.entries

    def get(self, state_id: str, site: str) -> WarmnessEntry | None:
        return self.entries.get((state_id, site))

    def fraction_warm(self, manifest: ServingGroupManifest, site: str) -> float:
        """Fraction of manifest's state objects that are warm at `site`.
        Returns 0.0 if the manifest has no state objects."""
        if not manifest.state_objects:
            return 0.0
        warm = sum(1 for sid in manifest.state_objects if self.is_warm(sid, site))
        return warm / len(manifest.state_objects)

    def states_warm_at(self, site: str) -> set[str]:
        return {sid for (sid, s) in self.entries if s == site}

    def sites_warm_for(self, state_id: str) -> set[str]:
        return {s for (sid, s) in self.entries if sid == state_id}

    def __len__(self) -> int:
        return len(self.entries)

    # -------- K4 mutators (return new instances) --------

    def with_added(self, state_id: str, site: str, mode: str,
                   age_s: float = 0.0) -> "WarmnessMap":
        """Return a new map with (state_id, site) marked warm."""
        new_entries = dict(self.entries)
        new_entries[(state_id, site)] = WarmnessEntry(
            state_id=state_id, site=site, mode=mode, age_s=age_s,
        )
        return WarmnessMap(entries=new_entries)

    def with_evicted(self, state_id: str, site: str) -> "WarmnessMap":
        """Return a new map without (state_id, site)."""
        new_entries = dict(self.entries)
        new_entries.pop((state_id, site), None)
        return WarmnessMap(entries=new_entries)

    def with_aged(self, delta_s: float) -> "WarmnessMap":
        """Return a new map where every entry's age has increased by `delta_s`."""
        new_entries = {
            key: dataclasses.replace(entry, age_s=entry.age_s + delta_s)
            for key, entry in self.entries.items()
        }
        return WarmnessMap(entries=new_entries)

    def lru_evict(self, site: str) -> tuple["WarmnessMap", str | None]:
        """Evict the oldest entry at `site`. Returns (new_map, evicted_state_id).
        If no entries at site, returns (self, None).

        Ordering: primary by descending age_s (largest = oldest);
        secondary tie_break by dict insertion order (earlier insertion =
        older). Python 3.7+ preserves dict insertion order, so this is
        deterministic.
        """
        # enumerate(items) gives insertion index; filter to entries at `site`.
        indexed = [(idx, key, entry) for idx, (key, entry) in enumerate(self.entries.items())
                   if key[1] == site]
        if not indexed:
            return self, None
        # Sort by (-age_s, insertion_idx) ascending: largest age_s first;
        # within ties, earliest insertion first.
        indexed.sort(key=lambda x: (-x[2].age_s, x[0]))
        _, oldest_key, _ = indexed[0]
        new_entries = dict(self.entries)
        del new_entries[oldest_key]
        return WarmnessMap(entries=new_entries), oldest_key[0]
