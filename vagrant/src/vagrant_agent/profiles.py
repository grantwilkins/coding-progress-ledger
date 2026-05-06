"""Model and site profile loaders.

MVP ships one model and two sites. The structure accommodates more.
Links are symmetric and keyed by an unordered site pair (alphabetic).

K-extension (Workstream K, 2026-05-05): SiteProfile carries optional
fluid-capacity fields used by K4's simulator
(`workspace_hydrate_bps`, `kv_memory_bytes`). Both default to math.inf
so existing 2-site MVP configs continue to load unchanged — capacity
is uncapped unless the YAML explicitly sets it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelProfile:
    name: str
    active_params_b: float
    kv_bytes_per_token: int
    single_stream_prefill_tok_s: float = 100_000.0  # T_ref=100k tok/s baseline; overridden per architecture
    notes: str = ""


@dataclass(frozen=True)
class SiteProfile:
    name: str
    prefill_tok_s: float
    workspace_hydrate_bps: float = math.inf  # K4 fluid capacity (bytes/s for workspace bring-up); inf = uncapped (MVP)
    kv_memory_bytes: float = math.inf        # K4 KV-resident capacity; inf = uncapped (MVP)


@dataclass(frozen=True)
class LinkProfile:
    site_a: str
    site_b: str
    effective_bps: float


@dataclass(frozen=True)
class ProfileBundle:
    model: ModelProfile
    sites: dict[str, SiteProfile]
    links: dict[tuple[str, str], LinkProfile]
    home_site: str

    def site(self, name: str) -> SiteProfile:
        if name not in self.sites:
            raise ValueError(f"unknown site: {name!r}")
        return self.sites[name]

    def link(self, site_a: str, site_b: str) -> LinkProfile:
        if site_a == site_b:
            raise ValueError("link requires two distinct sites")
        key = _link_key(site_a, site_b)
        if key not in self.links:
            raise ValueError(f"no link configured between {site_a!r} and {site_b!r}")
        return self.links[key]


def load_model(path: str | Path, name: str) -> ModelProfile:
    raw = _load_yaml(path)
    models = raw.get("models", {})
    if name not in models:
        raise ValueError(f"model {name!r} not found in {path}; available: {sorted(models)}")
    spec = models[name]
    return ModelProfile(
        name=name,
        active_params_b=float(spec["active_params_b"]),
        kv_bytes_per_token=int(spec["kv_bytes_per_token"]),
        single_stream_prefill_tok_s=float(spec.get("single_stream_prefill_tok_s", 100_000.0)),
        notes=spec.get("notes", ""),
    )


def load_sites(path: str | Path) -> tuple[dict[str, SiteProfile], dict[tuple[str, str], LinkProfile], str]:
    raw = _load_yaml(path)
    sites = {
        name: SiteProfile(
            name=name,
            prefill_tok_s=float(spec["prefill_tok_s"]),
            workspace_hydrate_bps=float(spec.get("workspace_hydrate_bps", math.inf)),
            kv_memory_bytes=float(spec.get("kv_memory_bytes", math.inf)),
        )
        for name, spec in raw.get("sites", {}).items()
    }
    if not sites:
        raise ValueError(f"no sites in {path}")
    for name in sites:
        if "-" in name:
            raise ValueError(f"site name {name!r} must not contain '-' (reserved for link keys)")
    home_site = raw.get("home_site")
    if home_site is None:
        raise ValueError(f"home_site missing from {path}")
    if home_site not in sites:
        raise ValueError(f"home_site {home_site!r} is not a configured site")
    links: dict[tuple[str, str], LinkProfile] = {}
    for label, spec in raw.get("links", {}).items():
        if "-" not in label:
            raise ValueError(f"link key {label!r} must be 'siteA-siteB'")
        a, b = label.split("-", 1)
        if a == b:
            raise ValueError(f"link {label!r} is a self-link; not allowed")
        if a not in sites or b not in sites:
            raise ValueError(f"link {label!r} references unknown site")
        links[_link_key(a, b)] = LinkProfile(
            site_a=a, site_b=b, effective_bps=float(spec["effective_bps"]),
        )
    return sites, links, home_site


def load_bundle(model_path: str | Path, sites_path: str | Path, model_name: str) -> ProfileBundle:
    model = load_model(model_path, model_name)
    sites, links, home_site = load_sites(sites_path)
    return ProfileBundle(model=model, sites=sites, links=links, home_site=home_site)


def _link_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))  # type: ignore[return-value]


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
