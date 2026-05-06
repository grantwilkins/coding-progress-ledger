from pathlib import Path

import pytest

from vagrant_agent.profiles import load_bundle, load_model, load_sites

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def test_load_model():
    m = load_model(MODELS, "compact_kv")
    assert m.name == "compact_kv"
    assert m.kv_bytes_per_token == 70656
    assert m.active_params_b == 49


def test_load_unknown_model_hard_fails():
    with pytest.raises(ValueError, match="not found"):
        load_model(MODELS, "no_such_model")


def test_self_link_at_load_time_hard_fails(tmp_path: Path):
    bad = tmp_path / "bad_sites.yaml"
    bad.write_text("home_site: a\nsites:\n  a:\n    prefill_tok_s: 1000\nlinks:\n  a-a:\n    effective_bps: 1000\n")
    with pytest.raises(ValueError, match="self-link"):
        load_sites(bad)


def test_dash_in_site_name_hard_fails(tmp_path: Path):
    bad = tmp_path / "bad_sites.yaml"
    bad.write_text(
        "home_site: phoenix\nsites:\n  phoenix:\n    prefill_tok_s: 1000\n"
        "  east-a:\n    prefill_tok_s: 1000\n"
    )
    with pytest.raises(ValueError, match="must not contain"):
        load_sites(bad)


def test_home_site_must_be_a_configured_site(tmp_path: Path):
    bad = tmp_path / "bad_sites.yaml"
    bad.write_text("home_site: ghost\nsites:\n  phoenix:\n    prefill_tok_s: 1000\n")
    with pytest.raises(ValueError, match="not a configured site"):
        load_sites(bad)


def test_load_sites_and_links():
    sites, links, home_site = load_sites(SITES)
    assert set(sites) == {"phoenix", "seattle"}
    assert sites["seattle"].prefill_tok_s == 45000
    assert ("phoenix", "seattle") in links
    assert links["phoenix", "seattle"].effective_bps == 5_000_000_000
    assert home_site == "phoenix"


def test_load_bundle_home_site():
    b = load_bundle(MODELS, SITES, "compact_kv")
    assert b.home_site == "phoenix"


def test_bundle_link_lookup_is_symmetric():
    b = load_bundle(MODELS, SITES, "compact_kv")
    a_to_b = b.link("phoenix", "seattle")
    b_to_a = b.link("seattle", "phoenix")
    assert a_to_b is b_to_a


def test_bundle_link_same_site_hard_fails():
    b = load_bundle(MODELS, SITES, "compact_kv")
    with pytest.raises(ValueError, match="distinct"):
        b.link("phoenix", "phoenix")


def test_bundle_unknown_site_hard_fails():
    b = load_bundle(MODELS, SITES, "compact_kv")
    with pytest.raises(ValueError, match="unknown site"):
        b.site("no_such")
