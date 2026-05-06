from agent_migrate_agent import events


def test_event_constants_distinct():
    assert len(set(events.ALL)) == 8


def test_event_constants_are_strings():
    for name in events.ALL:
        assert isinstance(name, str) and name


def test_no_overlap_with_ledger_event_types():
    from ledger_progress import EventType

    ledger_values = {e.value for e in EventType}
    assert ledger_values.isdisjoint(set(events.ALL))


def test_state_layers_cover_plan_taxonomy():
    expected = {"model_execution", "prompt_context", "subagent", "workspace", "memory", "semantic"}
    assert set(events.STATE_LAYERS) == expected


def test_materialization_modes_cover_plan_taxonomy():
    expected = {
        "warm_reuse", "kv_transfer", "context_replay", "text_transfer",
        "artifact_copy", "workspace_hydrate", "remote_workspace", "summary", "restart",
    }
    assert set(events.MATERIALIZATION_MODES) == expected
