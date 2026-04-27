"""
Claim:
Category-filtered scoring computes complete active leaf weight divided by total
active leaf weight after the normal ledger leaf reduction, restricted to the
included subtask categories; absent categories default to product, and replay
preserves category semantics.

Plausible wrong implementations:
- Filter by category before determining active leaves, causing completed parents
  to reappear when their children are in another category.
- Use parent categories for split subtrees even when children override category.
- Average category progress ratios instead of aggregating complete and active
  weights at the active-leaf level.
- Count incomplete run-management leaves in overall progress but accidentally
  drop them from the filtered denominator.
- Serialize enum reprs or omit explicit categories during JSONL roundtrip.
"""

from ledger_progress import (
    EventType,
    LedgerEvent,
    LedgerSession,
    SubtaskCategory,
    apply_event,
    from_jsonl,
    new_ledger,
    score,
)
from ledger_progress.serialization import event_to_dict


CODING_CATEGORIES = {
    SubtaskCategory.PRODUCT,
    SubtaskCategory.VALIDATION,
    SubtaskCategory.INVESTIGATION,
}


def event(step, event_type, subtask_id, payload, reason=None):
    return LedgerEvent(step, event_type, subtask_id, payload, reason)


def test_existing_ledgers_without_category_default_to_product_and_score_the_same():
    ledger = new_ledger("Fix parser")
    apply_event(ledger, event(1, EventType.ADD_SUBTASK, "S1", {"description": "Patch parser"}))
    apply_event(ledger, event(1, EventType.ADD_SUBTASK, "S2", {"description": "Run tests"}))
    apply_event(ledger, event(2, EventType.UPDATE_STATUS, "S1", {
        "status": "complete",
        "evidence": ["Patch applied."],
    }))

    overall = score(ledger)
    product_only = score(ledger, categories={SubtaskCategory.PRODUCT})

    assert ledger.subtasks["S1"].category is SubtaskCategory.PRODUCT
    assert ledger.subtasks["S2"].category is SubtaskCategory.PRODUCT
    assert (overall.complete_weight, overall.active_weight, overall.progress) == (1.0, 2.0, 0.5)
    assert (product_only.complete_weight, product_only.active_weight, product_only.progress) == (1.0, 2.0, 0.5)
    assert product_only.categories_included == (SubtaskCategory.PRODUCT,)


def test_artifact_subtasks_affect_overall_progress_but_not_coding_progress():
    session = LedgerSession("Fix parser")
    product = session.add("Patch parser", step=1, category=SubtaskCategory.PRODUCT)
    artifact = session.add("Export ledger artifacts", step=1, category=SubtaskCategory.ARTIFACT)
    session.complete(product, "Patch applied.", step=2)

    overall = session.score()
    coding = session.score(categories=CODING_CATEGORIES)
    run_management = session.score(categories={SubtaskCategory.ARTIFACT, SubtaskCategory.DOCUMENTATION})

    assert session.ledger.subtasks[artifact].category is SubtaskCategory.ARTIFACT
    assert (overall.complete_weight, overall.active_weight, overall.progress) == (1.0, 2.0, 0.5)
    assert (coding.complete_weight, coding.active_weight, coding.progress) == (1.0, 1.0, 1.0)
    assert (run_management.complete_weight, run_management.active_weight, run_management.progress) == (0, 1.0, 0.0)
    assert coding.active_leaf_count == 1
    assert run_management.active_leaf_count == 1


def test_split_children_inherit_parent_category():
    session = LedgerSession("Fix parser")
    parent = session.add("Prepare environment", step=1, category=SubtaskCategory.ENVIRONMENT)

    children = session.split(parent, ["Install deps", "Run setup"], step=2, reason="Environment task was broad")

    assert [session.ledger.subtasks[child].category for child in children] == [
        SubtaskCategory.ENVIRONMENT,
        SubtaskCategory.ENVIRONMENT,
    ]
    assert session.score(categories={SubtaskCategory.ENVIRONMENT}).active_leaf_count == 2
    assert session.score(categories={SubtaskCategory.PRODUCT}).active_leaf_count == 0


def test_explicit_split_child_category_override_works():
    session = LedgerSession("Fix parser")
    parent = session.add("Finish change", step=1, category=SubtaskCategory.PRODUCT)

    product_child, docs_child = session.split(
        parent,
        ["Patch parser", "Update run notes"],
        step=2,
        reason="Separate product work from notes",
        categories=[None, SubtaskCategory.DOCUMENTATION],
    )

    assert session.ledger.subtasks[product_child].category is SubtaskCategory.PRODUCT
    assert session.ledger.subtasks[docs_child].category is SubtaskCategory.DOCUMENTATION
    assert session.score(categories={SubtaskCategory.PRODUCT}).active_leaf_count == 1
    assert session.score(categories={SubtaskCategory.DOCUMENTATION}).active_leaf_count == 1


def test_jsonl_roundtrip_preserves_category(tmp_path):
    session = LedgerSession("Fix parser")
    artifact = session.add("Export progress CSV", step=1, category=SubtaskCategory.ARTIFACT)
    session.complete(artifact, "progress.csv written.", step=2)
    path = tmp_path / "ledger.jsonl"

    session.export_jsonl(str(path))
    loaded = from_jsonl(str(path))

    assert loaded.subtasks[artifact].category is SubtaskCategory.ARTIFACT
    assert score(loaded, categories={SubtaskCategory.ARTIFACT}) == session.score(categories={SubtaskCategory.ARTIFACT})
    assert event_to_dict(session.ledger.events[1])["payload"]["category"] == "artifact"


def test_category_filtering_happens_after_global_active_leaf_reduction():
    ledger = new_ledger("Fix parser")
    apply_event(ledger, event(1, EventType.ADD_SUBTASK, "P", {
        "description": "Completed product parent",
        "category": "product",
        "weight": 10,
    }))
    apply_event(ledger, event(2, EventType.UPDATE_STATUS, "P", {
        "status": "complete",
        "evidence": ["Parent looked complete before validation was split out."],
    }))
    apply_event(ledger, event(3, EventType.SPLIT_SUBTASK, "P", {"children": [
        {"id": "V1", "description": "Cross-category validation leaf", "category": "validation", "weight": 2},
    ]}))

    product = score(ledger, categories={SubtaskCategory.PRODUCT})
    validation = score(ledger, categories={SubtaskCategory.VALIDATION})
    overall = score(ledger)

    assert (product.complete_weight, product.active_weight, product.progress, product.active_leaf_count) == (0, 0, 0.0, 0)
    assert (validation.complete_weight, validation.active_weight, validation.progress, validation.active_leaf_count) == (0, 2, 0.0, 1)
    assert (overall.complete_weight, overall.active_weight, overall.progress, overall.active_leaf_count) == (0, 2, 0.0, 1)


def test_mixed_category_split_uses_each_child_category_and_weight():
    ledger = new_ledger("Fix parser")
    apply_event(ledger, event(1, EventType.ADD_SUBTASK, "P", {
        "description": "Finish broad change",
        "category": "product",
        "weight": 9,
    }))
    apply_event(ledger, event(2, EventType.SPLIT_SUBTASK, "P", {"children": [
        {"id": "P.1", "description": "Patch parser", "category": "product", "weight": 3},
        {"id": "P.2", "description": "Run regression suite", "category": "validation", "weight": 2},
        {"id": "P.3", "description": "Export progress ledger", "category": "artifact", "weight": 5},
    ]}, "Broad work decomposed into coding and run-management leaves"))
    apply_event(ledger, event(3, EventType.UPDATE_STATUS, "P.1", {
        "status": "complete",
        "evidence": ["Parser patched."],
    }))
    apply_event(ledger, event(4, EventType.UPDATE_STATUS, "P.3", {
        "status": "complete",
        "evidence": ["Ledger exported."],
    }))

    coding = score(ledger, categories=CODING_CATEGORIES)
    run_management = score(ledger, categories={SubtaskCategory.ARTIFACT, SubtaskCategory.DOCUMENTATION})
    overall = score(ledger)

    assert (coding.complete_weight, coding.active_weight, coding.progress, coding.active_leaf_count) == (3, 5, 3 / 5, 2)
    assert (run_management.complete_weight, run_management.active_weight, run_management.progress, run_management.active_leaf_count) == (5, 5, 1.0, 1)
    assert (overall.complete_weight, overall.active_weight, overall.progress, overall.active_leaf_count) == (8, 10, 4 / 5, 3)


def test_overall_progress_is_weighted_category_recomposition_not_mean_of_category_ratios():
    session = LedgerSession("Fix parser")
    product = session.add("Patch parser", step=1, weight=6, category=SubtaskCategory.PRODUCT)
    validation = session.add("Run validation", step=1, weight=2, category=SubtaskCategory.VALIDATION)
    artifact = session.add("Export artifact", step=1, weight=1, category=SubtaskCategory.ARTIFACT)
    documentation = session.add("Write run notes", step=1, weight=3, category=SubtaskCategory.DOCUMENTATION)
    environment = session.add("Prepare optional environment path", step=1, weight=100, category=SubtaskCategory.ENVIRONMENT)
    session.complete(product, "Patch applied.", step=2)
    session.complete(artifact, "Artifact exported.", step=2)
    session.invalidate(environment, step=3, reason="Environment path was not needed")

    coding = session.score(categories=CODING_CATEGORIES)
    run_management = session.score(categories={SubtaskCategory.ARTIFACT, SubtaskCategory.DOCUMENTATION})
    overall = session.score(categories={
        SubtaskCategory.PRODUCT,
        SubtaskCategory.VALIDATION,
        SubtaskCategory.ARTIFACT,
        SubtaskCategory.DOCUMENTATION,
    })

    assert (coding.complete_weight, coding.active_weight, coding.progress) == (6, 8, 3 / 4)
    assert (run_management.complete_weight, run_management.active_weight, run_management.progress) == (1, 4, 1 / 4)
    assert (overall.complete_weight, overall.active_weight, overall.progress) == (7, 12, 7 / 12)
    assert overall.progress != (coding.progress + run_management.progress) / 2
