from ledger_progress import (
    LedgerSession,
    SubtaskCategory,
    active_incomplete_coding_leaves,
    active_incomplete_leaves,
)
from scripts.list_active_incomplete_coding_leaves import main as list_active_incomplete_main


def test_active_incomplete_coding_leaves_excludes_completed_artifacts_and_split_parents():
    session = LedgerSession("Finish report")
    product = session.add("Patch behavior", step=1, category=SubtaskCategory.PRODUCT)
    validation = session.add("Run validation", step=1, category=SubtaskCategory.VALIDATION)
    artifact = session.add("Export artifact bundle", step=1, category=SubtaskCategory.ARTIFACT)
    docs = session.add("Document behavior", step=1, category=SubtaskCategory.DOCUMENTATION)
    session.complete(product, "final_diff.patch shows patch", step=2)
    parent = session.add("Finish broad coding work", step=3, category=SubtaskCategory.PRODUCT)
    child_done, child_remaining = session.split(
        parent,
        ["Patch first branch", "Patch second branch"],
        step=4,
        reason="Broad coding work split into leaves",
    )
    session.complete(child_done, "final_diff.patch shows first branch", step=5)

    leaves = active_incomplete_coding_leaves(session.ledger)

    assert [leaf.id for leaf in leaves] == [validation, child_remaining]
    assert artifact not in [leaf.id for leaf in leaves]
    assert docs not in [leaf.id for leaf in leaves]
    assert parent not in [leaf.id for leaf in leaves]


def test_active_incomplete_leaves_can_select_non_coding_categories():
    session = LedgerSession("Finish artifacts")
    artifact = session.add("Export artifact bundle", step=1, category=SubtaskCategory.ARTIFACT)
    docs = session.add("Document behavior", step=1, category=SubtaskCategory.DOCUMENTATION)
    product = session.add("Patch behavior", step=1, category=SubtaskCategory.PRODUCT)
    session.complete(docs, "README.md updated", step=2)
    session.invalidate(product, step=3, reason="Product change was not needed")

    leaves = active_incomplete_leaves(session.ledger, categories={SubtaskCategory.ARTIFACT, "documentation"})

    assert [leaf.id for leaf in leaves] == [artifact]


def test_active_incomplete_coding_leaves_cli_outputs_json(tmp_path, capsys):
    session = LedgerSession("Finish report")
    product = session.add("Patch behavior", step=1, category=SubtaskCategory.PRODUCT)
    validation = session.add("Run validation", step=1, category=SubtaskCategory.VALIDATION)
    session.complete(product, "final_diff.patch shows patch", step=2)
    path = tmp_path / "ledger.jsonl"
    session.export_jsonl(str(path))

    assert list_active_incomplete_main([str(path)]) == 0

    output = capsys.readouterr().out
    assert f'"id": "{validation}"' in output
    assert '"category": "validation"' in output
