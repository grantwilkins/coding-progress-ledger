from repair_controller import (
    Assignment, Attempt, AttemptUpdate, Failure, FeasibilityRepairController,
    ObservationBatch, ProposedDiff, RepairMove, RepairRequest, RepairResult,
    RevisedMaximum,
)


A = Assignment("kv_transfer", "d0", "p0")
B = Assignment("replay", "d0", "p0")


def controller(planned=5):
    attempt = Attempt("s", 0, A, "running", 100, 10, 0, planned)
    return FeasibilityRepairController(
        (attempt,), {"s", "other"}, 10, 10, 1,
        lambda sessions: 10 * len(sessions),
    )


def update(sample, completed=10, status="running", generation=0):
    return ObservationBatch(
        sample, float(sample),
        (AttemptUpdate("s", generation, status, 100, completed),),
    )


def test_soft_repair_fires_once_after_two_miss_batches():
    c = controller()

    assert c.observe(update(1)) is None
    request = c.observe(update(2))

    assert isinstance(request, RepairRequest)
    assert request.trigger == "soft:1"
    assert c.observe(update(3)) is None


def test_stale_attempt_generation_is_ignored():
    c = controller()
    c.attempts["s"] = Attempt("s", 1, A, "running", 100, 20, 0, 5)

    assert c.observe(update(1, 99, generation=0)) is None
    assert c.attempts["s"].completed_work == 20


def test_hard_failure_bypasses_soft_hysteresis_and_is_deduplicated():
    c = controller(planned=20)
    batch = ObservationBatch(1, 1, failures=(Failure("lost", "worker", ("s",)),))

    request = c.observe(batch)

    assert isinstance(request, RepairRequest)
    assert request.trigger == "hard:lost"


def test_only_target_restoring_result_proposes_a_diff():
    c = controller()
    c.observe(update(1)); request = c.observe(update(2))
    miss = c.complete_repair(RepairResult(
        request.request_id, c.budget_version, (), 0, False,
    ), 2.1)

    assert isinstance(miss, RevisedMaximum)

    c = controller()
    c.observe(update(1)); request = c.observe(update(2))
    proposal = c.complete_repair(RepairResult(
        request.request_id, c.budget_version,
        (RepairMove("other", B, 1, 100),), 10, True,
    ), 2.1)

    assert isinstance(proposal, ProposedDiff)
    assert {row.session_id for row in proposal.changes} == {"other", "s"}


def test_applied_repair_bumps_generation_and_waits_for_progress():
    c = controller()
    c.observe(update(1)); request = c.observe(update(2))
    proposal = c.complete_repair(RepairResult(
        request.request_id, c.budget_version,
        (RepairMove("s", B, 1, 100),), 10, True,
    ), 2.1)

    c.acknowledge(proposal.proposal_id, "applied", 2.2)

    assert c.attempts["s"].generation == 1
    assert c.attempts["s"].assignment == B
    assert c.progress_gate == {"s"}
