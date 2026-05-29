"""Convenience wrapper: run Stages 1->2->3 on an instance."""

from __future__ import annotations

from dataclasses import dataclass

from instance import ProblemInstance
from stage1 import Stage1Result, solve_stage1
from stage2 import Stage2Result, solve_stage2
from stage3 import Stage3Result, solve_stage3


@dataclass(frozen=True)
class PipelineResult:
    inst: ProblemInstance
    s1: Stage1Result
    s2: Stage2Result
    s3: Stage3Result


def run_pipeline(inst: ProblemInstance) -> PipelineResult:
    s1 = solve_stage1(inst)
    s2 = solve_stage2(inst, s1)
    s3 = solve_stage3(inst, s2)
    return PipelineResult(inst=inst, s1=s1, s2=s2, s3=s3)
