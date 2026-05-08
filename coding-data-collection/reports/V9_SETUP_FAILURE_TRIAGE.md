# V9 Setup Failure Triage

Environment setup failures are excluded from model outcome metrics and should not be counted as model failures.

| task_id | failed_checks | class | V10 decision |
| --- | --- | --- | --- |
| adaptive-rejection-sampler | hidden_image_artifacts_unreadable, r_runtime_available | hidden_artifact_leakage_risk | exclude from V10 until image rebuild removes or protects hidden paths |
| blind-maze-explorer-algorithm | hidden_image_artifacts_unreadable | hidden_artifact_leakage_risk | exclude from V10 until image rebuild removes or protects hidden paths |
| classifier-debug | python_imports_available | dependency_prebuild_required | fixable by image prebuild, but exclude until required imports pass |
| nginx-request-logging | nginx_available | dependency_prebuild_required | fixable by image prebuild, but exclude until nginx preflight passes |

## Policy

V10 excludes tasks with readable hidden image artifacts until the image is rebuilt or the hidden path is proven inaccessible from the agent container. Dependency-only failures may be reintroduced after an explicit image/dependency prebuild smoke passes.
