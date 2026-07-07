  # Queue-Haul Stage 1b Two-Instance Drain/Sink Testbed

  ## Summary

  Build a minimal gpt-oss-20b Docker proof with two vLLM instances on two GPUs: source on GPU 0, sink on GPU 1. Shape source egress with host tc, run replay first, then
  LMCache KV transfer, and reduce the run into deadline and solver-expectation metrics.

  ## Key Changes

  - Add one script: queue-haul/stage1b_drain_sink.py with runbook, driver, and reduce subcommands.
  - Generated runbook starts:
      - lmcache/vllm-openai:latest-cu129 by default, configurable with --image.
      - source vLLM on port 8100, sink vLLM on port 8200.
      - model openai/gpt-oss-20b, --max-model-len 32768, one GPU per container.
      - Docker bridge network, no host networking.

  - Apply shaping from the host into the source container netns:
      - nsenter -t $(docker inspect -f '{{.State.Pid}}' qh-source) -n tc qdisc replace dev eth0 root tbf rate <rate> burst <burst> latency <latency>.
      - Use tc because Linux shaping is egress-side.

  - Replay mode:
      - warm source with controlled prompts;
      - at t=0, run the driver inside the source container and POST selected prompts to the sink OpenAI-compatible endpoint;
      - record source eth0 tx bytes, streaming TTFT, completion time, and sink response.

  - KV mode:
      - start lmcache server as sink-side shared state;
      - run both vLLM instances with LMCacheMPConnector, using the LMCache gpt-oss recipe;
      - source stores KV to LMCache over the shaped link, sink retrieves/reuses it locally;
      - record source tx bytes, sink TTFT, and LMCache/vLLM logs.

  ## Validation

  - Reducer writes queue-haul/outputs/stage1b_gpt_oss_20b_drain_sink.csv with:
      - session_id, mode, T, rate_Bps, deadline_s
      - expected egress/rebuild/admission times from Queue-Haul movement formulas
      - measured source tx bytes, measured first-token/resume latency, completion time
      - egress_hit, rebuild_hit, matches_solver_expectation

  - Hard fail malformed runs: missing sessions, nonpositive bandwidth, missing source tx counters, missing first-token timing, or duplicate vLLM flags.
  - Acceptance: a row passes when measured resume time is within max(2s, 20%) of predicted deadline-safe behavior and the solver-expected hit/miss classification matches the
    measured hit/miss.

  ## Tests

  Use research-test-creator style tests in queue-haul/tests/test_stage1b_drain_sink.py:

  - runbook pins source/sink to distinct GPUs and shapes source eth0, not host loopback;
  - replay/KV expected times use bytes divided by shaped bandwidth plus the correct rebuild term;
  - reducer aligns rows by session_id and hard fails missing timing/counter fields;
  - hit/miss comparison fails if HTTP success is counted without deadline-realized first-token timing.

  Run uv run pytest after implementation, then update README.md with the new runbook command and commit all changes.

  ## Assumptions

  - Use two NVIDIA GPUs with host NET_ADMIN.
  - Preserve current dirty files unless explicitly touched by this task.
  - Use HF_TOKEN if the model is not already cached.
  - Sources checked: vLLM Docker docs, vLLM OpenAI-compatible server docs, LMCache installation/gpt-oss docs, Hugging Face openai/gpt-oss-20b model card, and Linux tc manual.

