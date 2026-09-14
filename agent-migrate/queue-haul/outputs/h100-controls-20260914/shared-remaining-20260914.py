import json, os, subprocess
from pathlib import Path
rows = []
for slug, model in (("qwen", "Qwen/Qwen3.8-27B"), ("gemma", "google/gemma-4-26B-A4B-it")):
    command = ["/datadrive/qh0912/.venv/bin/python", "shared_load_controls.py", "--model", model, "--cluster", "/datadrive/shared-controls-cluster.json", "--calibration", "/datadrive/c12.json", "--resident-rps", "0.25", "--run-root", f"/datadrive/shared-{slug}-20260914"]
    if slug == "qwen": command += ["--state-reference", "/datadrive/qf20/full_state_control.json"]
    print(json.dumps({"event": "start", "model": model}), flush=True)
    result = subprocess.run(command, env={**os.environ, "QH_MODEL_PROFILE": f"profiles/network-{slug}-h100.json"})
    rows.append({"model": model, "returncode": result.returncode})
    Path("/datadrive/shared-remaining-20260914.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows[-1]), flush=True)
if any(row["returncode"] for row in rows): raise SystemExit(1)
