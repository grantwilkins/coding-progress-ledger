from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thought_summary", "action"],
    "properties": {
        "thought_summary": {"type": "string"},
        "action": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "type",
                "path",
                "command",
                "content",
                "instruction",
                "summary",
                "start_line",
                "end_line",
                "pattern",
                "file_glob",
                "unified_diff",
            ],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "list_dir",
                        "find_files",
                        "grep",
                        "read_file",
                        "write_file",
                        "edit_file",
                        "apply_patch",
                        "shell",
                        "done",
                    ],
                },
                "path": {"type": ["string", "null"]},
                "command": {"type": ["string", "null"]},
                "content": {"type": ["string", "null"]},
                "instruction": {"type": ["string", "null"]},
                "summary": {"type": ["string", "null"]},
                "start_line": {"type": ["integer", "null"], "minimum": 1},
                "end_line": {"type": ["integer", "null"], "minimum": 1},
                "pattern": {"type": ["string", "null"]},
                "file_glob": {"type": ["string", "null"]},
                "unified_diff": {"type": ["string", "null"]},
            },
        },
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI Responses API adapter for ProviderModelClient.")
    parser.add_argument("--model", help="Override request model name.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=OPENAI_RESPONSES_URL)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--reasoning-effort", default="low")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.stderr.write(f"{args.api_key_env} is not set\n")
        return 2

    request_payload = json.loads(sys.stdin.read())
    payload = build_openai_payload(request_payload, model_override=args.model, reasoning_effort=args.reasoning_effort)
    try:
        response = call_openai(args.base_url, api_key=api_key, payload=payload, timeout_s=args.timeout_s)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"OpenAI HTTP {exc.code}: {body[-2000:]}\n")
        return 1
    except urllib.error.URLError as exc:
        sys.stderr.write(f"OpenAI request failed: {exc}\n")
        return 1

    action = extract_action(response)
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    requested_model = payload["model"]
    print(
        json.dumps(
            {
                "action": action,
                "usage": {
                    "tokens_in": usage.get("input_tokens", 0),
                    "tokens_out": usage.get("output_tokens", 0),
                    "estimated_cost_usd": None,
                },
                "provider": {
                    "adapter": "openai_responses",
                    "base_url": args.base_url,
                    "requested_model": requested_model,
                    "resolved_model": response.get("model") or requested_model,
                    "fallback_models": [],
                    "fallback_used": False,
                    "model_alias_or_route_changed": bool(response.get("model") and response.get("model") != requested_model),
                    "response_id": response.get("id"),
                    "provider_routing_policy": {"allow_fallbacks": False},
                },
            },
            sort_keys=True,
        )
    )
    return 0


def build_openai_payload(
    request_payload: dict[str, Any],
    *,
    model_override: str | None = None,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    model = model_override or request_payload.get("model", {}).get("name") or "gpt-5.1"
    max_tokens_out = int(request_payload.get("model", {}).get("max_tokens_out") or 2048)
    user_payload = {
        "task_prompt": request_payload.get("task_prompt", ""),
        "transcript_prefix": request_payload.get("transcript_prefix", []),
        "tool_specs": request_payload.get("tool_specs", []),
        "budget_state": request_payload.get("budget_state", {}),
    }
    return {
        "model": model,
        "instructions": request_payload.get("system_prompt", ""),
        "input": json.dumps(user_payload, sort_keys=True),
        "store": False,
        "max_output_tokens": max_tokens_out,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "model_tool_loop_action",
                "strict": True,
                "schema": ACTION_SCHEMA,
            },
            "verbosity": "low",
        },
    }


def call_openai(url: str, *, api_key: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_action(response: dict[str, Any]) -> dict[str, Any]:
    text = response.get("output_text")
    if isinstance(text, str) and text.strip():
        return json.loads(text)
    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunk = content["text"].strip()
                if not chunk:
                    continue
                try:
                    payload = json.loads(chunk)
                except json.JSONDecodeError:
                    chunks.append(chunk)
                    continue
                if _looks_like_model_action(payload):
                    return payload
                chunks.append(chunk)
    for chunk in chunks:
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if _looks_like_model_action(payload):
            return payload
    if chunks:
        return json.loads("".join(chunks))
    else:
        raise ValueError("OpenAI response did not include output text")


def _looks_like_model_action(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("action"), dict)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
