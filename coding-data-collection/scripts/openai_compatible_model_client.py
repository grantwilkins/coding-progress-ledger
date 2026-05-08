from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible Chat Completions adapter for ProviderModelClient."
    )
    parser.add_argument("--model", help="Override request model name.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", DEFAULT_OPENROUTER_BASE_URL))
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--fallback-model", action="append", default=[])
    parser.add_argument("--allow-fallbacks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-collection", choices=["allow", "deny"])
    parser.add_argument("--provider-order", action="append", default=[])
    parser.add_argument("--provider-only", action="append", default=[])
    parser.add_argument("--provider-ignore", action="append", default=[])
    parser.add_argument("--max-price-input")
    parser.add_argument("--max-price-output")
    parser.add_argument(
        "--response-format",
        choices=["none", "json_object", "json_schema"],
        default="json_object",
        help="Use json_schema only with models/providers known to support it.",
    )
    parser.add_argument("--referer")
    parser.add_argument("--title", default="coding-data-collection")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.stderr.write(f"{args.api_key_env} is not set\n")
        return 2

    request_payload = json.loads(sys.stdin.read())
    payload = build_chat_payload(request_payload, args=args)
    try:
        response = call_chat_completion(
            _chat_completions_url(args.base_url),
            api_key=api_key,
            payload=payload,
            timeout_s=args.timeout_s,
            referer=args.referer,
            title=args.title,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"OpenAI-compatible HTTP {exc.code}: {body[-2000:]}\n")
        return 1
    except urllib.error.URLError as exc:
        sys.stderr.write(f"OpenAI-compatible request failed: {exc}\n")
        return 1

    try:
        action = extract_action(response)
    except ValueError as exc:
        sys.stderr.write(f"{exc}: {_response_excerpt(response)}\n")
        return 1
    usage = _usage_payload(response.get("usage"))
    provider = _provider_payload(
        response,
        requested_model=payload.get("model"),
        fallback_models=list(args.fallback_model),
        provider_policy=payload.get("provider"),
        base_url=args.base_url,
    )
    print(json.dumps({"action": action, "usage": usage, "provider": provider}, sort_keys=True))
    return 0


def build_chat_payload(request_payload: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    model = args.model or request_payload.get("model", {}).get("name") or "openrouter/auto"
    max_tokens_out = int(request_payload.get("model", {}).get("max_tokens_out") or 2048)
    user_payload = {
        "task_prompt": request_payload.get("task_prompt", ""),
        "transcript_prefix": request_payload.get("transcript_prefix", []),
        "tool_specs": request_payload.get("tool_specs", []),
        "budget_state": request_payload.get("budget_state", {}),
    }
    provider = _provider_policy(args)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt(request_payload.get("system_prompt", ""))},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ],
        "temperature": float(request_payload.get("model", {}).get("temperature") or 0.0),
        "max_tokens": max_tokens_out,
    }
    if args.fallback_model:
        payload["models"] = [model, *args.fallback_model]
    if provider:
        payload["provider"] = provider
    if args.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif args.response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "model_tool_loop_action",
                "strict": True,
                "schema": ACTION_SCHEMA,
            },
        }
    return payload


def call_chat_completion(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout_s: int,
    referer: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_action(response: dict[str, Any]) -> dict[str, Any]:
    texts: list[str] = []
    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"].strip())
    for text in texts:
        for payload in _jsonish_payloads(text):
            if _looks_like_model_action(payload):
                return payload
    if texts:
        for payload in _jsonish_payloads("\n".join(texts)):
            if _looks_like_model_action(payload):
                return payload
    raise ValueError("chat completion response did not include a model action JSON object")


def _system_prompt(system_prompt: str) -> str:
    return (
        system_prompt
        + "\n\nReturn exactly one JSON object and no prose. Do not echo a JSON schema. "
        + "Do not include keys such as properties, required, or additionalProperties. "
        + "Use this concrete shape:\n"
        + '{"thought_summary":"Inspect the task.","action":{"type":"read_file","path":"task.md",'
        + '"command":null,"content":null,"instruction":null,"summary":null,'
        + '"start_line":null,"end_line":null,"pattern":null,"file_glob":null,"unified_diff":null}}\n'
        + "Allowed action.type values: list_dir, find_files, grep, read_file, write_file, "
        + "edit_file, apply_patch, shell, done."
    )


def _provider_policy(args: argparse.Namespace) -> dict[str, Any]:
    provider: dict[str, Any] = {"allow_fallbacks": bool(args.allow_fallbacks)}
    if args.require_parameters:
        provider["require_parameters"] = True
    if args.data_collection:
        provider["data_collection"] = args.data_collection
    if args.provider_order:
        provider["order"] = args.provider_order
    if args.provider_only:
        provider["only"] = args.provider_only
    if args.provider_ignore:
        provider["ignore"] = args.provider_ignore
    max_price: dict[str, float] = {}
    if args.max_price_input:
        max_price["input"] = float(args.max_price_input)
    if args.max_price_output:
        max_price["output"] = float(args.max_price_output)
    if max_price:
        provider["max_price"] = max_price
    return provider


def _usage_payload(usage: Any) -> dict[str, Any]:
    usage = usage if isinstance(usage, dict) else {}
    cost = usage.get("cost")
    return {
        "tokens_in": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "tokens_out": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "estimated_cost_usd": None,
        "cost_credits": cost,
        "total_tokens": usage.get("total_tokens"),
        "raw_usage": usage,
    }


def _provider_payload(
    response: dict[str, Any],
    *,
    requested_model: Any,
    fallback_models: list[str],
    provider_policy: Any,
    base_url: str,
) -> dict[str, Any]:
    resolved_model = response.get("model")
    requested = str(requested_model or "")
    resolved = str(resolved_model or "")
    explicit_fallback_used = bool(
        fallback_models
        and resolved
        and requested
        and resolved != requested
        and not _same_model_alias(resolved, requested)
    )
    return {
        "adapter": "openai_compatible_chat_completions",
        "base_url": base_url,
        "requested_model": requested,
        "fallback_models": fallback_models,
        "resolved_model": resolved_model,
        "model_alias_or_route_changed": bool(resolved and requested and resolved != requested),
        "fallback_used": explicit_fallback_used,
        "response_id": response.get("id"),
        "provider_routing_policy": provider_policy,
    }


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _parse_jsonish(text: str) -> Any:
    payloads = _jsonish_payloads(text)
    if payloads:
        return payloads[0]
    raise ValueError("no JSON object found")


def _jsonish_payloads(text: str) -> list[Any]:
    candidates = [_strip_json_fence(text.strip())]
    stripped = candidates[0]
    for index, char in enumerate(stripped):
        if char == "{":
            candidates.append(stripped[index:])
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payloads.append(json.loads(candidate))
            continue
        except json.JSONDecodeError:
            pass
        try:
            payload, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        payloads.append(payload)
    return payloads


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _looks_like_model_action(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("action"), dict)


def _same_model_alias(resolved: str, requested: str) -> bool:
    requested_base = requested.removesuffix(":free")
    resolved_base = resolved.removesuffix(":free")
    return resolved_base == requested_base or resolved_base.startswith(requested_base + "-")


def _response_excerpt(response: dict[str, Any]) -> str:
    excerpts: list[str] = []
    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            excerpts.append(str(message.get("content"))[:500])
    return json.dumps(
        {
            "id": response.get("id"),
            "model": response.get("model"),
            "content_excerpt": excerpts[:2],
        },
        sort_keys=True,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
