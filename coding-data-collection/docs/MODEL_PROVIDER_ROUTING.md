# Model Provider Routing

This harness supports provider-backed model loops through a provider adapter
command. The adapter receives one JSON request on stdin and returns one model
tool action on stdout.

## OpenRouter / OpenAI-Compatible Adapter

Use:

```text
scripts/openai_compatible_model_client.py
```

Default OpenRouter settings:

```text
base_url=https://openrouter.ai/api/v1
api_key_env=OPENROUTER_API_KEY
endpoint=/chat/completions
```

Example smoke command:

```bash
python scripts/openai_compatible_model_client.py \
  --model baidu/cobuddy:free \
  --response-format json_object \
  --no-allow-fallbacks
```

The adapter follows the current OpenRouter Chat Completions shape:

```text
model=<requested model id>
models=[primary, fallback...] when explicit fallbacks are configured
provider.allow_fallbacks=false by default
provider.require_parameters=true when strict parameter support is required
provider.data_collection=allow|deny when a data policy is specified
response.model is recorded as the resolved model
response.usage is recorded for token/cost accounting
```

Reference docs:

```text
https://openrouter.ai/docs/api-reference/chat-completion
https://openrouter.ai/docs/model-routing
https://openrouter.ai/docs/features/provider-routing
https://openrouter.ai/docs/use-cases/usage-accounting
```

## Provenance Rules

`OpenRouter` is a routing provider, not a model. Every provider-backed run must
record:

```text
model_provider
model_name
provider_calls
last_provider_call
resolved_models
fallback_call_count
```

For OpenRouter calls, each provider call records:

```text
requested_model
resolved_model
fallback_models
fallback_used
model_alias_or_route_changed
provider_routing_policy
response_id
base_url
```

## Comparable Route Policy

Allowed for comparable model-agent traces:

```text
fixed model id
allow_fallbacks=false
fallback_used=false
resolved_model recorded
```

OpenRouter may resolve a stable model alias such as `baidu/cobuddy:free` to a
dated slug. That is recorded as `model_alias_or_route_changed=true`; it is not
counted as explicit fallback unless a configured fallback model is used.

Use only for scouting/debug unless separately justified:

```text
openrouter/auto
multiple fallback models
allow_fallbacks=true
unknown or changing resolved_model
```

`baidu/cobuddy:free` is only for end-to-end smoke testing the adapter and
controller path. It should not be used as a benchmark-quality collection route.
