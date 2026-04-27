import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Network assumptions
# ----------------------------

rtt_seconds = 0.030

# Sweep effective application goodput, not advertised peak bandwidth.
bandwidths_gbps = [0.1, 0.5, 1, 3, 10, 25, 40, 100, 400, 800]

# Effective wall-clock replay rate for rebuilding this one request's usable state.
prefill_rates_tok_s = [300, 1_000, 3_000, 10_000, 30_000, 100_000]

context_lengths = np.unique(np.round(np.logspace(3, 7, 80)).astype(int))

token_id_bytes = 4
batch_size = 1

# ----------------------------
# MoE model configs
# ----------------------------

models = {
    "2T_MoE_moderate_GQA": {
        "total_params": 2e12,
        "layers": 96,
        "kv_heads": 8,
        "head_dim": 128,
        "kv_dtype_bytes": 2,
    },
    "5T_MoE_large_GQA": {
        "total_params": 5e12,
        "layers": 128,
        "kv_heads": 16,
        "head_dim": 128,
        "kv_dtype_bytes": 2,
    },
}

# Optional: use this if modeling fp8/int8 KV instead of bf16/fp16.
# for m in models.values():
#     m["kv_dtype_bytes"] = 1

# Optional: set retained_context_cap if the model only preserves a sliding KV window.
# For full dense attention, leave as None.
retained_context_cap = None
# retained_context_cap = 128_000

# ----------------------------
# Helpers
# ----------------------------

def kv_bytes_per_token(cfg):
    return (
        2
        * cfg["layers"]
        * cfg["kv_heads"]
        * cfg["head_dim"]
        * cfg["kv_dtype_bytes"]
    )

def retained_tokens(tokens):
    if retained_context_cap is None:
        return tokens
    return min(tokens, retained_context_cap)

def kv_bytes(tokens, cfg):
    return batch_size * retained_tokens(tokens) * kv_bytes_per_token(cfg)

def context_bytes(tokens):
    return batch_size * tokens * token_id_bytes

def transfer_time_seconds(num_bytes, bandwidth_gbps, rtt_s):
    return rtt_s + (8 * num_bytes) / (bandwidth_gbps * 1e9)

rows = []

for model_name, cfg in models.items():
    kv_bpt = kv_bytes_per_token(cfg)

    for T in context_lengths:
        kv_b = kv_bytes(T, cfg)
        ctx_b = context_bytes(T)

        for bw in bandwidths_gbps:
            t_kv = transfer_time_seconds(kv_b, bw, rtt_seconds)
            t_ctx = transfer_time_seconds(ctx_b, bw, rtt_seconds)

            slack_for_recompute = t_kv - t_ctx
            breakeven_rate = T / slack_for_recompute if slack_for_recompute > 0 else np.inf

            rows.append({
                "model": model_name,
                "tokens": T,
                "retained_tokens": retained_tokens(T),
                "bandwidth_gbps": bw,
                "rtt_ms": rtt_seconds * 1000,
                "kv_bytes_per_token": kv_bpt,
                "context_MB": ctx_b / 1e6,
                "kv_GB": kv_b / 1e9,
                "context_transfer_s": t_ctx,
                "kv_transfer_s": t_kv,
                "prefill_tok_s": np.nan,
                "context_plus_recompute_s": np.nan,
                "breakeven_prefill_tok_s": breakeven_rate,
            })

            for rate in prefill_rates_tok_s:
                rows.append({
                    "model": model_name,
                    "tokens": T,
                    "retained_tokens": retained_tokens(T),
                    "bandwidth_gbps": bw,
                    "rtt_ms": rtt_seconds * 1000,
                    "kv_bytes_per_token": kv_bpt,
                    "context_MB": ctx_b / 1e6,
                    "kv_GB": kv_b / 1e9,
                    "context_transfer_s": t_ctx,
                    "kv_transfer_s": t_kv,
                    "prefill_tok_s": rate,
                    "context_plus_recompute_s": t_ctx + T / rate,
                    "breakeven_prefill_tok_s": breakeven_rate,
                })

df = pd.DataFrame(rows)
df.to_csv("moe_migration_kv_vs_context.csv", index=False)

# ----------------------------
# Print selected summary
# ----------------------------

summary_lengths = [128_000, 1_000_000, 10_000_000]
summary_bandwidths = [1, 10, 100, 400]

summary = (
    df[
        df["tokens"].isin(summary_lengths)
        & df["bandwidth_gbps"].isin(summary_bandwidths)
        & df["prefill_tok_s"].isna()
    ][[
        "model",
        "tokens",
        "retained_tokens",
        "bandwidth_gbps",
        "context_MB",
        "kv_GB",
        "context_transfer_s",
        "kv_transfer_s",
        "breakeven_prefill_tok_s",
    ]]
    .sort_values(["model", "tokens", "bandwidth_gbps"])
)

print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ----------------------------
# Plot 1: KV transfer time for both models
# ----------------------------

chosen_bw = 100

plt.figure(figsize=(8, 5))
for model_name in models:
    tmp = df[
        (df["model"] == model_name)
        & (df["bandwidth_gbps"] == chosen_bw)
        & (df["prefill_tok_s"].isna())
    ]
    plt.plot(tmp["tokens"], tmp["kv_transfer_s"], label=f"{model_name}, KV")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Prompt context length (tokens)")
plt.ylabel("KV transfer time (s)")
plt.title(f"Full KV migration at {chosen_bw} Gbps effective goodput")
plt.legend()
plt.tight_layout()
plt.savefig("moe_kv_transfer_time.png", dpi=200)

# ----------------------------
# Plot 2: context + replay vs KV for 5T model
# ----------------------------

chosen_model = "5T_MoE_large_GQA"
chosen_bw = 100

plt.figure(figsize=(8, 5))

base = df[
    (df["model"] == chosen_model)
    & (df["bandwidth_gbps"] == chosen_bw)
    & (df["prefill_tok_s"].isna())
]
plt.plot(base["tokens"], base["kv_transfer_s"], linewidth=3, label=f"Send full KV @ {chosen_bw} Gbps")

for rate in prefill_rates_tok_s:
    tmp = df[
        (df["model"] == chosen_model)
        & (df["bandwidth_gbps"] == chosen_bw)
        & (df["prefill_tok_s"] == rate)
    ]
    plt.plot(tmp["tokens"], tmp["context_plus_recompute_s"], label=f"Send context + replay @ {rate:,} tok/s")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Prompt context length (tokens)")
plt.ylabel("Migration completion time (s)")
plt.title(f"{chosen_model}: migration boundary comparison")
plt.legend()
plt.tight_layout()
plt.savefig("moe_5t_context_replay_vs_kv.png", dpi=200)

# ----------------------------
# Plot 3: break-even replay rate vs bandwidth
# ----------------------------

plt.figure(figsize=(8, 5))

for model_name, cfg in models.items():
    kv_bpt = kv_bytes_per_token(cfg)
    bw = np.array(bandwidths_gbps) * 1e9
    # ignore token bytes; they are tiny compared to KV
    breakeven = bw / (8 * (kv_bpt - token_id_bytes))
    plt.plot(bandwidths_gbps, breakeven, marker="o", label=model_name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Effective inter-site goodput (Gbps)")
plt.ylabel("Break-even replay rate (tokens/s)")
plt.title("When does context replay beat full KV transfer?")
plt.legend()
plt.tight_layout()
plt.savefig("moe_breakeven_replay_rate.png", dpi=200)

print("\nWrote:")
print("  moe_migration_kv_vs_context.csv")
print("  moe_kv_transfer_time.png")
print("  moe_5t_context_replay_vs_kv.png")
print("  moe_breakeven_replay_rate.png")