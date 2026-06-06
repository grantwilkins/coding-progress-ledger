# Proportional-Fair Stage 1

## Formulation

$$
\max_{x^R, x^S, z \ge 0} \quad U(z) = \sum_{q} n_q \log\left(1 - \frac{z_q}{n_q}\right)
$$

subject to

$$
\begin{aligned}
\sum_{\ell} (x^R_{q\ell} + x^S_{q\ell}) + z_q &= n_q \quad \forall q \\
\sum_{q} (\beta_q T_q x^R_{q\ell} + \eta_q T_q x^S_{q\ell}) &\le \Lambda_\ell D \quad \forall \ell \\
\sum_{q \in \mathcal{Q}_m} \frac{T_q}{\rho_q} x^R_{q\ell} &\le W_{\ell m} D \quad \forall \ell, m \\
\sum_{q \in \mathcal{Q}_m} \eta_q T_q x^S_{q\ell} &\le W_{\ell m} \mu D \quad \forall \ell, m
\end{aligned}
$$

with $x^R_{q\ell}, x^S_{q\ell}, z_q \ge 0$.

The objective is proportional fairness on the migrated fraction $1 - z_q/n_q$, weighted by class size $n_q$. Equivalent minimization:

$$
\min_{x, z \ge 0} \; -U(z) = \sum_q n_q \log\frac{n_q}{n_q - z_q}
$$

The optimum $U^\star$ is carried into later stages as the constraint $U(z) \ge U^\star - \delta$ for a small slack $\delta$.

## Variables and parameters

- $x^R_{q\ell}$ — class-$q$ jobs rebuilt at destination $\ell$ by replay (ship context, recompute KV)
- $x^S_{q\ell}$ — class-$q$ jobs rebuilt at destination $\ell$ by state transfer (ship KV cache)
- $z_q$ — stranded jobs of class $q$ (miss the deadline)
- $n_q$ — number of jobs in class $q$
- $T_q$ — context length of class $q$ (tokens)
- $\beta_q$ — context bytes per token
- $\eta_q$ — KV bytes per token
- $\rho_q$ — prefill rate at $T_q$ (tokens/s)
- $\Lambda_\ell$ — network ingress at destination $\ell$ (bytes/s)
- $W_{\ell m}$ — warm instances of model $m$ at destination $\ell$
- $\mu$ — state-ingest rate per warm instance (bytes/s)
- $D$ — deadline (s)
- $\mathcal{Q}$ — classes; $\mathcal{Q}_m$ — classes of model $m$; $\mathcal{L}$ — destinations

## Base simulation values

GB = 1e9 bytes, KiB = 1024 bytes.

- $N = 10{,}000$ jobs (also run at 20,000)
- $D = 300$ s (swept 1 to 900)
- 6 models, 3 destinations
- $\beta_q = 4$ bytes/token for all classes
- $\mu = 512$ GB/s per instance (8 GPUs x 64 GB/s PCIe Gen5)
- $T_q$ clipped to $[10^3, 10^6]$ tokens
- 5 token buckets per model, so up to 30 classes
- 39 resource constraints (3 network + 18 prefill + 18 ingest)
- seed = 42

### Models

Each model gets round(N x fraction) jobs with token length drawn from Lognormal(ln(median), sigma^2), clipped to [1e3, 1e6], then binned into 5 log-spaced buckets. Per class, $n_q$ is the bucket count and $T_q$ is the bucket mean. $\eta_q$ in bytes/token = (KiB/tok) x 1024.

| Model | eta (KiB/tok) | fraction | jobs | median tok | sigma |
|---|---|---|---|---|---|
| DeepSeek V4 Pro | 9.7 | 0.25 | 2500 | 8000 | 1.5 |
| Kimi K2.6 | 68.6 | 0.25 | 2500 | 12000 | 1.8 |
| GLM 5 | 87.8 | 0.15 | 1500 | 6000 | 1.4 |
| Qwen3 235B | 188.0 | 0.15 | 1500 | 15000 | 1.9 |
| Qwen3.5 397B | 30.0 | 0.15 | 1500 | 20000 | 1.7 |
| Qwen3 Next 80B | 24.0 | 0.05 | 500 | 5000 | 1.3 |

### Prefill rate anchors (tokens/s)

$\rho_q$ is log-log interpolated in $T$ from these four anchors.

| Model | 1k | 10k | 100k | 1M |
|---|---|---|---|---|
| DeepSeek V4 Pro | 28000 | 25600 | 13900 | 2500 |
| Kimi K2.6 | 42500 | 36200 | 14700 | 2100 |
| GLM 5 | 33600 | 26200 | 8300 | 1100 |
| Qwen3 235B | 60800 | 46600 | 14000 | 1700 |
| Qwen3.5 397B | 80900 | 76000 | 47300 | 9900 |
| Qwen3 Next 80B | 454300 | 396800 | 175000 | 26600 |

### Destinations

| Site | Lambda (GB/s) | DeepSeek | Kimi | GLM | Qwen3-235B | Qwen3.5-397B | Qwen3-Next-80B |
|---|---|---|---|---|---|---|---|
| A | 25 | 2 | 1 | 1 | 2 | 1 | 1 |
| B | 12.5 | 1 | 2 | 1 | 1 | 3 | 1 |
| C | 50 | 1 | 1 | 2 | 1 | 1 | 2 |
