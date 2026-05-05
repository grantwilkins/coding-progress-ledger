FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux asciinema && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN cat > /app/sum_evens.py <<'PY'
import sys
xs = [int(t) for t in sys.stdin.read().strip().split(",") if t]
print(sum(x for x in xs if x % 2 == 1))
PY
