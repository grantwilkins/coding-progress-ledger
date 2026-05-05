#!/usr/bin/env bash
set -euo pipefail
cat > config.yaml <<'YAML'
service: api
replicas: 3
port: 8080
env: {}
resources:
  cpu: "500m"
  memory: "512Mi"
YAML
