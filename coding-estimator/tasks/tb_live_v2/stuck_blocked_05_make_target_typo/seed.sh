#!/usr/bin/env bash
set -euo pipefail
cat > Makefile <<'MK'
.PHONY: bulid clean

bulid:
	@mkdir -p dist
	@echo "built ok" > dist/output.txt

clean:
	@rm -rf dist
MK
