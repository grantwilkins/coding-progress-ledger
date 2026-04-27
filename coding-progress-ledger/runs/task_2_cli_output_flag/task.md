# TASK 2: CLI Output Flag Regression

Create a tiny Python CLI repo whose initial implementation accepts
`--output file.json` but ignores it and always writes JSON to stdout. Add a
regression-tested fix for writing to files while preserving stdout behavior for
the default and for `--output -`.
