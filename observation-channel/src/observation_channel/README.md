# observation_channel

Core modules:

- `models.py`: public dataclasses and `Category`.
- `annotator.py`: live-compatible work-unit state machine.
- `classify.py`: deterministic tool/bash category rules.
- `path_tracker.py`: first-write-target extraction.
- `readers.py`: source rows to canonical turns.
- `hf.py`: Hugging Face loading and raw JSONL caching.
- `runner.py`: annotate one file or a corpus.
- `cli.py`: `cache`, `preprocess`, `annotate`, `annotate-corpus`.

Readers should fail loudly on malformed source structure and must not use hidden thoughts as evidence.

Stuck detection requires three identical observation bodies that are either non-trivial in length or carry an explicit error marker; empty tool acknowledgements and short success acks are ignored.

Path tracking should abstain on editor line ranges and shell flags; it only returns concrete write paths.
