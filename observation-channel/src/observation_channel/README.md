# observation_channel

Core modules:

- `models.py`: public dataclasses and `Category`.
- `annotator.py`: live-compatible work-unit state machine and v1.6 prefix features.
- `classify.py`: deterministic tool/bash category rules.
- `path_tracker.py`: first-write-target extraction and source/scratch path classification.
- `readers.py`: source rows to canonical turns.
- `hf.py`: Hugging Face loading and raw JSONL caching.
- `runner.py`: annotate one file or a corpus.
- `empirical_bayes.py`: empirical final-unit lookup and evaluation artifacts.
- `belief_tracker.py`: replayable live final-work belief tracker over empirical-Bayes and GBM predictions.
- `progress_label_audit.py`: read-only audit for opened-unit progress as a remaining-work label.
- `together_replay_ensemble.py`: Together replay ablations over observer model, scalar prompt wording, context visibility, and ensemble size.
- `cli.py`: cache, preprocessing, annotation, diagnostics, and empirical-Bayes commands.

Readers should fail loudly on malformed source structure and must not use hidden thoughts as evidence.

Stuck detection requires three identical observation bodies that are either non-trivial in length or carry an explicit error marker; empty tool acknowledgements and short success acks are ignored.

Path tracking should abstain on editor line ranges and shell flags; it only returns concrete write paths. V1.6 source-touch features treat root scratch scripts as scratch and subdirectory/product paths as source.
