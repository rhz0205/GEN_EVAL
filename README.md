GEN_EVAL

GEN_EVAL is a lightweight refactor target for video generation evaluation code extracted from WorldLens.

Scope:
- generation evaluation only
- manifest-based sample loading
- lightweight metric registry and evaluator
- placeholder metric/model adapters for later implementation

Non-goals:
- editing or importing `worldlens/`
- model-dependent execution in this environment
- dataset-specific pipelines such as nuScenes loaders

Repository layout:
- `src/gen_eval/`: package source
- `configs/`: default configuration examples
- `examples/`: example manifest and evaluation config
- `scripts/`: CLI entrypoints

Quick start:
1. Prepare a manifest JSON matching `examples/manifest_example.json`.
2. Copy `configs/default.yaml` or `examples/eval_config_example.yaml`.
3. Run `python scripts/evaluate.py --config configs/default.yaml`.

Current status:
- package skeleton is in place
- metrics expose lightweight manifest-based evaluation wrappers with guarded skip behavior
- model loading is intentionally lazy and unimplemented
