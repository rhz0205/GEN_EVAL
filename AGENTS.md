# GEN_EVAL Refactor Rules

This repository contains the official WorldLens source under:

worldlens/

Treat `worldlens/` as a read-only reference source. Do not edit files under `worldlens/` unless explicitly requested.

The target lightweight package is:

src/gen_eval/

## Goal

Refactor only the video generation evaluation part of WorldLens into a lightweight package named GEN_EVAL.

Keep only generation-related evaluation code.

Do not migrate:
- reconstruction
- downstream task evaluation
- action-following
- lidar generation
- occupancy generation
- third_party heavy dependencies
- nuScenes-specific dataset pipelines

## Current constraints

The local machine does not have:
- WorldLens runtime environment
- pretrained weights
- full dataset

Therefore:
- Do not run model-dependent tests.
- Do not install packages.
- Do not download weights.
- Do not use network access.
- Focus on static refactoring, import cleanup, data-interface replacement, and clear TODOs.

## Source paths

Use these WorldLens source paths for reference:

worldlens/worldbench/videogen/generation/temporal_consistency
worldlens/worldbench/videogen/generation/subject_consistency
worldlens/worldbench/videogen/generation/temporal_semantic_consistency
worldlens/worldbench/videogen/generation/perceptual_fidelity/fvd
worldlens/worldbench/videogen/generation/depth_consistency
worldlens/worldbench/videogen/generation/object_coherence

## Target package layout

src/gen_eval/
  evaluator.py
  registry.py
  schemas.py
  dataset.py
  video_io.py
  result_writer.py
  metrics/
    fvd.py
    temporal_consistency.py
    subject_consistency.py
    temporal_semantic_consistency.py
    depth_consistency.py
    object_coherence.py
  models/
    backbones.py
    depth.py
    object.py
  utils/
    paths.py
    media.py

## Data interface

All metrics must use GenerationSample from src/gen_eval/schemas.py.

Do not use WorldLens dataset classes directly.
Do not use nuScenes-specific fields directly.

Replace dataset access with manifest-based fields:
- sample_id
- generated_video
- reference_video
- prompt
- objects
- metadata

## Metric interface

Each metric should expose:

class MetricName:
    name = "metric_name"

    def __init__(self, config: dict):
        ...

    def evaluate(self, samples: list[GenerationSample]) -> dict:
        ...

Metric code must not load pretrained weights at import time.
Model and weight loading must be lazy.

## Output format

Each metric should return a dict with:
- metric
- score
- num_samples
- details
- status
- reason, when skipped or failed

## Working style

Prefer small patches.
Do not over-engineer.
Do not create new nested directories unless necessary.
Do not introduce new dependencies unless explicitly requested.
Before editing, provide a short plan.
After editing, report:
- touched files
- changes made
- risks
- validation not run
- next steps