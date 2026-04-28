## Project Rules

1. `worldbench` is an offline evaluation framework for generated autonomous-driving videos.
2. It is not a training framework, not a heavy experiment-management platform, and not a dashboard system.
3. The canonical workflow is:
   data list
   -> dataset normalization
   -> reference data generation
   -> multi-dimensional evaluation modules
   -> result summarization
   -> visualization outputs
4. The only canonical config files are:
   `configs/dataset.yaml`
   `configs/metrics.yaml`
   `configs/reference.yaml`
   `configs/run.yaml`
5. Canonical dataset names are:
   `cosmos`
   `geely`
   `real`
   `sample`
6. Data file naming rule:
   `data/{dataset_name}_{data_count}_{timestamp}.json`
7. Output directory rule:
   `outputs/{dataset_name}/{data_count}_{timestamp}/`
8. Output layout:
   `results/`
   `logs/`
   `visualizations/`
9. Visualization folders under `visualizations/` must be exactly:
   `depth_raw`
   `semantic_raw`
   `multiview_match_raw`
   `depth_6v_image`
   `semantic_6v_image`
   `multiview_match_6v_image`
   `depth_6v_video`
   `semantic_6v_video`
   `multiview_match_6v_video`
10. Do not create `src/gen_eval/`.
11. Do not create `src/gen_eval/metrics/`.
12. All metric implementations belong in `src/modules/`.
13. Modules must contain real implementations, not wrappers around any metrics package.
14. Do not introduce `XXXMetric` legacy class names in the final structure.
15. Use concise class names:
    `VideoIntegrity`
    `TemporalConsistency`
    `AppearanceConsistency`
    `DepthConsistency`
    `SemanticConsistency`
    `InstanceConsistency`
    `ViewConsistency`
16. The dataset layer belongs under `src/dataset/`.
17. Do not create `src/gen_eval/dataset.py`.
18. Do not create `src/gen_eval/reference.py`.
19. The reference layer belongs under `src/reference/`.
20. The orchestration layer belongs under `src/models/`.
21. The visualization layer belongs under `src/visualization/`.
22. Shared schemas belong in `src/schemas.py`.
23. Do not create old `src/gen_eval/evaluator.py`, `src/gen_eval/execution.py`, or `src/gen_eval/registry.py`.
24. Scripts should be thin CLI entrypoints.
25. Scripts should add the project `src/` directory to `sys.path` when needed.
26. Scripts should import from the flat packages, for example:
    `from schemas import GenerationSample`
    `from dataset import build_dataset`
    `from modules import build_module`
    `from reference import ReferencePreparer`
    `from models import GenEval`
27. Do not use imports like:
    `from gen_eval.dataset import ...`
    `from gen_eval.modules import ...`
    `from gen_eval.reference import ...`
    `from gen_eval.models import ...`
28. Core logic should live under `src/`.
29. The reference layer should generate reference files before metric evaluation.
30. OpenSeeD should be part of the reference layer, not `semantic_consistency.py`.
31. `semantic_consistency.py` should only read prepared semantic masks.
32. OpenSeeD semantic reference protocol:
    `metadata.semantic_masks`
    `metadata.semantic_num_classes`
    `metadata.semantic_ignore_label`
33. Use `ignore_label = -1` for ignored semantic pixels.
34. For semantic consistency, fixed weights are:
    `S_SemC = 0.5 * S_LFR + 0.4 * S_SAC + 0.1 * S_CDS`
35. For no-reference temporal or appearance consistency, preferred aggregation is:
    `score = ACM / (1 + TJI)`
36. Do not change metric formulas unless explicitly requested.
37. Config field names should be unified:
    `weight_path`
    `repo_path`
    `model_path`
    `config_path`
    `device`
    `batch_size`
38. Avoid legacy config aliases unless explicitly needed.
39. Comments should be minimal.
40. If comments are necessary, use Chinese block-level comments.
41. Do not add unnecessary English implementation docstrings.
42. Do not add redundant comments such as:
    `# type: ignore`
    `# noqa: BLE001`
    `# type: ignore[assignment]`
43. Future Codex tasks should:
    - inspect old source only as reference
    - write only into `D:\Project\worldbench`
    - make limited changes
    - report touched files, risks, and validation commands
