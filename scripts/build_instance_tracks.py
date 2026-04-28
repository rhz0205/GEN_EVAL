#!/usr/bin/env python3
"""Build prepared instance tracks for a GEN_EVAL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gen_eval.dataset import load_manifest_payload
from gen_eval.instance_extraction import (
    InstanceTrackExtractor,
    UnavailableDetectionBackend,
    UnavailableEmbeddingBackend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a new manifest copy with metadata['instance_tracks']."
    )
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.manifest)
    output_path = Path(args.output)
    payload = load_manifest_payload(input_path)
    samples = _extract_samples(payload)

    extractor = InstanceTrackExtractor(
        detector=UnavailableDetectionBackend(),
        embedder=UnavailableEmbeddingBackend(),
    )

    processed = 0
    succeeded = 0
    skipped = 0

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        processed += 1
        sample_id = str(sample.get("sample_id") or "unknown")
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            sample["metadata"] = metadata

        extraction_result = extractor.extract_sample(sample_id, metadata)
        if extraction_result.status == "success":
            metadata["instance_tracks"] = extraction_result.instance_tracks
            metadata.pop("instance_tracks_status", None)
            succeeded += 1
        else:
            skipped += 1
            metadata["instance_tracks_status"] = {
                "status": extraction_result.status,
                "reason": extraction_result.reason,
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"input_manifest: {input_path}")
    print(f"output_manifest: {output_path}")
    print(f"processed_samples: {processed}")
    print(f"succeeded: {succeeded}")
    print(f"skipped: {skipped}")
    return 0


def _extract_samples(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return samples
    raise ValueError("Manifest JSON must be a list or an object with a 'samples' list.")


if __name__ == "__main__":
    raise SystemExit(main())
