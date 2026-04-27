from __future__ import annotations

from gen_eval.schemas import GenerationSample
from gen_eval.video_io import is_video_file


def count_existing_videos(samples: list[GenerationSample]) -> dict[str, int]:
    generated = 0
    reference = 0
    for sample in samples:
        if is_video_file(sample.generated_video):
            generated += 1
        if sample.reference_video and is_video_file(sample.reference_video):
            reference += 1
    return {
        "generated_video_entries": generated,
        "reference_video_entries": reference,
    }
