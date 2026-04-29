from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


DEFAULT_PKL_PATH = Path("/di/group/lishun/outputs/cosmos2/guojiaxiangmu_0424_with_hdmap.pkl")
DEFAULT_CONTAINER_KEYS = ("samples", "data", "items", "records", "infos", "entries")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and normalize pickle dataset files.")
    parser.add_argument("--path", default=str(DEFAULT_PKL_PATH), help="Path to the pickle file.")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of sample examples to print.")
    parser.add_argument("--max-items", type=int, default=10, help="Max top-level keys/items to print for previews.")
    parser.add_argument("--inspect-key", default=None, help="Optional top-level dict key to inspect in detail.")
    parser.add_argument("--dedup", action="store_true", help="Print deduplicated sample statistics for top-level dict payloads.")
    parser.add_argument("--normalize-output", default=None, help="Optional output pickle path for normalized samples.")
    return parser


def load_pickle(path: str | Path) -> Any:
    pkl_path = Path(path)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Pickle file does not exist: {pkl_path}")
    with pkl_path.open("rb") as file:
        return pickle.load(file)


def write_pickle(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(payload, file)


def infer_sample_container(payload: Any) -> tuple[str, Any]:
    if isinstance(payload, list):
        return "top-level list", payload
    if isinstance(payload, tuple):
        return "top-level tuple", payload
    if isinstance(payload, dict):
        for key in DEFAULT_CONTAINER_KEYS:
            value = payload.get(key)
            if isinstance(value, (list, tuple)):
                return f"payload['{key}']", value
        return "top-level dict", payload
    return type(payload).__name__, payload


def normalize_payload(payload: Any) -> list[Any]:
    _, container = infer_sample_container(payload)
    if isinstance(container, list):
        return list(container)
    if isinstance(container, tuple):
        return list(container)
    if isinstance(container, dict):
        return [value for value in container.values()]
    raise ValueError(f"Unsupported payload type for normalization: {type(payload).__name__}")


def preview_value(value: Any, *, max_items: int) -> Any:
    if isinstance(value, dict):
        preview: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                preview["..."] = f"{len(value) - max_items} more keys"
                break
            preview[str(key)] = preview_scalar_or_type(item)
        return preview
    if isinstance(value, (list, tuple)):
        preview = [preview_scalar_or_type(item) for item in value[:max_items]]
        if len(value) > max_items:
            preview.append(f"... ({len(value) - max_items} more items)")
        return preview
    return preview_scalar_or_type(value)


def preview_scalar_or_type(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def print_top_level_summary(payload: Any, *, max_items: int) -> None:
    print(f"top_level_type={type(payload).__name__}")
    if isinstance(payload, dict):
        print(f"top_level_num_keys={len(payload)}")
        print("top_level_preview=")
        print(preview_value(payload, max_items=max_items))
        print("top_level_list_lengths=")
        print(preview_top_level_list_lengths(payload, max_items=max_items))
    elif isinstance(payload, (list, tuple)):
        print(f"top_level_num_items={len(payload)}")
    else:
        print(f"top_level_repr={preview_scalar_or_type(payload)}")


def print_sample_examples(container_name: str, container: Any, *, num_examples: int, max_items: int) -> None:
    print(f"sample_container={container_name}")
    if isinstance(container, dict):
        print("sample_container_type=dict")
        print("sample_container_preview=")
        print(preview_value(container, max_items=max_items))
        return

    if not isinstance(container, (list, tuple)):
        print(f"sample_container_type={type(container).__name__}")
        print(f"sample_container_preview={preview_scalar_or_type(container)}")
        return

    print(f"sample_count={len(container)}")
    example_count = min(max(0, num_examples), len(container))
    for index in range(example_count):
        item = container[index]
        print(f"example[{index}]_type={type(item).__name__}")
        print(f"example[{index}]_preview=")
        print(preview_value(item, max_items=max_items))


def preview_top_level_list_lengths(payload: dict[str, Any], *, max_items: int) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for index, (key, value) in enumerate(payload.items()):
        if index >= max_items:
            preview["..."] = f"{len(payload) - max_items} more keys"
            break
        if isinstance(value, (list, tuple)):
            preview[str(key)] = len(value)
        else:
            preview[str(key)] = f"<{type(value).__name__}>"
    return preview


def print_key_examples(payload: Any, inspect_key: str, *, num_examples: int, max_items: int) -> None:
    if not isinstance(payload, dict):
        print("inspect_key_skipped=payload is not a dict")
        return
    if inspect_key not in payload:
        print(f"inspect_key_missing={inspect_key}")
        return

    value = payload[inspect_key]
    print(f"inspect_key={inspect_key}")
    print(f"inspect_key_type={type(value).__name__}")

    if isinstance(value, (list, tuple)):
        print(f"inspect_key_count={len(value)}")
        example_count = min(max(0, num_examples), len(value))
        for index in range(example_count):
            item = value[index]
            print(f"inspect_key_example[{index}]_type={type(item).__name__}")
            print(f"inspect_key_example[{index}]_preview=")
            print(preview_value(item, max_items=max_items))
        return

    print("inspect_key_preview=")
    print(preview_value(value, max_items=max_items))


def build_sample_key(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    video = item.get("video")
    hdmap = item.get("hdmap")
    if video is None and hdmap is None:
        return None
    return f"{video}|{hdmap}"


def collect_deduped_samples(payload: Any) -> tuple[int, dict[str, dict[str, Any]]]:
    total_items = 0
    deduped: dict[str, dict[str, Any]] = {}

    if not isinstance(payload, dict):
        return total_items, deduped

    for tag, value in payload.items():
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            total_items += 1
            sample_key = build_sample_key(item)
            if sample_key is None:
                continue
            if sample_key not in deduped:
                deduped[sample_key] = {
                    "item": item,
                    "tags": [str(tag)],
                }
            else:
                deduped[sample_key]["tags"].append(str(tag))
    return total_items, deduped


def print_dedup_summary(payload: Any, *, num_examples: int, max_items: int) -> None:
    if not isinstance(payload, dict):
        print("dedup_skipped=payload is not a top-level dict")
        return

    total_items, deduped = collect_deduped_samples(payload)
    unique_count = len(deduped)
    duplicate_count = max(0, total_items - unique_count)
    duplicate_ratio = (duplicate_count / total_items) if total_items > 0 else 0.0

    print(f"dedup_total_items={total_items}")
    print(f"dedup_unique_samples={unique_count}")
    print(f"dedup_duplicate_items={duplicate_count}")
    print(f"dedup_duplicate_ratio={duplicate_ratio:.6f}")

    keys = list(deduped.keys())
    example_count = min(max(0, num_examples), len(keys))
    for index in range(example_count):
        sample = deduped[keys[index]]
        print(f"dedup_example[{index}]_tags=")
        print(sample["tags"][:max_items])
        print(f"dedup_example[{index}]_item_preview=")
        print(preview_value(sample["item"], max_items=max_items))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    payload = load_pickle(args.path)
    print_top_level_summary(payload, max_items=max(1, args.max_items))
    container_name, container = infer_sample_container(payload)
    print_sample_examples(
        container_name,
        container,
        num_examples=max(0, args.num_examples),
        max_items=max(1, args.max_items),
    )
    if args.inspect_key:
        print_key_examples(
            payload,
            args.inspect_key,
            num_examples=max(0, args.num_examples),
            max_items=max(1, args.max_items),
        )
    if args.dedup:
        print_dedup_summary(
            payload,
            num_examples=max(0, args.num_examples),
            max_items=max(1, args.max_items),
        )
    if args.normalize_output:
        normalized = normalize_payload(payload)
        write_pickle(args.normalize_output, normalized)
        print(f"normalize_output={args.normalize_output}")
        print(f"normalize_count={len(normalized)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
