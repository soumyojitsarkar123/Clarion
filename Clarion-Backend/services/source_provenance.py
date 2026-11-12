"""Versioned source-location mapping for extracted and normalized document text."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Iterable

NORMALIZATION_MAP_VERSION = "clarion-normalization-offset-map-v2"
EXTRACTION_VERSION = "clarion-extraction-v2"
CHUNK_PROVENANCE_VERSION = "clarion-chunk-provenance-v1"
EMBEDDING_PROVENANCE_VERSION = "clarion-embedding-provenance-v1"

_TOKEN = re.compile(r"\s+|\S+", re.UNICODE)


def build_normalization_map(source_text: str, normalized_text: str) -> dict[str, Any]:
    """Map normalized character spans to conservative extracted-text intervals.

    Unchanged tokens have character-accurate linear mappings. A changed token
    maps to the complete source token that produced it; inserted characters are
    recorded as zero-width source anchors. Deleted source spans are retained in
    ``removed_source_intervals``. This deliberately reports coarse spans for OCR
    repairs and token splitting instead of implying character-level certainty.
    """
    source_tokens = list(_TOKEN.finditer(source_text or ""))
    normalized_tokens = list(_TOKEN.finditer(normalized_text or ""))
    matcher = SequenceMatcher(
        None,
        [m.group(0) for m in source_tokens],
        [m.group(0) for m in normalized_tokens],
        autojunk=True,
    )
    runs: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    def append_run(out_start: int, out_end: int, src_start: int, src_end: int, operation: str) -> None:
        if out_start >= out_end:
            return
        run = {
            "normalized_start": out_start,
            "normalized_end": out_end,
            "source_start": src_start,
            "source_end": src_end,
            "operation": operation,
        }
        if runs:
            prev = runs[-1]
            linear = operation == prev["operation"] == "copy" and (
                prev["normalized_end"] == out_start and prev["source_end"] == src_start
            )
            replaced = operation == prev["operation"] == "replacement" and (
                prev["normalized_end"] == out_start
                and prev["source_start"] == src_start
                and prev["source_end"] == src_end
            )
            if linear or replaced:
                prev["normalized_end"] = out_end
                if linear:
                    prev["source_end"] = src_end
                return
        runs.append(run)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        src_start = source_tokens[i1].start() if i1 < len(source_tokens) else len(source_text or "")
        src_end = source_tokens[i2 - 1].end() if i2 > i1 else src_start
        out_start = normalized_tokens[j1].start() if j1 < len(normalized_tokens) else len(normalized_text or "")
        out_end = normalized_tokens[j2 - 1].end() if j2 > j1 else out_start

        if tag == "equal":
            for src_token, out_token in zip(source_tokens[i1:i2], normalized_tokens[j1:j2]):
                append_run(out_token.start(), out_token.end(), src_token.start(), src_token.end(), "copy")
        elif tag == "replace":
            append_run(out_start, out_end, src_start, src_end, "replacement")
            removed.append({"source_start": src_start, "source_end": src_end, "operation": "replaced_by_normalization"})
        elif tag == "insert":
            anchor = src_start
            append_run(out_start, out_end, anchor, anchor, "inserted")
        elif tag == "delete":
            removed.append({"source_start": src_start, "source_end": src_end, "operation": "removed_by_normalization"})

    return {
        "version": NORMALIZATION_MAP_VERSION,
        "source_length": len(source_text or ""),
        "normalized_length": len(normalized_text or ""),
        "runs": runs,
        "removed_source_intervals": removed,
    }


def map_normalized_interval(
    offset_map: dict[str, Any] | None,
    start: int,
    end: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Project a normalized interval to source spans and note non-source edits."""
    if not offset_map or start < 0 or end < start:
        return [], ["source_offset_unavailable"]
    intervals: list[dict[str, Any]] = []
    operations: list[str] = []
    for run in offset_map.get("runs", []):
        overlap_start = max(start, int(run["normalized_start"]))
        overlap_end = min(end, int(run["normalized_end"]))
        if overlap_start >= overlap_end:
            continue
        operation = run.get("operation", "unknown")
        if operation == "copy":
            source_start = int(run["source_start"]) + overlap_start - int(run["normalized_start"])
            source_end = source_start + overlap_end - overlap_start
            intervals.append({"source_start": source_start, "source_end": source_end, "mapping": "exact"})
        elif operation == "replacement":
            intervals.append({
                "source_start": int(run["source_start"]),
                "source_end": int(run["source_end"]),
                "mapping": "source_token_span",
            })
            operations.append("normalization_replacement")
        else:
            operations.append("normalization_inserted_text")
    return _merge_intervals(intervals), sorted(set(operations))


def _merge_intervals(intervals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(intervals, key=lambda x: (int(x["source_start"]), int(x["source_end"])))
    merged: list[dict[str, Any]] = []
    for interval in ordered:
        start, end = int(interval["source_start"]), int(interval["source_end"])
        if merged and start <= int(merged[-1]["source_end"]):
            merged[-1]["source_end"] = max(int(merged[-1]["source_end"]), end)
            if merged[-1].get("mapping") != interval.get("mapping"):
                merged[-1]["mapping"] = "combined_source_span"
        else:
            merged.append({**interval, "source_start": start, "source_end": end})
    return merged


def attach_page_references(
    intervals: Iterable[dict[str, Any]], pages: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[int]]:
    """Split absolute source spans at known PDF page boundaries."""
    mapped: list[dict[str, Any]] = []
    page_numbers: set[int] = set()
    page_rows = list(pages or [])
    for interval in intervals:
        start, end = int(interval["source_start"]), int(interval["source_end"])
        overlaps = []
        for page in page_rows:
            pstart, pend = page.get("source_char_start"), page.get("source_char_end")
            if pstart is None or pend is None:
                continue
            left, right = max(start, int(pstart)), min(end, int(pend))
            if left < right or (start == end and int(pstart) <= start <= int(pend)):
                overlaps.append((page, left, right))
        if not overlaps:
            mapped.append({**interval, "page_number": None, "page_char_start": None, "page_char_end": None})
            continue
        for page, left, right in overlaps:
            number = int(page["page_number"])
            page_numbers.add(number)
            mapped.append({
                **interval,
                "source_start": left,
                "source_end": right,
                "page_number": number,
                "page_char_start": left - int(page["source_char_start"]),
                "page_char_end": right - int(page["source_char_start"]),
            })
    return mapped, sorted(page_numbers)


def project_content_interval(
    provenance: dict[str, Any] | None,
    content_start: int,
    content_end: int,
) -> dict[str, Any]:
    """Map a chunk/segment-relative interval through its retained content map."""
    provenance = provenance or {}
    intervals: list[dict[str, Any]] = []
    pages: set[int] = set()
    blocks: dict[str, dict[str, Any]] = {}
    operations: set[str] = set(provenance.get("normalization_operations", []))
    for mapping in provenance.get("content_mappings", []):
        left = max(content_start, int(mapping.get("content_start", 0)))
        right = min(content_end, int(mapping.get("content_end", 0)))
        if left >= right:
            continue
        norm_start = int(mapping["normalized_start"]) + left - int(mapping["content_start"])
        norm_end = norm_start + right - left
        mapped, found_operations = map_normalized_interval(
            {"runs": mapping.get("offset_map_runs", [])}, norm_start, norm_end
        )
        page_intervals, page_numbers = attach_page_references(mapped, mapping.get("page_boundaries", []))
        intervals.extend(page_intervals)
        pages.update(page_numbers)
        operations.update(found_operations)
        for block in mapping.get("block_references", []):
            if any(
                int(span["source_start"]) < int(block.get("source_char_end", -1))
                and int(span["source_end"]) > int(block.get("source_char_start", -1))
                for span in mapped
            ):
                blocks[str(block.get("block_id"))] = {
                    key: block.get(key) for key in ("block_id", "block_type", "block_index")
                }
    return {
        # Preserve page-split spans; joining adjacent ranges can miscite a page boundary.
        "source_intervals": intervals,
        "page_references": sorted(pages),
        "block_references": list(blocks.values()),
        "normalization_operations": sorted(op for op in operations if op),
        "location_available": bool(intervals or blocks),
        "location_unavailable_reason": None if (intervals or blocks) else "source_offset_map_unavailable",
    }
