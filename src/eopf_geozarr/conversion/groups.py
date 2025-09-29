"""Helper utilities for working with xarray DataTree measurement groups."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import xarray as xr

DEFAULT_REFLECTANCE_GROUPS: List[str] = [
    "/measurements/reflectance/r10m",
    "/measurements/reflectance/r20m",
    "/measurements/reflectance/r60m",
]

__all__ = [
    "DEFAULT_REFLECTANCE_GROUPS",
    "list_available_groups",
    "normalize_crs_groups",
    "normalize_measurement_groups",
]


def _normalized_datatree_path(raw_path: str | None) -> str:
    if not raw_path:
        return ""
    cleaned = "".join(part for part in raw_path.strip().replace("\\", "/") if part)
    stripped = "/".join(segment for segment in cleaned.split("/") if segment)
    return f"/{stripped}" if stripped else ""


def list_available_groups(dt: xr.DataTree) -> List[str]:
    """Return sorted list of dataset-bearing paths available in a DataTree."""
    return sorted(
        {node.path for node in dt.subtree if node.path and node.ds is not None}
    )


def normalize_measurement_groups(
    dt: xr.DataTree,
    groups: Optional[Sequence[str]] = None,
    *,
    default_groups: Sequence[str] | None = None,
) -> List[str]:
    """Normalize and validate requested measurement groups against a DataTree.

    The function normalises path formatting (ensuring absolute, deduplicated
    group identifiers) and verifies that each requested group exists. Any
    missing group raises ``ValueError`` rather than attempting to autofix the
    request, keeping behaviour predictable across different satellite products.
    """

    requested_candidates: Sequence[str] | None = groups or default_groups
    if requested_candidates is None:
        requested_candidates = DEFAULT_REFLECTANCE_GROUPS

    normalized: List[str] = []
    missing: List[str] = []
    seen: set[str] = set()

    for raw_path in requested_candidates:
        normalized_path = _normalized_datatree_path(raw_path)
        if not normalized_path or normalized_path in seen:
            continue
        seen.add(normalized_path)
        try:
            dt[normalized_path.lstrip("/")]
        except KeyError:
            missing.append(normalized_path)
        else:
            normalized.append(normalized_path)

    if missing:
        available = list_available_groups(dt)
        available_display = ", ".join(available) if available else "<none>"
        raise ValueError(
            "Missing required measurement groups: "
            + ", ".join(missing)
            + ". Available groups: "
            + available_display
        )

    if not normalized:
        raise ValueError("No measurement groups found; verify the input DataTree.")

    return normalized


def normalize_crs_groups(
    dt: xr.DataTree,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Normalize CRS group paths and report any missing entries."""
    if not groups:
        return [], []

    normalized: List[str] = []
    missing: List[Tuple[str, str]] = []
    seen: set[str] = set()

    for raw_path in groups:
        normalized_path = _normalized_datatree_path(raw_path).rstrip("/")
        if not normalized_path:
            continue
        try:
            dt[normalized_path.lstrip("/")]
        except KeyError:
            missing.append((normalized_path, raw_path))
        else:
            if normalized_path not in seen:
                normalized.append(normalized_path)
                seen.add(normalized_path)

    return normalized, missing
