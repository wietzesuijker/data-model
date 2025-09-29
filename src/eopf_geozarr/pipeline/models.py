"""Public payload helpers used by GeoZarr pipeline integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..conversion.geozarr import DEFAULT_REFLECTANCE_GROUPS


def _normalize_members(values: Iterable[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw = values.replace("\n", ",").replace("\t", ",")
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = [str(item).strip() for item in values]
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        if not part.startswith("/"):
            part = f"/{part}"
        part = part.rstrip("/")
        if part not in seen:
            cleaned.append(part)
            seen.add(part)
    return tuple(cleaned)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _maybe_bool(value: Any) -> bool | None:
    if isinstance(value, bool) or value is None:
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_overwrite(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "replace", "yes"}:
        return "replace"
    if text in {"0", "false", "", "no"}:
        return None
    return str(value)


@dataclass(slots=True)
class GeoZarrPayload:
    """Lightweight structured view over a GeoZarr payload mapping."""

    src_item: str
    output_zarr: str
    groups: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_REFLECTANCE_GROUPS)
    )
    crs_groups: tuple[str, ...] = field(default_factory=tuple)
    overwrite: str | None = None
    owner: str | None = None
    collection: str | None = None
    metrics_out: str | None = None
    register_mode: str | None = None
    profile: str | None = None
    dask_cluster: bool | None = None
    verbose: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GeoZarrPayload:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        data = dict(payload)
        groups = _normalize_members(
            data.pop("groups", DEFAULT_REFLECTANCE_GROUPS)
        ) or tuple(DEFAULT_REFLECTANCE_GROUPS)
        crs_groups = _normalize_members(data.pop("crs_groups", ()))
        return cls(
            src_item=_maybe_str(data.pop("src_item", "")) or "",
            output_zarr=_maybe_str(data.pop("output_zarr", "")) or "",
            groups=groups,
            crs_groups=crs_groups,
            overwrite=_normalize_overwrite(data.pop("overwrite", None)),
            owner=_maybe_str(data.pop("owner", None)),
            collection=_maybe_str(data.pop("collection", None)),
            metrics_out=_maybe_str(data.pop("metrics_out", None)),
            register_mode=_maybe_str(data.pop("register_mode", None)),
            profile=_maybe_str(data.pop("profile", None)),
            dask_cluster=_maybe_bool(data.pop("dask_cluster", None)),
            verbose=_maybe_bool(data.pop("verbose", None)),
            extras=data,
        )

    def ensure_required(self) -> None:
        missing = [
            name
            for name in ("src_item", "output_zarr", "groups")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "src_item": self.src_item,
            "output_zarr": self.output_zarr,
            "groups": ",".join(self.groups),
        }
        if self.crs_groups:
            payload["crs_groups"] = ",".join(self.crs_groups)
        for key in (
            "overwrite",
            "owner",
            "collection",
            "metrics_out",
            "register_mode",
            "profile",
        ):
            value = getattr(self, key)
            if value:
                payload[key] = value
        for key in ("dask_cluster", "verbose"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = bool(value)
        payload.update(self.extras)
        return payload


_EXAMPLE_PAYLOADS: dict[str, dict[str, Any]] = {
    "minimal": {
        "src_item": "https://example.com/stac/items/S2A_SAMPLE",
        "output_zarr": "s3://example/geozarr/S2A_SAMPLE.zarr",
        "collection": "sentinel-2-l2a",
        "groups": ",".join(DEFAULT_REFLECTANCE_GROUPS),
        "register_mode": "update",
    },
    "full": {
        "src_item": "https://example.com/stac/items/S2A_FULL",
        "output_zarr": "s3://example/geozarr/S2A_FULL.zarr",
        "collection": "sentinel-2-l2a",
        "groups": ",".join(DEFAULT_REFLECTANCE_GROUPS),
        "metrics_out": "s3://example/geozarr/S2A_FULL.json",
    },
}


def load_example_payload(name: str = "minimal") -> dict[str, Any]:
    try:
        return dict(_EXAMPLE_PAYLOADS[name])
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown example payload: {name}") from exc


def validate_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    required = ("src_item", "output_zarr", "groups")
    missing = [key for key in required if not str(payload.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Payload missing required keys: {', '.join(missing)}")
    if not _normalize_members(payload.get("groups")):
        raise ValueError("Payload must include at least one group entry")


__all__ = [
    "GeoZarrPayload",
    "load_example_payload",
    "validate_payload",
]
