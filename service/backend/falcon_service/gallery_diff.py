"""Deterministic comparison of two immutable gallery snapshots."""

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from sqlalchemy import select

from .catalog import scoped
from .db import Database
from .errors import ServiceError
from .models import Gallery, GalleryVersion, Observation, ObservationDigest
from .storage import ArtifactStore


@dataclass(frozen=True)
class _Entry:
    observation: Observation
    image_sha256: str

    @property
    def signature(self) -> tuple[str, tuple[int, ...]]:
        return self.image_sha256, tuple(self.observation.bbox)


def _load_entries(
    store: ArtifactStore,
    observations: list[Observation],
    digests: dict[str, str],
) -> list[_Entry]:
    """Read old snapshots too: digest rows were introduced additively."""
    return [
        _Entry(item, digests.get(item.id) or hashlib.sha256(store.read(item.image_key)).hexdigest())
        for item in observations
    ]


def _group_key(entry: _Entry) -> str:
    if entry.observation.external_id:
        return f"external:{entry.observation.external_id}"
    return f"sha256:{entry.image_sha256}"


def _entry_key(entry: _Entry) -> tuple:
    return (*entry.signature, entry.observation.ordinal, entry.observation.id)


def _item(identity: str, change: str, old: _Entry | None, new: _Entry | None) -> dict:
    image_changed = old is not None and new is not None and old.image_sha256 != new.image_sha256
    bbox_changed = old is not None and new is not None and old.observation.bbox != new.observation.bbox
    reference = new or old
    assert reference is not None
    return {
        "identity": identity,
        "external_id": reference.observation.external_id,
        "change": change,
        "from_observation_id": old.observation.id if old else None,
        "to_observation_id": new.observation.id if new else None,
        "image_changed": image_changed,
        "bbox_changed": bbox_changed,
        "from_bbox": old.observation.bbox if old else None,
        "to_bbox": new.observation.bbox if new else None,
    }


def _compare_entries(before: list[_Entry], after: list[_Entry]) -> tuple[list[dict], int]:
    """Pair repeated external IDs across both snapshots without inventing identities."""
    before_groups: dict[str, list[_Entry]] = defaultdict(list)
    after_groups: dict[str, list[_Entry]] = defaultdict(list)
    for entry in before:
        before_groups[_group_key(entry)].append(entry)
    for entry in after:
        after_groups[_group_key(entry)].append(entry)

    changes: list[dict] = []
    unchanged = 0
    for group_key in sorted(before_groups.keys() | after_groups.keys()):
        old_remaining = sorted(before_groups.get(group_key, []), key=_entry_key)
        new_remaining = sorted(after_groups.get(group_key, []), key=_entry_key)
        total_slots = max(len(old_remaining), len(new_remaining))
        slot = 0

        # Exact image+bbox matches win before deterministic pairing of changed rows.
        new_by_signature = defaultdict(deque)
        for new in new_remaining:
            new_by_signature[new.signature].append(new)
        matched_new_ids: set[str] = set()
        unmatched_old: list[_Entry] = []
        for old in old_remaining:
            matches = new_by_signature[old.signature]
            if not matches:
                unmatched_old.append(old)
            else:
                matched_new_ids.add(matches.popleft().observation.id)
        exact_count = len(old_remaining) - len(unmatched_old)
        new_remaining = [new for new in new_remaining if new.observation.id not in matched_new_ids]
        unchanged += exact_count
        slot += exact_count

        pair_count = min(len(unmatched_old), len(new_remaining))
        for index in range(pair_count):
            slot += 1
            identity = group_key if total_slots == 1 else f"{group_key}:{slot}"
            changes.append(_item(identity, "changed", unmatched_old[index], new_remaining[index]))
        for old in unmatched_old[pair_count:]:
            slot += 1
            identity = group_key if total_slots == 1 else f"{group_key}:{slot}"
            changes.append(_item(identity, "removed", old, None))
        for new in new_remaining[pair_count:]:
            slot += 1
            identity = group_key if total_slots == 1 else f"{group_key}:{slot}"
            changes.append(_item(identity, "added", None, new))
    return changes, unchanged


def compare_versions(
    db: Database,
    store: ArtifactStore,
    workspace: str,
    gallery_id: str,
    from_version_id: str,
    to_version_id: str,
    *,
    limit: int,
    offset: int,
) -> dict:
    """Return one bounded change page plus complete summary counts."""
    if from_version_id == to_version_id:
        raise ServiceError("same_version", "Для сравнения выберите две разные версии", 400)
    with db.transaction() as session:
        scoped(session, Gallery, gallery_id, workspace)
        before = scoped(session, GalleryVersion, from_version_id, workspace)
        after = scoped(session, GalleryVersion, to_version_id, workspace)
        if before.gallery_id != gallery_id or after.gallery_id != gallery_id:
            raise ServiceError("not_found", "Версия не принадлежит галерее", 404)
        if before.status != "ready" or after.status != "ready":
            raise ServiceError("gallery_not_ready", "Сравнивать можно только готовые версии", 409)
        before_rows = list(
            session.scalars(
                select(Observation)
                .where(
                    Observation.workspace == workspace,
                    Observation.version_id == before.id,
                )
                .order_by(Observation.ordinal, Observation.id)
            )
        )
        after_rows = list(
            session.scalars(
                select(Observation)
                .where(
                    Observation.workspace == workspace,
                    Observation.version_id == after.id,
                )
                .order_by(Observation.ordinal, Observation.id)
            )
        )
        if len(before_rows) != before.count or len(after_rows) != after.count:
            raise ServiceError("gallery_incomplete", "Снимок галереи неполон", 503)
        digests = {}
        observation_ids = [item.id for item in before_rows + after_rows]
        for start in range(0, len(observation_ids), 500):
            rows = session.scalars(
                select(ObservationDigest).where(
                    ObservationDigest.workspace == workspace,
                    ObservationDigest.observation_id.in_(observation_ids[start : start + 500]),
                )
            )
            digests.update((item.observation_id, item.sha256) for item in rows)

    changes, unchanged = _compare_entries(
        _load_entries(store, before_rows, digests),
        _load_entries(store, after_rows, digests),
    )
    counts = Counter(item["change"] for item in changes)
    page = changes[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "gallery_id": gallery_id,
        "from_version_id": before.id,
        "to_version_id": after.id,
        "from_number": before.number,
        "to_number": after.number,
        "added": counts["added"],
        "removed": counts["removed"],
        "changed": counts["changed"],
        "unchanged": unchanged,
        "total_changes": len(changes),
        "items": page,
        "next_offset": next_offset if next_offset < len(changes) else None,
    }
