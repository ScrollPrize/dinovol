from __future__ import annotations

import atexit
import json
import os
import random
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from dinovol_2.augmentation.pipelines import create_training_transforms
from dinovol_2.dataset.normalization import get_normalization
from dinovol_2.dataset.ssl_zarr_dataset import ZarrHandle, open_zarr_handle


def _as_2tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value), int(value)
    result = tuple(int(v) for v in value)
    if len(result) != 2:
        raise ValueError(f"expected 2 values, got {result}")
    return result


def _float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    return float(value[0]), float(value[1])


class _SSL2DBaseDataset(Dataset):
    def __init__(self, config: Mapping[str, Any], *, do_augmentations: bool = False) -> None:
        self.config = dict(config)
        self.do_augmentations = bool(do_augmentations)
        self.epoch_length = int(self.config.get("epoch_length", 1_000_000))
        self.global_crop_size = _as_2tuple(self.config.get("global_crop_size", self.config.get("crop_size")))
        self.local_crop_size = _as_2tuple(self.config.get("local_crop_size"))
        self.global_view_size = _as_2tuple(self.config.get("global_view_size", self.global_crop_size))
        self.local_view_size = _as_2tuple(self.config.get("local_view_size", self.local_crop_size))
        self.num_global_crops = int(self.config.get("num_global_crops", 2))
        self.num_local_crops = int(self.config.get("num_local_crops", 8))
        if self.num_global_crops != 2:
            raise ValueError(f"2D SSL datasets require exactly two global crops, got {self.num_global_crops}")
        if self.global_crop_size is None or self.global_view_size is None:
            raise ValueError("global_crop_size is required")
        default_source_size = tuple(2 * value for value in self.global_crop_size)
        self.source_sampling_size = _as_2tuple(
            self.config.get("source_sampling_size", self.config.get("source_crop_size", default_source_size))
        )
        if any(source < crop for source, crop in zip(self.source_sampling_size, self.global_crop_size)):
            raise ValueError(
                f"source_sampling_size must contain global_crop_size, got "
                f"{self.source_sampling_size} and {self.global_crop_size}"
            )
        self.global_crop_scale = _float_pair(self.config.get("global_crop_scale"), (0.32, 1.0))
        self.local_crop_scale = _float_pair(self.config.get("local_crop_scale"), (0.05, 0.32))
        self.normalizer = get_normalization(self.config.get("normalization_scheme", "robust"))
        self.max_sample_attempts = max(1, int(self.config.get("max_sample_attempts", 100)))
        self.max_view_attempts = max(1, int(self.config.get("max_view_attempts", 32)))
        self.reject_all_zero_views = bool(self.config.get("reject_all_zero_views", True))
        transform_kwargs = {
            "spatial_only": bool(self.config.get("spatial_augmentations_only", False)),
        }
        self.global_transforms = [
            create_training_transforms(self.global_view_size, **transform_kwargs)
            for _ in range(self.num_global_crops)
        ]
        self.local_transforms = (
            [
                create_training_transforms(self.local_view_size, **transform_kwargs)
                for _ in range(self.num_local_crops)
            ]
            if self.local_view_size is not None
            else []
        )

    def __len__(self) -> int:
        return self.epoch_length

    def _read_source(self) -> tuple[np.ndarray, np.ndarray | None, str] | None:
        raise NotImplementedError

    @staticmethod
    def _has_signal(array: np.ndarray) -> bool:
        return bool(array.size and np.any(array != 0))

    def _sample_crop_bounds(
        self,
        source_shape: tuple[int, int],
        scale_range: tuple[float, float],
    ) -> tuple[int, int, int, int]:
        scale = float(np.random.uniform(*scale_range))
        linear_scale = max(scale, 1e-8) ** 0.5
        crop_h = min(source_shape[0], max(1, int(round(self.source_sampling_size[0] * linear_scale))))
        crop_w = min(source_shape[1], max(1, int(round(self.source_sampling_size[1] * linear_scale))))
        y0 = int(np.random.randint(0, source_shape[0] - crop_h + 1))
        x0 = int(np.random.randint(0, source_shape[1] - crop_w + 1))
        return y0, x0, crop_h, crop_w

    def _materialize_view(
        self,
        source: np.ndarray,
        instances: np.ndarray | None,
        *,
        scale_range: tuple[float, float],
        target_size: tuple[int, int],
        transform: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        for _ in range(self.max_view_attempts):
            y0, x0, crop_h, crop_w = self._sample_crop_bounds(source.shape, scale_range)
            raw_crop = np.asarray(source[y0:y0 + crop_h, x0:x0 + crop_w])
            if self.reject_all_zero_views and not self._has_signal(raw_crop):
                continue

            image_crop = np.asarray(raw_crop, dtype=np.float32)
            if self.normalizer is not None:
                image_crop = self.normalizer.run(image_crop)
            image_tensor = torch.from_numpy(np.ascontiguousarray(image_crop)).unsqueeze(0)
            if tuple(image_tensor.shape[1:]) != target_size:
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
                ).squeeze(0)

            instance_tensor = None
            if instances is not None:
                instance_crop = np.asarray(instances[:, y0:y0 + crop_h, x0:x0 + crop_w])
                instance_tensor = torch.from_numpy(np.ascontiguousarray(instance_crop)).float()
                if tuple(instance_tensor.shape[1:]) != target_size:
                    instance_tensor = F.interpolate(
                        instance_tensor.unsqueeze(0), size=target_size, mode="nearest"
                    ).squeeze(0)
                instance_tensor = instance_tensor.round().long()

            if self.do_augmentations:
                transformed = transform(image=image_tensor, segmentation=instance_tensor)
                image_tensor = transformed["image"]
                instance_tensor = transformed.get("segmentation")
            return image_tensor, instance_tensor
        return None

    def _component_constraints(self, global_instances: list[torch.Tensor | None]) -> list[dict[str, int]]:
        return []

    def __getitem__(self, index: int) -> dict[str, Any]:
        del index
        for _ in range(self.max_sample_attempts):
            source_record = self._read_source()
            if source_record is None:
                continue
            source, instances, _source_key = source_record
            if self.reject_all_zero_views and not self._has_signal(source):
                continue

            global_views: list[torch.Tensor] = []
            global_instances: list[torch.Tensor | None] = []
            valid = True
            for transform in self.global_transforms:
                materialized = self._materialize_view(
                    source,
                    instances,
                    scale_range=self.global_crop_scale,
                    target_size=self.global_view_size,
                    transform=transform,
                )
                if materialized is None:
                    valid = False
                    break
                view, view_instances = materialized
                global_views.append(view)
                global_instances.append(view_instances)
            if not valid:
                continue

            local_views: list[torch.Tensor] = []
            if self.local_view_size is not None:
                for transform in self.local_transforms:
                    materialized = self._materialize_view(
                        source,
                        None,
                        scale_range=self.local_crop_scale,
                        target_size=self.local_view_size,
                        transform=transform,
                    )
                    if materialized is None:
                        valid = False
                        break
                    local_views.append(materialized[0])
            if not valid:
                continue

            return {
                "global_views": global_views,
                "local_views": local_views,
                "component_constraints": self._component_constraints(global_instances),
            }
        raise RuntimeError(
            f"failed to sample a nonzero 2D training example after {self.max_sample_attempts} attempts"
        )


@dataclass(frozen=True)
class _ManifestRecord:
    sample_id: str
    segment_id: str
    path: str
    scale: str


class SSLZarrSliceDataset(_SSL2DBaseDataset):
    """Random YX slices from the surface-volume Zarrs listed in a manifest."""

    def __init__(self, config: Mapping[str, Any], *, do_augmentations: bool = False) -> None:
        super().__init__(config, do_augmentations=do_augmentations)
        manifest_path = Path(self.config["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        grouped: dict[str, dict[str, list[_ManifestRecord]]] = defaultdict(lambda: defaultdict(list))
        for sample_id, sample in manifest.get("samples", {}).items():
            for segment_id, segment in sample.get("segments", {}).items():
                for match in segment.get("matches", []):
                    roots = match.get("zarr_roots") or []
                    if not roots:
                        continue
                    access_uris = roots[0].get("access_uris") or []
                    if not access_uris:
                        continue
                    root_uri = str(access_uris[0])
                    for scale in match.get("matching_scales", []):
                        internal_path = str(scale.get("internal_path", scale.get("level")))
                        grouped[str(sample_id)][str(segment_id)].append(
                            _ManifestRecord(str(sample_id), str(segment_id), root_uri, internal_path)
                        )
        self.records_by_sample = {
            sample_id: {segment_id: tuple(records) for segment_id, records in segments.items() if records}
            for sample_id, segments in grouped.items()
            if any(segments.values())
        }
        if not self.records_by_sample:
            raise ValueError(f"manifest contains no usable Zarr scale records: {manifest_path}")
        self.volume_auth = self.config.get("volume_auth")
        self.s3_storage_options = self.config.get("s3_storage_options")
        self.min_nonzero_fraction = float(self.config.get("min_nonzero_fraction", 0.0))
        if not 0.0 <= self.min_nonzero_fraction < 1.0:
            raise ValueError("min_nonzero_fraction must be in [0, 1)")
        self.vol_trim_pct = float(self.config.get("vol_trim_pct", 1.0))
        middle_z_radius = self.config.get("middle_z_radius")
        self.middle_z_radius = None if middle_z_radius is None else int(middle_z_radius)
        if self.middle_z_radius is not None and self.middle_z_radius < 0:
            raise ValueError("middle_z_radius must be nonnegative")
        self._volume_handles: dict[tuple[str, str], ZarrHandle] = {}
        self._handle_pid: int | None = None
        self._atexit_pid: int | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_volume_handles"] = {}
        state["_handle_pid"] = None
        state["_atexit_pid"] = None
        return state

    def _ensure_process_local_handles(self) -> None:
        pid = os.getpid()
        if self._handle_pid == pid:
            return
        self.close()
        self._handle_pid = pid
        if self._atexit_pid != pid:
            atexit.register(self.close)
            self._atexit_pid = pid

    def _get_array(self, record: _ManifestRecord):
        self._ensure_process_local_handles()
        key = (record.path, record.scale)
        handle = self._volume_handles.get(key)
        if handle is None:
            handle = open_zarr_handle(
                record.path,
                record.scale,
                auth=self.volume_auth,
                s3_storage_options=self.s3_storage_options,
            )
            self._volume_handles[key] = handle
        return handle.array

    def close(self) -> None:
        for handle in self._volume_handles.values():
            handle.close()
        self._volume_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _pick_record(self) -> _ManifestRecord:
        sample_id = random.choice(tuple(self.records_by_sample))
        segments = self.records_by_sample[sample_id]
        segment_id = random.choice(tuple(segments))
        return random.choice(segments[segment_id])

    def _z_sampling_bounds(self, depth: int) -> tuple[int, int]:
        if self.middle_z_radius is None:
            return 0, depth
        middle_z = depth // 2
        return max(0, middle_z - self.middle_z_radius), min(depth, middle_z + self.middle_z_radius + 1)

    def _read_source(self) -> tuple[np.ndarray, None, str] | None:
        record = self._pick_record()
        array = self._get_array(record)
        if len(array.shape) != 3:
            raise ValueError(f"expected a ZYX array at {record.path}/{record.scale}, got {array.shape}")
        depth, height, width = (int(v) for v in array.shape)
        crop_h, crop_w = self.source_sampling_size
        trim_h = max(crop_h, min(height, int(round(height * self.vol_trim_pct))))
        trim_w = max(crop_w, min(width, int(round(width * self.vol_trim_pct))))
        if depth <= 0 or trim_h < crop_h or trim_w < crop_w:
            return None
        y_trim0 = (height - trim_h) // 2
        x_trim0 = (width - trim_w) // 2
        z_low, z_high = self._z_sampling_bounds(depth)
        z = int(np.random.randint(z_low, z_high))
        y0 = int(np.random.randint(y_trim0, y_trim0 + trim_h - crop_h + 1))
        x0 = int(np.random.randint(x_trim0, x_trim0 + trim_w - crop_w + 1))
        source = np.asarray(array[z, y0:y0 + crop_h, x0:x0 + crop_w])
        fraction = float(np.count_nonzero(source)) / float(source.size) if source.size else 0.0
        if fraction <= self.min_nonzero_fraction:
            return None
        return source, None, f"{record.sample_id}/{record.segment_id}"


@dataclass(frozen=True)
class _TiffTriplet:
    key: str
    image: Path
    horizontal: Path
    vertical: Path


def _normalized_tiff_key(path: Path, kind: str) -> str:
    stem = path.stem.lower()
    patterns = {
        "image": r"(?:-og|_og_c(\d+))$",
        "horizontal": r"(?:-hor|_horiz_c(\d+))$",
        "vertical": r"(?:-vert|_vert_c(\d+))$",
    }
    match = re.search(patterns[kind], stem)
    if match is None:
        return stem
    suffix = f"_c{match.group(1)}" if match.group(1) is not None else ""
    return stem[:match.start()] + suffix


class PairedTiffComponentDataset(_SSL2DBaseDataset):
    def __init__(self, config: Mapping[str, Any], *, do_augmentations: bool = False) -> None:
        super().__init__(config, do_augmentations=do_augmentations)
        root = Path(self.config["root"])
        directories = {
            "image": root / self.config.get("image_dir", "images"),
            "horizontal": root / self.config.get("horizontal_label_dir", "labels_hz"),
            "vertical": root / self.config.get("vertical_label_dir", "labels_vt"),
        }
        maps = {
            kind: {_normalized_tiff_key(path, kind): path for path in directory.glob("*.tif*")}
            for kind, directory in directories.items()
        }
        keys = set(maps["image"]) & set(maps["horizontal"]) & set(maps["vertical"])
        unmatched = {kind: sorted(set(paths) - keys) for kind, paths in maps.items()}
        if any(unmatched.values()):
            raise ValueError(f"unmatched TIFF image/label stems under {root}: {unmatched}")
        self.triplets = tuple(
            _TiffTriplet(key, maps["image"][key], maps["horizontal"][key], maps["vertical"][key])
            for key in sorted(keys)
        )
        if not self.triplets:
            raise ValueError(f"no paired TIFFs found under {root}")
        self.exclude_orientation_overlap = bool(self.config.get("exclude_orientation_overlap", True))
        self.components_per_sample = max(2, int(self.config.get("components_per_sample", 4)))
        self.points_per_component = max(2, int(self.config.get("points_per_component", 4)))
        self.patch_size = _as_2tuple(self.config["patch_size"])
        view_delta = tuple(
            int(view_size) - int(crop_size)
            for view_size, crop_size in zip(self.global_view_size, self.global_crop_size)
        )
        if any(delta < 0 or delta % 2 != 0 for delta in view_delta):
            raise ValueError(
                "paired TIFF component labels require global_view_size to contain a centered "
                f"global_crop_size, got {self.global_view_size} and {self.global_crop_size}"
            )
        self.global_view_halo = tuple(delta // 2 for delta in view_delta)
        self.cache_size = max(0, int(self.config.get("tiff_cache_size", 8)))
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _load_triplet(self, triplet: _TiffTriplet) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(triplet.key)
        if cached is not None:
            self._cache.move_to_end(triplet.key)
            return cached
        image = np.asarray(Image.open(triplet.image))
        horizontal = np.asarray(Image.open(triplet.horizontal)) > 0
        vertical = np.asarray(Image.open(triplet.vertical)) > 0
        if image.ndim != 2 or image.shape != horizontal.shape or image.shape != vertical.shape:
            raise ValueError(
                f"shape mismatch for {triplet.key}: image={image.shape}, "
                f"horizontal={horizontal.shape}, vertical={vertical.shape}"
            )
        if self.exclude_orientation_overlap:
            overlap = horizontal & vertical
            horizontal = horizontal & ~overlap
            vertical = vertical & ~overlap
        structure = np.ones((3, 3), dtype=np.uint8)
        horizontal_ids, _ = ndimage.label(horizontal, structure=structure)
        vertical_ids, _ = ndimage.label(vertical, structure=structure)
        instances = np.stack((horizontal_ids, vertical_ids), axis=0).astype(np.int32, copy=False)
        result = np.asarray(image), instances
        if self.cache_size > 0:
            self._cache[triplet.key] = result
            self._cache.move_to_end(triplet.key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def _read_source(self) -> tuple[np.ndarray, np.ndarray, str] | None:
        triplet = random.choice(self.triplets)
        image, instances = self._load_triplet(triplet)
        crop_h, crop_w = self.source_sampling_size
        if image.shape[0] < crop_h or image.shape[1] < crop_w:
            return None
        y0 = int(np.random.randint(0, image.shape[0] - crop_h + 1))
        x0 = int(np.random.randint(0, image.shape[1] - crop_w + 1))
        return (
            np.asarray(image[y0:y0 + crop_h, x0:x0 + crop_w]),
            np.asarray(instances[:, y0:y0 + crop_h, x0:x0 + crop_w]),
            triplet.key,
        )

    def _component_constraints(self, global_instances: list[torch.Tensor | None]) -> list[dict[str, int]]:
        locations: dict[int, set[tuple[int, int]]] = defaultdict(set)
        patch_h, patch_w = self.patch_size
        halo_h, halo_w = self.global_view_halo
        crop_h, crop_w = self.global_crop_size
        grid_w = self.global_crop_size[1] // patch_w
        for view_index, instances in enumerate(global_instances):
            if instances is None:
                continue
            for orientation in range(int(instances.shape[0])):
                channel = instances[orientation]
                for component_id in torch.unique(channel).tolist():
                    component_id = int(component_id)
                    if component_id <= 0:
                        continue
                    coords = (channel == component_id).nonzero(as_tuple=False)
                    for y, x in coords.tolist():
                        crop_y = int(y) - halo_h
                        crop_x = int(x) - halo_w
                        if not (0 <= crop_y < crop_h and 0 <= crop_x < crop_w):
                            continue
                        patch_index = (crop_y // patch_h) * grid_w + (crop_x // patch_w)
                        locations[orientation * 1_000_000 + component_id].add((view_index, patch_index))

        candidates = [group_id for group_id, points in locations.items() if len(points) >= 2]
        if len(candidates) < 2:
            return []
        selected = random.sample(candidates, k=min(self.components_per_sample, len(candidates)))
        constraints: list[dict[str, int]] = []
        for group_id in selected:
            points = tuple(locations[group_id])
            chosen = random.sample(points, k=min(self.points_per_component, len(points)))
            constraints.extend(
                {"view_index": view_index, "patch_index": patch_index, "group_id": group_id}
                for view_index, patch_index in chosen
            )
        return constraints
