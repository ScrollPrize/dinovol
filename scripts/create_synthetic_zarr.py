from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.replace("x", ",").split(",") if part]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be SIZE or Z,Y,X")
    return tuple(parts)


def create_synthetic_zarr(path: Path, *, shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> None:
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    array = group.create_array("0", shape=shape, chunks=chunks, dtype="float32")

    z_coords = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    y_coords = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)
    x_coords = np.linspace(-1.0, 1.0, shape[2], dtype=np.float32)
    y = y_coords[None, :, None]
    x = x_coords[None, None, :]
    slab_depth = max(1, int(chunks[0]))
    for z0 in range(0, shape[0], slab_depth):
        z1 = min(shape[0], z0 + slab_depth)
        z = z_coords[z0:z1, None, None]
        volume = np.exp(-(z * z + y * y + x * x) * 6.0).astype(np.float32)
        volume += 0.15 * ((z + y + x + 3.0) / 6.0).astype(np.float32)
        array[z0:z1, :, :] = np.clip(volume, 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local synthetic 3D zarr volume for smoke tests.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--shape", type=_parse_shape, default=(224, 224, 224))
    parser.add_argument("--chunks", type=_parse_shape, default=(32, 32, 32))
    args = parser.parse_args()

    args.path.parent.mkdir(parents=True, exist_ok=True)
    create_synthetic_zarr(args.path, shape=args.shape, chunks=args.chunks)


if __name__ == "__main__":
    main()
