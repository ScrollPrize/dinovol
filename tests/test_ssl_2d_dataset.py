from __future__ import annotations

import unittest

import torch

from dinovol_2.dataset.ssl_2d_dataset import PairedTiffComponentDataset, SSLZarrSliceDataset


class SSLZarrSliceDatasetTests(unittest.TestCase):
    def test_middle_z_radius_uses_inclusive_window(self) -> None:
        dataset = object.__new__(SSLZarrSliceDataset)
        dataset.middle_z_radius = 5

        self.assertEqual(dataset._z_sampling_bounds(30), (10, 21))

    def test_middle_z_radius_clips_to_shallow_volume(self) -> None:
        dataset = object.__new__(SSLZarrSliceDataset)
        dataset.middle_z_radius = 5

        self.assertEqual(dataset._z_sampling_bounds(7), (0, 7))

    def test_unset_middle_z_radius_uses_full_depth(self) -> None:
        dataset = object.__new__(SSLZarrSliceDataset)
        dataset.middle_z_radius = None

        self.assertEqual(dataset._z_sampling_bounds(30), (0, 30))


class PairedTiffComponentDatasetTests(unittest.TestCase):
    def test_component_constraints_remove_centered_view_halo(self) -> None:
        dataset = object.__new__(PairedTiffComponentDataset)
        dataset.patch_size = (2, 2)
        dataset.global_crop_size = (4, 4)
        dataset.global_view_halo = (2, 2)
        dataset.components_per_sample = 4
        dataset.points_per_component = 4

        instances = torch.zeros((2, 8, 8), dtype=torch.long)
        instances[0, 2, 2] = 1
        instances[0, 5, 5] = 2
        instances[0, 0, 0] = 9

        constraints = dataset._component_constraints([instances, instances.clone()])

        grouped = {}
        for constraint in constraints:
            grouped.setdefault(constraint["group_id"], set()).add(
                (constraint["view_index"], constraint["patch_index"])
            )
        self.assertEqual(grouped, {
            1: {(0, 0), (1, 0)},
            2: {(0, 3), (1, 3)},
        })


if __name__ == "__main__":
    unittest.main()
