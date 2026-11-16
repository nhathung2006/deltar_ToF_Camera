import unittest

import numpy as np
import torch
from torch import nn

from apps.live_depth import make_input
from src.camera_tof_io import grid_rectangles
from src.models.fusion import TransformerFusion
from src.tof_projection import ManualToFProjector, draw_projected_zones, visible_zone_map


class ZoneMessage(nn.Module):
    def forward(self, query, source):
        return source.mean(dim=1, keepdim=True).expand_as(query)


class ManualGridTests(unittest.TestCase):
    def test_identity_preserves_roi_and_sensor_distances(self):
        projector = ManualToFProjector([96, 24, 448, 448])
        distances = np.arange(64, dtype=np.float64) + 1000
        result = projector.project(distances)
        np.testing.assert_allclose(result['polygons'][0], [[96, 24], [152, 24], [152, 80], [96, 80]])
        np.testing.assert_allclose(result['polygons'][-1, 2], [544, 472])
        np.testing.assert_array_equal(result['distance_mm'], distances)
        np.testing.assert_array_equal(result['sigma_scale'], np.ones(64))
        self.assertTrue(result['valid'].all())

    def test_left_rotation_and_right_shift_keep_cell_size_and_clip_only_at_image_edge(self):
        projector = ManualToFProjector([96, 24, 448, 448], 5, (1.5, 0))
        result = projector.project(np.full(64, 1000))
        np.testing.assert_allclose(projector.shift_pixels, [84, 0])
        # The grid centre moves right, independently of its rotation.
        centre = result['homography'][27] @ [1, 1, 1]
        np.testing.assert_allclose(centre[:2], [404, 248], atol=1e-6)
        top = result['homography'][3] @ [1, 0, 1]
        radians = np.deg2rad(5)
        np.testing.assert_allclose(top[:2], [404 - 224 * np.sin(radians),
                                           248 - 224 * np.cos(radians)], atol=1e-6)
        edge = result['polygons'][:, 1] - result['polygons'][:, 0]
        np.testing.assert_allclose(np.linalg.norm(edge, axis=1), 56, atol=1e-6)
        self.assertTrue((edge[:, 1] < 0).all())  # Right ends rise for counterclockwise rotation.
        self.assertGreater(result['polygons'][:, :, 0].max(), 640)
        self.assertGreater(result['polygons'][:, :, 1].max(), 480)
        image = draw_projected_zones(np.zeros((480, 640, 3), np.uint8), result)
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertTrue(image.any())

    def test_rotated_display_regions_match_actual_fusion_messages_and_invalid_holes(self):
        projector = ManualToFProjector([96, 24, 448, 448], 5, (1.5, 0))
        distances = np.full(64, 1000.)
        distances[18] = 0
        distances[2] = 5000
        result = projector.project(distances)
        owner = visible_zone_map(result, (640, 480))['owner']
        self.assertNotIn(18, owner)
        self.assertNotIn(2, owner)
        grids = projector.fusion_grids(result)
        for stride, stage in grids.items():
            np.testing.assert_array_equal(stage['coverage_mask'][0], owner[::stride, ::stride] >= 0)
        data, count = make_input(np.zeros((480, 640, 3), np.uint8), result['distance_mm'],
                                 grid_rectangles([96, 24, 448, 448]), .02, torch.device('cpu'),
                                 projection_grids=grids, sigma_scale=result['sigma_scale'])
        self.assertEqual(count, 62)
        fusion = TransformerFusion(8, (120, 160), attention_layer=['hist2image'])
        fusion.layers = nn.ModuleList([ZoneMessage()])
        with torch.no_grad():
            fusion.positional_encodings.zero_()
            fusion.positional_encodings2.zero_()
            features = torch.arange(1, 65, dtype=torch.float32)[None, :, None, None].expand(1, 64, 16, 8)
            actual = fusion(torch.zeros(1, 8, 120, 160), features, **data['additional'])
        expected = owner[::4, ::4]
        expected_values = np.where(expected >= 0, expected + 1, 0)
        np.testing.assert_allclose(actual[0, 0].numpy(), expected_values, atol=1e-4)

    def test_entirely_off_image_grid_supports_rgb_only_without_wrapping(self):
        projector = ManualToFProjector([96, 24, 448, 448], 5, (20, 0))
        result = projector.project(np.full(64, 1000))
        self.assertFalse(result['valid'].any())
        self.assertFalse(draw_projected_zones(np.zeros((480, 640, 3), np.uint8), result).any())
        data, count = make_input(np.zeros((480, 640, 3), np.uint8), result['distance_mm'],
                                 grid_rectangles([96, 24, 448, 448]), .02, torch.device('cpu'),
                                 projection_grids=projector.fusion_grids(result), allow_empty=True)
        self.assertEqual(count, 0)
        self.assertFalse(data['additional']['mask'].any())
        self.assertTrue(torch.isfinite(data['additional']['hist_data']).all())

    def test_invalid_manual_settings_are_rejected(self):
        for roi, angle, shift in [([96, 24, 0, 448], 5, (1.5, 0)),
                                  ([96, 24, 448, 448], float('nan'), (1.5, 0)),
                                  ([96, 24, 448, 448], 5, (float('inf'), 0))]:
            with self.subTest(roi=roi, angle=angle, shift=shift), self.assertRaises(ValueError):
                ManualToFProjector(roi, angle, shift)


if __name__ == '__main__':
    unittest.main()
