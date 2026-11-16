import unittest

import numpy as np
import torch
from torch import nn

from apps.live_depth import make_input
from src.camera_tof_io import grid_rectangles
from src.models.fusion import TransformerFusion
from src.tof_projection import ToFProjector, draw_projected_zones, visible_zone_map


def synthetic_calibration():
    focal = 224 / np.tan(np.deg2rad(22.5))
    return {
        'camera': {'image_size': [640, 480],
                   'K': [[focal, 0, 320], [0, focal, 240], [0, 0, 1]],
                   'distortion': [0] * 5},
        'R_tof_to_camera': np.eye(3).tolist(), 't_tof_to_camera_m': [0, 0, 0],
        'tof_fov_xy_deg': [45, 45], 'depth_convention': 'z',
        'rotate_tof_ccw': 180, 'flip_tof_x': False, 'flip_tof_y': False,
    }


class SourceMessage(nn.Module):
    def forward(self, query, source):
        return source.mean(dim=1, keepdim=True).expand_as(query)


class QueryMessage(nn.Module):
    def forward(self, query, source):
        return query


class ToFProjectionTests(unittest.TestCase):
    def overlapping_projection(self):
        calibration = synthetic_calibration()
        calibration['t_tof_to_camera_m'] = [.1, 0, 0]
        projector = ToFProjector(calibration)
        distances = np.full(64, 1000.)
        distances[28] = 400
        return projector, projector.project(distances)

    def test_nearest_zone_owns_overlap_and_no_far_edge_is_drawn_inside_it(self):
        projector, projection = self.overlapping_projection()
        owner = visible_zone_map(projection, (640, 480))['owner']
        self.assertEqual(owner[212, 500], 28)
        self.assertEqual(owner[212, 530], 30)
        image = draw_projected_zones(np.zeros((480, 640, 3), np.uint8), projection)
        # Zone 30's old left edge (~486 px) is hidden behind the closer zone 28.
        self.assertTrue(np.all(owner[212, 484:489] == 28))
        self.assertFalse(image[212, 484:489].any())
        self.assertTrue(image.any())

    def test_all_fusion_scales_use_the_visible_owner_without_mixing_zones(self):
        projector, projection = self.overlapping_projection()
        owner = visible_zone_map(projection, (640, 480))['owner']
        grids = projector.fusion_grids(projection)
        for stride, stage in grids.items():
            patch = int(np.ceil(56 / stride))
            ids = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 8, 8)
            tiles = ids.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)
            grid = torch.from_numpy(stage['scatter_grid']).unsqueeze(0)
            sampled = torch.nn.functional.grid_sample(tiles, grid, align_corners=True)[0, 0].numpy()
            expected = owner[::stride, ::stride]
            np.testing.assert_array_equal(stage['coverage_mask'][0], expected >= 0)
            np.testing.assert_allclose(sampled[expected >= 0], expected[expected >= 0] + 1, atol=1e-4)
            self.assertFalse(sampled[expected < 0].any())

    def test_invalid_zones_never_appear_in_visible_owner(self):
        projector = ToFProjector(synthetic_calibration())
        distances = np.zeros(64)
        distances[27] = 1000
        projection = projector.project(distances)
        owner = visible_zone_map(projection, (640, 480))['owner']
        np.testing.assert_array_equal(np.unique(owner), [-1, 27])
        empty = projector.project(np.zeros(64))
        self.assertFalse(draw_projected_zones(np.zeros((480, 640, 3), np.uint8), empty).any())

    def test_identity_geometry_matches_known_roi_and_depth(self):
        projector = ToFProjector(synthetic_calibration())
        projected = projector.project(np.full(64, 1000))
        np.testing.assert_allclose(projected['polygons'][0],
                                   [[96, 16], [152, 16], [152, 72], [96, 72]], atol=1e-6)
        np.testing.assert_allclose(projected['polygons'][-1, 2], [544, 464], atol=1e-6)
        np.testing.assert_allclose(projected['distance_mm'], 1000)
        self.assertTrue(projected['valid'].all())
        np.testing.assert_array_equal(projector.orient(np.arange(64)), np.arange(64)[::-1])

    def test_translation_has_depth_dependent_parallax_and_camera_z(self):
        calibration = synthetic_calibration()
        calibration['t_tof_to_camera_m'] = [.05, 0, .1]
        projector = ToFProjector(calibration)
        close = projector.project(np.full(64, 500))
        far = projector.project(np.full(64, 1000))
        np.testing.assert_allclose(close['distance_mm'], 600)
        np.testing.assert_allclose(far['distance_mm'], 1100)
        # Centre of the complete nominal FoV: analytic pinhole translation.
        for result, depth in [(close, .6), (far, 1.1)]:
            H = result['homography'][27]
            pixel = H @ [1, 1, 1]  # Zone (3,3)'s bottom-right is the ToF optical axis.
            self.assertAlmostEqual(pixel[0] / pixel[2], 320 + projector.K[0, 0] * .05 / depth)
            self.assertAlmostEqual(pixel[1] / pixel[2], 240)

    def test_radial_conversion_and_invalid_or_behind_camera_zones(self):
        calibration = synthetic_calibration()
        calibration['depth_convention'] = 'radial'
        projector = ToFProjector(calibration)
        result = projector.project(np.full(64, 1000))
        expected = 1000 / np.linalg.norm(projector.rays, axis=1)
        np.testing.assert_allclose(result['distance_mm'], expected, atol=1e-4)
        distances = np.full(64, 1000.)
        distances[:3] = [0, float('nan'), 5000]
        self.assertFalse(projector.project(distances)['valid'][:3].any())
        calibration['t_tof_to_camera_m'] = [0, 0, -2]
        behind = ToFProjector(calibration).project(np.full(64, 1000))
        self.assertFalse(behind['valid'].any())
        self.assertFalse(ToFProjector(calibration).fusion_grids(behind)[4]['coverage_mask'].any())

    def test_bad_rotation_is_rejected(self):
        calibration = synthetic_calibration()
        calibration['R_tof_to_camera'][0][0] = 2
        with self.assertRaisesRegex(ValueError, 'rotation'):
            ToFProjector(calibration)

    def test_fusion_preserves_zone_identity_and_does_not_fill_invalid_zones(self):
        projector = ToFProjector(synthetic_calibration())
        distances = np.full(64, 1000.)
        distances[18] = 0
        projection = projector.project(distances)
        grids = projector.fusion_grids(projection)
        data, count = make_input(np.zeros((480, 640, 3), np.uint8),
                                 projection['distance_mm'], grid_rectangles([96, 16, 448, 448]),
                                 .02, torch.device('cpu'), 10,
                                 projection_grids=grids, sigma_scale=projection['sigma_scale'])
        self.assertEqual(count, 63)
        fusion = TransformerFusion(8, (120, 160), attention_layer=['hist2image'])
        fusion.layers = nn.ModuleList([SourceMessage()])
        with torch.no_grad():
            fusion.positional_encodings.zero_()
            fusion.positional_encodings2.zero_()
            features = torch.arange(1, 65, dtype=torch.float32)[None, :, None, None].expand(1, 64, 16, 8)
            result = fusion(torch.zeros(1, 8, 120, 160), features, **data['additional'])
        for zone in range(64):
            row, col = divmod(zone, 8)
            x = int((96 + col * 56 + 28) / 4)
            y = int((16 + row * 56 + 28) / 4)
            expected = 0 if zone == 18 else zone + 1
            np.testing.assert_allclose(result[0, :, y, x].numpy(), expected, atol=1e-4)
        self.assertEqual(result[0, :, 0, 0].abs().sum().item(), 0)

    def test_empty_projection_is_a_supported_rgb_only_fallback(self):
        projector = ToFProjector(synthetic_calibration())
        projection = projector.project(np.zeros(64))
        inputs, count = make_input(np.zeros((480, 640, 3), np.uint8),
                                   projection['distance_mm'], grid_rectangles([96, 16, 448, 448]),
                                   .02, torch.device('cpu'), 10,
                                   projection_grids=projector.fusion_grids(projection), allow_empty=True)
        self.assertEqual(count, 0)
        self.assertFalse(inputs['additional']['mask'].any())
        self.assertTrue(torch.isfinite(inputs['additional']['hist_data']).all())

    def test_sampling_and_scatter_agree_on_a_translated_rgb_feature(self):
        calibration = synthetic_calibration()
        calibration['t_tof_to_camera_m'] = [.05, .025, 0]
        projector = ToFProjector(calibration)
        projection = projector.project(np.full(64, 1000))
        inputs, _ = make_input(np.zeros((480, 640, 3), np.uint8),
                               projection['distance_mm'], grid_rectangles([96, 16, 448, 448]),
                               .02, torch.device('cpu'), 10,
                               projection_grids=projector.fusion_grids(projection))
        fusion = TransformerFusion(8, (120, 160), attention_layer=['hist2image'])
        fusion.layers = nn.ModuleList([QueryMessage()])
        with torch.no_grad():
            fusion.positional_encodings.zero_()
            fusion.positional_encodings2.zero_()
            image_features = torch.arange(160, dtype=torch.float32)[None, None, None].expand(1, 8, 120, 160).clone()
            result = fusion(image_features, torch.zeros(1, 64, 16, 8), **inputs['additional'])
        # Bilinear gather and inverse scatter preserve a linear RGB feature.
        for y, x in [(30, 55), (58, 83), (86, 111)]:
            np.testing.assert_allclose(result[0, :, y, x].numpy(), x * 2, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
