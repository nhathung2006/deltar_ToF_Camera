import unittest

import numpy as np
import torch
from torch import nn

from apps.live_depth import make_input
from src.camera_tof_io import grid_rectangles
from src.models.fusion import TransformerFusion
from src.tof_projection import ToFProjector, ManualToFProjector
from src import tof_projection
from test_tof_projection import synthetic_calibration, SourceMessage


class ToFConfidenceTests(unittest.TestCase):
    def scene(self, flat=False):
        projector = ToFProjector(synthetic_calibration())
        distances = np.full((8, 8), 1800.)
        distances[:, :4] = 1800. if flat else 350.
        projection = projector.project(distances.ravel())
        # Depth cell ends at x=320; the RGB foreground ends earlier at x=296.
        image = np.full((480, 640, 3), 200, np.uint8)
        image[:, :296] = [70, 110, 150]
        return projector, projection, image

    def test_color_boundary_reduces_near_message_on_background_only(self):
        _, projection, image = self.scene()
        confidence = tof_projection.rgb_boundary_confidence(image, projection)
        self.assertLess(confidence[220, 312], .25)
        self.assertGreater(confidence[220, 280], .95)
        self.assertEqual(confidence[220, 350], 1.)

    def test_texture_on_one_depth_plane_keeps_full_confidence(self):
        _, projection, image = self.scene(flat=True)
        confidence = tof_projection.rgb_boundary_confidence(image, projection)
        np.testing.assert_array_equal(confidence[32:448, 112:528], 1.)
        self.assertEqual(confidence[0, 0], 0.)

    def test_equal_colors_do_not_invent_a_foreground_boundary(self):
        _, projection, image = self.scene()
        image[:] = 100
        confidence = tof_projection.rgb_boundary_confidence(image, projection)
        np.testing.assert_array_equal(confidence[32:448, 112:528], 1.)

    def test_missing_zones_are_not_filled_and_depths_are_unchanged(self):
        projector, _, image = self.scene()
        distances = np.full(64, 1800.)
        distances[27] = 0
        projection = projector.project(distances)
        before = projection['distance_mm'].copy()
        confidence = tof_projection.rgb_boundary_confidence(image, projection)
        self.assertEqual(confidence[212, 292], 0.)
        np.testing.assert_array_equal(projection['distance_mm'], before)

    def test_weighted_messages_reach_every_fusion_scale_without_changing_rgb(self):
        projector, projection, image = self.scene()
        confidence = np.full((480, 640), .25, np.float32)
        grids = projector.fusion_grids(projection, confidence=confidence)
        inputs, _ = make_input(image, projection['distance_mm'],
                              grid_rectangles([96, 16, 448, 448]), .02, 'cpu', 10,
                              projection_grids=grids, allow_empty=True)
        for stride in (4, 8, 16):
            fusion = TransformerFusion(8, (480//stride, 640//stride), attention_layer=['hist2image'])
            fusion.layers = nn.ModuleList([SourceMessage()])
            with torch.no_grad():
                fusion.positional_encodings.zero_()
                fusion.positional_encodings2.zero_()
                x = torch.full((1, 8, 480//stride, 640//stride), 2.)
                sources = torch.full((1, 64, 16, 8), 4.)
                result = fusion(x, sources, **inputs['additional'])
            np.testing.assert_allclose(result[0, :, 200//stride, 280//stride], 3.)
            np.testing.assert_allclose(result[0, :, 0, 0], 2.)

    def test_confidence_rejects_invalid_shapes_or_weights(self):
        projector, projection, image = self.scene()
        with self.assertRaises(ValueError):
            tof_projection.rgb_boundary_confidence(image.astype(np.float32), projection)
        for confidence in (np.ones((10, 10)), np.full((480, 640), np.nan), np.full((480, 640), 1.1)):
            with self.assertRaises(ValueError):
                projector.fusion_grids(projection, confidence=confidence)

    def test_uncovered_pixels_do_not_reduce_flat_plane_confidence_at_borders(self):
        projector = ManualToFProjector([96, 24, 448, 448])
        projection = projector.project(np.full(64, 1000.))
        confidence = tof_projection.rgb_boundary_confidence(np.full((480, 640, 3), 100, np.uint8), projection)
        for stage in projector.fusion_grids(projection, confidence=confidence).values():
            np.testing.assert_allclose(stage['tof_confidence'][stage['coverage_mask']], 1.)


if __name__ == '__main__':
    unittest.main()
