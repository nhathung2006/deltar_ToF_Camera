import unittest

import cv2
import numpy as np

from src.camera_tof_calibration import (CameraUndistorter, board_points, camera_for_plane_target, camera_plane, fit_point_to_plane, nominal_tof_rays,
                             summarize_static_tof, tof_points)


class CalibrationTests(unittest.TestCase):
    def test_zero_distortion_preserves_the_image(self):
        camera = {'image_size': [32, 24], 'K': [[40, 0, 16], [0, 40, 12], [0, 0, 1]], 'distortion': [0] * 5}
        correction = CameraUndistorter(camera, expected_size=(32, 24))
        image = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        np.testing.assert_array_equal(correction.apply(image), image)
        with self.assertRaisesRegex(ValueError, 'Expected image size'):
            correction.apply(image[:20])

    def test_remap_coordinates_use_original_intrinsics(self):
        K = np.array([[80., 0, 40], [0, 80, 30], [0, 0, 1]])
        dist = np.array([0.15, -0.03, 0.002, -0.001, 0.0])
        camera = {'image_size': [80, 60], 'K': K.tolist(), 'distortion': dist.tolist()}
        correction = CameraUndistorter(camera, expected_size=(80, 60))
        for x, y in [(10, 10), (60, 45), (40, 30)]:
            raw = np.array([[[correction.map_x[y, x], correction.map_y[y, x]]]], dtype=np.float64)
            ideal = cv2.undistortPoints(raw, K, dist, P=K).reshape(2)
            np.testing.assert_allclose(ideal, [x, y], atol=1e-3)

    def test_undistortion_rejects_incompatible_calibration(self):
        camera = {'image_size': [1280, 720], 'K': [[600, 0, 320], [0, 600, 240], [0, 0, 1]], 'distortion': [0] * 5}
        with self.assertRaisesRegex(ValueError, 'image size'):
            CameraUndistorter(camera)
        camera['image_size'] = [640, 480]
        camera['distortion'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'distortion'):
            CameraUndistorter(camera)

    def test_board_coordinates_are_metres_and_inner_corners(self):
        points = board_points(9, 6, 25)
        self.assertEqual(points.shape, (54, 3))
        np.testing.assert_allclose(points[1], [0.025, 0, 0], atol=1e-8)
        np.testing.assert_allclose(points[-1], [0.2, 0.125, 0], atol=1e-8)

    def test_checkerboard_pnp_and_corner_shapes(self):
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        for row in range(7):
            for col in range(10):
                if (row + col) % 2 == 0:
                    image[80 + row * 40:120 + row * 40, 100 + col * 40:140 + col * 40] = 0
        camera = {'image_size': [640, 480], 'columns': 9, 'rows': 6, 'square_mm': 25,
                  'K': [[600, 0, 320], [0, 600, 240], [0, 0, 1]], 'distortion': [0] * 5}
        normal, offset, corners, rms = camera_plane(image, camera)
        self.assertEqual(corners.shape, (54, 1, 2))
        self.assertLess(rms, 0.2)
        self.assertAlmostEqual(abs(normal[2]), 1.0, places=3)
        self.assertAlmostEqual(abs(offset), 600 * 0.025 / 40, places=3)

        # The same observed pixels on a 15 mm board imply 60% of the distance.
        # Lens intrinsics and the original intrinsic board metadata stay intact.
        target = {'columns': 9, 'rows': 6, 'square_mm': 15}
        plane_camera = camera_for_plane_target(camera, target)
        normal15, offset15, corners15, rms15 = camera_plane(image, plane_camera)
        np.testing.assert_allclose(normal15, normal, atol=1e-5)
        np.testing.assert_array_equal(corners15, corners)
        self.assertAlmostEqual(offset15 / offset, 0.6, places=5)
        self.assertAlmostEqual(rms15, rms, places=4)
        self.assertEqual(camera['square_mm'], 25)
        self.assertEqual(plane_camera['K'], camera['K'])
        self.assertEqual(plane_camera['distortion'], camera['distortion'])

    def test_plane_target_size_can_be_overridden_and_rejects_invalid_values(self):
        camera = {'columns': 9, 'rows': 6, 'square_mm': 25}
        self.assertEqual(camera_for_plane_target(camera)['square_mm'], 25)
        self.assertEqual(camera_for_plane_target(camera, {'square_mm': 15}, 20)['square_mm'], 20)
        for invalid in (0, -15, float('nan'), float('inf')):
            with self.subTest(square_mm=invalid), self.assertRaisesRegex(ValueError, 'square size'):
                camera_for_plane_target(camera, {'square_mm': invalid})

    def test_distance_conventions_are_distinct(self):
        rays = nominal_tof_rays(45, 45)
        mm = np.full(64, 1000.0)
        radial = tof_points(mm, rays, 'radial')
        axial = tof_points(mm, rays, 'z')
        np.testing.assert_allclose(np.linalg.norm(radial, axis=1), 1.0)
        np.testing.assert_allclose(axial[:, 2], 1.0)
        self.assertGreater(np.linalg.norm(axial[0]), 1.1)

    def test_invalid_and_unstable_measurements_are_filtered(self):
        samples = np.full((20, 64), 1000.0)
        samples[:, 0] = 0
        samples[:5, 1] = 0
        samples[:, 2] = np.arange(20) * 100
        median, stable = summarize_static_tof(samples)
        self.assertFalse(stable[:3].any())
        self.assertTrue(stable[3:].all())
        self.assertEqual(median[3], 1000)

    def test_extrinsics_recover_a_known_synthetic_rig(self):
        rays = nominal_tof_rays(45, 45)
        expected_R, _ = cv2.Rodrigues(np.array([0.07, -0.04, 0.02]))
        expected_t = np.array([0.035, -0.012, 0.008])
        observations = []
        for i in range(30):
            n = np.array([np.sin((i % 5 - 2) * 0.22), np.sin((i // 5 - 2) * 0.19), 1.0])
            n /= np.linalg.norm(n)
            offset = -0.5 - i * 0.025
            z = -(n @ expected_t + offset) / ((rays @ expected_R.T) @ n)
            observations.append({'points': rays * z[:, None], 'normal': n, 'offset': offset})
        R, t, metrics = fit_point_to_plane(observations, [1, 7, 13, 19, 25])
        np.testing.assert_allclose(R, expected_R, atol=1e-6)
        np.testing.assert_allclose(t, expected_t, atol=1e-6)
        self.assertLess(metrics['validation_mae_mm'], 1e-3)

    def test_parallel_planes_cannot_determine_a_rig(self):
        observations = [{'points': np.ones((64, 3)), 'normal': np.array([0, 0, 1.0]), 'offset': -1.0}
                        for _ in range(20)]
        with self.assertRaisesRegex(ValueError, 'normals'):
            fit_point_to_plane(observations, [1, 4, 8, 12])


if __name__ == '__main__':
    unittest.main()
