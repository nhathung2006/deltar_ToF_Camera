import unittest

import numpy as np

from src import depth_smoothing as smoothing


class SmoothingTests(unittest.TestCase):
    def test_valid_measurement_reinitializes_expired_history_without_zero_frame(self):
        f = smoothing.ToFTemporalFilter()
        f.update(np.full(64, 1700.), 0.)
        self.assertEqual(f.update(np.full(64, 320.), .2)[0], 320.)
        self.assertEqual(f.update(np.full(64, 320.), .4)[0], 320.)

    def test_reacquisition_after_dropout_uses_new_surface(self):
        f = smoothing.ToFTemporalFilter()
        f.update(np.full(64, 1700.), 0.)
        f.update(np.zeros(64), .1)
        result = f.update(np.full(64, 320.), .21)
        np.testing.assert_array_equal(result, 320.)

    def test_isolated_tof_spike_is_rejected_but_sustained_step_is_accepted(self):
        f = smoothing.ToFTemporalFilter()
        first = np.full(64, 1200., np.float32)
        f.update(first, 0.)
        spike = first.copy(); spike[27] = 240
        self.assertEqual(f.update(spike, .067)[27], 1200)
        self.assertEqual(f.update(first, .134)[27], 1200)
        f.update(spike, .201)
        self.assertAlmostEqual(f.update(spike, .268)[27], 240)

    def test_invalid_tof_is_held_briefly_then_expires(self):
        f = smoothing.ToFTemporalFilter()
        f.update(np.full(64, 1000.), 0.)
        self.assertEqual(f.update(np.zeros(64), .067)[0], 1000)
        self.assertEqual(f.update(np.zeros(64), .4)[0], 0)
        self.assertEqual(f.update(np.full(64, 1500.), 1.)[0], 1500)

    def test_bad_input_does_not_mutate_tof_state(self):
        f = smoothing.ToFTemporalFilter()
        f.update(np.full(64, 1000.), 1.)
        with self.assertRaises(ValueError):
            f.update(np.ones(63), 1.1)
        with self.assertRaises(ValueError):
            f.update(np.ones(64), float('nan'))
        self.assertTrue(np.all(f.update(np.full(64, 1000.), 1.2) == 1000))

    def test_guided_filter_smooths_false_seams_and_preserves_rgb_depth_edge(self):
        rgb = np.full((96, 192, 3), 30, np.uint8)
        rgb[:, 96:] = 230
        depth = np.ones((96, 192), np.float32)
        depth[:, 96:] = 2
        depth[:, 30:54] += .2
        out = smoothing.guided_depth_filter(rgb, depth, radius=16)
        self.assertLess(abs(float(out[48, 30]-out[48, 29])), .04)
        self.assertGreater(float(out[:, 102].mean()-out[:, 90].mean()), .75)
        self.assertTrue(np.isfinite(out).all())

    def test_guided_constant_depth_stays_metric_even_with_rgb_texture(self):
        rgb = np.random.default_rng(5).integers(0, 256, (47, 63, 3), dtype=np.uint8)
        depth = np.full((47, 63), 1.25, np.float32)
        np.testing.assert_allclose(smoothing.guided_depth_filter(rgb, depth), depth, atol=1e-4)

    def test_temporal_depth_reduces_jitter_and_releases_real_motion(self):
        f = smoothing.DepthTemporalFilter()
        rgb = np.full((48, 64, 3), 80, np.uint8)
        f.update(np.full((48, 64), 1., np.float32), rgb, 0.)
        out = f.update(np.full((48, 64), 1.04, np.float32), rgb, .067)
        self.assertLess(float(out.mean()), 1.025)
        changed = rgb.copy(); changed[:, 32:] = 220
        depth = np.full((48, 64), 1., np.float32); depth[:, 32:] = .5
        out = f.update(depth, changed, .134)
        np.testing.assert_allclose(out[:, 32:], .5, atol=1e-6)
        np.testing.assert_allclose(f.update(depth, changed, 2.), depth, atol=1e-6)

    def test_delayed_depth_follows_recent_local_rgb_motion(self):
        f = smoothing.DepthTemporalFilter()
        rgb = np.full((48, 64, 3), 80, np.uint8)
        near = np.ones((48, 64), np.float32)
        f.update(near, rgb, 0.)
        changed = rgb.copy(); changed[:, 32:] = 220
        # RGB observes departure first; depth responds one observation later.
        f.update(near, changed, .067)
        result = f.update(np.full_like(near, 2.), changed, .134)
        np.testing.assert_allclose(result[:, 40:], 2., atol=1e-6)
        # No RGB motion on the left: its isolated depth jump is still held.
        np.testing.assert_allclose(result[:, :24], 1., atol=1e-6)

    def test_recent_motion_expires_before_later_isolated_depth_spike(self):
        f = smoothing.DepthTemporalFilter()
        rgb = np.full((8, 8, 3), 80, np.uint8)
        changed = np.full_like(rgb, 220)
        near = np.ones((8, 8), np.float32)
        f.update(near, rgb, 0.)
        f.update(near, changed, .067)
        f.update(near, changed, .2)
        np.testing.assert_allclose(f.update(near * 2, changed, .3), near)

    def test_reset_gap_and_shape_change_clear_recent_motion(self):
        for restart in ('reset', 'gap', 'shape'):
            with self.subTest(restart=restart):
                f = smoothing.DepthTemporalFilter()
                rgb = np.full((8, 8, 3), 80, np.uint8)
                changed = np.full_like(rgb, 220)
                near = np.ones((8, 8), np.float32)
                f.update(near, rgb, 0.)
                f.update(near, changed, .067)
                timestamp = .1
                if restart == 'reset':
                    f.reset()
                elif restart == 'gap':
                    timestamp = 1.
                else:
                    changed = np.full((12, 12, 3), 220, np.uint8)
                    near = np.ones((12, 12), np.float32)
                f.update(near, changed, timestamp)
                np.testing.assert_allclose(f.update(near * 2, changed, timestamp + .067), near)

    def test_depth_spike_on_unchanged_rgb_needs_confirmation(self):
        f = smoothing.DepthTemporalFilter()
        rgb = np.full((48, 64, 3), 80, np.uint8)
        first = np.ones((48, 64), np.float32)
        f.update(first, rgb, 0.)
        spike = first.copy(); spike[12:36, 16:48] = 2.
        np.testing.assert_allclose(f.update(spike, rgb, .067), first, atol=1e-6)
        np.testing.assert_allclose(f.update(first, rgb, .134), first, atol=1e-6)
        f.update(spike, rgb, .201)
        # A consistent new surface is released at the second observation.
        np.testing.assert_allclose(f.update(spike, rgb, .268), spike, atol=1e-6)

    def test_unconfirmed_depth_jumps_cannot_freeze_history_forever(self):
        f = smoothing.DepthTemporalFilter()
        rgb = np.full((8, 8, 3), 80, np.uint8)
        f.update(np.ones((8, 8), np.float32), rgb, 0.)
        for timestamp, value in [(.067, 2.), (.134, 3.), (.201, 2.5)]:
            f.update(np.full((8, 8), value, np.float32), rgb, timestamp)
        result = f.update(np.full((8, 8), 3.2, np.float32), rgb, .268)
        self.assertGreater(float(result.mean()), 1.1)
        self.assertLess(float(result.mean()), 3.2)

    def test_clean_preview_keeps_rgb_pixels_and_has_no_tof_grid(self):
        from apps.live_depth import clean_preview_image
        rgb = np.full((480, 640, 3), [23, 47, 91], np.uint8)
        depth = np.ones((480, 640), np.float32)
        display = clean_preview_image(rgb, depth, 4., 15., 'combined', 48)
        np.testing.assert_array_equal(display[32:512, :640], rgb)
        self.assertEqual(display.shape, (552, 1280, 3))

    def test_comparison_preview_shows_rgb_original_and_smoothed_depth(self):
        from apps.live_depth import comparison_preview_image
        rgb = np.full((480, 640, 3), [23, 47, 91], np.uint8)
        original = np.full((480, 640), 1., np.float32)
        smoothed = np.full((480, 640), 2., np.float32)
        display = comparison_preview_image(rgb, original, smoothed, 4., 15., 'combined')
        self.assertEqual(display.shape, (384, 1280, 3))
        np.testing.assert_array_equal(display[180, 100], rgb[225, 150])
        self.assertFalse(np.array_equal(display[180, 640], display[180, 1080]))

if __name__ == '__main__':
    unittest.main()
