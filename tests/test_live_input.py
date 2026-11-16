import unittest

import numpy as np
import torch

from apps.live_depth import make_input, orient_tof
from src.camera_tof_io import grid_rectangles, parse_tof_line


class LiveInputTests(unittest.TestCase):
    def test_board_frame_and_wrapping_counter(self):
        for counter in (255, 0):
            text = f'F,{counter},' + ','.join(map(str, range(64))) + '\r\n'
            for raw in (text, text.encode(), bytearray(text.encode())):
                frame = parse_tof_line(raw, received_at=123.0)
                self.assertEqual(frame.counter, counter)
                self.assertEqual(frame.received_at, 123.0)
                np.testing.assert_array_equal(frame.distance_mm, np.arange(64))

    def test_bad_frames_are_not_used_as_measurements(self):
        valid = 'F,4,' + ','.join(['1000'] * 64)
        for text in ('I (100) Sensor ready', 'F,1,10,20', valid.replace('1000', 'oops', 1),
                     valid.replace('1000', '-1', 1), valid + ',1000', b'F,\xff'):
            self.assertIsNone(parse_tof_line(text))

    def test_f2_metadata_and_status_filtering(self):
        distances = [1000] * 64
        statuses = [5] * 64
        distances[3] = -10  # Raw invalid zones can contain negative distances.
        statuses[:4] = [9, 255, 4, 0]
        sigmas = list(range(64))
        text = ','.join(map(str, ['F2', 0, 176625, *distances, *statuses, *sigmas])) + '\r\n'
        for raw in (text, text.encode(), bytearray(text.encode())):
            frame = parse_tof_line(raw, received_at=123.0)
            self.assertEqual(frame.protocol, 'F2')
            self.assertEqual(frame.counter, 0)
            self.assertEqual(frame.sensor_timestamp, 176625)
            self.assertEqual(frame.received_at, 123.0)
            expected = np.full(64, 1000)
            expected[1:4] = 0
            np.testing.assert_array_equal(frame.distance_mm, expected)
            np.testing.assert_array_equal(frame.target_status, statuses)
            np.testing.assert_array_equal(frame.range_sigma_mm, sigmas)

    def test_malformed_f2_frames_are_rejected(self):
        fields = ['F2', '4', '176625'] + ['1000'] * 64 + ['5'] * 64 + ['20'] * 64
        cases = [fields[:-1], fields + ['20']]
        for index, invalid in ((1, '-1'), (2, '-1'), (2, 'oops'), (3, 'nan'),
                               (67, '256'), (67, '-1'), (131, '-1'), (131, 'nan'),
                               (131, 'inf')):
            changed = fields.copy()
            changed[index] = invalid
            cases.append(changed)
        for case in cases:
            with self.subTest(fields=case[:3]):
                self.assertIsNone(parse_tof_line(','.join(case)))

    def test_millimetres_masks_and_rgb_normalization(self):
        values = np.full(64, 1000, dtype=np.float32)
        values[0], values[1] = 0, 5000
        data, valid = make_input(np.zeros((480, 640, 3), dtype=np.uint8), values,
                                 grid_rectangles([96, 16, 448, 448]), 0.02, torch.device('cpu'))
        self.assertEqual(valid, 62)
        self.assertEqual(tuple(data['additional']['hist_data'].shape), (1, 64, 16))
        self.assertFalse(data['additional']['mask'][0, 0])
        self.assertFalse(data['additional']['mask'][0, 1])
        self.assertAlmostEqual(data['additional']['hist_data'][0, 2].mean().item(), 1.0, places=4)
        self.assertAlmostEqual(data['rgb'][0, 0, 0, 0].item(), -0.485 / 0.229, places=5)
        self.assertEqual(data['additional']['patch_info']['zone_num'].item(), 8)

    def test_orientation_keeps_zone_locations_consistent(self):
        values = np.arange(64, dtype=np.float32)
        oriented = orient_tof(values, rotation=90, flip_x=True)
        np.testing.assert_array_equal(oriented.reshape(8, 8), np.fliplr(np.rot90(values.reshape(8, 8))))
        rects = grid_rectangles([96, 16, 448, 448])
        np.testing.assert_array_equal(rects[0], [16, 96, 72, 152])
        np.testing.assert_array_equal(rects[-1], [408, 488, 464, 544])


if __name__ == '__main__':
    unittest.main()
