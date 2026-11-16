"""Exercise the real live loop with finite simulated camera/serial devices."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from apps import live_depth as run_live
from src.camera_tof_io import ToFFrame


class LiveDropoutTests(unittest.TestCase):
    def test_all_invalid_sensor_frames_update_output_instead_of_freezing(self):
        clock = [100.]

        class Camera:
            count = 0

            def read(self):
                self.count += 1
                if self.count > 3:
                    raise AssertionError('Live loop skipped the invalid observations')
                clock[0] = 100. + self.count * .2
                return True, np.full((480, 640, 3), 80, np.uint8)

            def release(self):
                pass

        camera = Camera()

        class Reader:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def snapshot(self):
                values = np.full(64, 1000. if camera.count == 1 else 0., np.float32)
                return ToFFrame(camera.count, values, clock[0]), camera.count, 0

        class CheapModel(torch.nn.Module):
            def forward(self, data):
                # Network cost is irrelevant to loop/dropout behavior. Real input
                # preparation and filtering still run; mask controls this output.
                value = 1. + data['additional']['mask'].float().mean()
                return None, torch.ones((1, 1, 120, 160)) * value

        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run_live, 'open_camera', return_value=(camera, 1)))
            stack.enter_context(patch.object(run_live, 'SerialToF', return_value=Reader()))
            stack.enter_context(patch.object(run_live, 'Deltar', return_value=CheapModel()))
            stack.enter_context(patch.object(run_live.torch, 'load', return_value={}))
            stack.enter_context(patch.object(run_live.time, 'monotonic', side_effect=lambda: clock[0]))
            stack.enter_context(patch('sys.argv', ['apps.live_depth', '--preview', '--sigma-m', '.02',
                                                 '--device', 'cpu', '--headless', '--frames', '3',
                                                 '--smoothing', 'tof', '--save-dir', folder]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            run_live.main()
            report = json.loads((Path(folder) / 'summary.json').read_text())
            self.assertEqual(report['frames'], 3)
            self.assertEqual(report['valid_tof_zones'], 0)
            self.assertEqual(report['filtered_valid_tof_zones'], 0)
            np.testing.assert_array_equal(np.load(Path(folder) / 'last_depth_m.npy'), 1.)
            np.testing.assert_array_equal(np.load(Path(folder) / 'last_tof_filtered_mm.npy'), 0.)


if __name__ == '__main__':
    unittest.main()
