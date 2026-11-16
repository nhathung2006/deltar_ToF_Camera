"""Check the Logitech camera and existing ESP32 firmware without flashing it."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from serial.tools import list_ports

from src.camera_tof_io import SerialToF, camera_inventory, open_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM9')
    parser.add_argument('--baud', type=int, default=921600)
    parser.add_argument('--camera', type=int)
    parser.add_argument('--seconds', type=float, default=4)
    parser.add_argument('--show-serial', action='store_true')
    parser.add_argument('--output', default='outputs/device_check.json')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    report = {
        'cameras': [{'index': c.index, 'name': c.name} for c in camera_inventory()],
        'ports': [{'port': p.device, 'description': p.description} for p in list_ports.comports()],
    }
    print(json.dumps(report, indent=2, ensure_ascii=True), flush=True)
    cap, index = open_camera(args.camera)
    try:
        ok, image = cap.read()
        if not ok:
            raise RuntimeError('Camera opened but returned no image.')
        report['camera_check'] = {'ok': True, 'index': index, 'width': image.shape[1], 'height': image.shape[0]}
    finally:
        cap.release()
    with SerialToF(args.port, args.baud) as reader:
        started = time.monotonic()
        last_received = None
        valid_counts = []
        positive_values = []
        while time.monotonic() - started < args.seconds:
            frame, count, ignored = reader.snapshot()
            if frame is not None and frame.received_at != last_received:
                last_received = frame.received_at
                positive = frame.distance_mm[frame.distance_mm > 0]
                valid_counts.append(len(positive))
                positive_values.extend(positive.tolist())
                if args.show_serial:
                    if frame.protocol == 'F2':
                        print(f'F2 counter={frame.counter}, board_timestamp={frame.sensor_timestamp}, '
                              f'valid_status_5_9={len(positive)}/64; filtered distances (mm):', flush=True)
                    print(f'F,{frame.counter},' + ','.join(str(int(v)) for v in frame.distance_mm), flush=True)
            time.sleep(0.01)
        frame, count, ignored = reader.snapshot()
        if frame is None:
            raise RuntimeError(f'No valid F or F2 frames from {args.port} at {args.baud} baud '
                               f'({ignored} ignored lines).')
        report['tof_check'] = {
            'ok': True, 'port': args.port, 'baud': args.baud, 'unit': 'mm',
            'frames_received': count, 'received_fps': round(count / (time.monotonic() - started), 2),
            'ignored_lines': ignored, 'last_counter': frame.counter,
            'protocol': frame.protocol, 'sensor_timestamp': frame.sensor_timestamp,
            'valid_zones_last': int(np.count_nonzero(frame.distance_mm)),
            'valid_zones_mean': round(float(np.mean(valid_counts)), 1),
            'positive_mm_min': min(positive_values) if positive_values else None,
            'positive_mm_max': max(positive_values) if positive_values else None,
            'last_distance_mm': frame.distance_mm.astype(int).reshape(8, 8).tolist(),
        }
        if frame.target_status is not None:
            report['tof_check']['last_target_status'] = frame.target_status.reshape(8, 8).tolist()
            report['tof_check']['last_range_sigma_mm'] = frame.range_sigma_mm.reshape(8, 8).tolist()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['camera_check'], indent=2))
    print(json.dumps(report['tof_check'], indent=2))
    print(f'PASS. Report: {target.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
