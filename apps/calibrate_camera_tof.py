"""Collect stationary checkerboard/ToF observations and estimate calibration.

Use --help or see README.md. No device is opened unless a capture
command is selected. ToF extrinsics use explicit, approximate per-zone rays.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.camera_tof_calibration import (board_points, camera_for_plane_target, camera_plane, find_board, fit_point_to_plane,
                             nominal_tof_rays, orient, read_camera, summarize_static_tof,
                             tof_points, write_json)
from src.camera_tof_io import SerialToF, open_camera


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write image: {path}')


def capture_camera(args):
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {'columns': args.columns, 'rows': args.rows, 'square_mm': args.square_mm,
                'image_size': [640, 480], 'kind': 'camera_intrinsic_capture'}
    settings_path = args.output / 'session.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('This folder has different target settings. Use a new --output folder.')
    write_json(settings_path, settings)
    index = len(list(args.output.glob('rgb_*.png')))
    cap, camera_index = open_camera(args.camera)
    print(f'Camera {camera_index}. Hold board still, S saves a raw image, Q exits. Need 30-50 diverse views.', flush=True)
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError('No camera image.')
            if image.shape[:2] != (480, 640):
                raise ValueError('Camera must return 640x480.')
            corners = find_board(image, args.columns, args.rows)
            display = image.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, (args.columns, args.rows), corners, True)
            cv2.putText(display, f'{"FOUND" if corners is not None else "NO BOARD"} | saved {index} | S save | Q quit',
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0) if corners is not None else (0, 0, 255), 2)
            cv2.imshow('Camera intrinsic capture - RAW image', display)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                if corners is None:
                    print('Not saved: all inner corners must be visible.', flush=True)
                    continue
                path = args.output / f'rgb_{index:04d}.png'
                while path.exists():
                    index += 1
                    path = args.output / f'rgb_{index:04d}.png'
                save_image(path, image)
                print(f'Saved {path}', flush=True)
                index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calibrate_camera(args):
    session = json.loads((args.input / 'session.json').read_text())
    objects, corners_list, names = [], [], []
    points = board_points(session['columns'], session['rows'], session['square_mm'])
    for path in sorted(args.input.glob('rgb_*.png')):
        image = load_image(path)
        if list(image.shape[1::-1]) != session['image_size']:
            raise ValueError(f'Wrong image size in {path}')
        corners = find_board(image, session['columns'], session['rows'])
        if corners is None:
            print(f'Skipped: checkerboard not detected in {path.name}')
            continue
        objects.append(points)
        corners_list.append(corners)
        names.append(path.name)
    if len(names) < 20:
        raise ValueError(f'Only {len(names)} usable images. Collect at least 20, preferably 30-50 diverse images.')
    held = set(np.random.default_rng(42).choice(len(names), max(4, len(names) // 5), replace=False).tolist())
    train = [i for i in range(len(names)) if i not in held]
    rms, K, dist, _, _ = cv2.calibrateCamera([objects[i] for i in train], [corners_list[i] for i in train],
                                           tuple(session['image_size']), None, None)
    report = []
    for i, (obj, corners, name) in enumerate(zip(objects, corners_list, names)):
        ok, rv, tv = cv2.solvePnP(obj, corners, K, dist)
        if not ok:
            raise ValueError(f'Pose solve failed for {name}')
        projected, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=2))))
        report.append({'image': name, 'rms_px': error, 'split': 'validation' if i in held else 'train'})
    result = dict(session, K=K.tolist(), distortion=dist.reshape(-1).tolist(),
                  train_rms_px=float(rms), views=report,
                  validation_mean_rms_px=float(np.mean([r['rms_px'] for r in report if r['split'] == 'validation'])),
                  valid_images=len(names), model='OpenCV pinhole + 5 distortion coefficients',
                  status='estimated; inspect image coverage and independent validation before use')
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ['K', 'distortion', 'valid_images', 'train_rms_px', 'validation_mean_rms_px']}, indent=2))
    print('Largest image errors:', sorted(report, key=lambda r: r['rms_px'], reverse=True)[:5])
    print(f'Saved {args.output.resolve()}')


def tof_panel(values):
    grid = values.reshape(8, 8)
    gray = np.clip(grid / 3000 * 255, 0, 255).astype(np.uint8)
    image = cv2.resize(cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA), (640, 480), interpolation=cv2.INTER_NEAREST)
    for r in range(8):
        for c in range(8):
            value = grid[r, c]
            cv2.putText(image, str(int(value)) if value > 0 else '--', (c * 80 + 8, r * 60 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


def capture_pairs(args):
    camera, _, _ = read_camera(args.intrinsics)
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / 'session.json'
    previous = json.loads(existing.read_text(encoding='utf-8')) if existing.exists() else None
    plane_camera = camera_for_plane_target(camera, previous.get('target') if previous else None,
                                           args.square_mm)
    target = {key: plane_camera[key] for key in ('columns', 'rows', 'square_mm')}
    settings = {'camera': camera, 'kind': 'stationary_plane_batches', 'unit': 'mm',
                'port': args.port, 'baud': args.baud, 'timestamps': 'host monotonic receipt; not hardware synchronized',
                'target': target}
    if previous is not None:
        previous.setdefault('target', {key: previous['camera'][key]
                                       for key in ('columns', 'rows', 'square_mm')})
    if previous is not None and previous != settings:
        raise ValueError('Capture settings changed; use a new output folder.')
    write_json(existing, settings)
    with ExitStack() as stack:
        cap, index = open_camera(args.camera)
        stack.callback(cap.release)
        reader = stack.enter_context(SerialToF(args.port, args.baud))
        number = len(list(args.output.glob('pose_*')))
        print('S: save stationary burst (confirm entire ToF view is on ONE flat target). Q: quit.', flush=True)
        message = 'Hold target still. S: capture burst. Q: quit.'
        while True:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError('No camera image.')
            corners = find_board(image, plane_camera['columns'], plane_camera['rows'])
            frame, _, _ = reader.snapshot()
            display = image.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, (plane_camera['columns'], plane_camera['rows']), corners, True)
            panel = tof_panel(frame.distance_mm if frame else np.zeros(64))
            joined = np.hstack([display, panel])
            cv2.putText(joined, f'poses={number} | {message}', (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow('Plane capture - RGB / RAW ToF mm (no alignment assumed)', joined)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                break
            if key != ord('s'):
                continue
            if corners is None:
                message = 'Rejected: checkerboard not detected'
                continue
            print('Capturing. Keep target STILL ...', flush=True)
            records, images, distances = [], [], []
            last_received = None
            started = time.monotonic()
            while len(images) < args.samples and time.monotonic() - started < 10:
                before = time.monotonic()
                ok, raw = cap.read()
                after = time.monotonic()
                if not ok:
                    raise RuntimeError('Camera read failed during burst.')
                tof, _, _ = reader.snapshot()
                if tof is None or tof.received_at == last_received or not 0 <= after - tof.received_at <= 0.15:
                    continue
                last_received = tof.received_at
                images.append(raw.copy())
                distances.append(tof.distance_mm.copy())
                records.append({'camera_read_start': before, 'camera_read_end': after,
                                'tof_host_received': tof.received_at, 'streamcount': tof.counter})
            if len(images) < args.samples:
                message = 'Rejected: insufficient fresh ToF frames'
                print(message, flush=True)
                continue
            check_indices = sorted({0, len(images) // 2, len(images) - 1})
            detected = [find_board(images[i], plane_camera['columns'], plane_camera['rows']) for i in check_indices]
            if any(c is None for c in detected):
                message = 'Rejected: board missing during burst'
                print(message, flush=True)
                continue
            baseline = detected[0]
            movement = max(min(float(np.sqrt(np.mean(np.sum((c - baseline) ** 2, axis=2)))),
                               float(np.sqrt(np.mean(np.sum((c[::-1] - baseline) ** 2, axis=2))))) for c in detected)
            if movement > 2.0:
                message = f'Rejected: board moved {movement:.1f}px; hold still and retry'
                print(message, flush=True)
                continue
            path = args.output / f'pose_{number:04d}'
            while path.exists():
                number += 1
                path = args.output / f'pose_{number:04d}'
            path.mkdir()
            for i, raw in enumerate(images):
                save_image(path / f'rgb_{i:03d}.png', raw)
            np.save(path / 'tof_mm.npy', np.stack(distances))
            write_json(path / 'metadata.json', {'frames': records, 'camera_index': index,
                       'checked_board_movement_px': movement, 'representative_image': f'rgb_{len(images)//2:03d}.png',
                       'human_required': 'All retained zones must observe the same flat backing board; not automatically verified.'})
            number += 1
            message = f'Saved {path.name}; change target tilt/distance'
            print(message, flush=True)
    cv2.destroyAllWindows()


def extrinsics(args):
    session = json.loads((args.input / 'session.json').read_text(encoding='utf-8'))
    camera = session['camera']
    plane_camera = camera_for_plane_target(camera, session.get('target'), args.square_mm)
    rays = nominal_tof_rays(*args.tof_fov_deg)
    observations, names, skipped = [], [], []
    for pose in sorted(args.input.glob('pose_*')):
        meta = json.loads((pose / 'metadata.json').read_text())
        median, stable = summarize_static_tof(np.load(pose / 'tof_mm.npy', allow_pickle=False))
        median = orient(median, args.rotate_tof, args.flip_tof_x, args.flip_tof_y)
        stable = orient(stable, args.rotate_tof, args.flip_tof_x, args.flip_tof_y).astype(bool)
        image = load_image(pose / meta['representative_image'])
        try:
            normal, offset, _, reprojection = camera_plane(image, plane_camera)
            if stable.sum() < 40:
                raise ValueError(f'Only {stable.sum()} stable valid ToF zones; need at least 40.')
            if reprojection > 1.0:
                raise ValueError(f'Board reprojection error {reprojection:.2f}px > 1px.')
        except ValueError as exc:
            skipped.append({'pose': pose.name, 'reason': str(exc)})
            continue
        observations.append({'points': tof_points(median, rays, args.depth_convention)[stable],
                             'normal': normal, 'offset': offset})
        names.append(pose.name)
    if len(observations) < 20:
        raise ValueError(f'Only {len(observations)} accepted poses; need at least 20. Skipped: {skipped}')
    held = sorted(np.random.default_rng(42).choice(len(names), max(4, len(names) // 5), replace=False).tolist())
    R, t, metrics = fit_point_to_plane(observations, held)
    result = {'status': 'APPROXIMATE_INITIALIZATION_REQUIRES_INDEPENDENT_VALIDATION',
              'transform': 'P_camera_m = R_tof_to_camera @ P_tof_m + t_tof_to_camera_m',
              'R_tof_to_camera': R.tolist(), 't_tof_to_camera_m': t.tolist(),
              'tof_fov_xy_deg': args.tof_fov_deg, 'depth_convention': args.depth_convention,
              'ray_model': 'nominal pinhole zone centres; not measured per-zone angular response',
              'rotate_tof_ccw': args.rotate_tof, 'flip_tof_x': args.flip_tof_x, 'flip_tof_y': args.flip_tof_y,
              'camera': camera,
              'target': {key: plane_camera[key] for key in ('columns', 'rows', 'square_mm')},
              'metrics': metrics, 'pose_order': names,
              'validation_poses': [names[i] for i in held], 'skipped_poses': skipped,
              'input_path': str(args.input.resolve()),
              'limitations': ['single-target distance is a zone measurement, not an exact centre-ray point',
                              'nominal FoV and range/Z convention must be validated for this sensor',
                              'host timestamps are not exposure synchronization',
                              'low plane residual alone does not establish correct pixel alignment']}
    write_json(args.output, result)
    print(json.dumps(metrics, indent=2))
    print(f'Saved approximate extrinsics: {args.output.resolve()}')
    print('Not a validated calibration. Check independent planes, edges, ray assumptions and physical translation.')


def inspect(args):
    result = json.loads(args.extrinsics.read_text(encoding='utf-8'))
    root = args.input if args.input else Path(result['input_path'])
    camera = result['camera']
    input_session = json.loads((root / 'session.json').read_text(encoding='utf-8'))
    if input_session['camera'] != camera:
        raise ValueError('Validation session uses different camera calibration.')
    plane_camera = camera_for_plane_target(camera, input_session.get('target', result.get('target')))
    rays = nominal_tof_rays(*result['tof_fov_xy_deg'])
    R = np.asarray(result['R_tof_to_camera'])
    t = np.asarray(result['t_tof_to_camera_m'])
    args.output.mkdir(parents=True, exist_ok=True)
    plane_report, all_errors = [], []
    for pose in sorted(root.glob('pose_*')):
        meta = json.loads((pose / 'metadata.json').read_text())
        image = load_image(pose / meta['representative_image'])
        pose_camera = dict(plane_camera, square_mm=result.get('board_square_mm_by_pose', {}).get(
            pose.name, plane_camera['square_mm']))
        median, valid = summarize_static_tof(np.load(pose / 'tof_mm.npy', allow_pickle=False))
        median = orient(median, result['rotate_tof_ccw'], result['flip_tof_x'], result['flip_tof_y'])
        valid = orient(valid, result['rotate_tof_ccw'], result['flip_tof_x'], result['flip_tof_y']).astype(bool)
        points = tof_points(median, rays, result['depth_convention']) @ R.T + t
        valid &= points[:, 2] > 0.01
        try:
            normal, offset, _, reprojection = camera_plane(image, pose_camera)
            if not valid.any():
                raise ValueError('No stable positive-depth ToF points.')
            errors = points[valid] @ normal + offset
            all_errors.extend(errors.tolist())
            plane_report.append({'pose': pose.name, 'valid_zones': int(valid.sum()),
                                 'board_square_mm': pose_camera['square_mm'],
                                 'board_reprojection_rms_px': reprojection,
                                 'plane_mae_mm': float(np.mean(np.abs(errors)) * 1000),
                                 'plane_p95_abs_mm': float(np.percentile(np.abs(errors), 95) * 1000)})
        except ValueError as exc:
            plane_report.append({'pose': pose.name, 'error': str(exc)})
        projected, _ = cv2.projectPoints(points[valid], np.zeros(3), np.zeros(3),
                                        np.asarray(camera['K']), np.asarray(camera['distortion']))
        for zone, xy in zip(np.flatnonzero(valid), projected.reshape(-1, 2)):
            if np.isfinite(xy).all() and 0 <= xy[0] < 640 and 0 <= xy[1] < 480:
                x, y = np.rint(xy).astype(int)
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(image, str(zone), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(image, 'APPROXIMATE zone CENTRES - not calibrated footprints', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        save_image(args.output / f'{pose.name}.png', image)
    report = {'status': 'plane_only_validation_under_approximate_ray_model',
              'input_path': str(root.resolve()), 'poses': plane_report,
              'point_count': len(all_errors),
              'plane_mae_mm': float(np.mean(np.abs(all_errors)) * 1000) if all_errors else None,
              'plane_p95_abs_mm': float(np.percentile(np.abs(all_errors), 95) * 1000) if all_errors else None,
              'limitation': 'Low plane error does not prove individual-zone footprint alignment.'}
    write_json(args.output / 'validation_report.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'point_count', 'plane_mae_mm', 'plane_p95_abs_mm']}, indent=2))
    print(f'Saved approximate centre overlays to {args.output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, handler in [('capture-camera', capture_camera), ('calibrate-camera', calibrate_camera),
                          ('capture-pairs', capture_pairs), ('fit-extrinsics', extrinsics), ('inspect', inspect)]:
        p = sub.add_parser(name)
        p.set_defaults(handler=handler)
        if name == 'capture-camera':
            p.add_argument('--columns', type=int, default=9, help='INNER corners per row, not squares')
            p.add_argument('--rows', type=int, default=6, help='INNER corners per column')
            p.add_argument('--square-mm', type=float, default=25, help='Actual measured printed square edge')
        if name in ('capture-camera', 'capture-pairs'):
            p.add_argument('--camera', type=int, default=1)
        if name in ('calibrate-camera', 'fit-extrinsics'):
            p.add_argument('--input', type=Path, required=True)
        if name in ('capture-pairs', 'fit-extrinsics'):
            p.add_argument('--square-mm', type=float,
                           help='Plane target square edge in mm; defaults to saved session target, then intrinsic target')
        if name == 'capture-pairs':
            p.add_argument('--intrinsics', type=Path, required=True)
            p.add_argument('--port', default='COM9')
            p.add_argument('--baud', type=int, default=921600)
            p.add_argument('--samples', type=int, default=20, help='Stationary unique ToF observations per pose')
        if name == 'fit-extrinsics':
            p.add_argument('--tof-fov-deg', type=float, nargs=2, required=True, metavar=('FOV_X', 'FOV_Y'))
            p.add_argument('--depth-convention', choices=['z', 'radial'], required=True)
            p.add_argument('--rotate-tof', type=int, choices=[0, 90, 180, 270], default=0)
            p.add_argument('--flip-tof-x', action='store_true')
            p.add_argument('--flip-tof-y', action='store_true')
        if name == 'inspect':
            p.add_argument('--extrinsics', type=Path, required=True)
            p.add_argument('--input', type=Path, help='Other held-out pose folder with the same session format')
        defaults = {'capture-camera': 'data/calibration/camera_images',
                    'calibrate-camera': 'configs/calibration/camera_intrinsics.json', 'capture-pairs': 'data/calibration/plane_pairs',
                    'fit-extrinsics': 'configs/calibration/rgb_tof_extrinsics.json', 'inspect': 'outputs/calibration_overlays'}
        p.add_argument('--output', type=Path, default=Path(defaults[name]))
    args = parser.parse_args()
    if getattr(args, 'square_mm', None) is not None and (not np.isfinite(args.square_mm) or args.square_mm <= 0):
        parser.error('Square size must be finite and positive.')
    if hasattr(args, 'columns') and (args.columns < 3 or args.rows < 3):
        parser.error('Need at least 3x3 inner corners.')
    if hasattr(args, 'samples') and args.samples < 10:
        parser.error('Use at least 10 samples per static pose.')
    args.handler(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        raise SystemExit(f'ERROR: {exc}')
