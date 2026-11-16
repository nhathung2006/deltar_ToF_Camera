"""Experimental DELTAR preview for Logitech + ESP32/VL53L8CX on Windows.

Supports legacy F frames and F2 frames with status and range sigma. A constant
Gaussian spread remains an explicit approximation, not the histogram
distribution used to train DELTAR or the sensor's range sigma.
The --preview option also uses an uncalibrated, regular 8x8 image ROI.
Optional --intrinsics applies camera lens correction; it does not align ToF.
--extrinsics applies lens correction plus experimental per-frame ToF projection.
--grid-angle-deg and --grid-shift-cells manually rotate/translate the preview ROI.
Default smoothing combines bounded ToF stabilization, RGB-guided spatial depth
filtering and motion-aware temporal filtering. --smoothing off restores raw output.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from src.camera_tof_calibration import CameraUndistorter
from src.depth_smoothing import ToFTemporalFilter, DepthTemporalFilter, guided_depth_filter
from src.camera_tof_io import SerialToF, grid_rectangles, open_camera
from src.models.deltar import Deltar
from src.tof_projection import ManualToFProjector, ToFProjector, draw_projected_zones, rgb_boundary_confidence
from src.utils.tof_sampling import patch_info_from_rect_data, sample_point_from_hist_parallel


ROOT = Path(__file__).resolve().parents[1]


def make_input(bgr, distance_mm, rectangles, sigma_m, device, max_tof_m=4.0,
               *, projection_grids=None, sigma_scale=None, allow_empty=False):
    image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 255
    image = (image - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    distances = torch.as_tensor(distance_mm.copy(), dtype=torch.float32).reshape(64) / 1000
    mask = torch.isfinite(distances) & (distances > 0) & (distances <= max_tof_m)
    if not mask.any() and not allow_empty:
        raise ValueError('No valid ToF distances within the configured range.')
    spread = torch.full_like(distances, sigma_m)
    if sigma_scale is not None:
        spread *= torch.as_tensor(sigma_scale, dtype=torch.float32)
    raw_hist = torch.stack([distances, spread], dim=1)
    options = SimpleNamespace(zone_sample_num=16, sample_uniform=False)
    samples = sample_point_from_hist_parallel(raw_hist, mask, options)
    samples[mask] = samples[mask].clamp(0.001, 10)
    rect = torch.as_tensor(rectangles.copy(), dtype=torch.float32)
    patches = default_collate([patch_info_from_rect_data(rect)])
    result = {
        'rgb': image.unsqueeze(0).to(device),
        'additional': {
            'hist_data': samples.unsqueeze(0).to(device),
            'rect_data': rect.unsqueeze(0).to(device),
            'mask': mask.unsqueeze(0).to(device),
            'patch_info': patches,
        },
    }
    if projection_grids is not None:
        result['additional']['zone_projection'] = {
            stride: {name: torch.from_numpy(value.copy()).unsqueeze(0).to(device)
                     for name, value in stage.items()}
            for stride, stage in projection_grids.items()}
    return result, int(mask.sum())


def orient_tof(distance_mm, rotation=0, flip_x=False, flip_y=False):
    grid = np.rot90(distance_mm.reshape(8, 8), rotation // 90)
    if flip_x:
        grid = np.fliplr(grid)
    if flip_y:
        grid = np.flipud(grid)
    return grid.copy().reshape(64)


def color_depth(depth_m, vmax):
    scaled = np.clip(np.nan_to_num(depth_m) / vmax * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_MAGMA)
    colored[~np.isfinite(depth_m) | (depth_m <= 0)] = 0
    return colored


def preview_image(bgr, distance_mm, depth, rects, title, vmax, sigma_m, fps, valid,
                  can_toggle_undistortion=False, projection=None, projected_valid=None):
    rgb = bgr.copy()
    if projection is not None:
        rgb = draw_projected_zones(rgb, projection)
    else:
        for sy, sx, ey, ex in rects:
            cv2.rectangle(rgb, (int(sx), int(sy)), (int(ex), int(ey)), (0, 220, 0), 1)
    raw = cv2.resize(color_depth(distance_mm.reshape(8, 8) / 1000, vmax), (640, 480), interpolation=cv2.INTER_NEAREST)
    for row in range(8):
        for col in range(8):
            mm = distance_mm.reshape(8, 8)[row, col]
            cv2.putText(raw, str(int(mm)) if mm > 0 else '--', (col * 80 + 12, row * 60 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    dense = color_depth(depth, vmax)
    mosaic = np.zeros((576, 1920, 3), dtype=np.uint8)
    rgb_caption = 'RGB + visible ToF regions' if projection is not None else 'RGB + ToF ROI'
    depth_caption = ('DELTAR - RGB ONLY (no projected ToF)'
                     if projection is not None and projected_valid == 0 else 'DELTAR - experimental')
    for index, (panel, caption) in enumerate([(rgb, rgb_caption), (raw, 'VL53L8CX raw (mm)'), (dense, depth_caption)]):
        x = index * 640
        mosaic[32:512, x:x + 640] = panel
        cv2.putText(mosaic, caption, (x + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    status = title
    if can_toggle_undistortion:
        status += ' | U: toggle lens correction'
    details = f'assumed spread={sigma_m * 1000:g}mm | {fps:.1f} FPS | valid={valid}/64 | scale=0..{vmax:g}m | Q: quit'
    if projected_valid is not None:
        details = details.replace(' | scale=', f' | projected={projected_valid}/64 | scale=')
    cv2.putText(mosaic, status, (8, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1)
    cv2.putText(mosaic, details, (8, 561), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1)
    return cv2.resize(mosaic, (1280, 384), interpolation=cv2.INTER_AREA)


def clean_preview_image(bgr, depth, vmax, fps, smoothing, radius, projected_valid=None):
    """Unmodified RGB beside metric depth, without ToF debug overlays."""
    mosaic = np.zeros((552, 1280, 3), dtype=np.uint8)
    mosaic[32:512, :640] = bgr
    mosaic[32:512, 640:] = color_depth(depth, vmax)
    cv2.putText(mosaic, 'RGB', (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    caption = 'Depth - ' + smoothing
    if projected_valid == 0:
        caption += ' / RGB ONLY'
    cv2.putText(mosaic, caption, (648, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    detail = f'{fps:.1f} FPS | 0..{vmax:g}m | S: smoothing on/off | C: compare | D: ToF diagnostics | Q: quit'
    cv2.putText(mosaic, detail, (8, 538), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 1)
    return mosaic


def comparison_preview_image(bgr, original_depth, smoothed_depth, vmax, fps, smoothing):
    """RGB, raw-model depth and the selected smoothing result on one scale."""
    mosaic = np.zeros((576, 1920, 3), dtype=np.uint8)
    caption = ('ToF + RGB guided + temporal' if smoothing == 'combined'
               else f'Depth - {smoothing}')
    panels = ((bgr, 'RGB'), (color_depth(original_depth, vmax), 'Original'),
              (color_depth(smoothed_depth, vmax), caption))
    for index, (panel, title) in enumerate(panels):
        x = index * 640
        mosaic[32:512, x:x + 640] = panel
        cv2.putText(mosaic, title, (x + 8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 1)
    detail = f'{fps:.1f} FPS | 0..{vmax:g}m | S: smoothing | C: clean | D: ToF diagnostics | Q: quit'
    cv2.putText(mosaic, detail, (8, 552), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (0, 220, 255), 1)
    return cv2.resize(mosaic, (1280, 384), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights' / 'deltar_nyu.pt')
    parser.add_argument('--port', default='COM9')
    parser.add_argument('--baud', type=int, default=921600)
    parser.add_argument('--camera', type=int, help='DirectShow index; otherwise auto-select Logitech')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--intrinsics', type=Path, help='Camera intrinsics JSON: enable lens correction; U toggles it in the viewer')
    parser.add_argument('--extrinsics', type=Path,
                        help='Experimental extrinsics JSON: use embedded camera intrinsics, saved ToF orientation and geometry-aware fusion')
    parser.add_argument('--preview', action='store_true', help='Explicitly allow an uncalibrated ROI for a functional preview')
    parser.add_argument('--roi', type=int, nargs=4, default=[96, 24, 448, 448], metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--grid-angle-deg', type=float, default=0.,
                        help='Manual ROI rotation about its centre: positive is counterclockwise')
    parser.add_argument('--grid-shift-cells', type=float, nargs=2, default=[0., 0.], metavar=('DX', 'DY'),
                        help='Translate the rotated ROI in screen X/Y by cell widths/heights; positive is right/down')
    parser.add_argument('--rects', type=Path, help='Calibrated regular 8x8 rectangles: .npy shape (64,4), order y1,x1,y2,x2')
    parser.add_argument('--sigma-m', type=float, help='Assumed Gaussian depth spread in METRES; not sensor range sigma')
    parser.add_argument('--rotate-tof', type=int, choices=[0, 90, 180, 270], default=180,
                        help='Counterclockwise rotation of ToF before inference and display (default: 180)')
    parser.add_argument('--flip-tof-x', action='store_true')
    parser.add_argument('--flip-tof-y', action='store_true')
    parser.add_argument('--max-tof-m', type=float, default=4.0)
    parser.add_argument('--max-age', type=float, default=0.3, help='Reject ToF older than this many seconds at host receipt')
    parser.add_argument('--vis-max-m', type=float, default=4.0)
    parser.add_argument('--smoothing', choices=['off', 'tof', 'guided', 'combined'], default='combined',
                        help='Default combined: stabilize ToF, RGB-guided depth, motion-aware temporal filter')
    parser.add_argument('--guided-radius', type=int, default=48,
                        help='RGB-guided depth filter radius in pixels (default: 48; smaller preserves finer detail)')
    parser.add_argument('--edge-aware-tof', action=argparse.BooleanOptionalAction, default=False,
                        help='Experimental RGB weighting near ToF depth boundaries; E toggles, smoothing off bypasses')
    parser.add_argument('--view', choices=['clean', 'diagnostic', 'comparison'], default='clean',
                        help='Clean RGB/depth, raw ToF diagnostic, or RGB/original/smoothed comparison')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0, help='Stop after N predictions; 0 means continuous')
    parser.add_argument('--save-dir', type=Path, help='Save the last depth and summary; no RGB images are saved')
    parser.add_argument('--self-test', action='store_true', help='Use synthetic inputs to test weights/GPU, without opening devices')
    args = parser.parse_args()
    if args.frames < 0 or args.max_age <= 0 or args.max_tof_m <= 0 or args.vis_max_m <= 0:
        parser.error('Frame count must be nonnegative; ranges and --max-age must be positive.')
    if not 1 <= args.guided_radius <= 128:
        parser.error('--guided-radius must be between 1 and 128 pixels.')
    if not np.isfinite(args.grid_angle_deg) or not np.isfinite(args.grid_shift_cells).all():
        parser.error('Manual grid angle and shifts must be finite.')
    manual_grid_enabled = args.grid_angle_deg != 0 or any(args.grid_shift_cells)
    if args.self_test:
        args.headless = True
        args.frames = args.frames or 1
        args.sigma_m = args.sigma_m if args.sigma_m is not None else 0.02
    elif not args.preview and args.rects is None and args.extrinsics is None:
        parser.error('Supply --preview, --rects or experimental --extrinsics.')
    if args.extrinsics is not None and args.rects is not None:
        parser.error('--extrinsics and --rects are separate alignment modes; use only one.')
    if manual_grid_enabled and (args.extrinsics is not None or args.rects is not None):
        parser.error('Manual grid options require a preview ROI; use them separately from --extrinsics or --rects.')
    if args.sigma_m is None or not np.isfinite(args.sigma_m) or args.sigma_m <= 0:
        parser.error('Supply --sigma-m with a positive assumed spread, e.g. --sigma-m 0.02 (experimental).')
    if args.headless and args.frames == 0:
        parser.error('--headless requires --frames N')
    if not args.weights.is_file():
        parser.error(f'Weights not found: {args.weights}')
    undistorter = None
    projector = None
    if args.extrinsics is not None:
        if not args.extrinsics.is_file():
            parser.error(f'Extrinsics not found: {args.extrinsics}')
        projector = ToFProjector.from_file(args.extrinsics)
        undistorter = projector.undistorter
        args.rotate_tof, args.flip_tof_x, args.flip_tof_y = projector.rotation, projector.flip_x, projector.flip_y
    if args.intrinsics is not None:
        if not args.intrinsics.is_file():
            parser.error(f'Camera calibration not found: {args.intrinsics}')
        camera_calibration = json.loads(args.intrinsics.read_text(encoding='utf-8'))
        if projector is not None and any(
                not np.array_equal(camera_calibration[key], projector.camera[key])
                for key in ('K', 'distortion', 'image_size')):
            parser.error('--intrinsics does not match the camera calibration embedded in --extrinsics.')
        undistorter = CameraUndistorter(camera_calibration)
    undistortion_enabled = undistorter is not None
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    rects = np.load(args.rects, allow_pickle=False) if args.rects else grid_rectangles(args.roi)
    if rects.shape != (64, 4) or not np.isfinite(rects).all() or np.any(rects[:, 2:] <= rects[:, :2]):
        parser.error('Rectangles must be finite, shape (64,4), with y2>y1 and x2>x1.')
    if args.rects is None and (np.any(rects < 0) or np.any(rects[:, [0, 2]] > 480) or np.any(rects[:, [1, 3]] > 640)):
        parser.error('Preview ROI must be inside the 640x480 image.')
    manual_projector = (ManualToFProjector(args.roi, args.grid_angle_deg, args.grid_shift_cells)
                        if manual_grid_enabled else None)
    zone_mapper = projector if projector is not None else manual_projector
    if args.edge_aware_tof and zone_mapper is None:
        parser.error('--edge-aware-tof requires --extrinsics or a manually transformed grid.')
    title = 'SYNTHETIC TEST' if args.self_test else 'UNCALIBRATED ROI' if args.rects is None else 'SUPPLIED CALIBRATION'
    if projector is not None:
        title = 'EXPERIMENTAL EXTRINSICS - NOT VALIDATED'
    elif manual_projector is not None:
        title = f'MANUAL ROI - UNCALIBRATED / CCW={args.grid_angle_deg:g}deg'
    print(f'Loading {args.weights} on {device} ...', flush=True)
    model = Deltar(n_bins=256, min_val=0.001, max_val=10, pretrained_backbone=False)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model = model.to(device).eval()
    print(f'Weights matched. {title}; Gaussian spread is APPROXIMATED, not measured.', flush=True)
    print(f'Smoothing: {args.smoothing}; guided radius: {args.guided_radius}px. '
          'S toggles smoothing, D toggles ToF diagnostics. Postprocessing is not a calibration.', flush=True)
    if zone_mapper is not None:
        print(f'RGB boundary weighting: {args.edge_aware_tof}. E toggles; only active with smoothing enabled.', flush=True)
    if not args.self_test:
        print('Frame pairing uses PC receipt time; hardware exposure synchronization is not available.', flush=True)
    if projector is not None:
        print(f'Experimental extrinsics: {args.extrinsics}; camera intrinsics and ToF orientation come from this file.', flush=True)
        print('RGB lens correction stays ON; measured ToF is converted to camera Z and projected at each frame.', flush=True)
        print('Footprints use nominal rays and a local-plane approximation; model weights are unchanged.', flush=True)
    elif undistorter is not None:
        print(f'Lens correction ON: {args.intrinsics}; output 640x480, same K, no crop. U toggles correction.', flush=True)
        print('This applies camera intrinsics only; RGB-ToF alignment is unchanged. Supplied --rects must match undistorted coordinates.', flush=True)
    if manual_projector is not None:
        print(f'Manual grid: {args.grid_angle_deg:g}deg counterclockwise, '
              f'shift={args.grid_shift_cells} cells = {manual_projector.shift_pixels.tolist()} pixels. '
              'Display and DELTAR use the same transformed cells; off-image parts are clipped.', flush=True)
    count = 0
    durations = []
    started = time.monotonic()
    last_success = started
    last_received = None
    summary = None
    smoothing_mode = args.smoothing
    enabled_smoothing = args.smoothing if args.smoothing != 'off' else 'combined'
    view_mode = args.view
    edge_aware = args.edge_aware_tof
    tof_filter = ToFTemporalFilter(max_distance_mm=args.max_tof_m * 1000)
    depth_filter = DepthTemporalFilter()
    with ExitStack() as stack, torch.inference_mode():
        if not args.self_test:
            cap, camera_index = open_camera(args.camera)
            stack.callback(cap.release)
            reader = stack.enter_context(SerialToF(args.port, args.baud))
            print(f'Camera {camera_index}, {args.port} at {args.baud} baud; distances in mm.', flush=True)
        else:
            synthetic = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        started = last_success = time.monotonic()
        while args.frames == 0 or count < args.frames:
            if args.self_test:
                bgr = synthetic
                distances = np.full(64, 1000, dtype=np.float32)
                frame_age = 0.0
                counter = count
                sample_time = count / 15.
            else:
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError('Camera stopped returning images.')
                if bgr.shape[:2] != (480, 640):
                    raise RuntimeError(f'Expected 640x480 camera frames, got {bgr.shape[1]}x{bgr.shape[0]}.')
                frame, received_count, ignored_count = reader.snapshot()
                frame_age = time.monotonic() - frame.received_at if frame else float('inf')
                if frame is None or frame_age > args.max_age or frame.received_at == last_received:
                    if time.monotonic() - last_success > 10:
                        raise RuntimeError(f'No fresh, usable ToF for 10 seconds '
                                           f'({received_count} F/F2 frames, {ignored_count} ignored lines). '
                                           'Check port, baud and sensor view.')
                    time.sleep(0.005)
                    continue
                last_received = frame.received_at
                counter = frame.counter
                distances = orient_tof(frame.distance_mm, args.rotate_tof, args.flip_tof_x, args.flip_tof_y)
                sample_time = frame.received_at
            valid = np.isfinite(distances) & (distances > 0) & (distances <= args.max_tof_m * 1000)
            tick = time.monotonic()
            raw_distances = distances.copy()
            if smoothing_mode in ('tof', 'combined'):
                distances = tof_filter.update(distances, sample_time)
            if undistortion_enabled:
                bgr = undistorter.apply(bgr)
            projection = None
            projected_count = None
            valid_count = int(valid.sum())
            edge_active = edge_aware and smoothing_mode != 'off' and zone_mapper is not None
            confidence = None
            if zone_mapper is not None:
                projection = zone_mapper.project(distances, args.max_tof_m)
                if edge_active:
                    confidence = rgb_boundary_confidence(bgr, projection)
                grids = zone_mapper.fusion_grids(projection, confidence=confidence)
                input_data, projected_count = make_input(
                    bgr, projection['distance_mm'], rects, args.sigma_m, device, 10.0,
                    projection_grids=grids, sigma_scale=projection['sigma_scale'], allow_empty=True)
            else:
                input_data, valid_count = make_input(bgr, distances, rects, args.sigma_m, device,
                                                     args.max_tof_m, allow_empty=True)
            inference_valid_count = int(input_data['additional']['mask'].sum().item())
            _, pred = model(input_data)
            depth = F.interpolate(pred, size=(480, 640), mode='bilinear', align_corners=True).clamp(0.001, 10)[0, 0].cpu().numpy()
            if not np.isfinite(depth).all():
                raise RuntimeError('Non-finite depth prediction.')
            raw_depth = depth.copy()
            original_depth = None
            comparison_raw_inference_ms = 0.
            if view_mode == 'comparison':
                original_depth = raw_depth.copy()
                if smoothing_mode in ('tof', 'combined'):
                    raw_tick = time.monotonic()
                    if zone_mapper is not None:
                        raw_projection = zone_mapper.project(raw_distances, args.max_tof_m)
                        raw_grids = zone_mapper.fusion_grids(raw_projection)
                        raw_input, _ = make_input(
                            bgr, raw_projection['distance_mm'], rects, args.sigma_m, device, 10.0,
                            projection_grids=raw_grids, sigma_scale=raw_projection['sigma_scale'], allow_empty=True)
                    else:
                        raw_input, _ = make_input(
                            bgr, raw_distances, rects, args.sigma_m, device,
                            args.max_tof_m, allow_empty=True)
                    _, raw_pred = model(raw_input)
                    original_depth = F.interpolate(
                        raw_pred, size=(480, 640), mode='bilinear', align_corners=True
                    ).clamp(0.001, 10)[0, 0].cpu().numpy()
                    if not np.isfinite(original_depth).all():
                        raise RuntimeError('Non-finite original depth prediction.')
                    comparison_raw_inference_ms = (time.monotonic() - raw_tick) * 1000
            post_tick = time.monotonic()
            if smoothing_mode in ('guided', 'combined'):
                depth = guided_depth_filter(bgr, depth, radius=args.guided_radius)
                depth = depth_filter.update(depth, bgr, sample_time)
            postprocess_ms = (time.monotonic() - post_tick) * 1000
            elapsed = time.monotonic() - tick
            durations.append(elapsed)
            count += 1
            last_success = time.monotonic()
            summary = {
                'source': 'synthetic' if args.self_test else 'live_devices', 'device': str(device),
                'frames': count, 'prediction_shape': list(depth.shape), 'all_finite': True,
                'prediction_min_m': float(depth.min()), 'prediction_max_m': float(depth.max()),
                'inference_ms_last': round(elapsed * 1000, 2),
                'inference_ms_mean': round(float(np.mean(durations)) * 1000, 2),
                'loop_fps': round(count / (time.monotonic() - started), 2),
                'valid_tof_zones': valid_count, 'last_counter': counter,
                'filtered_valid_tof_zones': int(np.count_nonzero(distances > 0)),
                'edge_aware_tof': edge_active,
                'edge_weighted_image_fraction': (float(np.mean((confidence > 0) & (confidence < .95)))
                                                  if confidence is not None else 0.),
                'tof_invalid_input_zones': int(np.count_nonzero(~valid)),
                'tof_status_counts': ({str(int(s)): int(n) for s, n in zip(*np.unique(frame.target_status, return_counts=True))}
                                      if not args.self_test and frame.target_status is not None else None),
                'smoothing': smoothing_mode, 'guided_radius_px': args.guided_radius,
                'postprocess_ms_last': round(postprocess_ms, 2), 'view': view_mode,
                'comparison_raw_inference_ms_last': round(comparison_raw_inference_ms, 2),
                'smoothing_parameters': {'tof_tau_s': tof_filter.tau_s, 'tof_hold_s': tof_filter.hold_s,
                                         'depth_tau_s': depth_filter.tau_s, 'depth_jump_hold_s': .18,
                                         'jump_confirmation_samples': 2, 'guided_epsilon': .003},
                'tof_protocol': 'synthetic' if args.self_test else frame.protocol,
                'tof_sensor_timestamp': None if args.self_test else frame.sensor_timestamp,
                'tof_age_at_selection_ms': round(frame_age * 1000, 2),
                'calibration_file': str(args.rects.resolve()) if args.rects else None,
                'extrinsics_file': str(args.extrinsics.resolve()) if args.extrinsics else None,
                'extrinsics_status': projector.calibration.get('status') if projector is not None else None,
                'extrinsics_fit_metrics': {
                    key: value for key, value in projector.calibration.get('metrics', {}).items()
                    if key != 'per_pose_mae_mm'} if projector is not None else None,
                'alignment': ('experimental projected footprints' if projector is not None else
                              'manual rotated and translated ROI' if manual_projector is not None else
                              'regular rectangles'),
                'manual_grid': ({'angle_ccw_deg': args.grid_angle_deg,
                                 'shift_cells_xy': args.grid_shift_cells,
                                 'shift_pixels_xy': manual_projector.shift_pixels.tolist()}
                                if manual_projector is not None else None),
                'projected_tof_zones': projected_count,
                'camera_intrinsics_file': str(args.intrinsics.resolve()) if args.intrinsics else None,
                'camera_intrinsics_from_extrinsics': projector is not None,
                'camera_undistortion_applied': undistortion_enabled,
                'camera_undistortion_geometry': 'original K; full image size; no crop' if undistortion_enabled else None,
                'preview_roi_xywh': args.roi if args.rects is None and projector is None else None,
                'assumed_sigma_m': args.sigma_m, 'distance_unit': 'mm',
                'rotation_ccw': args.rotate_tof, 'flip_x': args.flip_tof_x, 'flip_y': args.flip_tof_y,
                'pairing': 'synthetic' if args.self_test else 'latest host-received ToF; not exposure-synchronized',
            }
            if count == 1 or count % 30 == 0 or count == args.frames:
                print(json.dumps(summary), flush=True)
            if not args.headless:
                display_title = title + (' / RGB UNDISTORTED' if undistortion_enabled else ' / RGB RAW')
                if zone_mapper is not None:
                    display_title += ' / EDGE ' + ('ON' if edge_active else 'OFF') + ' [E]'
                if inference_valid_count == 0:
                    display_title += ' / NO ToF: RGB ONLY'
                fps = count / (time.monotonic() - started)
                if view_mode == 'clean':
                    mosaic = clean_preview_image(bgr, depth, args.vis_max_m, fps,
                                                 smoothing_mode + (' + RGB edge weights [E]' if edge_active else ''),
                                                 args.guided_radius, inference_valid_count)
                elif view_mode == 'comparison':
                    mosaic = comparison_preview_image(bgr, original_depth, depth,
                                                      args.vis_max_m, fps, smoothing_mode)
                else:
                    mosaic = preview_image(bgr, raw_distances, depth, rects, display_title,
                                           args.vis_max_m, args.sigma_m, fps, valid_count,
                                           can_toggle_undistortion=undistorter is not None and projector is None,
                                           projection=projection, projected_valid=projected_count)
                cv2.imshow('DELTAR - Logitech + VL53L8CX', mosaic)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27):
                    break
                if key == ord('d'):
                    view_mode = 'diagnostic' if view_mode != 'diagnostic' else args.view
                if key == ord('c'):
                    view_mode = 'clean' if view_mode == 'comparison' else 'comparison'
                if key == ord('e') and zone_mapper is not None:
                    edge_aware = not edge_aware
                    depth_filter.reset()
                    print(f'RGB boundary weighting requested: {edge_aware}; smoothing: {smoothing_mode}', flush=True)
                if key == ord('s'):
                    smoothing_mode = enabled_smoothing if smoothing_mode == 'off' else 'off'
                    tof_filter.reset()
                    depth_filter.reset()
                    print(f'Smoothing: {smoothing_mode}', flush=True)
                if key == ord('u') and undistorter is not None and projector is None:
                    undistortion_enabled = not undistortion_enabled
                    depth_filter.reset()
                    print(f'Lens correction {"ON" if undistortion_enabled else "OFF"}; applies to RGB display and DELTAR input.', flush=True)
        if args.save_dir and summary is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.save_dir / 'last_depth_m.npy', depth)
            np.save(args.save_dir / 'last_depth_before_postprocess_m.npy', raw_depth)
            if original_depth is not None:
                np.save(args.save_dir / 'last_depth_original_m.npy', original_depth)
            np.save(args.save_dir / 'last_tof_raw_mm.npy', raw_distances)
            np.save(args.save_dir / 'last_tof_filtered_mm.npy', distances)
            cv2.imwrite(str(args.save_dir / 'last_depth_mm.png'), np.rint(depth * 1000).astype(np.uint16))
            cv2.imwrite(str(args.save_dir / 'last_depth_color.png'), color_depth(depth, args.vis_max_m))
            (args.save_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(f'Saved last depth and summary to {args.save_dir.resolve()}', flush=True)
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Camera and serial port released.')
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
