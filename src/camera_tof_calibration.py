"""Camera/plane calibration helpers. Nominal ToF rays are an approximation."""
import json
from pathlib import Path

import cv2
import numpy as np


def board_points(columns, rows, square_mm):
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * (square_mm / 1000)
    return points


def find_board(image, columns, rows):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCornersSB(
        gray, (columns, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    # OpenCV 4/5 can return (N,1,2) or (N,2). Normalize before arithmetic.
    return corners.astype(np.float32).reshape(-1, 1, 2) if found else None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def read_camera(path):
    camera = json.loads(Path(path).read_text(encoding='utf-8'))
    return camera, np.asarray(camera['K'], dtype=np.float64), np.asarray(camera['distortion'], dtype=np.float64)


def camera_for_plane_target(camera, target=None, square_mm=None):
    """Use a plane target's dimensions without changing the intrinsic calibration."""
    target = target or {}
    settings = dict(camera)
    for key in ('columns', 'rows', 'square_mm'):
        settings[key] = target.get(key, camera[key])
    if square_mm is not None:
        settings['square_mm'] = square_mm
    if not np.isfinite(settings['square_mm']) or settings['square_mm'] <= 0:
        raise ValueError('Plane target square size must be finite and positive.')
    if settings['columns'] < 3 or settings['rows'] < 3:
        raise ValueError('Plane target needs at least 3x3 inner corners.')
    return settings


class CameraUndistorter:
    """Remove lens distortion, preserving image size and the original K.

    Does not estimate RGB-ToF alignment. Regions without source pixels are black.
    Any supplied model ROI must be expressed in these undistorted coordinates.
    """
    def __init__(self, camera, expected_size=(640, 480)):
        size = np.asarray(camera['image_size'], dtype=np.float64)
        K = np.asarray(camera['K'], dtype=np.float64)
        distortion = np.asarray(camera['distortion'], dtype=np.float64).reshape(-1)
        if size.shape != (2,) or not np.isfinite(size).all() or tuple(size) != tuple(expected_size):
            raise ValueError(f'Camera calibration must match image size {expected_size}.')
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError('Invalid camera matrix K.')
        if not np.allclose(K[2], [0, 0, 1]) or not np.allclose([K[0, 1], K[1, 0]], 0):
            raise ValueError('Expected an OpenCV pinhole camera matrix.')
        if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError('Invalid OpenCV pinhole distortion coefficients.')
        self.image_size = tuple(int(v) for v in size)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            K, distortion, np.eye(3), K, self.image_size, cv2.CV_32FC1)

    def apply(self, bgr):
        if bgr.shape[1::-1] != self.image_size:
            raise ValueError(f'Expected image size {self.image_size}, got {bgr.shape[1::-1]}.')
        return cv2.remap(bgr, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def camera_plane(image, camera):
    """Board Z=0 plane, in camera coordinates, using metre-valued object points."""
    if list(image.shape[1::-1]) != camera['image_size']:
        raise ValueError('Image size differs from intrinsic calibration.')
    corners = find_board(image, camera['columns'], camera['rows'])
    if corners is None:
        raise ValueError('Checkerboard not detected.')
    obj = board_points(camera['columns'], camera['rows'], camera['square_mm'])
    K = np.asarray(camera['K'], dtype=np.float64)
    dist = np.asarray(camera['distortion'], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        raise ValueError('No valid checkerboard pose.')
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=2))))
    return normal, -float(normal @ tvec[:, 0]), corners, rms


def nominal_tof_rays(fov_x_deg, fov_y_deg):
    """Uniform angular-footprint pinhole centres; not ST per-zone calibration."""
    if not 0 < fov_x_deg < 150 or not 0 < fov_y_deg < 150:
        raise ValueError('FoV must be between 0 and 150 degrees, separately for X/Y.')
    axis = (np.arange(8) + 0.5) / 8 * 2 - 1
    x, y = np.meshgrid(axis * np.tan(np.deg2rad(fov_x_deg) / 2),
                       axis * np.tan(np.deg2rad(fov_y_deg) / 2))
    return np.stack([x, y, np.ones_like(x)], axis=-1).reshape(64, 3)


def orient(values, rotation=0, flip_x=False, flip_y=False):
    grid = np.rot90(np.asarray(values).reshape(8, 8), rotation // 90)
    if flip_x:
        grid = np.fliplr(grid)
    if flip_y:
        grid = np.flipud(grid)
    return grid.copy().reshape(64)


def tof_points(distance_mm, rays, depth_convention):
    if depth_convention == 'radial':
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    elif depth_convention != 'z':
        raise ValueError('Depth convention must be z or radial.')
    return np.asarray(distance_mm).reshape(64, 1) / 1000 * rays


def summarize_static_tof(samples, minimum_fraction=0.8, maximum_std_mm=30):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 64 or len(samples) < 10:
        raise ValueError('Need at least 10 ToF frames, each with 64 distances.')
    median = np.zeros(64)
    stable = np.zeros(64, dtype=bool)
    for zone in range(64):
        valid = samples[:, zone][(samples[:, zone] > 0) & (samples[:, zone] <= 4000)]
        if len(valid):
            median[zone] = np.median(valid)
            stable[zone] = len(valid) / len(samples) >= minimum_fraction and np.std(valid) <= maximum_std_mm
    return median, stable


def fit_point_to_plane(observations, validation_indices):
    """Fit P_camera=R*P_tof+t. Only returns an approximation under fixed rays."""
    from scipy.optimize import least_squares

    held = set(validation_indices)
    train = [obs for i, obs in enumerate(observations) if i not in held]
    if len(train) < 12:
        raise ValueError('Need at least 12 training poses after holding out validation poses.')
    normals = np.array([obs['normal'] for obs in train])
    singular = np.linalg.svd(normals, compute_uv=False)
    normal_ratio = float(singular[-1] / singular[0])
    if normal_ratio < 0.04:
        raise ValueError('Plane normals are too similar. Tilt the target about BOTH horizontal and vertical axes.')

    def residual(params, items):
        R, _ = cv2.Rodrigues(params[:3])
        t = params[3:]
        return np.concatenate([(item['points'] @ R.T + t) @ item['normal'] + item['offset']
                               for item in items])

    solution = least_squares(lambda p: residual(p, train), np.zeros(6),
                             loss='soft_l1', f_scale=0.01, max_nfev=2000,
                             x_scale=[0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
    if not solution.success:
        raise ValueError(f'Optimization did not converge: {solution.message}')
    R, _ = cv2.Rodrigues(solution.x[:3])
    held_items = [observations[i] for i in sorted(held)]
    if not held_items:
        raise ValueError('Independent validation poses are required.')
    train_errors = residual(solution.x, train)
    test_errors = residual(solution.x, held_items)
    metrics = {
        'training_mae_mm': float(np.mean(np.abs(train_errors)) * 1000),
        'validation_mae_mm': float(np.mean(np.abs(test_errors)) * 1000),
        'validation_p95_abs_mm': float(np.percentile(np.abs(test_errors), 95) * 1000),
        'normal_diversity_ratio': normal_ratio,
        'training_poses': len(train), 'validation_poses': len(held),
        'translation_norm_mm': float(np.linalg.norm(solution.x[3:]) * 1000),
        'per_pose_mae_mm': [float(np.mean(np.abs(residual(solution.x, [obs]))) * 1000)
                            for obs in observations],
    }
    return R, solution.x[3:], metrics
