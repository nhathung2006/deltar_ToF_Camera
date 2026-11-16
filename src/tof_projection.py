"""Experimental ToF footprints in undistorted RGB coordinates.

Uses the nominal rays and R/t saved by apps.calibrate_camera_tof. Each zone is treated
as a small plane parallel to the ToF image plane at its measured centre Z.
This is a footprint approximation, especially at object edges, not a measured
per-zone angular response. The original calibration is never modified.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .camera_tof_calibration import CameraUndistorter, nominal_tof_rays, orient


class ToFProjector:
    def __init__(self, calibration):
        self.calibration = calibration
        self.camera = calibration['camera']
        self.undistorter = CameraUndistorter(self.camera)
        self.K = np.asarray(self.camera['K'], dtype=np.float64)
        self.R = np.asarray(calibration['R_tof_to_camera'], dtype=np.float64)
        self.t = np.asarray(calibration['t_tof_to_camera_m'], dtype=np.float64)
        if (self.R.shape != (3, 3) or self.t.shape != (3,)
                or not np.isfinite(self.R).all() or not np.isfinite(self.t).all()
                or not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(self.R), 1, atol=1e-4)):
            raise ValueError('Invalid ToF-to-camera rotation or translation.')
        self.rotation = calibration.get('rotate_tof_ccw', 0)
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError('Invalid calibrated ToF rotation.')
        self.flip_x = bool(calibration.get('flip_tof_x', False))
        self.flip_y = bool(calibration.get('flip_tof_y', False))
        self.convention = calibration['depth_convention']
        if self.convention not in ('z', 'radial'):
            raise ValueError('Unknown calibrated ToF distance convention.')
        fov = calibration['tof_fov_xy_deg']
        self.rays = nominal_tof_rays(*fov)
        half_fov = np.tan(np.deg2rad(fov) / 2)
        self.step = 2 * half_fov / 8
        row, col = np.divmod(np.arange(64), 8)
        self.x0 = -half_fov[0] + col * self.step[0]
        self.y0 = -half_fov[1] + row * self.step[1]

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def orient(self, raw_distance_mm):
        return orient(raw_distance_mm, self.rotation, self.flip_x, self.flip_y)

    def project(self, oriented_distance_mm, max_tof_m=4.0):
        """Input zones must already have the rotation/flips from this file."""
        distance = np.asarray(oriented_distance_mm, dtype=np.float64).reshape(64) / 1000
        valid = np.isfinite(distance) & (distance > 0) & (distance <= max_tof_m)
        ray_scale = (1 / np.linalg.norm(self.rays, axis=1)
                     if self.convention == 'radial' else np.ones(64))
        z = np.where(valid, distance * ray_scale, 1.0)
        centre = (self.rays * z[:, None]) @ self.R.T + self.t
        # Homography from local zone coordinates [s,t,1] to RGB pixels.
        plane = np.empty((64, 3, 3), dtype=np.float64)
        plane[:, :, 0] = z[:, None] * self.step[0] * self.R[:, 0]
        plane[:, :, 1] = z[:, None] * self.step[1] * self.R[:, 1]
        corner = np.stack([self.x0 * z, self.y0 * z, z], axis=1)
        plane[:, :, 2] = corner @ self.R.T + self.t
        homography = self.K @ plane
        square = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.]])
        projected = square @ homography.transpose(0, 2, 1)
        valid &= (centre[:, 2] > 0.001) & (centre[:, 2] <= 10)
        valid &= np.all(projected[:, :, 2] > 0.001, axis=1)
        polygon = projected[:, :, :2] / np.maximum(projected[:, :, 2:3], 1e-9)
        width, height = self.camera['image_size']
        valid &= ((polygon[:, :, 0].max(axis=1) >= 0)
                  & (polygon[:, :, 0].min(axis=1) < width)
                  & (polygon[:, :, 1].max(axis=1) >= 0)
                  & (polygon[:, :, 1].min(axis=1) < height))
        sigma_scale = np.abs((self.rays @ self.R.T)[:, 2] * ray_scale)
        return {'distance_mm': np.where(valid, centre[:, 2] * 1000, 0).astype(np.float32),
                'sigma_scale': sigma_scale.astype(np.float32), 'valid': valid,
                'polygons': polygon, 'homography': homography, 'plane': plane}

    def fusion_grids(self, projection, confidence=None):
        """Sampling/scatter grids for the existing 1/4, 1/8, 1/16 fusion stages.

        In overlaps, the closest local plane supplies the feature at a pixel.
        Tiles are clamped independently to prevent blending adjacent zones.
        """
        return fusion_grids_from_projection(projection, self.camera['image_size'], confidence)


class ManualToFProjector:
    """Rotate a fixed image ROI, then translate it in screen X/Y cell units.

    This is a 2D manual alignment. It never changes sensor distances, estimates
    extrinsics, or undistorts RGB. Pixels outside the image stay outside.
    """
    def __init__(self, roi, angle_ccw_deg=0., shift_cells=(0., 0.), image_size=(640, 480)):
        roi = np.asarray(roi, dtype=np.float64)
        shift = np.asarray(shift_cells, dtype=np.float64)
        if (roi.shape != (4,) or not np.isfinite(roi).all() or np.any(roi[2:] <= 0)
                or shift.shape != (2,) or not np.isfinite(shift).all()
                or not np.isfinite(angle_ccw_deg)):
            raise ValueError('Manual grid needs a finite ROI, angle and two cell shifts.')
        self.image_size = tuple(image_size)
        x, y, width, height = roi
        self.shift_pixels = shift * roi[2:] / 8
        affine = np.eye(3)
        affine[:2] = cv2.getRotationMatrix2D((x + width / 2, y + height / 2),
                                            float(angle_ccw_deg), 1.)
        affine[:2, 2] += self.shift_pixels
        row, col = np.divmod(np.arange(64), 8)
        local = np.tile(np.eye(3), (64, 1, 1))
        local[:, 0, 0] = width / 8
        local[:, 1, 1] = height / 8
        local[:, 0, 2] = x + col * width / 8
        local[:, 1, 2] = y + row * height / 8
        self.homography = affine @ local
        square = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.]])
        self.polygons = (square @ self.homography.transpose(0, 2, 1))[:, :, :2]
        image_width, image_height = self.image_size
        self.intersects_image = ((self.polygons[:, :, 0].max(axis=1) >= 0)
                                 & (self.polygons[:, :, 0].min(axis=1) < image_width)
                                 & (self.polygons[:, :, 1].max(axis=1) >= 0)
                                 & (self.polygons[:, :, 1].min(axis=1) < image_height))

    def project(self, oriented_distance_mm, max_tof_m=4.):
        distance = np.asarray(oriented_distance_mm, dtype=np.float64).reshape(64)
        valid = (np.isfinite(distance) & (distance > 0)
                 & (distance <= max_tof_m * 1000) & self.intersects_image)
        # Constant depth per cell is used only by the shared visibility map.
        plane = np.zeros((64, 3, 3))
        plane[:, 2, 2] = np.where(valid, distance / 1000, 1.)
        return {'distance_mm': np.where(valid, distance, 0).astype(np.float32),
                'sigma_scale': np.ones(64, dtype=np.float32), 'valid': valid,
                'polygons': self.polygons, 'homography': self.homography, 'plane': plane}

    def fusion_grids(self, projection, confidence=None):
        return fusion_grids_from_projection(projection, self.image_size, confidence)


def fusion_grids_from_projection(projection, image_size, confidence=None):
    """Share the same pixel ownership for display and all fusion scales."""
    width, height = image_size
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        if (confidence.shape != (height, width) or not np.isfinite(confidence).all()
                or np.any(confidence < 0) or np.any(confidence > 1)):
            raise ValueError('Confidence must match the image and be finite in [0, 1].')
    visibility = visible_zone_map(projection, (width, height))
    result = {}
    for stride in (4, 8, 16):
        h, w = height // stride, width // stride
        patch = math.ceil(56 / stride)
        s, t = np.meshgrid((np.arange(patch) + .5) / patch,
                           (np.arange(patch) + .5) / patch)
        local = np.stack([s, t, np.ones_like(s)], axis=-1)
        samples = local[None] @ projection['homography'].transpose(0, 2, 1)[:, None]
        samples = samples[:, :, :, :2] / np.maximum(samples[:, :, :, 2:3], 1e-9)
        samples = samples / stride / np.array([w - 1, h - 1]) * 2 - 1
        samples = np.clip(samples, -10, 10).astype(np.float32)
        owner = visibility['owner'][::stride, ::stride]
        coordinates = visibility['local'][::stride, ::stride]
        row, col = np.divmod(np.maximum(owner, 0), 8)
        tile_x = col * patch + np.clip(coordinates[:, :, 0] * patch - .5, 0, patch - 1)
        tile_y = row * patch + np.clip(coordinates[:, :, 1] * patch - .5, 0, patch - 1)
        scatter = np.stack([tile_x, tile_y], axis=-1) / (8 * patch - 1) * 2 - 1
        scatter[owner < 0] = 2
        result[stride] = {'sample_grid': samples, 'scatter_grid': scatter,
                          'coverage_mask': (owner >= 0)[None]}
        if confidence is not None:
            # Use the same pixel coordinates as ownership, without averaging
            # uncovered zeros into valid boundary pixels.
            result[stride]['tof_confidence'] = confidence[::stride, ::stride][None].copy()
    return result


def rgb_boundary_confidence(bgr, projection):
    """Conservative, experimental attenuation of near-zone fusion messages.

    Only near zones with an actually measured farther neighbor are candidates.
    Colors unlike the median central color get less ToF influence, retaining
    at least 15%. RGB features and metric measurements are never replaced.
    This heuristic cannot resolve same-color surfaces or a wrong center sample.
    Missing zones stay uncovered; texture on a single depth plane is untouched.
    """
    image = np.asarray(bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR image in projection coordinates.')
    height, width = image.shape[:2]
    visibility = visible_zone_map(projection, (width, height))
    owner, local = visibility['owner'], visibility['local']
    confidence = (owner >= 0).astype(np.float32)
    distances = projection['distance_mm'].reshape(8, 8)
    candidates = []
    for zone in np.flatnonzero(projection['valid']):
        row, col = divmod(zone, 8)
        neighbors = distances[max(0, row-1):min(8, row+2), max(0, col-1):min(8, col+2)]
        depth = distances[row, col]
        if np.any(neighbors > depth + max(250., .25 * depth)):
            candidates.append(zone)
    if not candidates:
        return confidence
    color = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    central = ((local[:, :, 0] > .25) & (local[:, :, 0] < .75)
               & (local[:, :, 1] > .25) & (local[:, :, 1] < .75))
    for zone in candidates:
        region = owner == zone
        core = region & central
        if np.count_nonzero(core) < 16:
            continue
        reference = np.median(color[core], axis=0)
        difference = np.linalg.norm(color[region] - reference, axis=1)
        confidence[region] = np.maximum(.15, np.exp(-(difference / 35.) ** 2))
    return confidence


def visible_zone_map(projection, image_size):
    """Assign each RGB pixel to exactly one visible ToF zone, or -1.

    A z-buffer keeps the nearest local plane where projected footprints overlap.
    The renderer and all fusion scales share this ownership map. Uncovered image
    pixels stay uncovered; no regular grid or invented depth fills those gaps.
    """
    width, height = image_size
    cached = projection.get('visibility')
    if cached is not None and cached['owner'].shape == (height, width):
        return cached
    owner = np.full((height, width), -1, dtype=np.int16)
    closest = np.full((height, width), np.inf)
    local = np.zeros((height, width, 2), dtype=np.float32)
    tolerance = 1e-7
    for zone in np.flatnonzero(projection['valid']):
        polygon = projection['polygons'][zone]
        left = max(0, int(np.floor(polygon[:, 0].min())))
        right = min(width, int(np.ceil(polygon[:, 0].max())) + 1)
        top = max(0, int(np.floor(polygon[:, 1].min())))
        bottom = min(height, int(np.ceil(polygon[:, 1].max())) + 1)
        homography = projection['homography'][zone]
        if left >= right or top >= bottom or abs(np.linalg.det(homography)) < 1e-12:
            continue
        inverse = np.linalg.inv(homography)
        xx = np.arange(left, right)[None, :]
        yy = np.arange(top, bottom)[:, None]
        denominator = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
        safe = np.where(np.abs(denominator) > 1e-12, denominator, 1)
        s = (inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]) / safe
        t = (inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]) / safe
        plane = projection['plane'][zone]
        depth = plane[2, 0] * s + plane[2, 1] * t + plane[2, 2]
        region_depth = closest[top:bottom, left:right]
        inside = ((np.abs(denominator) > 1e-12)
                  & (s >= -tolerance) & (s <= 1 + tolerance)
                  & (t >= -tolerance) & (t <= 1 + tolerance)
                  & (depth > .001) & (depth < region_depth - tolerance))
        region_depth[inside] = depth[inside]
        owner[top:bottom, left:right][inside] = zone
        local[top:bottom, left:right, 0][inside] = np.clip(s[inside], 0, 1)
        local[top:bottom, left:right, 1][inside] = np.clip(t[inside], 0, 1)
    result = {'owner': owner, 'local': local}
    projection['visibility'] = result
    return result


def draw_projected_zones(bgr, projection):
    """Draw only visible zone boundaries; hidden polygon edges are omitted."""
    image = bgr.copy()
    owner = visible_zone_map(projection, bgr.shape[1::-1])['owner']
    for zone in np.unique(owner[owner >= 0]):
        region = (owner == zone).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 220, 255), 1, cv2.LINE_8)
    return image
