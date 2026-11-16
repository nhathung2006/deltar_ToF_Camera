"""Optional ToF stabilization and RGB-guided depth postprocessing.

Depth stays in metres. Postprocessing reduces artifacts; it does not supply
missing sensor measurements or certify absolute accuracy. No new dependencies.
"""
import math

import cv2
import numpy as np


class ToFTemporalFilter:
    """Per-zone EMA, jump confirmation and bounded dropout hold.

    Timestamp must be monotonic seconds for distinct sensor observations.
    No spatial averaging: adjacent zones may observe different surfaces.
    """
    def __init__(self, tau_s=.12, hold_s=.18, max_distance_mm=4000.):
        if not np.isfinite([tau_s, hold_s, max_distance_mm]).all() or min(tau_s, hold_s, max_distance_mm) <= 0:
            raise ValueError('Filter durations and maximum distance must be positive and finite.')
        self.tau_s, self.hold_s, self.max_distance_mm = tau_s, hold_s, max_distance_mm
        self.reset()

    def reset(self):
        self.value = np.zeros(64, np.float32)
        self.last_good = np.full(64, -np.inf)
        self.pending = np.zeros(64, np.float32)
        self.last_time = None

    def update(self, distance_mm, timestamp):
        d = np.asarray(distance_mm, dtype=np.float32)
        if d.shape != (64,) or not np.isfinite(timestamp):
            raise ValueError('Expected 64 distances and a finite monotonic timestamp.')
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError('Each ToF observation must have a newer timestamp.')
        if self.last_time is not None and timestamp - self.last_time > .5:
            self.reset()
        dt = .067 if self.last_time is None else timestamp - self.last_time
        self.last_time = timestamp
        valid = np.isfinite(d) & (d > 0) & (d <= self.max_distance_mm)
        # Expire stale surfaces BEFORE classifying this observation. Otherwise a
        # valid new surface can be rejected as a jump and then replaced by zero.
        self.value[timestamp - self.last_good > self.hold_s] = 0
        new = valid & (self.value == 0)
        jump = valid & ~new & (np.abs(d - self.value) > np.maximum(120., .12 * self.value))
        confirmed = jump & (self.pending > 0) & (np.abs(d - self.pending) <= np.maximum(80., .08 * self.pending))
        ordinary = valid & ~new & ~jump
        alpha = 1 - math.exp(-dt / self.tau_s)
        self.value[ordinary] += alpha * (d[ordinary] - self.value[ordinary])
        self.value[new | confirmed] = d[new | confirmed]
        self.last_good[new | confirmed | ordinary] = timestamp
        self.pending = np.where(jump & ~confirmed, d, 0).astype(np.float32)
        return self.value.copy()


def _validate_images(bgr, depth):
    image = np.asarray(bgr)
    depth = np.asarray(depth, dtype=np.float32)
    if (depth.ndim != 2 or min(depth.shape) < 2 or image.shape != (*depth.shape, 3)
            or image.dtype != np.uint8 or not np.isfinite(depth).all() or np.any(depth <= 0)):
        raise ValueError('Expected uint8 BGR and matching finite positive depth in metres.')
    return image, depth


def guided_depth_filter(bgr, depth_m, radius=32, epsilon=.003, subsample=4):
    """Fast color guided filter (local linear RGB model), not a depth blur.

    Solve on a reduced image and upsample coefficients, then evaluate against
    full-resolution color. RGB edges guide smoothing; texture may transfer and
    a true depth edge without any RGB contrast may be softened.
    """
    image, depth = _validate_images(bgr, depth_m)
    if (not isinstance(radius, int) or radius < 1 or not isinstance(subsample, int)
            or subsample < 1 or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Radius/subsample must be positive integers; epsilon must be positive.')
    h, w = depth.shape
    size = (max(2, w // subsample), max(2, h // subsample))
    guide = image.astype(np.float32) / 255.
    small = cv2.resize(guide, size, interpolation=cv2.INTER_AREA)
    p = cv2.resize(depth, size, interpolation=cv2.INTER_AREA)
    r = max(1, int(round(radius / subsample)))

    def box(x):
        return cv2.boxFilter(x, -1, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT)

    mean_i, mean_p = box(small), box(p)
    covariance = np.empty((*p.shape, 3, 3), np.float32)
    cross = box(small * p[..., None]) - mean_i * mean_p[..., None]
    for i in range(3):
        for j in range(i, 3):
            cov = box(small[..., i] * small[..., j]) - mean_i[..., i] * mean_i[..., j]
            covariance[..., i, j] = covariance[..., j, i] = cov
        covariance[..., i, i] += epsilon
    a = np.linalg.solve(covariance, cross[..., None])[..., 0]
    b = mean_p - np.sum(a * mean_i, axis=2)
    a = cv2.resize(box(a), (w, h), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(box(b), (w, h), interpolation=cv2.INTER_LINEAR)
    result = np.sum(a * guide, axis=2) + b
    # Local linear fits can overshoot; keep output within the measured map range.
    return np.clip(result, depth.min(), depth.max()).astype(np.float32)


class DepthTemporalFilter:
    """Adaptive EMA with RGB-motion release and depth-jump confirmation.

    RGB motion bypasses history locally for 0.2 seconds, allowing a delayed
    depth response to follow the same motion. Otherwise a large depth jump needs
    two consistent samples; unconfirmed holds last at most 0.18 seconds, then
    resume EMA. Long frame gaps reset history. This trades one sample of delay
    on low-contrast motion for rejection of isolated sensor-driven depth flashes.
    """
    def __init__(self, tau_s=.18):
        if not np.isfinite(tau_s) or tau_s <= 0:
            raise ValueError('Temporal time constant must be finite and positive.')
        self.tau_s = tau_s
        self.reset()

    def reset(self):
        self.previous_depth = self.previous_rgb = self.last_time = None
        self.pending_depth = self.pending_since = None
        self.last_motion = None

    def update(self, depth_m, bgr, timestamp):
        image, depth = _validate_images(bgr, depth_m)
        if not np.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('Depth timestamps must be finite and strictly increasing.')
        small_rgb = cv2.GaussianBlur(image.astype(np.float32) / 255., (5, 5), 0)
        if (self.previous_depth is None or self.previous_depth.shape != depth.shape
                or timestamp - self.last_time > .5):
            result = depth.copy()
            self.pending_depth = np.zeros_like(depth)
            self.pending_since = np.full(depth.shape, np.nan)
            self.last_motion = np.full(depth.shape, -np.inf)
        else:
            alpha = 1 - math.exp(-(timestamp - self.last_time) / self.tau_s)
            movement = np.max(np.abs(small_rgb - self.previous_rgb), axis=2) > .06
            movement = cv2.dilate(movement.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            # RGB can change before the sensor/model reports the new surface.
            self.last_motion[movement] = timestamp
            movement |= timestamp - self.last_motion <= .2
            discontinuity = np.abs(depth - self.previous_depth) > np.maximum(.20, .12 * depth)
            jump = discontinuity & ~movement
            confirmed = jump & (self.pending_depth > 0) & (
                np.abs(depth - self.pending_depth) <= np.maximum(.10, .08 * depth))
            new_pending = jump & np.isnan(self.pending_since)
            self.pending_since[new_pending] = timestamp
            holding = jump & ~confirmed & (timestamp - self.pending_since < .18)
            weight = np.where(movement | confirmed, 1., np.where(holding, 0., alpha)).astype(np.float32)
            result = self.previous_depth + weight * (depth - self.previous_depth)
            self.pending_depth = np.where(jump & ~confirmed, depth, 0).astype(np.float32)
            self.pending_since[~jump | confirmed] = np.nan
        self.previous_depth = result.copy()
        self.previous_rgb = small_rgb
        self.last_time = timestamp
        return result
