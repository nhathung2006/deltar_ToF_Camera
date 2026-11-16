"""Windows camera and serial input for the ESP32's F and F2 ToF protocols."""
from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np
import serial
from cv2_enumerate_cameras import enumerate_cameras


@dataclass(frozen=True)
class ToFFrame:
    counter: int
    distance_mm: np.ndarray
    received_at: float
    protocol: str = 'F'
    sensor_timestamp: int | None = None
    target_status: np.ndarray | None = None
    range_sigma_mm: np.ndarray | None = None


def parse_tof_line(line, received_at=None):
    """Parse F or F2, masking F2 zones to target_status 5/9.

    F:  counter, 64 distances (mm; already masked by firmware).
    F2: counter, board timestamp, 64 distances, 64 statuses, 64 range sigmas.
    Range sigma is measurement uncertainty, not a DELTAR depth distribution.
    Ignore board logs, partial lines, and malformed frames. Counter can wrap.
    """
    try:
        text = line.decode('ascii') if isinstance(line, (bytes, bytearray)) else line
        if not isinstance(text, str):
            return None
        fields = text.strip().split(',')
        protocol = fields[0]
        if (protocol, len(fields)) not in (('F', 66), ('F2', 195)):
            return None
        counter = int(fields[1])
        sensor_timestamp = None
        target_status = None
        range_sigma_mm = None
        if protocol == 'F2':
            sensor_timestamp = int(fields[2])
            values = np.array([int(v) for v in fields[3:67]], dtype=np.float32)
            statuses = [int(v) for v in fields[67:131]]
            range_sigma_mm = np.array([float(v) for v in fields[131:195]], dtype=np.float32)
            if (sensor_timestamp < 0 or any(s < 0 or s > 255 for s in statuses)
                    or np.any(range_sigma_mm < 0) or not np.all(np.isfinite(range_sigma_mm))):
                return None
            target_status = np.array(statuses, dtype=np.uint8)
            if not np.all(np.isfinite(values)):
                return None
            values = np.where(np.isin(target_status, (5, 9)) & (values > 0), values, 0)
        else:
            values = np.array([int(v) for v in fields[2:]], dtype=np.float32)
        if counter < 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
            return None
        return ToFFrame(counter, values, time.monotonic() if received_at is None else received_at,
                        protocol, sensor_timestamp, target_status, range_sigma_mm)
    except (ValueError, UnicodeError, OverflowError):
        return None


class SerialToF:
    """Drain serial continuously so inference does not build up a frame backlog."""
    def __init__(self, port='COM9', baud=921600):
        self.serial = serial.Serial()
        self.serial.port = port
        self.serial.baudrate = baud
        self.serial.timeout = 0.15
        self.serial.dtr = False
        self.serial.rts = False
        self.latest = None
        self.frame_count = 0
        self.ignored_lines = 0
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def __enter__(self):
        self.serial.open()
        self.serial.reset_input_buffer()
        self.thread.start()
        return self

    def _read(self):
        pending = bytearray()
        try:
            while not self.stop.is_set():
                pending.extend(self.serial.read(min(max(self.serial.in_waiting, 1), 4096)))
                while b'\n' in pending:
                    raw, _, rest = pending.partition(b'\n')
                    pending = bytearray(rest)
                    frame = parse_tof_line(raw)
                    with self.lock:
                        if frame is not None:
                            self.latest = frame
                            self.frame_count += 1
                        else:
                            self.ignored_lines += 1
                if len(pending) > 16384:
                    pending.clear()
        except Exception as exc:
            self.error = exc

    def snapshot(self):
        if self.error is not None:
            raise RuntimeError(f'Serial connection failed: {self.error}')
        with self.lock:
            return self.latest, self.frame_count, self.ignored_lines

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=1)
        self.serial.close()


def camera_inventory():
    return list(enumerate_cameras(cv2.CAP_DSHOW))


def open_camera(index=None):
    if index is None:
        logitech = next((c for c in camera_inventory() if c.vid == 0x046D), None)
        if logitech is None:
            raise RuntimeError('Logitech camera not found. Run python -m apps.check_camera_tof or pass --camera INDEX.')
        index = logitech.index
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open camera {index}. Close other camera applications.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap, index


def grid_rectangles(roi):
    """ROI is x,y,width,height; model rectangles are y1,x1,y2,x2, row-major."""
    x, y, width, height = roi
    if width <= 0 or height <= 0:
        raise ValueError('ROI width and height must be positive.')
    xs = np.linspace(x, x + width, 9)
    ys = np.linspace(y, y + height, 9)
    return np.array([[ys[r], xs[c], ys[r + 1], xs[c + 1]]
                     for r in range(8) for c in range(8)], dtype=np.float32)
