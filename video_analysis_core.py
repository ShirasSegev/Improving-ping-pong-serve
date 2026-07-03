"""
Core video-analysis logic for the ball trajectory extraction GUI.
Kept separate from the NiceGUI callbacks so each piece can be tested
independently (see test_video_analysis_core.py).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Video metadata / frame access
# ---------------------------------------------------------------------------
def get_video_metadata(video_path: str) -> dict:
    """Returns basic metadata for a video file: fps, frame_count, width, height, duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise ValueError("Could not read FPS from video.")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


def get_frame(video_path: str, frame_index: int) -> np.ndarray:
    """Reads a single frame (BGR, as OpenCV returns it) by its index."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_index}.")
    return frame


# ---------------------------------------------------------------------------
# Ball detection (Hough circle transform), matching Appendix C.2
# ---------------------------------------------------------------------------
DEFAULT_HOUGH_PARAMS = {
    "dp": 1.2,
    "min_dist": 80,
    "param1": 100,
    "param2": 30,
    "min_radius": 15,
    "max_radius": 40,
}


def detect_ball(frame: np.ndarray, roi: dict, hough_params: dict = None):
    """
    Detects a single ball inside the given ROI of one frame.

    roi: dict with keys x, y, w, h (pixels, relative to the full frame)
    hough_params: dict with keys dp, min_dist, param1, param2, min_radius, max_radius

    Returns a dict {"x": full_frame_x, "y": full_frame_y, "r": radius} or None if not found.
    """
    hough_params = {**DEFAULT_HOUGH_PARAMS, **(hough_params or {})}

    roi_x, roi_y, roi_w, roi_h = roi["x"], roi["y"], roi["w"], roi["h"]
    cropped = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    if cropped.size == 0:
        return None

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=hough_params["dp"],
        minDist=hough_params["min_dist"],
        param1=hough_params["param1"],
        param2=hough_params["param2"],
        minRadius=hough_params["min_radius"],
        maxRadius=hough_params["max_radius"],
    )

    if circles is None:
        return None

    x, y, r = circles[0][0]
    return {"x": float(x + roi_x), "y": float(y + roi_y), "r": float(r)}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_by_ball_radius(mean_radius_px: float, ball_radius_m: float) -> float:
    """Returns the meters-per-pixel scale factor using a known ball radius."""
    return ball_radius_m / mean_radius_px


def calibrate_by_two_points(p1_px, p2_px, known_distance_m: float) -> float:
    """Returns the meters-per-pixel scale factor using two clicked points a known distance apart."""
    pixel_distance = np.hypot(p2_px[0] - p1_px[0], p2_px[1] - p1_px[1])
    if pixel_distance == 0:
        raise ValueError("The two calibration points cannot be identical.")
    return known_distance_m / pixel_distance


# ---------------------------------------------------------------------------
# Full trajectory extraction over a time window
# ---------------------------------------------------------------------------
def extract_trajectory(
    video_path: str,
    t_start: float,
    t_end: float,
    roi: dict,
    hough_params: dict = None,
    scale: float = None,
    ball_radius_m: float = 0.02,
) -> dict:
    """
    Runs ball detection over every frame in [t_start, t_end] and returns the
    trajectory in meters. If `scale` is not given, it is computed from the
    mean detected radius and `ball_radius_m` (default: a ping-pong ball).

    Returns a dict with: t, x_px, y_px, x, y, radii_px, scale, fps.
    """
    meta = get_video_metadata(video_path)
    fps = meta["fps"]

    cap = cv2.VideoCapture(video_path)
    times, centers, radii = [], [], []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_frame = frame_index / fps
        if t_start <= t_frame <= t_end:
            detection = detect_ball(frame, roi, hough_params)
            if detection is not None:
                times.append(t_frame)
                centers.append((detection["x"], detection["y"]))
                radii.append(detection["r"])

        frame_index += 1

    cap.release()

    if len(centers) == 0:
        raise ValueError("No ball detected in the selected time window / ROI / parameters.")

    centers = np.array(centers)
    t = np.array(times)
    radii = np.array(radii)
    x_px = centers[:, 0]
    y_px = centers[:, 1]

    if scale is None:
        scale = calibrate_by_ball_radius(np.mean(radii), ball_radius_m)

    x = scale * (x_px - x_px[0])
    y = -scale * (y_px - y_px[0])

    return {
        "t": t, "x_px": x_px, "y_px": y_px,
        "x": x, "y": y, "radii_px": radii,
        "scale": scale, "fps": fps,
    }


# ---------------------------------------------------------------------------
# Bonus: overlay the detected trajectory onto the video as a trail
# ---------------------------------------------------------------------------
def create_overlay_video(
    video_path: str,
    output_path: str,
    t_start: float,
    t_end: float,
    roi: dict,
    hough_params: dict = None,
) -> int:
    """
    Writes a new video file where every processed frame has the accumulated
    trail of detected ball centers drawn on it. Returns the number of
    detections drawn.
    """
    meta = get_video_metadata(video_path)
    fps = meta["fps"]

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (meta["width"], meta["height"]))

    trail = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_frame = frame_index / fps
        if t_start <= t_frame <= t_end:
            detection = detect_ball(frame, roi, hough_params)
            if detection is not None:
                trail.append((int(detection["x"]), int(detection["y"])))

            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 165, 255), 3)
            if trail:
                cv2.circle(frame, trail[-1], 6, (0, 0, 255), -1)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    return len(trail)
