"""
Part C - NiceGUI Application for Video-Based Serve Trajectory Extraction
===========================================================================
Interactive tool that lets the user upload a serve video and, step by step,
calibrate everything needed to extract the ball's trajectory:

  1. Upload a video, browse frames with a slider, pick a time window.
  2. Hover the mouse over the frame to read pixel coordinates and set the ROI.
  3. Run ball detection (Hough circle transform) on the current frame and
     tune the detector's parameters live.
  4. Choose a calibration method (known ball radius, or two clicked points
     a known distance apart) to convert pixels to meters.
  5. Run the full extraction over the video and plot the trajectory.
  6. (Bonus) Export a copy of the video with the detected trail overlaid.

All heavy-lifting functions live in video_analysis_core.py so they can be
unit-tested independently of the UI.
"""

import base64
import uuid
from pathlib import Path

import cv2
from nicegui import events, ui

from video_analysis_core import (
    DEFAULT_HOUGH_PARAMS,
    calibrate_by_two_points,
    create_overlay_video,
    detect_ball,
    extract_trajectory,
    get_frame,
    get_video_metadata,
)

UPLOAD_DIR = Path(__file__).parent / "uploaded_videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def frame_to_data_url(frame) -> str:
    """Encodes a BGR OpenCV frame as a base64 JPEG data URL for ui.interactive_image."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode frame as JPEG.")
    b64 = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_ui():
    """Builds the Part C UI inside whatever container is currently open
    (a full page when run standalone, or a tab panel when embedded)."""
    # -----------------------------------------------------------------
    # Per-session state
    # -----------------------------------------------------------------
    state = {
        "video_path": None,
        "meta": None,
        "frame_index": 0,
        "roi": {"x": 0, "y": 0, "w": 100, "h": 100},
        "hough": dict(DEFAULT_HOUGH_PARAMS),
        "last_detection": None,
        "calibration_mode": "radius",   # "radius" or "two_points"
        "ball_radius_m": 0.02,
        "calib_points": [],             # up to 2 (x, y) pixel points
        "calib_distance_m": 1.0,
        "result": None,
    }

    ui.label("Serve Trajectory Extractor (from Video)").classes("text-2xl font-bold q-mb-md")

    # ===================================================================
    # STEP 1 - Upload video, browse frames, choose time window
    # ===================================================================
    with ui.card().classes("w-full max-w-4xl q-pa-md"):
        ui.label("Step 1 - Upload video and select a time window").classes("text-lg font-semibold")

        info_label = ui.label("No video loaded yet.").classes("text-grey-7")

        image = ui.interactive_image(
            "", events=["mousemove", "click"], cross=True,
        ).classes("w-full").style("max-width: 900px")

        coord_label = ui.label("Mouse position: -").classes("text-caption text-grey-6")

        frame_slider = ui.slider(min=0, max=1, value=0, step=1).classes("w-full")
        frame_slider_label = ui.label("Frame: 0 / 0 (t = 0.00 s)")

        with ui.row().classes("w-full q-gutter-md items-end"):
            t_start_input = ui.number(label="t_start (s)", value=0.0, step=0.1, min=0.0).classes("col")
            t_end_input = ui.number(label="t_end (s)", value=0.0, step=0.1, min=0.0).classes("col")

        # ---------------------------------------------------------
        def redraw_overlay():
            """Draws the ROI rectangle, last detected circle, and calibration
            points as an SVG overlay on top of the current frame."""
            roi = state["roi"]
            svg_parts = [
                f'<rect x="{roi["x"]}" y="{roi["y"]}" width="{roi["w"]}" height="{roi["h"]}" '
                f'fill="none" stroke="lime" stroke-width="3" />'
            ]
            det = state["last_detection"]
            if det is not None:
                svg_parts.append(
                    f'<circle cx="{det["x"]:.1f}" cy="{det["y"]:.1f}" r="{det["r"]:.1f}" '
                    f'fill="none" stroke="red" stroke-width="3" />'
                )
            for (px, py) in state["calib_points"]:
                svg_parts.append(f'<circle cx="{px}" cy="{py}" r="6" fill="yellow" />')
            if len(state["calib_points"]) == 2:
                (x1, y1), (x2, y2) = state["calib_points"]
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="yellow" stroke-width="2" stroke-dasharray="6,4" />'
                )
            image.content = "".join(svg_parts)

        def load_frame(index: int):
            if state["video_path"] is None:
                return
            index = max(0, min(index, state["meta"]["frame_count"] - 1))
            state["frame_index"] = index
            frame = get_frame(state["video_path"], index)
            image.set_source(frame_to_data_url(frame))
            t = index / state["meta"]["fps"]
            frame_slider_label.text = f"Frame: {index} / {state['meta']['frame_count'] - 1} (t = {t:.2f} s)"
            state["last_detection"] = None
            redraw_overlay()

        async def on_upload(e: events.UploadEventArguments):
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{e.file.name}"
            await e.file.save(dest)

            state["video_path"] = str(dest)
            meta = get_video_metadata(str(dest))
            state["meta"] = meta

            info_label.text = (
                f"Loaded '{e.file.name}': {meta['width']}x{meta['height']} px, "
                f"{meta['fps']:.1f} fps, {meta['frame_count']} frames, "
                f"{meta['duration']:.2f} s"
            )

            # Default ROI = full frame; default time window = full duration
            state["roi"] = {"x": 0, "y": 0, "w": meta["width"], "h": meta["height"]}
            roi_x_input.value, roi_y_input.value = 0, 0
            roi_w_input.value, roi_h_input.value = meta["width"], meta["height"]
            t_start_input.value = 0.0
            t_end_input.value = round(meta["duration"], 2)

            frame_slider.min = 0
            frame_slider.max = meta["frame_count"] - 1
            middle = meta["frame_count"] // 2
            frame_slider.value = middle
            load_frame(middle)

        ui.upload(label="Upload a serve video (mp4)", auto_upload=True, on_upload=on_upload) \
            .props("accept=video/*").classes("w-full")

        def on_mouse(e: events.MouseEventArguments):
            coord_label.text = f"Mouse position: x={e.image_x:.0f} px, y={e.image_y:.0f} px"
            if e.type == "click" and state["calibration_mode"] == "two_points":
                if len(state["calib_points"]) >= 2:
                    state["calib_points"] = []
                state["calib_points"].append((e.image_x, e.image_y))
                redraw_overlay()

        image.on_mouse(on_mouse)

        frame_slider.on_value_change(lambda e: load_frame(int(e.value)))

    # ===================================================================
    # STEP 2 - Region Of Interest (ROI)
    # ===================================================================
    with ui.card().classes("w-full max-w-4xl q-pa-md q-mt-md"):
        ui.label("Step 2 - Region of Interest (ROI)").classes("text-lg font-semibold")
        ui.label(
            "Move the mouse over the frame above to read pixel coordinates, "
            "then set the ROI rectangle that contains the ball's motion."
        ).classes("text-caption text-grey-6")

        with ui.row().classes("w-full q-gutter-md"):
            roi_x_input = ui.number(label="roi_x", value=0, min=0, step=10).classes("col")
            roi_y_input = ui.number(label="roi_y", value=0, min=0, step=10).classes("col")
            roi_w_input = ui.number(label="roi_w", value=100, min=1, step=10).classes("col")
            roi_h_input = ui.number(label="roi_h", value=100, min=1, step=10).classes("col")

        def update_roi():
            state["roi"] = {
                "x": int(roi_x_input.value), "y": int(roi_y_input.value),
                "w": int(roi_w_input.value), "h": int(roi_h_input.value),
            }
            redraw_overlay()

        for inp in (roi_x_input, roi_y_input, roi_w_input, roi_h_input):
            inp.on_value_change(lambda e: update_roi())

    # ===================================================================
    # STEP 3 - Ball detection & Hough parameter tuning
    # ===================================================================
    with ui.card().classes("w-full max-w-4xl q-pa-md q-mt-md"):
        ui.label("Step 3 - Ball detection (Hough circle transform)").classes("text-lg font-semibold")

        detection_label = ui.label("No detection yet.").classes("text-body1")

        def run_detection():
            if state["video_path"] is None:
                ui.notify("Please upload a video first.", color="warning")
                return
            frame = get_frame(state["video_path"], state["frame_index"])
            det = detect_ball(frame, state["roi"], state["hough"])
            state["last_detection"] = det
            detection_label.text = (
                f"Ball found at x={det['x']:.1f}, y={det['y']:.1f}, r={det['r']:.1f} px"
                if det is not None else "No ball detected with the current parameters."
            )
            redraw_overlay()

        ui.button("Detect ball in current frame", icon="my_location", on_click=run_detection) \
            .props("color=primary")

        ui.label("Hough transform parameters (adjust and re-detect):").classes("q-mt-sm text-body2")

        with ui.grid(columns=3).classes("w-full q-gutter-sm"):
            dp_input = ui.number(label="dp", value=state["hough"]["dp"], step=0.1, min=0.1)
            min_dist_input = ui.number(label="minDist", value=state["hough"]["min_dist"], step=5)
            param1_input = ui.number(label="param1", value=state["hough"]["param1"], step=5)
            param2_input = ui.number(label="param2", value=state["hough"]["param2"], step=1)
            min_radius_input = ui.number(label="minRadius", value=state["hough"]["min_radius"], step=1)
            max_radius_input = ui.number(label="maxRadius", value=state["hough"]["max_radius"], step=1)

        def update_hough():
            state["hough"] = {
                "dp": dp_input.value, "min_dist": min_dist_input.value,
                "param1": param1_input.value, "param2": param2_input.value,
                "min_radius": min_radius_input.value, "max_radius": max_radius_input.value,
            }

        for inp in (dp_input, min_dist_input, param1_input, param2_input, min_radius_input, max_radius_input):
            inp.on_value_change(lambda e: update_hough())

    # ===================================================================
    # STEP 4 - Calibration
    # ===================================================================
    with ui.card().classes("w-full max-w-4xl q-pa-md q-mt-md"):
        ui.label("Step 4 - Calibration (pixels to meters)").classes("text-lg font-semibold")

        calib_radio = ui.radio(
            {"radius": "Known ball radius", "two_points": "Two clicked points (known distance)"},
            value="radius",
        ).props("inline")

        radius_row = ui.row().classes("items-center q-gutter-md")
        with radius_row:
            ball_radius_input = ui.number(
                label="Ball radius (m)", value=0.02, step=0.001, format="%.3f"
            )
            ui.label("Default 0.02 m = standard ping-pong ball.").classes("text-caption text-grey-6")

        points_row = ui.row().classes("items-center q-gutter-md")
        with points_row:
            ui.label("Click two points on the frame above (e.g. the ends of a ruler).")
            calib_distance_input = ui.number(label="Real distance (m)", value=1.0, step=0.01)
            calib_points_label = ui.label("Points selected: 0 / 2")

        points_row.set_visibility(False)

        def on_calib_mode_change():
            state["calibration_mode"] = calib_radio.value
            state["calib_points"] = []
            radius_row.set_visibility(calib_radio.value == "radius")
            points_row.set_visibility(calib_radio.value == "two_points")
            redraw_overlay()

        calib_radio.on_value_change(lambda e: on_calib_mode_change())

        def refresh_calib_points_label():
            calib_points_label.text = f"Points selected: {len(state['calib_points'])} / 2"
        ui.timer(0.5, refresh_calib_points_label)

    # ===================================================================
    # STEP 5 - Run full analysis
    # ===================================================================
    with ui.card().classes("w-full max-w-4xl q-pa-md q-mt-md"):
        ui.label("Step 5 - Run full trajectory extraction").classes("text-lg font-semibold")

        run_button = ui.button("Run full analysis", icon="play_arrow").props("color=positive")
        results_label = ui.label("").classes("text-body1 q-mt-sm")
        plot_container = ui.column().classes("w-full")

        def run_full_analysis():
            if state["video_path"] is None:
                ui.notify("Please upload a video first.", color="warning")
                return

            scale = None
            if state["calibration_mode"] == "two_points":
                if len(state["calib_points"]) != 2:
                    ui.notify("Please click exactly two calibration points on the frame.", color="warning")
                    return
                scale = calibrate_by_two_points(
                    state["calib_points"][0], state["calib_points"][1], calib_distance_input.value
                )

            try:
                result = extract_trajectory(
                    state["video_path"],
                    t_start=t_start_input.value, t_end=t_end_input.value,
                    roi=state["roi"], hough_params=state["hough"],
                    scale=scale, ball_radius_m=ball_radius_input.value,
                )
            except ValueError as err:
                ui.notify(str(err), color="negative")
                return

            state["result"] = result
            flight_time = result["t"][-1] - result["t"][0]
            horizontal_range = abs(result["x"][-1] - result["x"][0])

            results_label.text = (
                f"Detections: {len(result['t'])} | Flight time: {flight_time:.2f} s | "
                f"Horizontal range: {horizontal_range:.2f} m | "
                f"Scale: {result['scale']*1000:.3f} mm/px"
            )

            plot_container.clear()
            with plot_container:
                fig = {
                    "data": [{
                        "x": result["x"].tolist(), "y": result["y"].tolist(),
                        "mode": "markers+lines", "name": "Extracted trajectory",
                    }],
                    "layout": {
                        "title": "Ball trajectory extracted from video",
                        "xaxis": {"title": "x (m)"},
                        "yaxis": {"title": "y (m)"},
                        "height": 420, "margin": {"t": 40},
                    },
                }
                ui.plotly(fig).classes("w-full")

        run_button.on_click(run_full_analysis)

    # ===================================================================
    # STEP 6 (bonus) - Overlay trajectory on video
    # ===================================================================
    with ui.card().classes("w-full max-w-4xl q-pa-md q-mt-md q-mb-lg"):
        ui.label("Step 6 (bonus) - Export video with trajectory overlay").classes("text-lg font-semibold")
        overlay_status = ui.label("").classes("text-body1")

        def export_overlay_video():
            if state["video_path"] is None:
                ui.notify("Please upload a video first.", color="warning")
                return
            out_path = UPLOAD_DIR / f"overlay_{uuid.uuid4().hex}.mp4"
            n = create_overlay_video(
                state["video_path"], str(out_path),
                t_start=t_start_input.value, t_end=t_end_input.value,
                roi=state["roi"], hough_params=state["hough"],
            )
            overlay_status.text = f"Overlay video created ({n} points drawn). Starting download..."
            ui.download(out_path, filename="trajectory_overlay.mp4")

        ui.button("Create & download overlay video", icon="movie", on_click=export_overlay_video) \
            .props("color=secondary")


if __name__ in {"__main__", "__mp_main__"}:
    @ui.page("/")
    def main_page():
        build_ui()

    ui.run(title="Serve Trajectory Extractor", port=8080, reload=False)
