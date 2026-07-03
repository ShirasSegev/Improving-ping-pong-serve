# Improving Your Ping-Pong Serve

An interactive tool that analyzes a ping-pong serve and gives the player
feedback on their serve angle, based on a physical model of projectile
motion with air resistance and on real ball-trajectory extraction from
video.

Built as a mini-project for [course name], combining a physics simulation
(Euler's method) with a computer-vision pipeline (OpenCV Hough circle
transform) and an interactive NiceGUI interface.

## What it does

1. **Physics model** - solves the equations of motion of a ball under
   gravity and air resistance (`f = -(bv + cv^2)v_hat`), using the Euler
   method, and checks whether a given serve (height, speed, angle) clears
   the net and lands inside the court.
2. **Feedback mechanism** - for a given height and speed, scans the range
   of angles that would result in a valid serve, and tells the player
   whether their actual angle was too low (hits the net) or too high
   (lands out).
3. **Video analysis** - extracts the real ball trajectory from a serve
   video: lets the user browse frames, set a Region Of Interest, tune the
   Hough circle detector parameters live, calibrate pixels-to-meters
   (by known ball radius or by two clicked reference points), and plots
   the resulting trajectory. Can also export the video with the detected
   trail overlaid.

## Project structure

| File | Description |
|---|---|
| `app.py` | **Main entry point.** Combines both parts below into one app with two tabs. Run this one. |
| `serve_physics_base.py` | Part A - physics core (no GUI): equations of motion, Euler integration, net/court constraint checks, valid-angle-range feedback logic. |
| `serve_app.py` | Part B - NiceGUI interface for the physics model, with a "simulated scan" button standing in for video extraction. |
| `video_analysis_core.py` | Ball-detection core (no GUI): frame access, Hough circle detection, calibration, full trajectory extraction, trail-overlay video export. |
| `video_gui_app.py` | Part C - NiceGUI interface for uploading a real video and calibrating/running the extraction pipeline interactively. |
| `Ping_Pong.mp4` | Sample serve video used for testing the video-analysis pipeline. |

## Running it

Requires Python 3.10+.

```bash
pip install nicegui opencv-python-headless matplotlib numpy plotly
python app.py
```

Then open the URL printed in the terminal (`http://localhost:8080`) in a
browser.

To run one part on its own instead of the combined app:

```bash
python serve_app.py       # physics model only
python video_gui_app.py   # video analysis only
```

Note: `serve_app.py` and `video_gui_app.py` both default to port 8080,
so don't run two of them (or one of them alongside `app.py`) at the same
time - stop one with `Ctrl+C` first.

## Physical model

Forces acting on the ball: gravity and air drag
`f_vec = -(b + c*v) * v_vec`, where `v` is the ball's speed. Newton's
second law gives:

```
ax = -(b + c*v)/m * vx
ay = -g - (b + c*v)/m * vy
```

solved numerically with the Euler method from the serve height `h` and
initial velocity components `v0*cos(alpha)`, `v0*sin(alpha)`.

## Acknowledgements

Video ball-detection code is based on the course's data-acquisition
appendix (Hough circle transform via OpenCV), adapted and extended into
an interactive calibration tool with AI assistance (Claude).
