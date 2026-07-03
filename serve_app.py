"""
Part B - NiceGUI Application for Volleyball Serve Analysis
=============================================================
Interactive web application built on top of the base physics code
(serve_physics_base.py). The app lets the player:
  1. Enter the serve height (h) - required.
  2. Either run a "simulated video scan" that generates example v0/alpha
     values (standing in for real computer-vision extraction, which is out
     of scope for this project), OR enter v0/alpha manually.
  3. See the resulting trajectory on an interactive plot, the flight time,
     the horizontal range, and a feedback message about the serve angle.
"""

import random

from nicegui import ui

from serve_physics_base import (
    simulate_serve,
    evaluate_serve,
    build_feedback_message,
    NET_HEIGHT,
    NET_X,
    COURT_LENGTH,
)

# ---------------------------------------------------------------------------
# "Simulated scan" - stands in for a real video-analysis pipeline.
# Produces plausible v0/alpha values as if they were extracted from a serve
# video. This intentionally does NOT process any real video file, per the
# project's requirement to simulate this step rather than implement full CV.
# ---------------------------------------------------------------------------
def simulate_video_scan():
    """Returns a (v0, alpha) pair that mimics values extracted from a serve video."""
    v0 = round(random.uniform(14.0, 24.0), 1)
    alpha = round(random.uniform(5.0, 30.0), 1)
    return v0, alpha


# ---------------------------------------------------------------------------
# UI state (kept as a simple dict so callbacks can read/write it easily)
# ---------------------------------------------------------------------------
state = {
    "h": 1.80,
    "v0": 18.0,
    "alpha": 10.0,
    "use_manual": False,
}


def build_ui():
    """Builds the Part B UI inside whatever container is currently open
    (a full page when run standalone, or a tab panel when embedded)."""
    with ui.column().classes("w-full items-center q-pa-md"):
        ui.label("ניתוח והדרכת הגשת פינג-פונג").classes("text-2xl font-bold q-mb-md")

        # -------------------------------------------------------------
        # Input card
        # -------------------------------------------------------------
        with ui.card().classes("w-full max-w-2xl q-pa-md"):
            ui.label("שלב 1: נתוני הגשה").classes("text-lg font-semibold")

            h_input = ui.number(
                label="גובה נקודת ההגשה h (מ')",
                value=state["h"], min=1.0, max=3.5, step=0.05, format="%.2f",
            ).classes("w-full")

            ui.separator().classes("q-my-sm")
            ui.label("שלב 2: מקור נתוני v0 ו-α").classes("text-md font-medium")

            with ui.row().classes("items-center q-gutter-md"):
                scan_result_label = ui.label("לא בוצעה סריקה עדיין").classes("text-grey-7")

            def run_scan():
                v0, alpha = simulate_video_scan()
                state["v0"], state["alpha"] = v0, alpha
                v0_input.value = v0
                alpha_input.value = alpha
                scan_result_label.text = f"תוצאות הסריקה המדומה: v0={v0} מ'/ש', α={alpha}°"

            ui.button("הרץ סריקה מדומה מהסרטון", icon="videocam", on_click=run_scan) \
                .props("color=primary")

            ui.label(
                "* הסריקה מדמה חילוץ נתונים מווידאו של השחקן (v0, α) - "
                "אינה מנתחת קובץ וידאו אמיתי במסגרת פרויקט זה."
            ).classes("text-caption text-grey-6")

            ui.separator().classes("q-my-sm")
            ui.label("או הזיני נתונים ידנית (override לסריקה):").classes("text-md font-medium")

            with ui.row().classes("w-full q-gutter-md"):
                v0_input = ui.number(
                    label="מהירות התחלתית v0 (מ'/ש')",
                    value=state["v0"], min=1.0, max=40.0, step=0.5, format="%.1f",
                ).classes("col")
                alpha_input = ui.number(
                    label="זווית α (מעלות)",
                    value=state["alpha"], min=0.0, max=89.0, step=0.5, format="%.1f",
                ).classes("col")

            calc_button = ui.button("חשב מסלול ונתח הגשה", icon="calculate") \
                .props("color=positive").classes("q-mt-md")

        # -------------------------------------------------------------
        # Results card (built once, updated on each calculation)
        # -------------------------------------------------------------
        with ui.card().classes("w-full max-w-2xl q-pa-md q-mt-md"):
            ui.label("שלב 3: תוצאות הניתוח").classes("text-lg font-semibold")

            with ui.row().classes("w-full q-gutter-md"):
                flight_time_label = ui.label("זמן מעוף: -").classes("text-body1")
                range_label = ui.label("טווח אופקי: -").classes("text-body1")
                net_height_label = ui.label("גובה מעל הרשת: -").classes("text-body1")

            status_label = ui.label("").classes("text-h6 q-mt-sm")
            feedback_label = ui.label("").classes("text-body1 q-mt-sm")

            plot_container = ui.column().classes("w-full")

        # -------------------------------------------------------------
        # Main calculation callback
        # -------------------------------------------------------------
        def calculate():
            h = h_input.value
            v0 = v0_input.value
            alpha = alpha_input.value

            if h is None or v0 is None or alpha is None:
                ui.notify("נא למלא את כל השדות: h, v0, α", color="negative")
                return

            result = simulate_serve(v0=v0, alpha_deg=alpha, h=h)
            evaluation = evaluate_serve(result)
            message, is_valid, angle_range = build_feedback_message(v0, alpha, h, evaluation)

            flight_time_label.text = f"זמן מעוף: {result['flight_time']:.2f} שנ'"
            range_label.text = f"טווח אופקי: {result['horizontal_range']:.2f} מ'"
            net_height_label.text = (
                f"גובה מעל הרשת: {evaluation['height_at_net']:.2f} מ'"
                if evaluation["height_at_net"] is not None else "גובה מעל הרשת: לא רלוונטי"
            )

            if is_valid:
                status_label.text = "✅ הגשה תקינה"
                status_label.classes(remove="text-red", add="text-green-8")
            else:
                status_label.text = "❌ הגשה לא תקינה"
                status_label.classes(remove="text-green-8", add="text-red")

            feedback_label.text = message

            draw_plot(result, alpha)

        def draw_plot(result, alpha):
            plot_container.clear()
            with plot_container:
                fig = {
                    "data": [
                        {
                            "x": result["x"], "y": result["y"],
                            "mode": "lines", "name": "מסלול הכדור",
                            "line": {"width": 3},
                        },
                        {
                            "x": [NET_X], "y": [NET_HEIGHT],
                            "mode": "markers", "name": "גובה הרשת",
                            "marker": {"size": 12, "color": "red"},
                        },
                        {
                            "x": [0, COURT_LENGTH], "y": [0, 0],
                            "mode": "lines", "name": "קרקע",
                            "line": {"width": 1, "color": "gray", "dash": "dot"},
                        },
                    ],
                    "layout": {
                        "title": f"מסלול הגשה (α={alpha}°)",
                        "xaxis": {"title": "מרחק אופקי x (מ')"},
                        "yaxis": {"title": "גובה y (מ')"},
                        "height": 420,
                        "margin": {"t": 40},
                    },
                }
                ui.plotly(fig).classes("w-full")

        calc_button.on_click(calculate)

        # run once on load with default values so the page isn't empty
        calculate()


if __name__ in {"__main__", "__mp_main__"}:
    @ui.page("/")
    def main_page():
        ui.query("body").style("background-color: #f4f6f8")
        build_ui()

    ui.run(title="Ping-Pong Serve Analyzer", port=8080, reload=False)
