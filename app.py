"""
Unified entry point - combines Part B (simulated-scan physics demo) and
Part C (real video-based trajectory extraction) into a single running
application, presented as two tabs of one NiceGUI page.

Run this file (not serve_app.py / video_gui_app.py directly) to see the
finished product as one app:

    python app.py
"""

from nicegui import ui

import serve_app
import video_gui_app


@ui.page("/")
def main_page():
    ui.query("body").style("background-color: #f4f6f8")

    ui.label("פרויקט ניתוח הגשת פינג-פונג").classes("text-2xl font-bold q-mb-md text-center w-full")

    with ui.tabs().classes("w-full") as tabs:
        tab_b = ui.tab("חלק ב' - הדגמה מבוססת פיזיקה")
        tab_c = ui.tab("חלק ג' - ניתוח מווידאו אמיתי")

    with ui.tab_panels(tabs, value=tab_b).classes("w-full"):
        with ui.tab_panel(tab_b):
            serve_app.build_ui()
        with ui.tab_panel(tab_c):
            video_gui_app.build_ui()


ui.run(title="Ping-Pong Serve Project", port=8080, reload=False)
