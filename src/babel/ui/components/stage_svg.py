"""Custom HTML/SVG communication stage renderer for Replay Theater & Human Arena."""

from __future__ import annotations

SHAPE_NAMES = ("circle", "square", "triangle")
COLOR_HEX = {
    0: "#e63946",  # red
    1: "#2a9d8f",  # green
    2: "#277da1",  # blue
}
COLOR_NAMES = ("red", "green", "blue")
SYMBOL_COLORS = ("#3a86ff", "#ff006e", "#fb5607", "#8338ec", "#06d6a0", "#ffbe0b", "#118ab2", "#073b4c", "#6c757d")


def _shape_svg(shape_id: int, color_id: int, size: int = 72) -> str:
    fill = COLOR_HEX.get(int(color_id), "#6c757d")
    if int(shape_id) == 0:
        geom = f'<circle cx="{size//2}" cy="{size//2}" r="{size//2 - 8}" fill="{fill}" stroke="#1f2937" stroke-width="2.5"/>'
    elif int(shape_id) == 1:
        geom = f'<rect x="10" y="10" width="{size - 20}" height="{size - 20}" rx="6" fill="{fill}" stroke="#1f2937" stroke-width="2.5"/>'
    else:
        p1 = f"{size//2},8"
        p2 = f"{size - 8},{size - 10}"
        p3 = f"8,{size - 10}"
        geom = f'<polygon points="{p1} {p2} {p3}" fill="{fill}" stroke="#1f2937" stroke-width="2.5"/>'
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{geom}</svg>'


def render_stage_html(
    *,
    message: list[int],
    is_correct: bool,
    true_label_text: str,
    pred_label_text: str,
    synthetic_truth: tuple[int, int] | None = None,
    synthetic_pred: tuple[int, int] | None = None,
    reward_value: float | None = None,
) -> str:
    """Render Speaker -> discrete symbols -> Listener stage with green/red reward pulse."""
    pulse_color = "#16a34a" if is_correct else "#dc2626"
    pulse_bg = "#dcfce7" if is_correct else "#fee2e2"
    status_text = "MATCH (+Reward)" if is_correct else "MISMATCH (0 Reward)"
    if reward_value is not None:
        status_text = f"{status_text} | r = {reward_value:.3f}"

    pills = "".join(
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:44px;height:44px;border-radius:10px;margin:0 4px;'
        f'background:{SYMBOL_COLORS[int(sym) % len(SYMBOL_COLORS)]};color:#fff;'
        f'font-weight:700;font-size:18px;box-shadow:0 2px 6px rgba(0,0,0,0.2);">m={int(sym)}</span>'
        for sym in message
    )

    left_visual = _shape_svg(*synthetic_truth) if synthetic_truth is not None else ""
    right_visual = _shape_svg(*synthetic_pred) if synthetic_pred is not None else ""

    return f"""
    <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px;background:#0f172a;color:#f8fafc;font-family:Inter,system-ui,sans-serif;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
        <div style="flex:1;min-width:150px;text-align:center;padding:12px;border-radius:10px;background:#1e293b;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;">World Input &rarr; Speaker</div>
          <div style="margin:10px 0;">{left_visual}</div>
          <div style="font-weight:600;font-size:14px;">{true_label_text}</div>
        </div>
        <div style="flex:1;min-width:180px;text-align:center;padding:12px;border:2px dashed #475569;border-radius:10px;background:#0b1120;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#38bdf8;">Discrete Channel Boundary</div>
          <div style="margin:14px 0;">{pills}</div>
          <div style="font-size:12px;color:#94a3b8;">Integer IDs only &mdash; zero raw features leaked</div>
        </div>
        <div style="flex:1;min-width:150px;text-align:center;padding:12px;border-radius:10px;background:#1e293b;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;">Listener &rarr; Prediction</div>
          <div style="margin:10px 0;">{right_visual}</div>
          <div style="font-weight:600;font-size:14px;">{pred_label_text}</div>
        </div>
      </div>
      <div style="margin-top:14px;padding:8px 14px;border-radius:8px;text-align:center;font-weight:700;background:{pulse_bg};color:{pulse_color};border:1px solid {pulse_color};">
        {status_text}
      </div>
    </div>
    """
