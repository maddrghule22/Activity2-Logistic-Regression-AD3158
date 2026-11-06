"""
Smart Student Performance & Placement Prediction System
UI Components Module: Custom CSS styles, KPI badges, metric cards, and theme support.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
/* Global aesthetics */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Metric card styling */
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease-in-out;
}

.metric-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
    border-color: #cbd5e1;
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.80rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Prediction badge */
.badge-placed {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 8px 18px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}

.badge-not-placed {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 8px 18px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}

/* Pill badge general */
.pill-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #dbeafe;
    margin-right: 6px;
}

/* Clean section header */
.section-header {
    border-left: 4px solid #2563eb;
    padding-left: 12px;
    margin: 18px 0 14px 0;
}

.section-header h3 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
}

.section-header p {
    margin: 2px 0 0 0;
    color: #64748b;
    font-size: 0.88rem;
}
</style>
"""

def inject_custom_css():
    """Injects high quality modern CSS styles into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtitle: str = "", delta: str = None):
    """Renders a styled KPI metric card."""
    delta_html = f"<span style='color: #10b981; font-weight: 600;'>{delta}</span>" if delta else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{delta_html} {subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_section_header(title: str, subtitle: str = ""):
    """Renders an accented section header."""
    html = f"""
    <div class="section-header">
        <h3>{title}</h3>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_prediction_badge(prediction: str, probability: float, confidence: float):
    """Renders a prominent prediction banner."""
    is_placed = (prediction == 'Placed')
    badge_class = "badge-placed" if is_placed else "badge-not-placed"
    icon = "🎉" if is_placed else "⚠️"
    status_text = "LIKELY TO BE PLACED" if is_placed else "PLACEMENT UNLIKELY (AT RISK)"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 24px 16px; background: {'#f0fdf4' if is_placed else '#fef2f2'}; border: 1px solid {'#bbf7d0' if is_placed else '#fecaca'}; border-radius: 14px; margin-bottom: 20px;">
        <span class="{badge_class}">{icon} {status_text}</span>
        <div style="margin-top: 14px; font-size: 1.05rem; color: {'#166534' if is_placed else '#991b1b'};">
            Placement Probability: <b>{probability * 100:.1f}%</b> &nbsp;|&nbsp; Model Confidence: <b>{confidence:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
