CSS_STYLE = """
<style>
body {background-color:#0e1117;color:#e6edf3;}
section.main {background-color:#131a26;}
.stButton>button {background-color:#1f77b4;color:#ffffff;}
.metric-card {padding:18px;border-radius:16px;background:#161f2d;border:1px solid #23304a;margin-bottom:12px;}
.result-box {padding:18px;border-radius:16px;color:#ffffff;margin-top:16px;}
.result-normal {background:#2d7d46;}
.result-fraud {background:#b02a2a;}
.result-risk {background:#d68d1a;}
</style>
"""

PLOTLY_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': '#0e1117',
    'plot_bgcolor': '#0e1117',
    'font': {'color': '#e6edf3'},
    'legend': {'bgcolor': '#131a26', 'bordercolor': '#333', 'borderwidth': 1},
}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
