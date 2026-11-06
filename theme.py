"""Visual layer: CSS, chart styling, and small layout helpers. No business logic here."""
import streamlit as st

INK, MUTED, ACCENT, GRID = "#111827", "#9CA3AF", "#0F766E", "#ECECE8"
ACCENT_FILL = "rgba(15,118,110,0.12)"
COLORWAY = [ACCENT, "#1D4ED8", "#B45309", "#7C3AED", "#BE123C", "#4B5563"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Inter', system-ui, sans-serif; }

/* chrome: hide Streamlit furniture, keep the sidebar toggle */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1240px; }

/* typography */
.app-title { font-size: 1.55rem; font-weight: 600; letter-spacing: -0.02em; color: #111827; }
.app-sub   { font-size: .92rem; color: #6B7280; margin: .15rem 0 1.4rem 0; }
.sect      { font-size: .74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
             color: #6B7280; margin: 1.6rem 0 .4rem 0; }
.sect span { font-weight: 400; text-transform: none; letter-spacing: 0; margin-left: .5rem; color: #9CA3AF; }
.lede      { font-size: 1.02rem; line-height: 1.6; color: #1F2937; margin: 1.1rem 0 .2rem 0; max-width: 62rem; }

/* KPI cards */
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px 16px; }
[data-testid="stMetricLabel"] p { font-size: .7rem; font-weight: 600; letter-spacing: .06em;
                                  text-transform: uppercase; color: #6B7280; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight: 600; letter-spacing: -0.01em; }

/* tabs */
button[data-baseweb="tab"] { font-size: .9rem; font-weight: 500; color: #6B7280; }
button[data-baseweb="tab"][aria-selected="true"] { color: #111827; }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; }
[data-testid="stSidebar"] h3 { font-size: .72rem; font-weight: 600; letter-spacing: .08em;
                               text-transform: uppercase; color: #6B7280; margin-top: 1.3rem; }

/* inputs and tables */
[data-testid="stTextInput"] input { border-radius: 8px; }
[data-testid="stDataFrame"] { border: 1px solid #E5E7EB; border-radius: 10px; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, sub):
    st.markdown(f'<div class="app-title">{title}</div><div class="app-sub">{sub}</div>',
                unsafe_allow_html=True)


def section(title, sub=""):
    extra = f"<span>{sub}</span>" if sub else ""
    st.markdown(f'<div class="sect">{title}{extra}</div>', unsafe_allow_html=True)


def lede(html):
    st.markdown(f'<p class="lede">{html}</p>', unsafe_allow_html=True)


def style_fig(fig):
    fig.update_layout(
        template="simple_white", colorway=COLORWAY,
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        margin=dict(l=0, r=0, t=36, b=0),
        hoverlabel=dict(font_family="Inter, system-ui, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#D1D5DB", title_font_size=12)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title_font_size=12)
    return fig


def chart(fig):
    st.plotly_chart(style_fig(fig))
