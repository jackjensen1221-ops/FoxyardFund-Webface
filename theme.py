"""Visual layer: dark, card-based look. CSS, chart styling, and small HTML components."""
import html

import streamlit as st

BG, CARD, BORDER = "#0B0D12", "#12151C", "#1F232D"
TEXT, MUTED = "#F3F4F6", "#8B91A0"
GREEN, RED, GRID = "#2FD283", "#FF5C6C", "#1A1E27"
ACCENT, BENCH = GREEN, "#7C8394"
COLORWAY = [GREEN, "#5B8DEF", "#F5B544", "#B07CFF", "#FF7AB6", "#46C2CB"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stSidebar"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: radial-gradient(1100px 480px at 50% -8%, #10261D 0%, {BG} 60%); }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1180px; }}

.brand {{ display:flex; align-items:center; gap:.6rem; font-size:1.25rem; font-weight:600; letter-spacing:-0.02em; }}
.dot {{ width:12px; height:12px; border-radius:50%; background:{GREEN}; box-shadow:0 0 14px {GREEN}; }}
.app-sub {{ font-size:.9rem; color:{MUTED}; margin:.25rem 0 1.2rem 0; }}

.hero-label {{ color:{MUTED}; font-size:.85rem; }}
.hero-value {{ font-size:3.1rem; font-weight:600; letter-spacing:-0.03em; line-height:1.1;
              font-variant-numeric:tabular-nums; }}
.hero-delta {{ display:inline-block; font-weight:600; margin-top:.4rem; padding:.15rem .65rem;
              border-radius:999px; font-size:.9rem; }}
.hero-delta.pos {{ background:rgba(47,210,131,.13); color:{GREEN}; }}
.hero-delta.neg {{ background:rgba(255,92,108,.13); color:{RED}; }}
.hero-sub {{ color:{MUTED}; font-size:.85rem; margin-left:.7rem; }}
.pos {{ color:{GREEN}; }} .neg {{ color:{RED}; }}

.chips {{ display:flex; flex-wrap:wrap; gap:.5rem; margin:1rem 0 .3rem 0; }}
.chip {{ background:{CARD}; border:1px solid {BORDER}; border-radius:999px; padding:.28rem .8rem;
        font-size:.78rem; color:#C9CDD6; }}

.sect {{ font-size:.74rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
        color:{MUTED}; margin:1.6rem 0 .5rem 0; }}
.sect span {{ font-weight:400; text-transform:none; letter-spacing:0; margin-left:.5rem; opacity:.75; }}

.picks {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:.7rem; }}
.pick {{ display:flex; gap:.75rem; align-items:center; background:{CARD}; border:1px solid {BORDER};
        border-radius:16px; padding:.85rem .95rem; }}
.avatar {{ width:40px; height:40px; border-radius:50%; background:linear-gradient(135deg,#1E3A32,#14231F);
          display:flex; align-items:center; justify-content:center; font-weight:600; color:{GREEN}; }}
.pick-t {{ font-weight:600; }} .pick-r {{ font-size:.9rem; font-variant-numeric:tabular-nums; }}
.tag {{ font-size:.62rem; background:rgba(255,92,108,.15); color:{RED}; border-radius:6px;
       padding:1px 6px; margin-left:6px; font-weight:500; }}

[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:14px 16px; }}
[data-testid="stMetricLabel"] p {{ font-size:.7rem; font-weight:600; letter-spacing:.06em;
                                   text-transform:uppercase; color:{MUTED}; }}
[data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; font-weight:600; }}

div[data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:none; }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display:none; }}
button[data-baseweb="tab"] {{ background:transparent; border:1px solid transparent; border-radius:999px;
                              padding:.35rem .95rem; color:{MUTED}; font-weight:500; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background:{CARD}; border-color:{BORDER}; color:{TEXT}; }}

[data-testid="stSidebar"] {{ border-right:1px solid {BORDER}; }}
[data-testid="stSidebar"] h3 {{ font-size:.72rem; font-weight:600; letter-spacing:.08em;
                                text-transform:uppercase; color:{MUTED}; margin-top:1.3rem; }}
[data-testid="stTextInput"] input {{ border-radius:12px; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:14px; }}
button[kind="primary"] {{ border-radius:12px; font-weight:600; color:#06120C; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def header(sub):
    st.markdown(f'<div class="brand"><span class="dot"></span>Backtest Lab</div>'
                f'<div class="app-sub">{html.escape(sub)}</div>', unsafe_allow_html=True)


def section(title, sub=""):
    extra = f"<span>{html.escape(sub)}</span>" if sub else ""
    st.markdown(f'<div class="sect">{html.escape(title)}{extra}</div>', unsafe_allow_html=True)


def hero(label, value, delta, sub):
    cls, arrow = ("pos", "▲") if delta >= 0 else ("neg", "▼")
    st.markdown(
        f'<div class="hero-label">{html.escape(label)}</div><div class="hero-value">{html.escape(value)}</div>'
        f'<span class="hero-delta {cls}">{arrow} {abs(delta):.1%}</span>'
        f'<span class="hero-sub">{html.escape(sub)}</span>', unsafe_allow_html=True)


def chips(items):
    st.markdown('<div class="chips">' + "".join(f'<span class="chip">{html.escape(i)}</span>' for i in items)
                + "</div>", unsafe_allow_html=True)


def picks(items):
    """items: list of (ticker, return, stopped_out)"""
    tiles = ""
    for t, r, stopped in items:
        cls = "pos" if r >= 0 else "neg"
        tag = '<span class="tag">stopped</span>' if stopped else ""
        tiles += (f'<div class="pick"><div class="avatar">{html.escape(t[:1])}</div><div>'
                  f'<div class="pick-t">{html.escape(t)}{tag}</div><div class="pick-r {cls}">{r:+.1%}</div></div></div>')
    st.markdown(f'<div class="picks">{tiles}</div>', unsafe_allow_html=True)


def color_cols(styler, cols):
    fn = lambda v: f"color: {GREEN}" if v > 0 else (f"color: {RED}" if v < 0 else "")
    return (styler.map if hasattr(styler, "map") else styler.applymap)(fn, subset=cols)


def style_fig(fig):
    fig.update_layout(
        template="plotly_dark", colorway=COLORWAY,
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#C9CDD6"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        margin=dict(l=0, r=0, t=36, b=0),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font_family="Inter, system-ui, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, showline=False, title_font_size=12)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title_font_size=12)
    return fig


def chart(fig):
    st.plotly_chart(style_fig(fig))
