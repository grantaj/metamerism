"""Spectral paint mixing explorer — GOLDEN Heavy Body catalogue."""

from __future__ import annotations

import io

import colour
import matplotlib.pyplot as plt
import streamlit as st

from metamerism.mixing import mix_spectra
from metamerism.pigments import load_golden_heavy_body_paints
from metamerism.visualization import perceived_colour_rgb

OBSERVER_OPTIONS: dict[str, colour.MultiSpectralDistributions] = {
    "CIE 1964 10°": colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"],
    "CIE 1931 2°": colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
}

_CSS = """
<style>
.stApp { background: #f8f9fa; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; }

/* give sliders breathing room below their label */
section[data-testid="stSidebar"] [data-testid="stSlider"] { margin-top: 0.5rem; }
</style>
"""


# ── data / caching ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_paints():
    """Load catalogue once; cache_resource returns the same object with no copy overhead."""
    return load_golden_heavy_body_paints()


@st.cache_resource(show_spinner=False)
def _paint_lookup() -> dict:
    return {p.paint_name: p for p in _load_paints()}


@st.cache_data(show_spinner=False)
def _all_swatch_rgbs(observer_key: str) -> dict[str, tuple[int, int, int]]:
    """Pre-compute perceived colours for all paints at once on first use."""
    cmfs = OBSERVER_OPTIONS[observer_key]
    result = {}
    for p in _load_paints():
        rgb = perceived_colour_rgb(p.reflectance, cmfs=cmfs)
        result[p.paint_name] = tuple(round(x * 255) for x in rgb)
    return result


def _swatch_rgb(paint_name: str, observer_key: str) -> tuple[int, int, int]:
    return _all_swatch_rgbs(observer_key)[paint_name]  # type: ignore[return-value]


@st.cache_data(show_spinner=False)
def _mix_result(
    paint_names: tuple[str, ...],
    weights: tuple[float, ...],
    observer_key: str,
) -> tuple[tuple[int, int, int], bytes]:
    """Mix colour (sRGB 0–255) + spectrum PNG — cached per unique combination."""
    lookup = _paint_lookup()
    selected = [lookup[n] for n in paint_names]
    mix = mix_spectra([p.ks for p in selected], list(weights), mode="ks")

    cmfs = OBSERVER_OPTIONS[observer_key]
    rgb_f = perceived_colour_rgb(mix, cmfs=cmfs)
    rgb_i: tuple[int, int, int] = tuple(round(x * 255) for x in rgb_f)  # type: ignore[assignment]

    wl = mix.wavelengths
    refl_pct = mix.values * 100.0

    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(wl, refl_pct, color="#0f172a", linewidth=1.8)
    ax.fill_between(wl, refl_pct, alpha=0.06, color="#0f172a")
    ax.set_xlabel("Wavelength (nm)", fontsize=10, color="#475569")
    ax.set_ylabel("Reflectance (%)", fontsize=10, color="#475569")
    ax.set_xlim(float(wl[0]), float(wl[-1]))
    ax.set_ylim(0, 102)
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(True, color="#e2e8f0", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return rgb_i, buf.read()


# ── helpers ──────────────────────────────────────────────────────────────────

def _css_rgb(rgb: tuple[int, int, int]) -> str:
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _normalised_weights(raw: list[int]) -> tuple[float, ...]:
    """Normalise integer parts to fractions — integers hash cleanly so no rounding needed."""
    total = sum(raw)
    if total <= 0:
        return tuple(0.0 for _ in raw)
    return tuple(r / total for r in raw)


# ── sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(paint_names: list[str]) -> tuple[list[str], str, list[float]]:
    st.sidebar.markdown("## Mixing Lab")

    observer_key: str = st.sidebar.selectbox(
        "Observer",
        list(OBSERVER_OPTIONS.keys()),
        index=0,
        help="Standard observer for reflectance → colour conversion.",
    )  # type: ignore[assignment]

    selected_names: list[str] = st.sidebar.multiselect(
        "Paints",
        paint_names,
        default=paint_names[:2],
        placeholder="Choose paints…",
    )

    if not selected_names:
        return selected_names, observer_key, []

    st.sidebar.divider()

    if st.sidebar.button("Equal split", use_container_width=True):
        for name in selected_names:
            st.session_state[f"w::{name}"] = 50
        st.rerun()

    # Read current slider values first so we can compute shares before rendering.
    for name in selected_names:
        if f"w::{name}" not in st.session_state:
            st.session_state[f"w::{name}"] = 50

    raw_weights: list[int] = [st.session_state[f"w::{name}"] for name in selected_names]
    total = sum(raw_weights)

    for i, name in enumerate(selected_names):
        key = f"w::{name}"
        rgb = _swatch_rgb(name, observer_key)
        share_pct = f"{raw_weights[i] / total:.0%}" if total > 0 else "—"

        st.sidebar.markdown(
            f"""<div style="display:flex;align-items:center;gap:9px;margin-top:0.7rem">
              <div style="
                width:28px;height:28px;border-radius:5px;flex-shrink:0;
                background:{_css_rgb(rgb)};border:1px solid rgba(0,0,0,0.15);
              "></div>
              <span style="font-size:0.84rem;font-weight:500;color:#0f172a;flex:1;
                           white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                {name}
              </span>
              <span style="font-size:0.84rem;font-weight:600;color:#0f172a;flex-shrink:0">
                {share_pct}
              </span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.sidebar.slider(
            name,
            min_value=0,
            max_value=100,
            step=1,
            key=key,
            label_visibility="collapsed",
        )
        raw_weights[i] = st.session_state[key]  # pick up any change from this render

    return selected_names, observer_key, raw_weights


# ── main area ────────────────────────────────────────────────────────────────

def render_app() -> None:
    st.set_page_config(page_title="Mixing Lab", layout="wide", page_icon="🎨")
    st.markdown(_CSS, unsafe_allow_html=True)

    paint_names = [p.paint_name for p in _load_paints()]
    # Warm swatch cache for all observers so first selection is instant.
    for _ok in OBSERVER_OPTIONS:
        _all_swatch_rgbs(_ok)
    selected_names, observer_key, raw_weights = _render_sidebar(paint_names)

    if not selected_names:
        st.markdown("## Mixing Lab")
        st.info("Select paints in the sidebar to begin.", icon="🎨")
        return

    if sum(raw_weights) <= 0:
        st.warning("Set at least one weight above zero.", icon="⚠️")
        return

    norm_weights = _normalised_weights(raw_weights)
    mix_rgb, spectrum_png = _mix_result(tuple(selected_names), norm_weights, observer_key)

    # ── mix colour hero ──────────────────────────────────────────────────────
    luma = 0.299 * mix_rgb[0] + 0.587 * mix_rgb[1] + 0.114 * mix_rgb[2]
    text_col = "#ffffff" if luma < 128 else "#0f172a"

    st.markdown(
        f"""<div style="
            background:{_css_rgb(mix_rgb)};
            border-radius:16px;
            border:1px solid rgba(0,0,0,0.1);
            height:200px;
            display:flex;
            align-items:flex-end;
            padding:1rem 1.25rem;
            margin-bottom:0.5rem;
        ">
            <span style="font-size:0.85rem;font-weight:500;color:{text_col};opacity:0.75">
                Predicted mix
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;margin-bottom:1.5rem'>"
        "Kubelka–Munk arithmetic K/S blend · GOLDEN Heavy Body"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── reflectance spectrum ─────────────────────────────────────────────────
    st.image(spectrum_png, use_container_width=True)
    st.caption("Mixed reflectance spectrum")


if __name__ == "__main__":
    render_app()
