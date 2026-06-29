"""Spectral paint mixing explorer — GOLDEN Heavy Body catalogue."""

from __future__ import annotations

import io

import colour
import matplotlib.pyplot as plt
import pandas as pd
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
) -> tuple[tuple[int, int, int], tuple[float, float, float], bytes]:
    """Mix colour, CIE Lab, and spectrum PNG — cached per unique combination."""
    lookup = _paint_lookup()
    selected = [lookup[n] for n in paint_names]
    mix = mix_spectra([p.ks for p in selected], list(weights), mode="ks")

    cmfs = OBSERVER_OPTIONS[observer_key]
    rgb_f = perceived_colour_rgb(mix, cmfs=cmfs)
    rgb_i: tuple[int, int, int] = tuple(round(x * 255) for x in rgb_f)  # type: ignore[assignment]

    # CIE Lab — always D65/10° to match measurement conditions
    _shape = colour.SpectralShape(380, 780, 5)
    _cmfs10 = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"].copy().align(_shape)
    _ill = colour.SDS_ILLUMINANTS["D65"].copy().align(_shape)
    xyz = colour.sd_to_XYZ(mix.sd.copy().align(_shape), cmfs=_cmfs10, illuminant=_ill)
    lab_arr = colour.XYZ_to_Lab(xyz / 100.0)
    lab: tuple[float, float, float] = (float(lab_arr[0]), float(lab_arr[1]), float(lab_arr[2]))

    # Spectrum chart
    wl = mix.wavelengths
    refl_pct = mix.values * 100.0
    all_swatches = _all_swatch_rgbs(observer_key)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Component curves — use each paint's perceived colour, darkened for legibility
    for name, paint in zip(paint_names, selected):
        r, g, b = all_swatches[name]
        luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        if luma > 0.62:
            scale = 0.62 / luma
            r, g, b = int(r * scale), int(g * scale), int(b * scale)
        ax.plot(
            paint.reflectance.wavelengths,
            paint.reflectance.values * 100.0,
            color=(r / 255, g / 255, b / 255),
            linewidth=1.2,
            alpha=0.75,
            label=name,
        )

    # Mix curve — bold, dark, on top
    ax.plot(wl, refl_pct, color="#0f172a", linewidth=2.2, label="Mix", zorder=5)
    ax.fill_between(wl, refl_pct, alpha=0.05, color="#0f172a", zorder=4)

    ax.set_xlabel("Wavelength (nm)", fontsize=10, color="#475569")
    ax.set_ylabel("Reflectance (%)", fontsize=10, color="#475569")
    ax.set_xlim(float(wl[0]), float(wl[-1]))
    ax.set_ylim(0, 102)
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(True, color="#e2e8f0", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right", ncol=max(1, len(paint_names) // 4))
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return rgb_i, lab, buf.read()


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
    mix_rgb, mix_lab, spectrum_png = _mix_result(tuple(selected_names), norm_weights, observer_key)

    # ── mix colour hero ──────────────────────────────────────────────────────
    luma = 0.299 * mix_rgb[0] + 0.587 * mix_rgb[1] + 0.114 * mix_rgb[2]
    text_col = "#ffffff" if luma < 128 else "#0f172a"

    st.markdown(
        f"""<div style="
            background:{_css_rgb(mix_rgb)};
            border-radius:16px;
            border:1px solid rgba(0,0,0,0.1);
            height:180px;
            display:flex;
            align-items:flex-end;
            padding:1rem 1.25rem;
            margin-bottom:0.4rem;
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

    # ── Lab table: component paints + mix row ────────────────────────────────
    lookup = _paint_lookup()
    L, a, b = mix_lab
    rows = []
    for name in selected_names:
        p = lookup[name]
        L_p, a_p, b_p = p.lab_d65_10
        rows.append({"Paint": name, "L*": round(L_p, 1), "a*": round(a_p, 1), "b*": round(b_p, 1)})
    rows.append({"Paint": "Mix (predicted)", "L*": round(L, 1), "a*": round(a, 1), "b*": round(b, 1)})

    st.markdown(
        "<p style='font-size:0.78rem;color:#64748b;margin-top:1.2rem;margin-bottom:0.3rem'>"
        "CIE L*a*b* — D65 · 10° · components measured, mix predicted</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Paint": st.column_config.TextColumn("Paint", width="large"),
            "L*": st.column_config.NumberColumn("L*", format="%.1f"),
            "a*": st.column_config.NumberColumn("a*", format="%.1f"),
            "b*": st.column_config.NumberColumn("b*", format="%.1f"),
        },
    )


if __name__ == "__main__":
    render_app()
