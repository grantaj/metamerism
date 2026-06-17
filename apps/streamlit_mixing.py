"""Streamlit MVP for Golden paint mixing."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from metamerism.mixing import mix_spectra
from metamerism.pigments import load_golden_heavy_body_paints
from metamerism.visualization import plot_perceived_colour, plot_spectrum

MODE_CHOICES = {
    "Reflectance blend": "reflectance",
    "K/S blend": "ks",
}


@st.cache_data(show_spinner=False)
def load_paints():
    """Load the Golden paint catalogue once per session."""
    return load_golden_heavy_body_paints()


def inject_styles() -> None:
    """Apply a restrained light theme to the app."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #edf2f7 100%);
            color: #0f172a;
        }
        section[data-testid="stSidebar"] {
            background: #eef3f8;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebarContent"] {
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_mix_figure(mixture) -> plt.Figure:
    """Render spectrum and perceived-colour previews for a mixture."""
    fig, (ax_spectrum, ax_colour) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    plot_spectrum(mixture, ax=ax_spectrum, color="black")
    ax_spectrum.set_title("Mixed reflectance")
    plot_perceived_colour(mixture, ax=ax_colour, title="Perceived colour")
    return fig


def ensure_weight_state(selected_names: list[str]) -> None:
    """Initialize raw weights for newly selected paints."""
    for name in selected_names:
        key = f"weight::{name}"
        if key not in st.session_state:
            st.session_state[key] = 1.0


def render_weight_controls(selected_paints) -> list[float]:
    """Render relative-weight controls and return raw values."""
    st.subheader("Recipe")
    if st.button("Equal split", use_container_width=True):
        for paint in selected_paints:
            st.session_state[f"weight::{paint.paint_name}"] = 1.0
        st.rerun()

    raw_weights: list[float] = []
    for paint in selected_paints:
        key = f"weight::{paint.paint_name}"
        row = st.columns([2.4, 1.3], vertical_alignment="center")
        row[0].markdown(f"**{paint.paint_name}**")
        raw_weights.append(
            row[1].number_input(
                "Relative weight",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state[key]),
                step=0.1,
                key=key,
                label_visibility="collapsed",
            )
        )

    frame = pd.DataFrame(
        {
            "Paint": [paint.paint_name for paint in selected_paints],
            "Raw": raw_weights,
        }
    )
    raw_series = pd.Series(raw_weights, dtype="float64")
    total = float(raw_series.sum())
    frame["Share"] = (
        raw_series / total if total > 0 else raw_series
    ).map(lambda value: f"{value:.1%}")

    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption("Relative weights are normalized automatically.")
    return raw_weights


def render_app() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Metamerism Mixing Lab", layout="wide")
    inject_styles()

    st.title("Metamerism Mixing Lab")

    paints = load_paints()
    paint_names = [paint.paint_name for paint in paints]
    paint_lookup = {paint.paint_name: paint for paint in paints}

    mode_label = st.sidebar.radio(
        "Mixing model",
        list(MODE_CHOICES.keys()),
    )
    selected_names = st.sidebar.multiselect(
        "Paints",
        paint_names,
        default=paint_names[:3],
    )

    if not selected_names:
        st.info("Select at least one paint to build a mix.")
        st.stop()

    ensure_weight_state(selected_names)
    selected_paints = [paint_lookup[name] for name in selected_names]

    left_col, right_col = st.columns([2.3, 1], gap="large")

    with right_col:
        with st.container(border=True):
            raw_weights = render_weight_controls(selected_paints)

        with st.container(border=True):
            st.subheader("Model")
            st.write(mode_label)
            st.caption("The current model is applied to the raw weights above.")

    if sum(raw_weights) <= 0:
        st.warning("At least one relative weight must be positive.")
        st.stop()

    mixture = mix_spectra(
        [
            paint.reflectance
            if MODE_CHOICES[mode_label] == "reflectance"
            else paint.ks
            for paint in selected_paints
        ],
        raw_weights,
        mode=MODE_CHOICES[mode_label],
    )

    recipe = pd.DataFrame(
        {
            "Paint": [paint.paint_name for paint in selected_paints],
            "Raw": pd.Series(raw_weights, dtype="float64").round(3),
            "Share": (
                pd.Series(raw_weights, dtype="float64")
                / max(sum(raw_weights), 1e-12)
            ).map(lambda value: f"{value:.1%}"),
        }
    )

    with left_col:
        with st.container(border=True):
            fig = build_mix_figure(mixture)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        with st.container(border=True):
            st.subheader("Mix summary")
            st.dataframe(recipe, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    render_app()
