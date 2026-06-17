"""Streamlit MVP for Golden paint mixing."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from metamerism.mixing import mix_spectra
from metamerism.pigments import load_golden_heavy_body_reflectance
from metamerism.visualization import plot_perceived_colour, plot_spectrum


@st.cache_data(show_spinner=False)
def load_paints():
    """Load the Golden paint catalogue once per session."""
    return load_golden_heavy_body_reflectance()


def build_mix_figure(mixture) -> plt.Figure:
    """Render spectrum and perceived-colour previews for a mixture."""
    fig, (ax_spectrum, ax_colour) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    plot_spectrum(mixture, ax=ax_spectrum, color="black")
    ax_spectrum.set_title("Mixed reflectance")
    plot_perceived_colour(mixture, ax=ax_colour, title="Perceived colour")
    return fig


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Metamerism Mixing Lab", layout="wide")
    st.title("Metamerism Mixing Lab")

    paints = load_paints()
    paint_names = [paint.paint_name for paint in paints]
    paint_lookup = {paint.paint_name: paint for paint in paints}

    mode = st.sidebar.selectbox("Mixing model", ["reflectance", "ks"])
    selected_names = st.sidebar.multiselect(
        "Paints",
        paint_names,
        default=paint_names[:3],
    )

    if not selected_names:
        st.info("Select at least one paint to build a mix.")
        st.stop()

    st.sidebar.subheader("Weights")
    weight_values: list[float] = []
    for name in selected_names:
        key = f"weight::{name}"
        default = 1.0 / len(selected_names)
        weight_values.append(
            st.sidebar.slider(
                name,
                min_value=0.0,
                max_value=1.0,
                value=default,
                step=0.01,
                key=key,
            )
        )

    selected_paints = [paint_lookup[name] for name in selected_names]
    mixture = mix_spectra(
        [paint.spectrum for paint in selected_paints],
        weight_values,
        mode=mode,
    )

    col_mix, col_info = st.columns([2, 1], gap="large")
    with col_mix:
        fig = build_mix_figure(mixture)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with col_info:
        st.subheader("Selection")
        for paint, weight in zip(selected_paints, weight_values, strict=True):
            st.write(f"{paint.paint_name}: {weight:.2f}")

        st.subheader("Result")
        st.write(f"Mode: {mode}")
        st.write(
            f"Mix provenance: {mixture.provenance.notes or mixture.provenance.source}"
        )


if __name__ == "__main__":
    main()
