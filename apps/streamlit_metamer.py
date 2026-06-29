"""Metamer search diagnostic app — ideal reflectance partner generation.

Allows the user to pick a Golden paint as a reference, choose conceal and
reveal illuminants, run bounded null-space generation, and inspect the
resulting candidates visually.

All search logic lives in metamerism.search; this file only drives the UI.
"""

from __future__ import annotations

import io

import colour
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from metamerism.core import PROJECT_SHAPE
from metamerism.illuminants import get_illuminant
from metamerism.pigments import load_golden_heavy_body_paints
from metamerism.pigments.golden import GoldenPaintSpectrum
from metamerism.search.generate import CandidateGenerationConfig, MetamerCandidate
from metamerism.search.metamer import ViewingCondition, score_condition
from metamerism.search.objectives import CandidateScore, RankingConfig
from metamerism.search.pool import SearchConfig, SearchResult, run_search
from metamerism.search.realisability import (
    PaintRealisation,
    SparseRealisabilityConfig,
    find_sparse_metameric_mixture,
)

# ── constants ─────────────────────────────────────────────────────────────────

ILLUMINANT_OPTIONS = ["D65", "D50", "A", "FL2", "FL7", "FL11"]

_DEFAULT_CONCEAL = "D65"
_DEFAULT_REVEAL = ["A"]

_CSS = """
<style>
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; }
</style>
"""

# ── data loading ───────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _paint_lookup() -> dict:
    return {p.paint_name: p for p in load_golden_heavy_body_paints()}


@st.cache_resource(show_spinner=False)
def _paint_library() -> list[GoldenPaintSpectrum]:
    return load_golden_heavy_body_paints()


# ── colour helpers ─────────────────────────────────────────────────────────────


def _xyz_to_hex(xyz: np.ndarray) -> str:
    """Convert XYZ (Y=100 scale) to a CSS hex colour for display."""
    rgb = colour.XYZ_to_sRGB(np.clip(xyz / 100.0, 0.0, None))
    r, g, b = (round(float(np.clip(c, 0.0, 1.0)) * 255) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _swatch_html(hex_colour: str, label: str, size: int = 56) -> str:
    return (
        f'<div style="display:inline-block;text-align:center;margin:0 8px 4px 0">'
        f'<div style="width:{size}px;height:{size}px;background:{hex_colour};'
        f'border-radius:6px;border:1px solid #cbd5e1;margin-bottom:4px"></div>'
        f'<div style="font-size:0.72rem;color:#64748b">{label}</div>'
        f"</div>"
    )


# ── plotting ───────────────────────────────────────────────────────────────────


def _reflectance_figure(
    ref_vals: np.ndarray,
    cand_vals: np.ndarray,
    wavelengths: np.ndarray,
    title: str,
) -> bytes:
    fig, (ax_r, ax_d) = plt.subplots(
        2, 1, figsize=(8, 4.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in (ax_r, ax_d):
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#e2e8f0", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")

    ax_r.plot(
        wavelengths, ref_vals, color="#2563eb", lw=1.5, label="Reference", zorder=3
    )
    ax_r.plot(
        wavelengths, cand_vals, color="#d97706", lw=1.5, label="Candidate", zorder=3
    )
    ax_r.set_ylim(-0.02, 1.05)
    ax_r.set_ylabel("Reflectance", fontsize=9, color="#475569")
    ax_r.legend(fontsize=8, framealpha=0.9)
    ax_r.set_title(title, fontsize=10, color="#1e293b", pad=6)

    diff = cand_vals - ref_vals
    ax_d.fill_between(
        wavelengths, diff, 0, where=(diff >= 0), color="#16a34a", alpha=0.5
    )
    ax_d.fill_between(
        wavelengths, diff, 0, where=(diff < 0), color="#dc2626", alpha=0.5
    )
    ax_d.axhline(0, color="#94a3b8", lw=0.8)
    ax_d.set_ylabel("Difference", fontsize=9, color="#475569")
    ax_d.set_xlabel("Wavelength (nm)", fontsize=9, color="#475569")

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _realisation_figure(
    ref_vals: np.ndarray,
    realised_vals: np.ndarray,
    wavelengths: np.ndarray,
    ideal_vals: np.ndarray | None,
    title: str,
) -> bytes:
    fig, (ax_r, ax_d) = plt.subplots(
        2, 1, figsize=(8, 4.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in (ax_r, ax_d):
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#e2e8f0", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")

    ax_r.plot(
        wavelengths, ref_vals, color="#2563eb", lw=1.5, label="Reference", zorder=3
    )
    ax_r.plot(
        wavelengths, realised_vals, color="#d97706", lw=1.5,
        label="Paint mixture", zorder=3,
    )
    if ideal_vals is not None:
        ax_r.plot(
            wavelengths, ideal_vals, color="#16a34a", lw=1.0,
            ls="--", label="Ideal candidate", zorder=2,
        )
    ax_r.set_ylim(-0.02, 1.05)
    ax_r.set_ylabel("Reflectance", fontsize=9, color="#475569")
    ax_r.legend(fontsize=8, framealpha=0.9)
    ax_r.set_title(title, fontsize=10, color="#1e293b", pad=6)

    diff_real = realised_vals - ref_vals
    ax_d.fill_between(
        wavelengths, diff_real, 0,
        where=(diff_real >= 0), color="#d97706", alpha=0.4,
    )
    ax_d.fill_between(
        wavelengths, diff_real, 0,
        where=(diff_real < 0), color="#d97706", alpha=0.4,
    )
    ax_d.plot(wavelengths, diff_real, color="#d97706", lw=0.8)
    if ideal_vals is not None:
        diff_ideal = ideal_vals - ref_vals
        ax_d.plot(
            wavelengths, diff_ideal, color="#16a34a", lw=0.8, ls="--", alpha=0.7
        )
    ax_d.axhline(0, color="#94a3b8", lw=0.8)
    ax_d.set_ylabel("Difference", fontsize=9, color="#475569")
    ax_d.set_xlabel("Wavelength (nm)", fontsize=9, color="#475569")

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── sidebar ────────────────────────────────────────────────────────────────────


def _section(label: str) -> None:
    st.sidebar.markdown(
        f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:#64748b;margin:1rem 0 0.25rem">'
        f"{label}</p>",
        unsafe_allow_html=True,
    )


def _sidebar() -> dict | None:
    """Render the sidebar and return the search parameters dict, or None."""
    with st.sidebar:
        _section("Reference paint")
        lookup = _paint_lookup()
        paint_names = sorted(lookup.keys())
        default_idx = (
            paint_names.index("Diarylide Yellow")
            if "Diarylide Yellow" in paint_names
            else 0
        )
        paint_name = st.selectbox(
            "Paint",
            paint_names,
            index=default_idx,
            help=(
                "The fixed member of the metameric pair. "
                "The search finds partner spectra that match this paint "
                "under the conceal illuminant but differ under the reveal illuminants."
            ),
        )

        _section("Illuminants")
        conceal_name = st.selectbox(
            "Conceal illuminant",
            ILLUMINANT_OPTIONS,
            index=0,
            help="The pair looks identical under this illuminant.",
        )
        reveal_names = st.multiselect(
            "Reveal illuminants",
            ILLUMINANT_OPTIONS,
            default=_DEFAULT_REVEAL,
            help="The pair looks different under these illuminants.",
        )

        _section("Search")
        n_directions = st.slider(
            "Directions to sample",
            20, 400, 100, step=10,
            help="Number of random null-space directions explored. More = larger pool.",
        )
        seed = int(st.number_input(
            "Random seed", min_value=0, max_value=99999, value=42,
            help="Fix the seed for reproducible results.",
        ))
        conceal_limit = st.slider(
            "Conceal dE00 limit",
            0.1, 3.0, 1.0, step=0.1,
            help="Max colour difference under the conceal illuminant (1 JND = 1.0).",
        )
        n_diverse = st.slider(
            "Max candidates shown",
            1, 10, 5,
            help="How many diverse candidates to display from the Pareto front.",
        )

        st.divider()
        run = st.button("Run search", type="primary", use_container_width=True)

        _section("Paint realisation")
        mixing_mode = st.radio(
            "Mixing model",
            ["ks", "reflectance"],
            format_func=lambda m: "Kubelka-Munk (K/S)" if m == "ks" else "Linear",
            help=(
                "K/S arithmetic mixing is the better physical model for "
                "single-substrate drawdown data. Linear is a transparent baseline."
            ),
        )
        max_components = st.slider(
            "Max paint components",
            1, 6, 4,
            help="Greedy forward selection adds paints one at a time up to this limit.",
        )
        has_candidates = (
            "result" in st.session_state
            and bool(st.session_state["result"].selected)
        )
        realise = st.button(
            "Realise top candidate",
            use_container_width=True,
            disabled=not has_candidates,
            help=(
                "Find paint mixture fractions that match the reference under the "
                "conceal illuminant while maximising reveal difference. "
                "Run the ideal search first."
            ),
        )

        if not reveal_names:
            st.warning("Select at least one reveal illuminant.")
            return None

    return {
        "paint_name": paint_name,
        "conceal_name": conceal_name,
        "reveal_names": tuple(reveal_names),
        "n_directions": n_directions,
        "seed": seed,
        "conceal_limit": conceal_limit,
        "n_diverse": n_diverse,
        "run": run,
        "mixing_mode": mixing_mode,
        "max_components": max_components,
        "realise": realise,
    }


# ── search execution ───────────────────────────────────────────────────────────


def _execute_search(params: dict) -> SearchResult:
    lookup = _paint_lookup()
    reference = lookup[params["paint_name"]].reflectance
    conceal = ViewingCondition(
        name=params["conceal_name"],
        illuminant=get_illuminant(params["conceal_name"]),
    )
    reveals = tuple(
        ViewingCondition(name=n, illuminant=get_illuminant(n))
        for n in params["reveal_names"]
    )
    config = SearchConfig(
        conceal_condition=conceal,
        reveal_conditions=reveals,
        generation=CandidateGenerationConfig(
            seed=params["seed"],
            n_directions=params["n_directions"],
            conceal_delta_e00_limit=params["conceal_limit"],
        ),
        ranking=RankingConfig(conceal_limit=params["conceal_limit"]),
        n_diverse=params["n_diverse"],
    )
    return run_search(reference, config)


# ── realisation ───────────────────────────────────────────────────────────────


def _execute_realisation(params: dict, result: SearchResult) -> PaintRealisation:
    lookup = _paint_lookup()
    reference = lookup[params["paint_name"]].reflectance
    conceal = result.config.conceal_condition
    reveals = list(result.config.reveal_conditions)
    ideal_candidate, _ = result.selected[0]
    cfg = SparseRealisabilityConfig(
        conceal_delta_e00_limit=params["conceal_limit"],
        max_components=params["max_components"],
        mixing_mode=params["mixing_mode"],
        seed=params["seed"],
        use_omp_warmstart=True,
    )
    return find_sparse_metameric_mixture(
        reference,
        _paint_library(),
        conceal,
        reveals,
        cfg,
        ideal_candidate=ideal_candidate,
    )


def _display_realisation(real: PaintRealisation, result: SearchResult) -> None:
    conceal_cond = result.config.conceal_condition
    reveal_conds = list(result.config.reveal_conditions)

    # ── feasibility warnings ──────────────────────────────────────────────────

    realised_reveal = sum(r.delta_e00 for r in real.reveal_results)
    try:
        conceal_limit = float(real.model_assumptions["conceal_delta_e00_limit"])
    except (KeyError, ValueError):
        conceal_limit = None

    if conceal_limit is not None and real.conceal_delta_e00 > conceal_limit + 0.01:
        st.warning(
            f"Conceal constraint violated: mixture ΔE₀₀ = "
            f"{real.conceal_delta_e00:.3f} exceeds limit {conceal_limit:.2f}. "
            "The paint library may not contain mixtures that match the reference "
            "under this illuminant at this tolerance. Try raising the limit or "
            "switching to the Kubelka-Munk mixing model.",
            icon="⚠️",
        )
    elif realised_reveal < real.conceal_delta_e00:
        st.warning(
            f"Reveal difference ({realised_reveal:.3f}) is less than conceal "
            f"difference ({real.conceal_delta_e00:.3f}). The mixture is more "
            "distinguishable under the conceal illuminant than under the reveal — "
            "this is not a useful metameric pair. Consider a different reference "
            "paint or adding more reveal illuminants.",
            icon="⚠️",
        )

    # ── metrics ──────────────────────────────────────────────────────────────
    ceiling = real.ideal_reveal_delta_e00
    n_components = int(real.model_assumptions.get("n_components_used", "?"))

    cols = st.columns(4)
    cols[0].metric(
        "Conceal ΔE₀₀", f"{real.conceal_delta_e00:.3f}",
        help="Should be ≤ conceal limit for a valid metameric pair.",
    )
    cols[1].metric(
        "Reveal ΔE₀₀ (sum)", f"{realised_reveal:.2f}",
        help="Sum of reveal differences across all reveal illuminants.",
    )
    if ceiling is not None:
        gap_pct = realised_reveal / ceiling * 100 if ceiling > 0 else 0.0
        cols[2].metric(
            "Ideal ceiling", f"{ceiling:.2f}",
            help="Ideal candidate's reveal ΔE₀₀ — theoretical upper bound.",
        )
        cols[3].metric(
            "Achieved", f"{gap_pct:.0f}%",
            help="Fraction of the ideal ceiling the paint mixture achieves.",
        )
    else:
        cols[2].metric("Components", n_components,
                       help="Number of paints in the mixture.")

    # ── paint fractions ───────────────────────────────────────────────────────

    st.markdown("**Paint mixture**")
    active = {k: v for k, v in real.fractions.items() if v > 0.002}
    sorted_fracs = sorted(active.items(), key=lambda x: x[1], reverse=True)
    for name, frac in sorted_fracs:
        st.markdown(
            f'<div style="margin-bottom:6px">'
            f'<span style="font-size:0.85rem;color:#1e293b">{name}</span>'
            f'<span style="float:right;font-size:0.85rem;font-weight:600;'
            f'color:#475569">{frac:.1%}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(float(frac))

    # ── reflectance plot ──────────────────────────────────────────────────────

    wavelengths = np.asarray(
        real.reference.sd.copy().align(PROJECT_SHAPE).wavelengths, dtype=float
    )
    ref_vals = np.asarray(
        real.reference.sd.copy().align(PROJECT_SHAPE).values, dtype=float
    )
    realised_vals = np.asarray(
        real.realised.sd.copy().align(PROJECT_SHAPE).values, dtype=float
    )
    ideal_vals: np.ndarray | None = None
    if real.ideal_candidate is not None:
        ideal_vals = np.asarray(
            real.ideal_candidate.sd.copy().align(PROJECT_SHAPE).values, dtype=float
        )

    conceal_de00 = real.conceal_delta_e00
    min_reveal = real.minimum_reveal_delta_e00
    model = real.model_assumptions.get("model", real.model_name)
    title = (
        f"Conceal ΔE₀₀ = {conceal_de00:.3f}  |  "
        f"Min reveal ΔE₀₀ = {min_reveal:.3f}  |  {model}"
    )
    png = _realisation_figure(ref_vals, realised_vals, wavelengths, ideal_vals, title)
    st.image(png, use_container_width=True)

    # ── colour swatches ───────────────────────────────────────────────────────

    st.markdown(
        "**Colour appearance** (reference → paint mixture under each illuminant)"
    )
    all_conditions = [conceal_cond, *reveal_conds]
    swatch_html = ""
    for cond in all_conditions:
        cr = score_condition(real.reference, real.realised, cond)
        hex_ref = _xyz_to_hex(cr.xyz_first)
        hex_mix = _xyz_to_hex(cr.xyz_second)
        role = "conceal" if cond is conceal_cond else "reveal"
        swatch_html += (
            f'<div style="display:inline-block;margin-right:20px;margin-bottom:8px">'
            f'<div style="font-size:0.78rem;font-weight:600;color:#475569;'
            f'margin-bottom:4px">{cond.name} ({role})</div>'
            f"{_swatch_html(hex_ref, 'ref')}{_swatch_html(hex_mix, 'mix')}"
            f'<div style="font-size:0.76rem;color:#64748b;margin-top:2px">'
            f"ΔE₀₀ = {cr.delta_e00:.3f}</div>"
            f"</div>"
        )
    st.markdown(swatch_html, unsafe_allow_html=True)

    # ── model assumptions ─────────────────────────────────────────────────────

    with st.expander("Model assumptions"):
        for k, v in real.model_assumptions.items():
            st.markdown(f"**{k}**: {v}")


# ── result display ─────────────────────────────────────────────────────────────


def _display_result(result: SearchResult, params: dict) -> None:
    diag = result.pool_diagnostics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generated", diag.n_total)
    c2.metric("Passing conceal", diag.n_passing_conceal)
    c3.metric("Pareto front", len(result.pareto_scored))
    c4.metric("Selected", len(result.selected))

    if not result.selected:
        st.info(
            "No candidates passed the conceal threshold. "
            "Try raising the ΔE₀₀ limit or increasing directions."
        )
        return

    # Aligned reference for display (project grid 380-780 nm)
    ref_spectrum = result.selected[0][0].source_reference
    ref_aligned_sd = ref_spectrum.sd.copy().align(PROJECT_SHAPE)
    ref_vals = np.asarray(ref_aligned_sd.values, dtype=float)
    wavelengths = np.asarray(ref_aligned_sd.wavelengths, dtype=float)

    conceal_cond = result.config.conceal_condition
    reveal_conds = result.config.reveal_conditions

    tab_labels = [f"Candidate {i + 1}" for i in range(len(result.selected))]
    tabs = st.tabs(tab_labels)

    for tab, (candidate, score) in zip(tabs, result.selected, strict=True):
        with tab:
            _display_candidate(
                candidate, score, ref_vals, wavelengths,
                conceal_cond, reveal_conds, ref_spectrum,
            )


def _display_candidate(
    candidate: MetamerCandidate,
    score: CandidateScore,
    ref_vals: np.ndarray,
    wavelengths: np.ndarray,
    conceal_cond: ViewingCondition,
    reveal_conds: tuple[ViewingCondition, ...],
    ref_spectrum,
) -> None:
    cand_vals = np.asarray(
        candidate.spectrum.sd.copy().align(PROJECT_SHAPE).values, dtype=float
    )

    title = (
        f"Conceal ΔE₀₀ = {score.conceal_delta_e00:.3f}  |  "
        f"Min reveal ΔE₀₀ = {score.minimum_reveal_delta_e00:.3f}"
    )
    png = _reflectance_figure(ref_vals, cand_vals, wavelengths, title)
    st.image(png, use_container_width=True)

    # Score the pair under each condition to get XYZ for swatches
    all_conditions = [conceal_cond, *list(reveal_conds)]
    all_results = [
        score_condition(ref_spectrum, candidate.spectrum, cond)
        for cond in all_conditions
    ]

    st.markdown("**Colour appearance** (reference → candidate under each illuminant)")
    swatch_html = ""
    for cond, cr in zip(all_conditions, all_results, strict=True):
        hex_ref = _xyz_to_hex(cr.xyz_first)
        hex_cand = _xyz_to_hex(cr.xyz_second)
        role = "conceal" if cond is conceal_cond else "reveal"
        swatch_html += (
            f'<div style="display:inline-block;margin-right:20px;margin-bottom:8px">'
            f'<div style="font-size:0.78rem;font-weight:600;color:#475569;'
            f'margin-bottom:4px">{cond.name} ({role})</div>'
            f"{_swatch_html(hex_ref, 'ref')}{_swatch_html(hex_cand, 'cand')}"
            f'<div style="font-size:0.76rem;color:#64748b;margin-top:2px">'
            f"ΔE₀₀ = {cr.delta_e00:.3f}</div>"
            f"</div>"
        )
    st.markdown(swatch_html, unsafe_allow_html=True)

    # Diagnostics row
    d1, d2, d3 = st.columns(3)
    d1.metric("Roughness", f"{score.roughness:.4g}")
    d2.metric("Boundary saturation", f"{score.boundary_saturation:.3f}")
    d3.metric("Spectral distance", f"{score.spectral_dist:.4f}")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Metamer search",
        page_icon=":artist_palette:",
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("Metamer search")
    st.caption(
        "Choose a reference paint and two illuminants. "
        "The search finds partner reflectance spectra that look identical to the "
        "reference under the **conceal** illuminant, but visibly different under "
        "the **reveal** illuminants — metameric pairs."
    )

    params = _sidebar()
    if params is None:
        return

    if params["run"]:
        st.session_state.pop("realisation", None)
        with st.spinner("Running metamer search…"):
            result = _execute_search(params)
        st.session_state["result"] = result
        st.session_state["result_params"] = params
        st.rerun()  # re-render sidebar so Realise button becomes enabled

    if params["realise"] and "result" in st.session_state:
        search_result: SearchResult = st.session_state["result"]
        if search_result.selected:
            with st.spinner("Finding paint mixture…"):
                real = _execute_realisation(params, search_result)
            st.session_state["realisation"] = real

    if "result" in st.session_state:
        result: SearchResult = st.session_state["result"]
        rp = st.session_state["result_params"]
        st.subheader(
            f"Ideal candidates — {rp['paint_name']} · conceal {rp['conceal_name']}"
            f" · reveal {', '.join(rp['reveal_names'])}"
        )
        _display_result(result, rp)
    else:
        st.info(
            "Configure the search in the sidebar and click **Run search** to begin."
        )

    if "realisation" in st.session_state and "result" in st.session_state:
        st.subheader("Paint realisation")
        st.caption(
            "Best sparse paint mixture found by greedy forward selection. "
            "Reflectance plot shows reference (blue), mixture (orange), "
            "and ideal candidate (green dashed) for comparison."
        )
        _display_realisation(
            st.session_state["realisation"],
            st.session_state["result"],
        )


if __name__ == "__main__":
    main()
