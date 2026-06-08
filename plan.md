# Project Plan: Metameric Paint–Light Exploration Tool

## 1. Project Aim

The aim of this project is to develop a software-assisted experimental system for discovering, testing, and exploiting metameric relationships between artist pigments under controlled illumination.

The practical artistic goal is to identify paint mixtures that appear visually similar under one lighting condition but diverge strongly under another. These results will be used in physical paintings and installations where programmable illumination forms part of the work.

The project combines:

- artist paint reflectance spectra;
- mixture modelling;
- programmable RGB/RGBW LED illumination;
- colour science and human cone response modelling;
- numerical search and optimisation;
- physical swatch testing;
- eventual empirical calibration using a spectrophotometer.

The software is not intended to replace studio experimentation. Its purpose is to guide experimentation by narrowing the search space and identifying promising paint–light candidates.

The software is also intended to be a high-quality open-source contribution: well-tested, well-documented, and structured for extension by others working at the intersection of colour science and generative/programmable art practice.

---

## 2. Core Concept

A painted surface is not a colour in itself. It is a spectral reflector. Its perceived colour depends on:

```text
illumination spectrum × paint reflectance spectrum × human visual response
```

Two physically different reflectance spectra may produce the same perceived colour under one illuminant but different perceived colours under another. This is metamerism.

The project extends this idea by treating the illuminant as programmable. Instead of searching only for paint mixtures that are metameric under fixed daylight or gallery lighting, the tool searches for complete paint–light systems:

```text
paint mixture A
paint mixture B
lighting state 1: conceal
lighting state 2: reveal
```

The ideal result is:

```text
A ≈ B under lighting state 1
A ≠ B under lighting state 2
```

This makes controlled illumination an active compositional material.

---

## 3. Research Questions

The project is guided by the following questions:

1. Can public artist-paint spectral data be used to predict useful metameric paint candidates?
2. Can approximate pigment-mixture models generate sufficiently good candidates for physical testing?
3. How much does controlled RGB or RGBW illumination amplify metameric effects compared with broad-spectrum illumination?
4. Can the software search simultaneously over paint mixtures and LED illumination states?
5. At what stage do measured reflectance spectra from actual painted swatches become necessary?
6. Can the final tool become a practical studio instrument for designing paintings that change under programmable light?

---

## 4. System Overview

The software consists of several separable modules arranged in a strict dependency hierarchy. The hierarchy is enforced by the package structure: lower layers have no knowledge of higher ones.

```text
core/           ← no internal dependencies; foundation for everything
colorimetry/    ← depends on core only
illuminants/    ← depends on core only
pigments/       ← depends on core only
mixing/         ← depends on core, pigments
search/         ← depends on all of the above
io/             ← depends on core, search (candidates)
```

The most important architectural principle is that all colour calculations are spectral-first. RGB values are outputs for preview only, never the internal representation.

Visualisation and user interface code lives outside the importable library (in `notebooks/` or a separate `app/` directory) and is not a package module. This keeps the scientific core dependency-free from display libraries.

---

## 5. Canonical Wavelength Grid

All spectral data in the package is represented on a single canonical internal grid:

```text
Range:   380–780 nm
Spacing: 5 nm
Points:  81
```

This grid is chosen to match CIE standard observer tables exactly. It is defined in `core/spectrum.py` as `CANONICAL_GRID`.

All data from other grids (e.g. GOLDEN 400–700 nm at 10 nm) is resampled onto `CANONICAL_GRID` on ingestion. Resampling is always explicit, never silent. The interpolation method and any extrapolation applied are recorded in the `Provenance` attached to the resulting `Spectrum`.

Default interpolation methods by domain:

```text
REFLECTANCE:    PCHIP  (monotone, avoids overshoot)
TRANSMITTANCE:  PCHIP
ILLUMINANT:     linear (conservative for power distributions)
CMF:            linear
EMISSION:       linear
```

Default extrapolation policies by domain:

```text
REFLECTANCE:    CLAMP  (repeat boundary value)
TRANSMITTANCE:  CLAMP
ILLUMINANT:     ZERO   (no emission outside measured range)
CMF:            ZERO
EMISSION:       ZERO
```

---

## 6. Module 1 — Core Layer (`core/`)

### Purpose

Provide the foundational types used by all other modules. This layer has no dependencies on anything else in the package.

### `spectrum.py` — The `Spectrum` type

The `Spectrum` is the central value type of the entire package. It is an immutable frozen dataclass.

```python
@dataclass(frozen=True)
class Spectrum:
    wavelengths: NDArray[np.float64]   # monotonically increasing, nm
    values:      NDArray[np.float64]   # sampled spectral values
    domain:      SpectrumDomain        # physical meaning of values
    provenance:  Provenance            # origin and processing history
```

**Domains** (`SpectrumDomain` enum):

```text
REFLECTANCE    — values in [0, 1]
ILLUMINANT     — values >= 0, arbitrary scale
CMF            — colour matching function values
EMISSION       — emitted power, values >= 0
TRANSMITTANCE  — values in [0, 1]
UNKNOWN        — no range constraint
```

**Construction** is via `Spectrum.from_arrays(wavelengths, values, domain=..., provenance=...)`, which validates inputs and domain constraints. Pass `validate=False` to skip range checks when constructing intermediate results.

**Operations** return new `Spectrum` objects; no method mutates the receiver:

```text
resample(target_grid)    — interpolate onto a new wavelength grid
to_canonical()           — convenience: resample onto CANONICAL_GRID
normalise(to=1.0)        — scale so peak value equals `to`
clip(low, high)          — clamp values to [low, high]
integrate()              — trapezoidal integration: ∫ values dλ
dot(other)               — inner product: ∫ self * other dλ
* / + -                  — pointwise arithmetic (grids must match)
allclose(other)          — numerical equality with tolerance
```

All arithmetic operators raise `GridMismatchError` if grids differ. Call `resample()` or `to_canonical()` first.

**`Provenance`** records origin and processing history:

```python
@dataclass(frozen=True)
class Provenance:
    source:               str         # human-readable origin description
    instrument:           str | None  # e.g. "X-Rite i1Pro 3"
    measurement_geometry: str | None  # e.g. "45°/0°"
    illuminant_used:      str | None  # e.g. "D65"
    substrate:            str | None
    medium:               str | None
    film_thickness_um:    float | None
    confidence:           float       # 0–1; 1.0 = directly measured
    notes:                str | None  # resampling history, warnings
```

Confidence convention:

```text
1.0  — directly measured under controlled conditions
0.7  — manufacturer or reference data, assumed reliable
0.4  — approximated (Gaussian LED model, extrapolated values)
0.1  — rough guess or placeholder
```

### `observer.py` — The `Observer` type

The observer (colour matching functions) is an explicit parameter in all colorimetric calculations. It is never a hidden global.

```python
@dataclass(frozen=True)
class Observer:
    name:                 str                  # e.g. "CIE1931_2deg"
    cmf:                  NDArray[np.float64]  # shape (81, 3) on CANONICAL_GRID
    integration_weights:  NDArray[np.float64]  # shape (81,), nm spacing
```

Standard observers to be provided:

```text
CIE 1931 2-degree   — default; matches most published ΔE formulae
CIE 2006 2-degree   — more physiologically accurate
CIE 2006 10-degree  — appropriate for paintings viewed at normal distance
```

The choice of observer is a genuine scientific decision with artistic implications. The 10° observer is more relevant for paintings viewed from typical gallery distances.

### `exceptions.py` — Package exceptions

All package exceptions are defined in one place and imported from here:

```text
SpectrumError        — base class
GridMismatchError    — operation requires matching grids
ExtrapolationError   — extrapolation required under RAISE policy
DomainError          — domain-specific constraint violated
```

---

## 7. Module 2 — Colorimetry (`colorimetry/`)

### Purpose

Compute perceived colour from spectral data. Depends on `core/` only.

### `tristimulus.py`

```text
xyz(reflectance, illuminant, observer) → XYZ
```

Computes tristimulus values by integrating `I(λ) · R(λ) · CMF(λ)` over the canonical grid using the trapezoidal rule.

The observer is always passed explicitly. The canonical integration is:

```text
X = ∫ I(λ) R(λ) x̄(λ) dλ
Y = ∫ I(λ) R(λ) ȳ(λ) dλ
Z = ∫ I(λ) R(λ) z̄(λ) dλ
```

### `spaces.py`

Colour space conversions:

```text
xyz_to_lab(XYZ, whitepoint)  → Lab
lab_to_lch(Lab)              → LCh
xyz_to_oklab(XYZ)            → OKLab
xyz_to_srgb(XYZ)             → sRGB  (preview only)
```

### `difference.py`

Colour difference metrics:

```text
delta_e_76(Lab_a, Lab_b)     → float
delta_e_00(Lab_a, Lab_b)     → float
spectral_distance(r_a, r_b)  → float  (RMS spectral difference)
```

### Note on chromatic adaptation

Under RGBW illumination that differs significantly from D65, von Kries or CIECAM02 adaptation should be applied before computing ΔE, or the visual system's adaptation to the illuminant will be ignored and colour differences will be overestimated. This is particularly important for the conceal condition, where a coloured illuminant is partially adapted out.

Chromatic adaptation is a known limitation in v1. The normalisation policy (per-illuminant normalisation simulates a fully adapted observer; preserving absolute luminance does not) must be documented as an explicit parameter with a named default, not an implicit assumption.

### Dependency

Use the `colour-science` library for standard colorimetric conversions (ΔE00, XYZ→Lab, D65/D50 whitepoints, standard observers) rather than implementing these from scratch. This reduces the risk of numerical errors in well-specified calculations. The novel contributions of this project are the mixture models and search logic, not the standard colorimetry.

---

## 8. Module 3 — Illuminants (`illuminants/`)

### Purpose

Represent programmable and standard illuminants as spectral power distributions. Depends on `core/` only.

### `standard.py`

Standard reference illuminants loaded from `data/illuminants/`:

```text
D65   — CIE standard daylight (6500 K)
D50   — CIE standard daylight (5000 K)
A     — incandescent
```

### `led.py`

LED channel models. A general multi-channel illuminant:

```text
I(λ) = c₁E₁(λ) + c₂E₂(λ) + ... + cₙEₙ(λ)
```

For RGBW:

```text
I(λ) = r·R(λ) + g·G(λ) + b·B(λ) + w·W(λ)
```

Initial Gaussian approximations for WS2812B / NeoPixel:

```text
Red:   centre ~625 nm, FWHM ~20 nm
Green: centre ~525 nm, FWHM ~30 nm
Blue:  centre ~467 nm, FWHM ~20 nm
```

These are placeholders only. The architecture should be developed against them before measured LED spectra are available. Confidence is set to 0.4 for Gaussian models.

### `measured.py`

Loader for measured LED spectra (from spectroradiometer CSV export).

### Preferred early hardware

SK6812 RGBW is the recommended early upgrade from standard NeoPixel RGB. The additional white channel provides a phosphor component spanning the visible range alongside the narrow RGB channels, significantly expanding the accessible null space for the conceal search.

---

## 9. Module 4 — Pigments (`pigments/`)

### Purpose

Store and query spectral reflectance data for artist pigments. Depends on `core/` only.

### `database.py`

```python
@dataclass(frozen=True)
class PigmentEntry:
    brand:           str          # e.g. "Golden"
    range_:          str          # e.g. "Heavy Body"
    name:            str          # e.g. "Cadmium Red Medium Hue"
    pigment_codes:   list[str]    # e.g. ["PR108"]
    reflectance:     Spectrum     # on CANONICAL_GRID
    opacity:         str | None   # "opaque", "semi-transparent", "transparent"
    colour_index:    str | None
    entry_type:      PigmentEntryType
    provenance:      Provenance
```

`PigmentEntryType` enum:

```text
SINGLE_PIGMENT     — single-pigment paint
MULTI_PIGMENT      — commercial multi-pigment formulation
MEASURED_SWATCH    — measured physical swatch
COMPUTED_MIXTURE   — result of mixture model
```

This distinction is important because artist paint names can conceal complex pigment formulations. Single-pigment paints are preferred for mixture modelling.

### `loaders.py`

CSV ingestion for GOLDEN Heavy Body spectral data. Resamples all loaded spectra onto `CANONICAL_GRID` and records resampling parameters in `Provenance`.

---

## 10. Module 5 — Mixture Models (`mixing/`)

### Purpose

Predict the reflectance spectrum of a paint mixture from component spectra. Depends on `core/` and `pigments/`.

### Protocol

All mixture models implement a common protocol:

```python
class MixtureModel(Protocol):
    name:             str
    accuracy_regime:  str   # e.g. "opaque films", "transparent washes"

    def predict(
        self,
        components: list[tuple[Spectrum, float]],  # (pigment_reflectance, weight)
    ) -> MixturePrediction: ...

@dataclass(frozen=True)
class MixturePrediction:
    reflectance:          Spectrum
    confidence:           float           # 0–1
    uncertainty_spectrum: NDArray | None  # per-wavelength std dev
    model_name:           str
    warnings:             list[str]
```

The `confidence` and `uncertainty_spectrum` fields are not optional decoration — they propagate into search scoring and candidate ranking.

### Implementations

**`linear.py`** — linear reflectance mixing:

```text
R_mix(λ) = Σ wᵢ Rᵢ(λ)
```

Physically weak; useful as a baseline. High confidence only at very low pigment concentrations.

**`log_reflectance.py`** — log-reflectance mixing:

```text
log R_mix(λ) = Σ wᵢ log Rᵢ(λ)
```

Better approximates subtractive behaviour. Absorption compounds multiplicatively.

**`kubelka_munk.py`** — Kubelka–Munk mixing in K/S space:

```text
K/S = (1 − R)² / (2R)
K/S_mix = Σ wᵢ (K/S)ᵢ
```

Valid only for opaque films at full coverage. The two-flux model is required for transparent or layered applications. The single-constant model (implemented first) must carry a warning for transparent pigments.

### Important caveat

All mixture models are candidate generators, not exact predictors. Real paint mixing is affected by pigment concentration, medium, film thickness, substrate, opacity, scattering, gloss, application method, and drying behaviour. The pluggable design allows the model to be replaced or improved without touching search logic.

---

## 11. Module 6 — Metameric Search (`search/`)

### Purpose

Find pairs of paint mixtures metameric under one illuminant and divergent under another, together with optimal illumination states for both conditions. Depends on all layers above.

### Two-level search architecture

The search operates at two levels with very different computational costs.

**Level 1 — Paint search (expensive, combinatorial)**

Enumerate candidate paint pairs. For each pair, invoke the Level 2 search to find the optimal lighting states. Score the result.

**Level 2 — Light search (cheap, linear algebra)**

For a fixed paint pair A and B, the response difference under LED state `u` is:

```text
c_A − c_B = (M_A − M_B) u
```

where `M_A`, `M_B` are matrices mapping LED channel values to cone responses.

The conceal light solves:

```text
(M_A − M_B) u ≈ 0   →   null space of (M_A − M_B)
```

The reveal light maximises:

```text
‖(M_A − M_B) u‖   →   leading eigenvector of (M_A − M_B)ᵀ(M_A − M_B)
```

Both are closed-form solutions. Level 2 is the inner loop; it is essentially free once `M_A` and `M_B` are computed. The scoring function for LED states is therefore exact (for fixed paints), not approximate.

### `paint_search.py`

Combinatorial enumeration of paint pairs. Initial strategies:

```text
grid search over two-pigment mixtures (main early approach)
random search over three-pigment mixtures
Bayesian optimisation (later)
```

### `light_search.py`

Linear algebra LED search as described above. Applies brightness and channel constraint projections after finding the null space / leading eigenvector.

### `scoring.py`

Composite score for a candidate:

```text
J = ΔE_reveal / (ΔE_conceal + ε)
```

With penalties for:

```text
too dark or too low chroma under conceal light
too dark or too low chroma under reveal light
too many pigments in recipe
tiny or unstable mixture ratios
low mixture model confidence
out-of-gamut sRGB preview
unpleasant or impractical illumination state
```

Lexicographic ranking: candidates with `ΔE_conceal > threshold` are rejected before ranking by `J`.

### `candidates.py`

`CandidatePair` dataclass and JSON serialisation. All candidates are saved with full provenance.

---

## 12. Module 7 — Visualisation and Simulation

### Purpose

Provide visual feedback for candidate pairs and lighting states.

### Scope

Visualisation code lives **outside the importable library** — in `notebooks/` or a separate `app/` directory. This avoids dragging display dependencies (matplotlib, streamlit, etc.) into the scientific core.

### Core views

```text
paint A and B under conceal and reveal light (colour swatches)
reflectance curves of A and B
illumination spectra for light 1 and light 2
resulting reflected spectra
Lab / LCh / sRGB values
ΔE values
mixture recipes
confidence warnings
```

### Important limitation

The screen cannot display the true spectral phenomenon. It can only show predicted colour appearance under each modelled illuminant. The display is a decision aid, not the artwork.

---

## 13. Module 8 — Empirical Measurement and Calibration

### Purpose

Close the loop between predicted and actual behaviour.

### Measurements needed

```text
actual paint swatches
actual paint mixtures
actual LED channels
actual mixed lighting states
possibly substrates and varnishes
```

### Calibration loop

```text
predict mixture
paint swatch
measure reflectance
compare measured vs predicted
fit correction parameters
update mixture model
rerun search
paint next candidates
```

### When spectrophotometer data becomes critical

A spectrophotometer is not critical for the first software prototype. It becomes critical when the project moves from candidate discovery to reliable physical production. The key failure point is mixture prediction.

---

## 14. Module 9 — I/O (`io/`)

### Purpose

Import and export data at the boundaries of the library. Depends on `core/` and `search/` (for candidate types).

### `export.py`

```text
candidate report (PDF or HTML)
recipe sheet (CSV)
reflectance curve export
illumination state export
```

### `spectrophotometer.py`

Ingest measured swatch spectra from spectrophotometer export formats. Resamples onto `CANONICAL_GRID` and attaches instrument provenance.

---

## 15. Repository Structure

```text
metamerism/
│
├── pyproject.toml
├── README.md
├── .python-version
├── .gitignore
│
├── src/
│   └── metamerism/
│       ├── __init__.py
│       │
│       ├── core/                        # no internal dependencies
│       │   ├── __init__.py
│       │   ├── spectrum.py              # Spectrum, Provenance, CANONICAL_GRID ✓
│       │   ├── observer.py              # Observer, CMF loading
│       │   └── exceptions.py            # all package exceptions
│       │
│       ├── colorimetry/                 # depends on core only
│       │   ├── __init__.py
│       │   ├── tristimulus.py
│       │   ├── spaces.py
│       │   └── difference.py
│       │
│       ├── illuminants/                 # depends on core only
│       │   ├── __init__.py
│       │   ├── standard.py
│       │   ├── led.py
│       │   └── measured.py
│       │
│       ├── pigments/                    # depends on core only
│       │   ├── __init__.py
│       │   ├── database.py
│       │   └── loaders.py
│       │
│       ├── mixing/                      # depends on core, pigments
│       │   ├── __init__.py
│       │   ├── base.py                  # MixtureModel protocol, MixturePrediction
│       │   ├── linear.py
│       │   ├── log_reflectance.py
│       │   └── kubelka_munk.py
│       │
│       ├── search/                      # depends on everything above
│       │   ├── __init__.py
│       │   ├── candidates.py
│       │   ├── paint_search.py
│       │   ├── light_search.py
│       │   └── scoring.py
│       │
│       └── io/                          # depends on core, search
│           ├── __init__.py
│           ├── export.py
│           └── spectrophotometer.py
│
├── data/
│   ├── observers/                       # CIE 1931, CIE 2006 CMF tables (CSV)
│   ├── illuminants/                     # D65, D50, A spectral data (CSV)
│   ├── pigments/                        # Golden et al. reflectance data (CSV)
│   └── examples/                        # example candidate JSON files
│
├── tests/
│   ├── core/
│   │   └── test_spectrum.py             # ✓ 68 tests passing
│   ├── colorimetry/
│   ├── illuminants/
│   ├── pigments/
│   ├── mixing/
│   ├── search/
│   └── io/
│
├── notebooks/                           # exploratory Jupyter work
│
└── docs/
```

`✓` marks files that have been implemented.

---

## 16. Technology Choices

### Language and runtime

```text
Python 3.12+
```

### Core dependencies

```text
numpy >= 2.0          — spectral arrays, linear algebra
scipy >= 1.14         — interpolation (PCHIP), optimisation
colour-science        — standard colorimetry (ΔE00, observers, whitepoints)
pandas                — tabular data, pigment database queries
matplotlib            — plots (notebooks and app only, not core library)
```

### Development tools

```text
uv                    — package and environment management
pytest >= 8.0         — test runner
ruff                  — linting and formatting
ty                    — type checking
```

### Build

```text
hatchling             — build backend
src layout            — src/metamerism/
```

### JSON serialisation (candidates)

Pydantic v2 is suitable for the candidate data model and spectrophotometer ingest. `Provenance` maps directly to a Pydantic model. `Spectrum` itself should not be a Pydantic model; use custom serialisation for the numpy array fields. All candidate JSON includes a `schema_version` field from the first version.

---

## 17. Testing Strategy

Colour science code is susceptible to subtle numerical bugs that produce plausible-looking wrong answers. The test suite is structured around known ground truths.

### Test categories

**Round-trip tests** (core correctness):

```text
resample to canonical and back; verify values within tolerance
XYZ of perfect white diffuser under D65: Y = 100.00 (CIE standard)
```

**Synthetic metamer identity tests**:

```text
construct two artificial reflectance spectra analytically metameric
under a known illuminant; verify ΔE_conceal ≈ 0
```

**Null-space tests**:

```text
construct M_A, M_B for synthetic paint pair
run light_search; verify conceal light produces ΔE ≈ 0
```

**Mixture model regression tests**:

```text
fix known pigment spectra and weights
verify predictions do not drift across refactors
```

**Domain validation tests**:

```text
reflectance and transmittance values in [0, 1]
illuminant and emission values >= 0
```

Writing the synthetic metamer tests before implementing search is the single highest-leverage investment for quality. Tests catch grid mismatch errors, normalisation inconsistencies, and mixture model overconfidence early.

### Current test status

```text
tests/core/test_spectrum.py    — 68 tests, all passing
```

---

## 18. Candidate Data Model

A candidate metameric pair is saved with all necessary metadata. The `schema_version` field is mandatory from v1 to support future migrations.

```json
{
  "schema_version": "1.0",
  "candidate_id": "cand_0001",
  "paint_a": {
    "recipe": [
      {"pigment": "PY35", "amount": 0.7},
      {"pigment": "PR108", "amount": 0.3}
    ],
    "predicted_reflectance_file": "cand_0001_A.csv",
    "mixture_model": "kubelka_munk",
    "mixture_confidence": 0.6
  },
  "paint_b": {
    "recipe": [
      {"pigment": "PO20", "amount": 1.0}
    ],
    "predicted_reflectance_file": "cand_0001_B.csv",
    "mixture_model": "kubelka_munk",
    "mixture_confidence": 0.7
  },
  "conceal_light": {
    "source": "SK6812_RGBW_approx",
    "channels": {"r": 0.42, "g": 0.61, "b": 0.18, "w": 0.75},
    "search_method": "null_space"
  },
  "reveal_light": {
    "source": "SK6812_RGBW_approx",
    "channels": {"r": 1.0, "g": 0.15, "b": 0.05, "w": 0.0},
    "search_method": "leading_eigenvector"
  },
  "observer": "CIE1931_2deg",
  "chromatic_adaptation": "none_v1",
  "metrics": {
    "delta_e_conceal": 1.2,
    "delta_e_reveal": 14.8,
    "strength_ratio": 12.3
  },
  "status": "predicted_only",
  "notes": "High-priority physical swatch candidate."
}
```

---

## 19. Physical Experiment Program

### Stage 1 — Software-only exploration

Use public pigment spectra and approximate LED spectra.

Goals:

```text
build pipeline
verify calculations
generate early candidate pairs
identify promising pigment families
identify useful illumination states
```

### Stage 2 — Physical swatch tests without spectrophotometer

Paint small swatches from predicted candidate recipes.

Test under:

```text
D65-ish white LED
warm white LED
cool white LED
NeoPixel RGB
SK6812 RGBW
individual R/G/B/W channels
conceal/reveal programmed states
```

Document with:

```text
camera photos
manual notes
lighting settings
paint recipes
subjective visibility
```

### Stage 3 — Measured LED spectra

Measure actual LED channels. Update the illumination model. Rerun search.

### Stage 4 — Measured paint swatches

Measure single paints and mixtures. Compare predicted and measured reflectance. Improve the mixture model.

### Stage 5 — Studio-scale panels

Move from small swatches to painted panels. Investigate scale, viewing distance, gloss, application method, illumination uniformity.

### Stage 6 — Finished painting/installation studies

Possible formats:

```text
static conceal/reveal painting
slowly changing illumination painting
hidden structure revealed by spectral choreography
viewer-triggered lighting transitions
apparently monochrome painting that differentiates under LED states
```

---

## 20. Hardware Plan

### Immediate experimental hardware

```text
SK6812 RGBW LED strip
ESP32 controller
5 V power supply with adequate current
diffuser material
small test enclosure or light box
paint swatch cards
artist acrylic paints
neutral substrate/ground
camera for documentation
```

### Later measurement hardware

Required:

```text
spectrophotometer for reflectance curves
```

Desirable:

```text
emission spectrometer or spectroradiometer for LED spectra
```

If one instrument can handle both reflectance and emission sufficiently well, that is ideal. Reflectance measurement is the higher priority for paint modelling; approximate or manufacturer LED data may suffice temporarily.

### Gallery-scale lighting

Possible later options:

```text
RGBW LED bars
theatrical RGBWA or RGBAL fixtures
DMX-controlled multi-channel LED fixtures
custom multi-primary LED array
diffused controlled light box
```

---

## 21. Success Criteria

### Software success

The software is successful if it can:

```text
import pigment reflectance curves
model RGB/RGBW illumination
compute perceptual colour under arbitrary illumination
search for candidate metameric pairs using two-level paint/light search
rank candidates meaningfully
export recipes and lighting states
incorporate measured swatch data later
```

### Experimental success

The experimental program is successful if it can produce physical swatches where:

```text
two painted areas are difficult to distinguish under conceal lighting
the same areas become clearly distinguishable under reveal lighting
the effect is repeatable
the effect survives scale-up to painting format
```

### Artistic success

The artistic outcome is successful if the phenomenon is not merely a colour-science demonstration but becomes part of the conceptual and perceptual structure of the work.

The painting should not simply say "this is metamerism." It should use metamerism as a material condition: a latent image, hidden structure, unstable monochrome, or controlled perceptual event.

---

## 22. Risks and Mitigations

### Risk 1 — Mixture predictions are inaccurate

Mitigation:

```text
treat software as candidate generator
test physical swatches early
add empirical calibration
keep mixture model pluggable via MixtureModel protocol
expose uncertainty_spectrum per prediction
```

### Risk 2 — LED spectra are too approximate

Mitigation:

```text
start with Gaussian approximations (confidence = 0.4)
measure actual LED channels later
allow imported LED spectra via measured.py
prefer RGBW or multi-channel fixtures for wider null space
```

### Risk 3 — Screen simulation is misleading

Mitigation:

```text
show spectral plots and metrics, not just colour swatches
label sRGB previews as approximate
prioritise physical swatch testing
```

### Risk 4 — Effects only work in dark/neutral colours

Mitigation:

```text
add high-key constraints to search scoring
add chroma/value constraints
rank candidates by artistic usefulness
two-level search optimises LED states for each paint pair separately
```

### Risk 5 — Surface effects dominate

Mitigation:

```text
standardise substrate and medium
compare matte/gloss surfaces
measure actual swatches
test at painting scale
control viewing geometry
```

### Risk 6 — Gallery lighting is impractical

Mitigation:

```text
design lighting rig early
test diffuser and brightness requirements
consider DMX / theatrical fixtures for final work
```

### Risk 7 — Chromatic adaptation not modelled

Mitigation:

```text
document as known v1 limitation
add normalisation_policy parameter (per_illuminant / absolute_luminance)
plan CIECAM02 adaptation for v2
```

---

## 23. Implementation Sequence

### Step 1 — Core layer ✓ (in progress)

```text
Spectrum type with Provenance            ✓
CANONICAL_GRID (380–780 nm, 5 nm, 81 pt) ✓
Resampling with interpolation/extrapolation policies ✓
Arithmetic operators with grid mismatch guards ✓
Full test suite (68 tests passing)       ✓
Observer type
```

### Step 2 — Colorimetry

```text
XYZ integration using colour-science CMFs
Lab, LCh, OKLab conversions
ΔE76 and ΔE00
round-trip test: white diffuser under D65 → Y = 100
```

### Step 3 — Illuminants

```text
D65, D50, A from data/
Gaussian LED models (WS2812B, SK6812 RGBW)
first end-to-end calculation: two paints, two lights, ΔE
```

### Step 4 — Pigments

```text
PigmentEntry dataclass
GOLDEN Heavy Body CSV ingestion
pigment database with query interface
```

### Step 5 — Mixture models

```text
MixtureModel protocol and MixturePrediction
linear reflectance mixing
log-reflectance mixing
Kubelka–Munk (single-constant, opaque films)
mixture regression tests
```

### Step 6 — Search

```text
light_search.py: null-space and eigenvector solutions
paint_search.py: grid search over two-pigment pairs
scoring.py: composite J with penalties
synthetic metamer identity tests
first real candidate generation run
```

### Step 7 — Candidates and export

```text
CandidatePair with schema_version field
JSON serialisation (Pydantic v2)
CSV recipe export
reflectance and illumination curve plots (notebooks)
```

### Step 8 — Physical swatch testing

Translate top candidates into paint recipes and test physically.

### Step 9 — Calibration

Incorporate measured LED and swatch spectra. Update models. Rerun search.

---

## 24. Long-Term Vision

The long-term vision is a studio research tool that allows an artist to explore colour not as fixed surface appearance, but as a relationship between material, light, and observer.

The software should eventually allow the artist to ask:

```text
Find me two high-key mixtures that look identical under this white light
but separate under this RGBW lighting transition.
```

or:

```text
Given this measured paint surface, find the lighting state
that reveals maximum contrast.
```

or:

```text
Given these paints in my studio, search for a latent image palette
for a controlled-light painting.
```

The final artistic medium is not simply acrylic paint, nor simply LED light. It is the interaction between pigment spectra, programmable illumination, and human colour vision.
