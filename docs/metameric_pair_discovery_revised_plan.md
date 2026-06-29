# Revised Implementation Plan: Metameric Pair Discovery

## Status and purpose

This plan extends the existing `metamerism` repository to discover, rank, and eventually realise metameric paint pairs.

The goal is to find two reflectance spectra, and later two paint mixtures, which are visually close under a **conceal** illuminant and clearly different under one or more **reveal** illuminants.

The repository already provides the project foundations:

- `colour-science` is the authoritative numerical colour engine.
- `Spectrum` provides provenance-aware spectral data and domain semantics.
- the project default spectral shape is 380–780 nm at 5 nm intervals;
- Golden paint reflectance data and a mixing MVP already exist;
- `search/` is the natural home for metamer-search workflow logic;
- Streamlit already provides a local exploratory surface.

This plan must not create a second colour-science stack or a competing spectral abstraction layer.

---

# Non-negotiable architecture rules

1. **Use `colour-science` for standard spectral and colourimetric calculations.**

   This includes alignment, interpolation, `sd_to_XYZ`, `XYZ_to_Lab`, `XYZ_to_sRGB`, and `delta_E`.

2. **Do not add project-local replacements for general colour operations.**

   The search subsystem may construct a sampled XYZ projection matrix for null-space and optimisation calculations. This is a specialised internal optimisation representation, not a general replacement for `colour`.

3. **Use existing project types and modules.**

   Extend `core`, `colorimetry`, `illuminants`, `mixing`, `pigments`, and especially `search`; do not introduce parallel package trees such as `spectral/` or a nested `metamerism/metamerism/`.

4. **Validate all search-generated results through direct `colour-science` calculations.**

   The projection matrix is an acceleration and geometry tool. Direct `colour` results are authoritative for acceptance, reporting, and ranking.

5. **Keep UI thin.**

   Streamlit should call stable core/search APIs and display results. It must not contain numerical search logic.

---

# Scientific model

Let reflectance be sampled on the project wavelength grid:

\[
r \in \mathbb{R}^{n}.
\]

For a chosen observer and conceal illuminant, construct an internal sampled projection matrix:

\[
A_0 \in \mathbb{R}^{3 \times n},
\]

such that approximately:

\[
XYZ = A_0 r.
\]

Two spectra are reference metamers when:

\[
A_0r_1 = A_0r_2.
\]

Equivalently, the difference is in the null space:

\[
A_0(r_1-r_2)=0.
\]

If \(N\) is a basis for \(\ker(A_0)\), a spectrum metameric with a centre spectrum \(r_c\) can be written:

\[
r = r_c + Nz.
\]

The first search stages find bounded, smooth candidates in this affine slice:

\[
0 \le r(\lambda) \le 1.
\]

Candidates are then ranked by direct CIEDE2000 differences under reveal illuminants.

The practical long-term objective is:

\[
\max \quad \Delta E_{\mathrm{reveal}}(r_1, r_2)
\]

subject to:

\[
\Delta E_{\mathrm{conceal}}(r_1, r_2) \le \epsilon,
\]

plus reflectance plausibility and later paint-mixture realisability constraints.

---

# Target module layout

Extend the existing package approximately as follows:

```text
src/metamerism/
  core/
    ... existing Spectrum, provenance, domains ...

  colorimetry/
    ... existing colour-science-facing helpers ...

  illuminants/
    ... existing named/project illuminants ...

  pigments/
    ... existing Golden data loading and paint library ...

  mixing/
    ... existing mixture models ...

  search/
    __init__.py
    metamer.py          # public domain types and scores
    projection.py       # sampled XYZ projection + null-space basis
    feasibility.py      # reflectance-bound geometry
    generate.py         # candidate generation
    objectives.py       # conceal/reveal scoring and filters
    diagnostics.py      # numerical checks and result summaries
    realisability.py    # later: approximation by available paints
    optimise.py         # later: constrained/direct optimisation

apps/
  streamlit_mixing.py   # extend only after stable search APIs exist

tests/
  search/
    test_metamer.py
    test_projection.py
    test_feasibility.py
    test_generate.py
    test_objectives.py
    test_realisability.py
```

Do not create `spectral/`, `visualisation/`, or `experiments/` package trees unless a later concrete need proves the current structure insufficient.

---

# Phase A — Conceal/reveal scoring vertical slice

## Goal

Turn the existing search placeholder into a tested workflow primitive that compares two spectra under a conceal and reveal condition.

## Scope

Implement or extend:

```text
src/metamerism/search/metamer.py
src/metamerism/search/objectives.py
tests/search/test_metamer.py
```

Suggested public types:

```python
@dataclass(frozen=True)
class ViewingCondition:
    name: str
    illuminant: Spectrum
    observer_name: str

@dataclass(frozen=True)
class ConditionResult:
    condition_name: str
    xyz_first: NDArray[np.float64]
    xyz_second: NDArray[np.float64]
    lab_first: NDArray[np.float64]
    lab_second: NDArray[np.float64]
    delta_e00: float

@dataclass(frozen=True)
class MetamerScore:
    conceal: ConditionResult
    reveal: tuple[ConditionResult, ...]
    weighted_reveal_score: float
    minimum_reveal_delta_e00: float
```

## Required behaviour

- Convert spectra to XYZ and Lab using direct `colour` calls.
- Calculate CIEDE2000 under each named condition.
- Separate **conceal error** from **reveal strength**.
- Reject or flag incompatible spectral domains/grids through existing project conventions.
- Preserve condition names and provenance in the result.

## Acceptance criteria

- Identical reflectance spectra produce zero difference under all conditions.
- A deliberately changed spectrum produces non-zero difference.
- Direct `colour` calculations and the project result agree within tolerance.
- Tests are deterministic and do not rely on UI.
- Existing style/type/test commands pass.

---

# Phase B — Internal projection and null-space geometry

## Goal

Add the linear-algebra machinery needed to construct reference metamers.

## Scope

Add:

```text
src/metamerism/search/projection.py
src/metamerism/search/feasibility.py
tests/search/test_projection.py
tests/search/test_feasibility.py
```

## Projection representation

```python
@dataclass(frozen=True)
class XYZProjection:
    matrix: NDArray[np.float64]          # shape (3, n_wavelengths)
    wavelengths_nm: NDArray[np.float64]
    observer_name: str
    illuminant_name: str
    normalisation: str
    singular_values: NDArray[np.float64]
```

The projection must be built from the same observer, illuminant, wavelength alignment, and normalisation assumptions as the direct `colour` path.

The preferred convention is that a perfect reflector has:

\[
Y = 100.
\]

## Required functions

```python
def build_xyz_projection(
    condition: ViewingCondition,
    *,
    spectral_shape: colour.SpectralShape = PROJECT_SHAPE,
) -> XYZProjection:
    ...

def compute_null_space(
    projection: XYZProjection,
    *,
    relative_tolerance: float | None = None,
) -> NullSpace:
    ...

def projection_xyz(
    reflectance_values: NDArray[np.float64],
    projection: XYZProjection,
) -> NDArray[np.float64]:
    ...

def feasible_step_interval(
    centre: NDArray[np.float64],
    direction: NDArray[np.float64],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, float] | None:
    ...
```

Use SVD to obtain the null-space basis. The tolerance must be documented and recorded in diagnostics.

## Important implementation boundary

`projection_xyz` is only for optimisation and numerical checks. The public score path must still use `colour.sd_to_XYZ` and related `colour` functions.

## Acceptance criteria

- The matrix has expected shape and numerical rank 3 for normal conditions.
- Matrix XYZ agrees with direct `colour` XYZ for:
  - perfect reflector;
  - flat 50% reflector;
  - several smooth synthetic spectra;
  - at least one Golden spectrum.
- `A_0N` is numerically near zero.
- Random null-space perturbations preserve projected XYZ.
- `feasible_step_interval` guarantees wavelength-wise reflectance bounds.
- Every tolerance is named, documented, and centralised.

---

# Phase C — Bounded ideal-metamer generation

## Goal

Generate valid reflectance spectra that match a supplied reference spectrum under the conceal condition.

## Scope

Add:

```text
src/metamerism/search/generate.py
tests/search/test_generate.py
```

Start with a transparent directional algorithm. Do not begin with vertex enumeration, a full polytope solver, evolutionary optimisation, or joint nonlinear pair search.

## Algorithm

For a supplied reference reflectance \(r_c\):

1. Align it to the project spectral shape.
2. Construct the conceal projection \(A_0\).
3. Compute its null-space basis \(N\).
4. Sample a direction \(d = Nz\).
5. Compute the feasible interval \([\alpha_-, \alpha_+]\) satisfying:

   \[
   0 \le r_c+\alpha d \le1.
   \]

6. Generate one or both endpoint candidates:

   \[
   r_- = r_c+\alpha_-d,\qquad r_+=r_c+\alpha_+d.
   \]

7. Convert candidates to project `Spectrum` objects with complete provenance.
8. Verify conceal difference through direct `colour` scoring.
9. Retain only candidates meeting the named conceal tolerance.

## Suggested API

```python
@dataclass(frozen=True)
class CandidateGenerationConfig:
    seed: int
    n_directions: int
    endpoint_fraction: float
    conceal_delta_e00_limit: float
    reflectance_bound_tolerance: float

@dataclass(frozen=True)
class MetamerCandidate:
    spectrum: Spectrum
    source_reference: Spectrum
    null_space_coordinates: NDArray[np.float64]
    max_bound_violation: float
    roughness: float
    provenance: Provenance

def generate_directional_metamers(
    reference: Spectrum,
    conceal_condition: ViewingCondition,
    config: CandidateGenerationConfig,
) -> list[MetamerCandidate]:
    ...
```

## Acceptance criteria

- Generates multiple distinct bounded partner spectra from a smooth reference spectrum.
- Every retained candidate satisfies:
  - reflectance bounds within tolerance;
  - conceal \(\Delta E_{00}\) below configured threshold;
  - direct-colour verification.
- Search is reproducible for a fixed seed/configuration.
- Candidate provenance identifies reference spectrum, condition, null-space tolerance, direction seed, and endpoint choice.

---

# Phase D — Smoothness, diagnostics, and practical ranking

## Goal

Stop mathematically valid but implausible comb-like spectra from dominating the search.

## Scope

Extend:

```text
src/metamerism/search/objectives.py
src/metamerism/search/diagnostics.py
tests/search/test_objectives.py
```

## Metrics

Implement explicit, reportable metrics:

### Second-difference roughness

\[
R(r) = \|D^2r\|_2^2.
\]

### Spectral separation

\[
D(r_1,r_2)=\|W(r_1-r_2)\|_2.
\]

Initially use \(W=I\), but structure the implementation so wavelength weighting can be introduced later.

### Boundary saturation

Measure the fraction of wavelength samples near 0 or 1. This is diagnostic initially, then optionally a penalty.

### Reveal score

For reveal conditions \(j\):

\[
S_{\mathrm{weighted}}=\sum_jw_j\Delta E_{00,j}.
\]

Also report:

\[
S_{\min}=\min_j\Delta E_{00,j}.
\]

## Ranking policy

Rank candidates or pairs by a transparent configurable policy:

1. conceal difference must meet threshold;
2. reveal mismatch is maximised;
3. roughness is penalised or screened;
4. boundary saturation is reported and optionally penalised;
5. raw spectral distance remains secondary diagnostic information.

Do not hide regularisation inside the generator. All penalties and filters must be config values recorded in provenance.

## Acceptance criteria

- A rougher candidate can be rejected or ranked below an otherwise comparable smooth candidate.
- Result objects expose conceal error, per-reveal differences, roughness, saturation, and spectral distance.
- A known regression fixture can detect accidental changes in scoring/ranking.

---

# Phase E — Multiple reveal conditions and candidate-pool selection

## Goal

Find a useful set of diverse candidates rather than one brittle optimum.

## Scope

Allow configurations containing:

- one conceal condition;
- one or more reveal conditions;
- reveal weights;
- smoothness/saturation settings;
- a random seed and candidate count.

Use the existing illuminant infrastructure for named or project-defined SPDs.

## Candidate selection process

1. Generate a pool through Phase C.
2. Score every candidate against the reference under conceal and reveal conditions.
3. Form candidate pairs where needed.
4. Filter by conceal threshold and feasibility.
5. Compute a Pareto set over:
   - maximise weighted reveal mismatch;
   - maximise minimum reveal mismatch;
   - minimise roughness;
   - minimise boundary saturation;
   - maximise spectral distance as a diagnostic.
6. Apply a simple diversity pass, initially greedy farthest-point selection in spectral space or null-space coordinates.

Do not introduce a genetic algorithm yet.

## Acceptance criteria

- Results include multiple distinct high-value candidates, not near duplicates.
- Fixed configuration and seed reproduce the selected set.
- Search results are serialisable, including full conditions/configuration and numeric diagnostics.
- At least one small end-to-end fixture is retained for regression testing.

---

# Phase F — Early Streamlit diagnostic integration

## Goal

Make the mathematical search inspectable early, without coupling it to the UI.

## Scope

Extend the existing `apps/streamlit_mixing.py` only after Phases A–E provide stable public APIs.

Add a small metamer-search panel that can:

- choose a reference paint spectrum from the existing Golden library;
- choose conceal and reveal illuminants;
- run bounded directional generation with a small candidate count;
- show the best candidate or candidate pair;
- plot:
  - reference and partner reflectance;
  - their difference;
  - conceal and reveal colour swatches;
  - conceal/reveal \(\Delta E_{00}\);
  - roughness and bound diagnostics.

The UI should use library APIs and configuration objects, not reproduce any search logic.

## Acceptance criteria

- The same configuration yields the same result from UI and command-line/script use.
- No search or colour calculation is implemented in the app module.
- A user can visually inspect whether the result is an obvious numerical artefact.

---

# Phase G — Paint realisability using current mixing infrastructure

## Goal

Evaluate whether ideal metamer targets are plausibly reachable using the available Golden paint library.

## Scope

Add:

```text
src/metamerism/search/realisability.py
tests/search/test_realisability.py
```

Use existing `pigments/` loading and `mixing/` interfaces. Do not create a second forward-model hierarchy.

## First realisability model

Begin with an explicitly labelled baseline approximation model, even if it is only a constrained linear/convex mixture in reflectance space.

For an ideal target \(r_t\), solve:

\[
\min_x
\quad
\|W(R_{\mathrm{mix}}(x)-r_t)\|_2^2
+
\lambda_{\mathrm{conceal}}
\|XYZ_{\mathrm{conceal}}(R_{\mathrm{mix}}(x))-
  XYZ_{\mathrm{conceal}}(r_t)\|_2^2
\]

subject to:

\[
x_i \ge 0,\qquad \sum_i x_i=1.
\]

The target is not accepted as “paint-realizable” merely because the solver returns a result.

## Required output

```python
@dataclass(frozen=True)
class PaintApproximation:
    target: Spectrum
    realised: Spectrum
    fractions: Mapping[str, float]
    spectral_error: float
    conceal_delta_e00: float
    reveal_results: tuple[ConditionResult, ...]
    model_name: str
    model_assumptions: Mapping[str, str]
```

## Acceptance criteria

- Pure-paint targets recover the corresponding paint under the baseline model.
- Approximation output exposes reference and reveal degradation honestly.
- Candidate reporting distinguishes:
  - ideal spectral potential;
  - achieved approximation quality;
  - realised metameric strength.
- No physical accuracy claim is made for the baseline model.

---

# Phase H — Improve the paint forward model

## Goal

Replace or supplement the baseline approximation only when the paint-model assumptions are understood.

## Scope

Review the current mixing code and Golden measurement conditions before choosing a Kubelka–Munk implementation.

Important limitation:

Golden drawdown reflectances are paint-film measurements under particular thickness/substrate conditions. They are not automatically intrinsic pigment \(K\) and \(S\) functions.

Before adding Kubelka–Munk:

- define what data are available for each paint;
- document substrate and thickness assumptions;
- decide how \(K\) and \(S\) are obtained or estimated;
- add validation against measured mixed drawdowns where available.

## Acceptance criteria

- The chosen physical model has documented assumptions.
- Its behaviour is tested at pure-paint limits.
- Predictions are compared against actual measurements, not only internal numerical consistency.
- Existing baseline results remain available for comparison.

---

# Phase I — Direct optimisation over paint-mixture pairs

## Goal

Search directly within the spectra accessible to the paint model.

## Formulation

For paint mixture fractions \(x\) and \(y\):

\[
\max_{x,y}
\quad
S_{\mathrm{reveal}}(R_{\mathrm{mix}}(x),R_{\mathrm{mix}}(y))
\]

subject to:

\[
\Delta E_{00,\mathrm{conceal}}
(R_{\mathrm{mix}}(x),R_{\mathrm{mix}}(y))
\le\epsilon,
\]

\[
x_i,y_i\ge0,\qquad
\sum_i x_i=\sum_i y_i=1.
\]

This is non-convex. Do not claim global optimality.

## Suggested first approach

1. Sample mixture simplex points.
2. Group or index mixtures by conceal colour.
3. find close conceal matches with strong estimated reveal mismatch;
4. refine promising pairs using constrained local optimisation;
5. use multi-start;
6. retain and compare multiple local optima.

## Acceptance criteria

- Finds actual mixture pairs meeting conceal tolerance.
- Reports all mixture fractions and forward-model assumptions.
- Repeats reliably with a fixed seed/configuration.
- Compares direct-search results to the ideal-target-then-approximate pipeline.

---

# Cross-cutting requirements

## Configuration and provenance

All searches must be configured by serialisable data, preferably JSON or YAML. Each result must record:

- reference spectrum or paint;
- conceal and reveal condition definitions;
- observer;
- wavelength shape;
- null-space/SVD tolerance;
- random seed;
- generator and ranking parameters;
- paint-library and mixing-model versions where applicable.

## Numerical tolerances

Define named constants/configuration fields. Do not scatter magic numbers.

Examples:

```python
DEFAULT_CONCEAL_DELTA_E00_LIMIT = ...
DEFAULT_REFLECTANCE_BOUND_TOLERANCE = ...
DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE = ...
DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE = ...
```

## Testing

Use three levels:

1. **Unit tests**
   - projection construction;
   - null-space dimensions;
   - feasible intervals;
   - direct conceal/reveal scoring;
   - roughness and ranking;
   - input validation.

2. **Numerical invariant tests**
   - null-space perturbations preserve projected XYZ;
   - direct `colour` verification agrees with projection calculations;
   - candidates satisfy bounds and conceal threshold.

3. **Regression fixtures**
   - one reference spectrum;
   - a small fixed illuminant configuration;
   - seeded candidate generation;
   - expected top-candidate diagnostics within tolerance;
   - later, one paint approximation and one actual mixture pair.

Do not require bitwise-identical results across platforms.

## Tooling

Use the repository’s existing workflow:

```text
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check
```

Use NumPy/SciPy for linear algebra and initial optimisation. Add CVXPY only when a clearly convex subproblem benefits from it.

---

# First implementation milestone

## Deliverable

Implement an end-to-end, tested **ideal metamer discovery vertical slice**:

> Given a Golden reference paint spectrum, one conceal illuminant, and one or more reveal illuminants, generate a seeded pool of bounded smooth null-space partner spectra; validate each with direct colour-science calculations; rank candidates by reveal CIEDE2000 mismatch subject to a conceal CIEDE2000 threshold; expose the top result in a small Streamlit diagnostic view.

## In scope

- Phases A–D;
- a minimal single-reveal version of Phase E;
- a small UI display after core APIs are stable.

## Explicitly out of scope

- paint inverse optimisation;
- Kubelka–Munk;
- global polytope vertex enumeration;
- evolutionary/global optimisation;
- experimental hardware control;
- claims that ideal candidates are physically paint-realizable.

## Definition of done

A script or command-line example can:

1. Load a Golden reflectance spectrum.
2. Choose declared conceal and reveal conditions.
3. Build an internal sampled projection and null-space basis.
4. Generate multiple bounded candidate partners.
5. Verify each candidate with direct `colour` calculations.
6. Rank candidates with:
   - conceal \(\Delta E_{00}\);
   - reveal \(\Delta E_{00}\);
   - roughness;
   - spectral distance;
   - boundary saturation;
   - full provenance.
7. Export a serialisable result/configuration.
8. Display the best result in the existing Streamlit app.

This milestone validates the central research proposition before substantial effort is spent on paint-mixture inversion.
