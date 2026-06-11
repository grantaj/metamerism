# Colour-Science-First Refactor Plan

## Goal

Make `colour-science` the primary library for all colour and spectral computation in this repository.

The repository should stop reimplementing spectral storage, resampling, observer handling, tristimulus integration, whitepoint handling, and other core colour-science functionality that is already provided by `colour-science`.

## Architectural rule

Use this rule for every change:

- **`colour-science` owns colour math and spectral representation.**
- **`metamerism` owns loading, provenance, domain semantics, and project workflows.**

If a new function mainly wraps a standard colour-science operation, prefer calling `colour-science` directly instead of adding new project-local logic.

## Non-goals

- `Spectrum` is **not** a numerical mini-library.
- `Spectrum` is **not** the canonical internal spectral storage format.
- The codebase should **not** keep reimplementing `colour-science` methods behind custom APIs unless project-specific behavior is being added.
- The repository should **not** maintain its own observer integration machinery when the standard data and operations already exist in `colour-science`.

## Current problems to fix

### 1. `Spectrum` is doing too much

`src/metamerism/core/spectrum.py` currently owns:

- spectral storage as raw wavelength/value arrays;
- interpolation and extrapolation policy;
- resampling implementation;
- arithmetic operators;
- integration and dot products;
- the notion of a canonical internal wavelength grid.

This duplicates too much of `colour-science` and makes the wrapper too heavy.

### 2. Observer handling is reimplemented

`src/metamerism/core/observer.py` currently:

- defines a custom `Observer` dataclass;
- loads standard CMFs from `colour`;
- manually resamples them;
- manually constructs integration weights.

That code should instead use `colour` CMF objects directly.

### 3. XYZ conversion is reimplemented

`src/metamerism/colorimetry/tristimulus.py` currently performs manual tristimulus integration and normalization. That responsibility should move to `colour.sd_to_XYZ` and related `colour` functions.

### 4. The public API encourages local reimplementation

The current module layout still suggests that project-local utilities are the main path for colour work. The public API and docs should instead make it clear that:

- `colour-science` is the default tool for colour computation;
- `metamerism` adds project-specific wrappers and workflows only where needed.

## Target architecture

### `Spectrum`

Refactor `Spectrum` into a thin wrapper around `colour.SpectralDistribution`.

`Spectrum` should own:

- a `colour.SpectralDistribution` instance;
- project metadata such as `domain`;
- provenance metadata;
- loading/construction helpers;
- explicit conversion to and from `colour` objects.

`Spectrum` should not own:

- interpolation algorithms;
- extrapolation algorithms;
- arithmetic operator semantics;
- integration logic;
- observer projection logic;
- implicit internal canonicalization rules.

Recommended shape:

```python
@dataclass(frozen=True)
class Spectrum:
    sd: colour.SpectralDistribution
    domain: SpectrumDomain = SpectrumDomain.UNKNOWN
    provenance: Provenance = field(default_factory=Provenance)
```

This is only an outline, not a required exact signature.

### Preferred project shape

If the project wants a preferred wavelength sampling, define it as a convenience constant such as:

```python
PROJECT_SHAPE = colour.SpectralShape(380, 780, 5)
```

Treat this as a project default for alignment and ingestion, not as a bespoke spectral model.

### Observers and illuminants

Use `colour` standard datasets directly:

- `colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER`
- `colour.SDS_ILLUMINANTS`

If convenient aliases are useful, export thin constants that point to these objects without rewrapping them into custom data structures.

### Colorimetry

Use `colour` directly for:

- `sd_to_XYZ`
- `XYZ_to_Lab`
- `XYZ_to_sRGB`
- `delta_E`
- standard whitepoints and illuminants

Keep project-local colorimetry code only if it expresses project-specific semantics that `colour` does not already provide.

## Implementation plan

### Phase 1: Introduce a thin `Spectrum` wrapper

#### Deliverables

- Refactor `Spectrum` to hold a `colour.SpectralDistribution`.
- Add constructors/helpers such as:
  - `Spectrum.from_arrays(...)`
  - `Spectrum.from_colour(...)`
  - `Spectrum.to_colour()` or `.sd`
- Preserve `domain` and `provenance`.

#### Requirements

- `from_arrays(...)` should construct a `colour.SpectralDistribution` instead of storing parallel NumPy arrays as the primary representation.
- Keep input validation that is genuinely project-specific, especially domain validation and provenance handling.
- Expose explicit conversion helpers so downstream code can use `colour` APIs naturally.

#### Notes

- It is acceptable to keep lightweight convenience accessors like wavelengths/values if they are simple views over `sd`.
- Avoid reintroducing custom numerical behavior during this refactor.

### Phase 2: Remove homegrown spectral operations from `Spectrum`

#### Deliverables

Deprecate and then remove project-local numerical methods that duplicate `colour-science`, including as many of these as feasible in one pass:

- `resample`
- `to_canonical`
- interpolation/extrapolation enums and logic
- arithmetic operators (`__mul__`, `__add__`, `__sub__`, etc.)
- `normalise`
- `integrate`
- `dot`

#### Requirements

- Replace internal callers with direct `colour` operations.
- Do not move duplicated logic somewhere else under a different name.

#### Decision rule

If the behavior is standard spectral math, it belongs in `colour` usage, not in `Spectrum`.

### Phase 3: Replace custom observer handling

#### Deliverables

- Remove or drastically simplify `src/metamerism/core/observer.py`.
- Stop manually resampling CMFs and constructing integration weights.
- Replace custom `Observer` usage with direct `colour` CMFs.

#### Requirements

- Preserve convenient exported constants where helpful.
- Those constants should be aliases over `colour` data, not custom observer wrappers with duplicated state.

### Phase 4: Rebuild colorimetry as delegation

#### Deliverables

- Rewrite `src/metamerism/colorimetry/tristimulus.py` to delegate to `colour.sd_to_XYZ`.
- Simplify `src/metamerism/colorimetry/difference.py` to use `colour` APIs for ΔE and remove duplicated standard-data handling.
- Remove hard-coded standard whitepoint values if `colour` already provides the equivalent data.

#### Requirements

- Project wrappers may remain if they improve ergonomics, but they should be thin and explicit.
- Any wrapper should add project semantics, not reimplement colour-science internals.

#### Example direction

- Convert `Spectrum` inputs to `colour.SpectralDistribution`
- Pass standard observer CMFs directly
- Pass illuminants directly
- Let `colour` perform the numerical integration

### Phase 5: Update public exports and docs

#### Deliverables

- Update `src/metamerism/core/__init__.py`
- Update `src/metamerism/colorimetry/__init__.py`
- Update README and any relevant docs

#### Requirements

- The docs should clearly state:
  - use `colour-science` for colour computations;
  - use `metamerism` for provenance-aware loading and project workflows.
- Remove language that implies metamerism provides its own core colour engine.

### Phase 6: Rewrite tests around the new boundary

#### Deliverables

- Rewrite `tests/core/test_spectrum.py`
- Update `tests/test_colorimetry.py`

#### Test strategy

Tests should now verify:

- `Spectrum` stores and exposes a `colour.SpectralDistribution`;
- provenance is preserved across construction and any remaining wrapper operations;
- domain validation still works;
- wrapper functions produce the same outputs as direct `colour` calls;
- project-specific helpers behave correctly.

Tests should stop validating:

- custom interpolation details;
- custom integration weights;
- bespoke observer resampling behavior;
- arithmetic semantics that no longer belong to `Spectrum`.

## Suggested file-by-file work order

1. `src/metamerism/core/spectrum.py`
2. `src/metamerism/core/observer.py`
3. `src/metamerism/colorimetry/tristimulus.py`
4. `src/metamerism/colorimetry/difference.py`
5. `src/metamerism/core/__init__.py`
6. `src/metamerism/colorimetry/__init__.py`
7. `tests/core/test_spectrum.py`
8. `tests/test_colorimetry.py`
9. `README.md`

## Constraints for the coding agent

- Prefer deletion over adaptation when code only duplicates `colour-science`.
- Keep `Spectrum` thin.
- Keep project-specific provenance and domain semantics.
- Do not add a new abstraction layer that merely renames `colour` concepts.
- Avoid compatibility shims that preserve old behavior if that behavior exists only because the codebase previously reimplemented `colour`.
- Preserve explicitness: no hidden global observer or hidden implicit conversions unless clearly documented and intentionally retained.

## Acceptance criteria

The refactor is complete when all of the following are true:

1. `Spectrum` is a provenance-aware wrapper over `colour.SpectralDistribution`, not an independent spectral math type.
2. Standard observer data is consumed directly from `colour`.
3. XYZ conversion delegates to `colour.sd_to_XYZ`.
4. Standard whitepoint and ΔE handling come from `colour`, not project-local copies.
5. The main spectral/color workflow in docs points users toward `colour-science` first.
6. Tests validate the new architecture rather than the removed implementation details.
7. There is no remaining project-local numerical colour-science subsystem hiding behind different names.

## Practical review checklist

Use this checklist during implementation review:

- Does this code call `colour` directly where possible?
- Is `Spectrum` only carrying project metadata plus a `colour` spectral object?
- Did we remove duplicated interpolation/integration logic rather than relocating it?
- Are standard observers and illuminants sourced directly from `colour`?
- Does every remaining wrapper add project-specific value?
- Would a new contributor clearly understand that `colour-science` is the core colour engine?

## Final guidance

When in doubt, use this question:

**"Should this just be a direct `colour-science` call?"**

If the answer is yes, it probably should not be implemented as custom repository logic.
