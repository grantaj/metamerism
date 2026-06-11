# Project Plan: Metameric Paint–Light Exploration Tool

## Aim

Build a software-assisted system for finding paint/illumination pairs that are visually similar under one light state and clearly different under another.

The software narrows experimental search space for studio work; it does not replace physical testing.

## Core scientific model

Perceived colour is driven by:

`illumination spectrum × reflectance spectrum × observer response`

Metameric behavior is evaluated across multiple programmable illuminants.

## Architecture rule (non-negotiable)

1. `colour-science` is the colour engine.
2. `metamerism` provides ingestion, provenance, domain semantics, and workflow orchestration.
3. Do not implement local replacements for spectral math provided by `colour-science`.

If a function is standard spectral/colorimetric math, call `colour` directly.

## What `Spectrum` is

`Spectrum` is a thin provenance-aware container:

- sampled wavelength/value data;
- domain metadata (`REFLECTANCE`, `ILLUMINANT`, etc.);
- provenance metadata.

`Spectrum` exists to carry project context and interoperate with `colour`.

### What `Spectrum` is not

- not a numerical operations API;
- not a resampling/interpolation engine;
- not an arithmetic layer;
- not an integration/dot-product layer.

## Spectral computation policy

Use native `colour` objects and operations for computation:

- `colour.SpectralDistribution`
- `align(...)`, `interpolate(...)`
- `sd_to_XYZ(...)`
- `XYZ_to_Lab(...)`, `XYZ_to_sRGB(...)`
- `delta_E(...)`

No project-local `ops.py` replacement layer for colour operations.

## Preferred sampling

Project default spectral shape:

- 380–780 nm, 5 nm interval
- represented as `PROJECT_SHAPE = colour.SpectralShape(380, 780, 5)`

This is a convenience default, not a custom spectral system.

## Observer policy

Use standard observer data directly from `colour` datasets.

Project aliases may exist for ergonomics, but must remain thin references to colour data, not duplicated observer implementations.

## Testing policy

Tests should validate:

- `Spectrum` ingestion/provenance/domain behavior;
- correct delegation to `colour`;
- parity between wrapper outputs and direct `colour` results.

Tests should not validate project-local implementations of interpolation, integration, or arithmetic primitives.

## Execution phases

1. Remove in-repo spectral operation layers (`ops.py`, local arithmetic helpers, interpolation/extrapolation APIs).
2. Ensure all computation paths use native `colour` calls.
3. Keep `Spectrum` focused on data + provenance + conversion.
4. Update tests/docs to reflect colour-first operation boundaries.
5. Keep artistic/search workflow modules focused on orchestration and experiment logic.

## Review checklist

- Does this code call `colour` directly for spectral math?
- Does `Spectrum` only hold data/provenance/domain + interop?
- Did we delete duplicate logic instead of moving it?
- Do docs make `colour-science` the default path?
- Would a new contributor clearly see that metamerism is workflow/provenance, not a custom colour engine?
