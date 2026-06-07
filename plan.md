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

The software will consist of several separable modules.

```text
1. Spectral data module
2. Pigment database module
3. LED illuminant module
4. Paint mixture prediction module
5. Colour/perception module
6. Metamer search module
7. Visualisation/simulation module
8. Empirical calibration module
9. Studio workflow/export module
```

Each module should be replaceable or improvable without breaking the rest of the system.

The most important architectural principle is that all colour calculations should be spectral first. RGB values should be outputs for preview only, not the internal representation.

---

## 5. Module 1 — Spectral Data Infrastructure

### Purpose

Provide common handling of wavelength-indexed spectral data.

### Responsibilities

- Define a standard wavelength grid, probably 380–730 nm or 380–780 nm.
- Resample all spectra onto the common grid.
- Load spectral data from CSV or similar formats.
- Interpolate missing values.
- Normalise spectra where appropriate.
- Store metadata and confidence levels.

### Data types

```text
Reflectance spectrum: R(λ), usually 0–1
Illuminant spectrum: I(λ), arbitrary relative power
Cone fundamentals or colour matching functions
LED channel spectra
Measured swatch spectra
```

### Initial implementation

Use NumPy arrays and simple CSV input/output.

Example internal form:

```python
wavelength_nm: np.ndarray
values: np.ndarray
```

Later this can be wrapped in richer classes.

---

## 6. Module 2 — Pigment Database

### Purpose

Store and access spectral reflectance curves for artist pigments and paints.

### Initial data source

The first useful source is the available spectral data for GOLDEN Heavy Body acrylic pigments. These can be used as a starting pigment library for simulation and candidate generation.

### Data model

Each pigment entry should include:

```text
paint brand
paint range
pigment name
pigment code, e.g. PO20, PY35, PR108
reflectance spectrum
opacity / transparency if known
colour index name if available
metadata source
confidence level
```

### Important distinction

The database should distinguish between:

```text
single-pigment paints
multi-pigment commercial paints
measured physical swatches
computed mixtures
```

This is important because artist paint names can conceal complex pigment formulations.

---

## 7. Module 3 — LED Illumination Model

### Purpose

Represent programmable LED illumination as spectral power distributions.

For RGB LEDs:

```text
I(λ) = rR(λ) + gG(λ) + bB(λ)
```

For RGBW LEDs:

```text
I(λ) = rR(λ) + gG(λ) + bB(λ) + wW(λ)
```

For a general multi-channel illuminant:

```text
I(λ) = c₁E₁(λ) + c₂E₂(λ) + ... + cₙEₙ(λ)
```

### Initial light sources

The first lighting models should include:

```text
WS2812B / NeoPixel RGB approximation
SK6812 RGBW approximation
generic RGB LED model
generic RGB + warm white model
D65 daylight reference
D50 reference
warm white LED approximation
cool white LED approximation
```

### NeoPixel approximation

For early software work, each channel can be approximated by a Gaussian or asymmetric Gaussian:

```text
Red:   centre ~625 nm, FWHM ~20 nm
Green: centre ~525 nm, FWHM ~30 nm
Blue:  centre ~467 nm, FWHM ~20 nm
```

This is not final scientific data. It is a placeholder model that allows the architecture and search algorithms to be developed before measured LED spectra are available.

### Preferred early hardware

The most useful readily available upgrade from standard NeoPixel RGB is likely SK6812 RGBW, because the additional white channel provides a broader phosphor component as well as narrow RGB channels.

For experimentation:

```text
SK6812 RGBW strip
ESP32 or Arduino controller
stable 5 V power supply
diffuser
small enclosed test box or controlled lighting rig
```

---

## 8. Module 4 — Paint Mixture Prediction

### Purpose

Predict the reflectance spectrum of paint mixtures from component paint spectra.

This is the most uncertain part of the modelling.

### Initial mixture models

The software should support several mixture models.

#### 1. Linear reflectance mixing

```text
R_mix(λ) = aR₁(λ) + bR₂(λ) + ...
```

This is physically weak but useful as a baseline.

#### 2. Log-reflectance mixing

```text
log R_mix(λ) = a log R₁(λ) + b log R₂(λ) + ...
```

This better approximates subtractive behaviour because absorption compounds multiplicatively.

#### 3. Kubelka–Munk-inspired mixing

Use a simplified opaque-paint model based on K/S values, where possible.

For opaque layers:

```text
K/S = (1 - R)² / (2R)
```

Mixtures can then be approximated in K/S space.

This will likely become more important once measured swatch data is available.

### Important caveat

Real paint mixing depends on:

```text
pigment concentration
medium
film thickness
substrate
opacity
scattering
surface gloss
application method
drying behaviour
```

The model should therefore be treated as a candidate generator, not as an exact predictor.

---

## 9. Module 5 — Colour and Perception Calculation

### Purpose

Compute perceived colour from a reflectance spectrum under a given illuminant.

### Calculation pipeline

Given:

```text
paint reflectance R(λ)
illuminant I(λ)
colour matching functions x̄(λ), ȳ(λ), z̄(λ)
```

Compute:

```text
X = ∫ I(λ) R(λ) x̄(λ) dλ
Y = ∫ I(λ) R(λ) ȳ(λ) dλ
Z = ∫ I(λ) R(λ) z̄(λ) dλ
```

Then convert to:

```text
XYZ
Lab
LCh
sRGB preview
possibly OKLab
possibly CAM16 later
```

### Difference metrics

At minimum:

```text
ΔE76
ΔE00
spectral distance
LMS difference
brightness / luminance difference
```

### Role in the project

This module should be implemented early and carefully. Unlike mixture modelling, this part is relatively standard and reliable.

---

## 10. Module 6 — Metameric Search

### Purpose

Find pairs of paint mixtures that are visually similar under one illuminant and different under another.

### Basic search target

For two candidate paint mixtures A and B:

```text
conceal_error = ΔE(A, B under light_1)
reveal_difference = ΔE(A, B under light_2)
```

A strong candidate has:

```text
conceal_error small
reveal_difference large
```

### Candidate scoring

A simple score might be:

```text
score = reveal_difference / (conceal_error + ε)
```

With penalties for:

```text
too dark
too low chroma
too low brightness
unpleasant or impractical illumination
too many pigments in mixture
tiny unstable mixture ratios
poor predicted physical robustness
out-of-gamut preview
```

### Search variables

The search can operate over:

```text
paint mixture A recipe
paint mixture B recipe
lighting state 1
lighting state 2
```

For RGBW lighting:

```text
light_1 = [r1, g1, b1, w1]
light_2 = [r2, g2, b2, w2]
```

### Constraint examples

```text
ΔE_conceal < 2.0
ΔE_reveal > 10.0
Y_conceal within acceptable brightness range
Y_reveal within acceptable brightness range
maximum 2 or 3 pigments per mixture
minimum reflectance / high-key constraint
similar value under conceal light
```

### Search strategies

Begin with simple brute force and random search.

Then add:

```text
grid search over two-pigment mixtures
random search over three-pigment mixtures
Bayesian optimisation
genetic algorithms
linear algebra screening for LED states
constrained numerical optimisation
```

The first goal is not elegance. The first goal is to find physically testable candidates.

---

## 11. Linear Algebra Insight for LED Search

For a fixed paint reflectance and fixed LED channel spectra, cone response is approximately linear in LED drive values.

For one paint:

```text
c = M_P u
```

where:

```text
c = [L, M, S]ᵀ
u = [r, g, b]ᵀ
```

For two paints A and B:

```text
c_A - c_B = (M_A - M_B)u
```

This gives a useful method for searching lighting states.

The conceal light should satisfy:

```text
(M_A - M_B)u ≈ 0
```

The reveal light should maximise:

```text
||(M_A - M_B)u||
```

subject to practical constraints:

```text
0 ≤ channel values ≤ 1
brightness sufficient
not too saturated unless desired
does not exceed LED current limits
does not cause flicker or thermal instability
```

This suggests that the software should search not just for metameric paint pairs, but for paint pairs with useful controllable light-response differences.

---

## 12. Module 7 — Visualisation and Simulation

### Purpose

Provide visual feedback for candidate pairs and lighting states.

### Core views

The interface should show:

```text
paint A under conceal light
paint B under conceal light
paint A under reveal light
paint B under reveal light
reflectance curves of A and B
illumination spectra for light 1 and light 2
resulting reflected spectra
Lab / LCh / sRGB values
ΔE values
mixture recipes
confidence warnings
```

### Important limitation

The screen cannot display the true spectral phenomenon. It can only show the predicted colour appearance under each modelled illuminant.

The display is therefore a decision aid, not the artwork itself.

### Useful interface modes

```text
candidate browser
side-by-side comparison
illumination slider
RGB/RGBW control panel
reflectance curve plot
ranking table
exportable swatch recipe sheet
```

---

## 13. Module 8 — Empirical Measurement and Calibration

### Purpose

Close the loop between predicted and actual behaviour.

### Measurements needed

Eventually the project needs measured spectra for:

```text
actual paint swatches
actual paint mixtures
actual LED channels
actual mixed lighting states
possibly substrates and varnishes
```

### Reflectance measurements

For each painted swatch:

```text
paint recipe
substrate
ground
application method
film thickness if known
medium
drying time
measured reflectance spectrum
lighting/viewing geometry
instrument settings
```

### LED measurements

For each light source:

```text
red channel spectrum
green channel spectrum
blue channel spectrum
white channel spectrum if present
mixed states for validation
channel response vs drive value
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

A spectrophotometer is not critical for the first software prototype.

It becomes critical when the project moves from:

```text
candidate discovery
```

to:

```text
reliable physical production
```

The key failure point is mixture prediction. Public pigment reflectance data may suggest promising candidates, but actual painted mixtures will vary with medium, thickness, surface, substrate, and application.

---

## 14. Physical Experiment Program

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

This phase will be rough but artistically informative.

### Stage 3 — Measured LED spectra

Measure the actual LED channels.

Update the illumination model.

Rerun search.

This improves the reliability of the predicted reveal/conceal lighting states.

### Stage 4 — Measured paint swatches

Measure single paints and mixtures.

Compare predicted and measured reflectance.

Improve the mixture model.

### Stage 5 — Studio-scale panels

Move from small swatches to painted panels.

Investigate:

```text
surface texture
scale
viewing distance
gloss
application method
edge conditions
illumination uniformity
viewer movement
```

### Stage 6 — Finished painting/installation studies

Develop completed works using controlled illumination.

Possible formats:

```text
static conceal/reveal painting
slowly changing illumination painting
hidden structure revealed by spectral choreography
viewer-triggered lighting transitions
apparently monochrome painting that differentiates under LED states
```

---

## 15. Hardware Plan

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

If one instrument can handle both reflectance and emission sufficiently well, that is ideal. If not, reflectance measurement is the higher priority for paint modelling, while approximate or manufacturer LED data may suffice temporarily.

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

## 16. Software Implementation Plan

### Language and libraries

Initial implementation in Python.

Likely libraries:

```text
numpy
scipy
pandas
matplotlib
colour-science
scikit-learn or scipy.optimize
streamlit, tkinter, textual, or a simple desktop/web UI
```

Given prior experience with the colourshift project, a Python desktop GUI or lightweight web interface would be appropriate.

### Repository structure

```text
metamerism/
    metamer/
        spectra.py
        pigments.py
        illuminants.py
        observers.py
        colorimetry.py
        mixing.py
        search.py
        simulate.py
        calibration.py
        io.py

    data/
        pigments/
        illuminants/
        observers/
        swatches/
        candidates/

    notebooks/
        001_colour_pipeline.ipynb
        002_led_model.ipynb
        003_mixture_model_tests.ipynb
        004_search_experiments.ipynb

    scripts/
        import_golden.py
        generate_led_approx.py
        search_candidates.py
        render_candidate_report.py

    tests/
        test_colorimetry.py
        test_mixing.py
        test_illuminants.py
        test_search.py
```

### First milestone code path

The first working version should be able to:

```text
load two reflectance curves
load or generate an LED illuminant
compute XYZ/Lab/sRGB
compute ΔE between two paints under two lights
display the result
```

This is the minimum viable colour engine.

---

## 17. Candidate Data Model

A candidate metameric pair should be saved with all necessary metadata.

Example:

```json
{
  "candidate_id": "cand_0001",
  "paint_a": {
    "recipe": [
      {"pigment": "PY35", "amount": 0.7},
      {"pigment": "PR108", "amount": 0.3}
    ],
    "predicted_reflectance_file": "cand_0001_A.csv"
  },
  "paint_b": {
    "recipe": [
      {"pigment": "PO20", "amount": 1.0}
    ],
    "predicted_reflectance_file": "cand_0001_B.csv"
  },
  "conceal_light": {
    "source": "SK6812_RGBW_approx",
    "channels": {"r": 0.42, "g": 0.61, "b": 0.18, "w": 0.75}
  },
  "reveal_light": {
    "source": "SK6812_RGBW_approx",
    "channels": {"r": 1.0, "g": 0.15, "b": 0.05, "w": 0.0}
  },
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

## 18. Success Criteria

### Software success

The software is successful if it can:

```text
import pigment reflectance curves
model RGB/RGBW illumination
compute perceptual colour under arbitrary illumination
search for candidate metameric pairs
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

The painting should not simply say “this is metamerism.” It should use metamerism as a material condition: a latent image, hidden structure, unstable monochrome, or controlled perceptual event.

---

## 19. Risks and Mitigations

### Risk 1 — Mixture predictions are inaccurate

Mitigation:

```text
treat software as candidate generator
test physical swatches early
add empirical calibration
keep mixture model pluggable
```

### Risk 2 — LED spectra are too approximate

Mitigation:

```text
start with approximate models
measure actual LED channels later
allow imported LED spectra
prefer RGBW or multi-channel fixtures
```

### Risk 3 — Screen simulation is misleading

Mitigation:

```text
show spectral plots and metrics
label previews as approximate
prioritise physical swatch testing
```

### Risk 4 — Effects only work in dark/neutral colours

Mitigation:

```text
add high-key constraints
add chroma/value constraints
rank candidates by artistic usefulness
search over LED states as well as pigments
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
avoid solutions requiring extremely dim or unpleasant lighting
```

---

## 20. Recommended Next Steps

### Step 1 — Build the colour engine

Implement:

```text
spectral data loading
standard wavelength grid
illuminant × reflectance calculation
XYZ/Lab/sRGB conversion
ΔE comparison
```

### Step 2 — Add approximate LED models

Create approximate spectral curves for:

```text
WS2812B RGB
SK6812 RGBW
D65
warm white LED
cool white LED
```

### Step 3 — Import pigment spectra

Import public GOLDEN Heavy Body pigment spectra into the software.

Normalise and resample them onto the standard wavelength grid.

### Step 4 — Implement first mixture models

Add:

```text
linear reflectance mixing
log-reflectance mixing
simple Kubelka–Munk-inspired mixing
```

### Step 5 — Build first search script

Search over simple two-pigment recipes and two LED states.

Initial target:

```text
ΔE_conceal < 2
ΔE_reveal as large as possible
```

### Step 6 — Generate candidate report

For each candidate, export:

```text
recipes
predicted colour panels
reflectance curves
illumination spectra
ΔE values
lighting settings
confidence warnings
```

### Step 7 — Paint first swatches

Use the highest-ranked candidates to create physical tests.

Start small and systematic.

### Step 8 — Measure and calibrate

Once a spectrophotometer is available, measure:

```text
single-pigment swatches
mixture swatches
LED channel spectra if possible
```

Then update the model.

---

## 21. Long-Term Vision

The long-term vision is a studio research tool that allows an artist to explore colour not as fixed surface appearance, but as a relationship between material, light, and observer.

The software should eventually allow the artist to ask:

```text
Find me two high-key mixtures that look identical under this white light
but separate under this RGBW lighting transition.
```

or:

```text
Given this measured paint surface, find the lighting state that reveals maximum contrast.
```

or:

```text
Given these paints in my studio, search for a latent image palette for a controlled-light painting.
```

The final artistic medium is not simply acrylic paint, nor simply LED light. It is the interaction between pigment spectra, programmable illumination, and human colour vision.
