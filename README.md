# Metamerism

Experimental Python package for exploring metameric paint-light systems.

The project models:

- pigment and paint reflectance spectra;
- programmable RGB/RGBW illumination spectra;
- colour matching / observer projection;
- approximate paint mixture models;
- search for metameric paint pairs under controlled illumination.

## Development

This repository uses `uv`, `ruff` and `ty`.

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check
```

Notes on colour handling

- This project uses colour-science for all numerical colour and spectral
  operations. Use `colour` (https://www.colour-science.org/) for conversions
  (sd_to_XYZ, XYZ_to_Lab, XYZ_to_sRGB, delta_E, etc.).
- Use `metamerism` for provenance-aware loading, domain semantics, and
  project-specific workflows only.

Golden demo

```bash
uv run python scripts/demo_golden_paint.py --paint "Alizarin Crimson Hue"
```

The demo loads a GOLDEN Heavy Body reflectance spectrum, plots the line
chart, and shows a D65 preview swatch.

