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


