# AlphaJet

AI across the subsurface workflow, from seismic to economics: **alphajet.dev**.

Interactive, precomputed demos with explanations of how the AI works and how far to trust it (uncertainty, physics checks, surrogate-vs-simulator parity). Built on open data only.

## Layout
| Path | Purpose |
|---|---|
| `site/` | Static site (HTML/CSS/JS) plus minimal PHP in `includes/` and `unlisted/` |
| `pipeline/` | Python package `alphajet`: seismic, geomodel, fluids, pore pressure, simulation, surrogate, economics |
| `data/` | `raw/` and `interim/` are git-ignored; see `docs/data-licenses.md` |
| `outputs/web/` | Artifacts the site consumes (JSON, images, videos) |
| `docs/` | Data licenses, how-it-works content |
| `infra/` | S3/CloudFront deployment (to be added) |
| `PLAN.md` | Architecture, timeline, costs, GitHub split |

## Local development
```bash
# site
cd site && php -S localhost:8000

# pipeline
cd pipeline && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest
```

## Status
Phase 0: setup. See `PLAN.md`.
