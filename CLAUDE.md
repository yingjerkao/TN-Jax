# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

- Always open a PR instead of pushing directly to `main`.
- Merge PRs with `gh pr merge <number> --squash --delete-branch --auto` so CI must pass first.
- `main` has branch protection: `Tests (Python 3.11)` and `Tests (Python 3.12)` must pass.



## Documentation

- When adding new public API (algorithms, classes, functions), update both `README.md` (features list, example sections) and `src/tnjax/__init__.py` (`__all__` exports).
- Sphinx docs live in `docs/`; build with `cd docs && make html`.
- Keep `README.md` example code consistent with actual function signatures and test usage.

## Known Gotchas

- **NumPy ≥2.0**: Adding a Python `complex` scalar (even `1+0j`) into a `float64` array raises `UFuncOutputCastingError`. Extract `.real` or use `complex128` dtype explicitly.
- **Local tests**: `uv run pytest` fails on macOS x86_64 (jaxlib has no wheel for this platform). Tests only run reliably in CI.
