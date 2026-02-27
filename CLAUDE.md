# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

- Always open a PR instead of pushing directly to `main`.
- Merge PRs with `gh pr merge <number> --squash --delete-branch --auto` so CI must pass first.
- `main` has branch protection: `Tests (Python 3.11)` and `Tests (Python 3.12)` must pass.
- Branch protection requires the PR branch to be up-to-date with `main`. If behind, merge main into the branch (`git merge origin/main`) rather than rebasing — rebase can get stuck on `--continue`.



## Documentation

- When adding new public API (algorithms, classes, functions), update both `README.md` (features list, example sections) and `src/tnjax/__init__.py` (`__all__` exports).
- Sphinx docs live in `docs/`; build with `cd docs && make html`.
- Keep `README.md` example code consistent with actual function signatures and test usage.

## Known Gotchas

- **Float64 default / JAX x64**: All tensors and algorithms default to `float64`. Importing `tnjax` enables x64 automatically via `jax.config.update("jax_enable_x64", True)`. CI tests that import `tnjax` will have x64 enabled. Tests that use JAX directly without importing `tnjax` should use `jax.config.x64_enabled` to pick the expected dtype.
- **MPO index convention**: `W[w_l, ket, bra, w_r]`. The DMRG effective-Hamiltonian matvec einsum is `"abc,apqd,bpse,eqtf,dfg->cstg"` where `p,q` are ket and `s,t` are bra physical indices.
- **NumPy ≥2.0**: Adding a Python `complex` scalar (even `1+0j`) into a `float64` array raises `UFuncOutputCastingError`. Extract `.real` or use `complex128` dtype explicitly.
- **Local tests**: `uv run pytest` may fail on macOS x86_64 if jaxlib has no wheel. CI does not enable JAX x64, so default `float64` is truncated to `float32` at runtime.
- **Odd-circumference cylinders**: `build_bulk_mpo_heisenberg_cylinder` only accepts even `Ly`. Odd circumference frustrates AFM order on the square lattice.
