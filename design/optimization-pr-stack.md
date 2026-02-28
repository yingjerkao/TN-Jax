# Optimization PR Stack — EXECUTED

All 5 PRs merged. See #22–#26.

## Context

Benchmarks revealed 5 remaining optimization hotspots after previous performance work landed. This plan creates a stack of 5 independent PRs (one per fix), each small and reviewable.

---

## PR 1: Fix network cache-key correctness

**Problem:** `network.py:360` uses `frozenset(nodes)` in the cache key, but output label order depends on node iteration order when `output_labels=None`. Two calls with the same nodes in different order return the same cached result with wrong leg ordering.

**File:** `src/tnjax/network/network.py`

**Change (line 360):**
```python
# Before:
cache_key = (frozenset(nodes), tuple(output_labels or ()), optimize)
# After:
cache_key = (tuple(nodes), tuple(output_labels or ()), optimize)
```

**Test:** Add a test in `tests/test_network.py` contracting nodes `['A','B']` vs `['B','A']` with `output_labels=None`, asserting different label orders in the results.

---

## PR 2: Remove duplicate SVD in DMRG truncation

**Problem:** `dmrg.py:786` calls `truncated_svd`, then `dmrg.py:804` does a second full SVD just for truncation error. The first SVD already computes all singular values at `contractor.py:1039`.

**Files:**
- `src/tnjax/contraction/contractor.py` — `truncated_svd()` (line ~1039)
- `src/tnjax/algorithms/dmrg.py` — `_svd_and_truncate_site()` (line ~786)

**Changes:**
1. In `truncated_svd`, save `s_full = s` before truncation (line 1039-1064). Return `(U_tensor, s_truncated, Vh_tensor, s_full)` — 4-tuple instead of 3.
2. Update the single caller in `_svd_and_truncate_site` to unpack the 4th element and compute truncation error from `s_full` directly, removing lines 798-804.

**Test:** Existing DMRG tests cover correctness. Add a unit test for `truncated_svd` verifying `s_full` contains all singular values while `s_truncated` is properly truncated.

---

## PR 3: JIT-compile HOTRG step functions

**Problem:** `hotrg.py` step functions (`_hotrg_step_horizontal`, `_hotrg_step_vertical`, `_compute_hosvd_isometry`) lack `@jax.jit`, causing recompilation overhead. Benchmark: HOTRG 13.39s vs TRG 0.013s.

**File:** `src/tnjax/algorithms/hotrg.py`

**Changes:**
1. Mark `_hotrg_step_horizontal` and `_hotrg_step_vertical` with `@jax.jit` using `static_argnums=(1,)` for `max_bond_dim`.
2. Mark `_compute_hosvd_isometry` with `@jax.jit` using `static_argnums=(1, 2)` for `axis` and `chi_target`.
3. The conditional `if T_new.shape[0] > max_bond_dim` (lines 143, 197) needs care — shape is known at trace time since `max_bond_dim` is static, so this should work as-is.

**Test:** Existing HOTRG tests + benchmark comparison. Expect ~10-50x improvement.

---

## PR 4: JIT CTM sweep loop for iPEPS

**Problem:** `ipeps.py:924` runs CTM iterations in a Python loop with `float()` host sync at line 940 for convergence checking, preventing JIT compilation.

**File:** `src/tnjax/algorithms/ipeps.py`

**Changes:**
1. Extract the 4-move sweep body (left/right/top/bottom + optional renormalize) into a helper.
2. Replace the Python `for` loop + `float()` convergence check with `jax.lax.while_loop`:
   - Carry: `(env, prev_sv, iteration, converged)`
   - Condition: `~converged & (iteration < max_iter)` (all JAX arrays)
   - Body: one full sweep + JAX-native convergence check (no `float()`)
3. Same treatment for `ctm_2site` (lines ~1357-1358) which has the same pattern.

**Test:** Existing iPEPS tests. Benchmark `ipeps` and `ipeps_ad` before/after.

---

## PR 5: Batch excitation H/N assembly with vmap

**Problem:** `ipeps_excitations.py:521` loops over `basis_size` (D^4*d) gradient evaluations serially, writing columns to NumPy arrays one at a time.

**File:** `src/tnjax/algorithms/ipeps_excitations.py`

**Changes in `_build_H_and_N` (lines 503-532):**
1. Stack basis tensors: `B_stacked = jnp.stack(basis)` — shape `(basis_size, D, D, D, D, d)`.
2. Replace the `for m in range(basis_size)` loop with:
   ```python
   H_grads = jax.vmap(jax.grad(energy_fn))(B_stacked)
   N_grads = jax.vmap(jax.grad(norm_fn))(B_stacked)
   ```
3. Reshape to matrices: `H_eff = np.array(H_grads.reshape(basis_size, basis_size))`.
4. Single host transfer at the end instead of per-iteration `np.array()`.

**Test:** Existing excitation tests. Verify H_eff and N_mat match previous values within tolerance.

---

## Execution order

PRs are independent — all can be developed in parallel on separate branches. Suggested merge order by risk (lowest first):

1. ~~PR 1 (cache-key) — 1-line fix + test~~ → merged as #22
2. ~~PR 2 (duplicate SVD) — small signature change~~ → merged as #23
3. ~~PR 3 (HOTRG JIT) — decorator additions~~ → merged as #24
4. ~~PR 5 (vmap excitations) — loop replacement~~ → merged as #26
5. ~~PR 4 (CTM while_loop) — most complex refactor~~ → merged as #25

## Verification

After each PR:
- `uv run pytest -q` — all 519 tests pass
- `uv run python -m benchmarks --algorithm <relevant> --size small` — compare timing
- For PR 1: add dedicated regression test for cache correctness
