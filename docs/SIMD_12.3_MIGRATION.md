# Pillow-SIMD 12.3 migration plan

Branch: `simd/12.3.x` (base: upstream Pillow **12.3.0**)

Status: **WIP** — merge started, partial SIMD port. Do not use in production yet.

## Staging benchmark baseline (2026-07-06)

Measured on `platform-toolbox` (staging, x86_64), image 2560×1600 RGB:

| Operation | pillow-simd 9.5 | stock Pillow 9.5 | stock Pillow 12.3 |
|---|---:|---:|---:|
| resize BICUBIC → 1536×864 | 307 Mpx/s | 73 Mpx/s | 70 Mpx/s |
| resize LANCZOS → 4109×2311 | 46 Mpx/s | 16 Mpx/s | 15 Mpx/s |
| gaussian blur r=10 | 51 Mpx/s | 22 Mpx/s | 17 Mpx/s |

**Target:** restore ≥90% of 9.5 SIMD throughput on the same ops after port.

---

## Work breakdown (tickets)

### Phase 1 — Core port (pillow-simd repo)

| ID | Title | Scope | Est. | Status |
|---|---|---|---|---|
| SIMD-1 | Branch `simd/12.3.x` from Pillow 12.3.0 | git, version `12.3.0.post0` | 0.5d | **done (WIP)** |
| SIMD-2 | Infra merge | `_version.py`, `setup.py` `-msse4`, `MANIFEST.in`, drop `setup.cfg` | 0.5d | **done** |
| SIMD-3 | Bands SSE4 | `Bands.c` — shuffle_epi8 getband/split | 0.5d | **done** |
| SIMD-4 | Filter SSE4/AVX2 | `Filter.c` dispatch + `FilterSIMD_*.c` | 1d | **done** |
| SIMD-5 | Convert SSE4/AVX2 | `Convert.c` — merged cleanly from 9.5 | 0.5d | **done** |
| SIMD-6 | **Resample SSE4/AVX2** | Port SIMD hooks into `_ImagingResampleHorizontal_8bpc` / vertical; 12.3 changed coef types (`INT32`/`normalize_coeffs_8bpc` return) | 2–3d | **TODO** |
| SIMD-7 | **Reduce SSE4** | `Reduce.c` heavily rewritten in 10–12.x | 1–2d | **TODO** |
| SIMD-8 | **BoxBlur SSE4** | `BoxBlur.c` — new sequential-box path in upstream | 1d | **TODO** |
| SIMD-9 | **AlphaComposite SSE4/AVX2** | Reconcile with Pillow 12 `rgba8` + restrict refactor | 1–2d | **TODO** |
| SIMD-10 | **ColorLUT SSE4/AVX2** | Port overflow fixes from simd/master | 1d | **TODO** |
| SIMD-11 | Build & CI | `pip install -e .`, Pillow test suite, `-mavx2` job in CI | 1–2d | **TODO** |
| SIMD-12 | Release | PyPI `12.3.0.post0`, git tag, release notes | 0.5d | **TODO** |

### Phase 2 — Platform integration (uploadcare/platform)

| ID | Title | Scope | Est. | Depends on |
|---|---|---|---|---|
| PLAT-1 | Bump pillow-simd pin | `pyproject.toml` git rev → `simd/12.3.x` tag | 0.5d | SIMD-12 |
| PLAT-2 | Update pip-stubs | `pip-stubs/pillow` → Pillow 12.3 API surface | 0.5d | SIMD-12 |
| PLAT-3 | `uv lock` + docker build | `Dockerfile-platform` compiles from source on x86_64 | 1d | PLAT-1 |
| PLAT-4 | Platform tests | functional + monkeypatches + filelens probes | 1–2d | PLAT-3 |
| PLAT-5 | Staging deploy + perf | Re-run benchmark script on toolbox; compare to baseline | 0.5d | PLAT-4 |
| PLAT-6 | Production rollout | Canary workers → full rollout | 1d | PLAT-5 |

### Phase 3 — Long-term (optional)

| ID | Title | Notes |
|---|---|---|
| UP-1 | Push Pillow PR #8209 | Merge SIMD into upstream; retire fork |
| UP-2 | Runtime CPU dispatch | SSE4 vs AVX2 without recompile (`CC="cc -mavx2"`) |

---

## PR sequence (pillow-simd)

1. **PR-1:** `simd/12.3.x` WIP scaffold (this commit)
   - Branch, version bump, Bands/Filter/Convert SIMD, docs
   - Resample includes added; hooks not wired yet

2. **PR-2:** Resample + Reduce (highest CDN impact)

3. **PR-3:** BoxBlur + AlphaComposite + ColorLUT

4. **PR-4:** CI green + `12.3.0.post0` release

5. **PR-5 (platform):** Pin + staging validation

---

## How to build locally

```bash
git checkout simd/12.3.x
pip uninstall -y pillow pillow-simd
# SSE4 (default):
pip install -U --force-reinstall .
# AVX2:
CC="cc -mavx2" pip install -U --force-reinstall .
```

## How to benchmark (staging)

```bash
POD=$(kubectl get pods -n platform -l app.kubernetes.io/instance=platform-toolbox -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n platform "$POD" -- python -c "
import time
from PIL import Image, ImageFilter
SIZE=(2560,1600); im=Image.new('RGB',SIZE); runs=11
def bench(name, fn):
    t=sorted(time.perf_counter() or fn() or time.perf_counter() for _ in range(runs))
    med=t[runs//2]; print(f'{name}: {med:.4f}s {SIZE[0]*SIZE[1]/med/1e6:.1f} Mpx/s')
print('Pillow', Image.__version__)
bench('resize BICUBIC', lambda: im.resize((1536,864), Image.BICUBIC))
bench('blur r=10', lambda: im.filter(ImageFilter.GaussianBlur(10)))
"
```

---

## Risk register

| Risk | Mitigation |
|---|---|
| Resample port breaks edge sizes | `Tests/test_image_resample.py`, pillow-perf `scale` suite |
| Security regressions vs Pillow 12.3 | Stay on upstream 12.3.0 base; only add SIMD diffs |
| Platform ARM builds | Keep `pillow==9.5.0` stub path for non-x86 until stock Pillow 12 on ARM |
| perf regression after port | Block release if staging Mpx/s < 80% of 9.5 baseline |

## Owners

- **SIMD C port:** homm / platform infra (C + Pillow experience)
- **Platform integration:** platform team
- **Release sign-off:** staging benchmarks + PLAT-4 tests green
