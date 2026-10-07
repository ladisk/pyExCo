# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**pyExCo** — "excitation control for closed-loop random vibration testing: the
PSD update rule, and nothing else." A tiny, dependency-light Python library
(numpy only) implementing one control law for shaker-table random-vibration
testing:

```
S_drive <- S_drive * clip(S_target / S_response) ** alpha
```

Given the drive PSD you played, the response PSD you measured, and the
response PSD you wanted, `update_psd` returns the drive PSD to play next. It
is Newton's method in the log domain: for a linear time-invariant plant
(`S_response = |H|^2 * S_drive`), the unknown `|H|^2` cancels algebraically at
`alpha=1` and the loop lands on the exact answer in one step. `alpha=0.6`
(the default) trades convergence speed for averaging out Welch-estimator
scatter.

The package is deliberately narrow: no sampling rate, no frequency vector, no
`nperseg` anywhere in the core API. The three PSDs (drive, response, target)
arrive on one common frequency grid and everything is per-line arithmetic.
Estimating spectra (`scipy.signal.welch`), designing target profiles,
synthesizing waveforms (`pyExSi`), talking to hardware, and running the
iteration loop are all explicitly left to the caller — see README.rst for the
rationale ("Why frequency domain, and not time domain?").

SISO only (single shaker). Multi-shaker / cross-spectral control is explicitly
out of scope ("a genuinely different problem and a different package").

## Public API (4 names, all in `pyExCo/__init__.py` re-exporting from `_core.py`)

- `update_psd(drive_psd, response_psd, target_psd, alpha=0.6, *, mask=None, normalize=True, max_step_db=6.0, max_shape_db=30.0, floor_frac=1e-2, return_info=False)` — the control step.
- `error_db(response_psd, target_psd, *, mask=None, normalize=True)` — per-line error in power dB, same normalization convention as `update_psd`.
- `metrics(err_db, *, freq=None, mask=None, alarm_db=3.0, abort_db=6.0)` — summary dict (rms/spread/max dB, worst frequency if `freq` given).
- `UpdateInfo` — `NamedTuple(step_clamped_fraction, shape_clamped_fraction, floored_fraction)`, returned by `update_psd` when `return_info=True`; diagnoses whether the loop is fighting something structural.

Conventions worth knowing before touching `_core.py`:
- **Power dB throughout** (`10*log10`), never amplitude dB (`20*log10`).
- `normalize=True` (default) divides drive/response/target by their own
  in-band median first, so the loop controls *shape* only, blind to overall
  level.
- In-band mask defaults to `target_psd > 0` (a target is zero outside its
  control band by construction); pass `mask=` to override.
- `max_step_db` clamps the ratio *before* the `alpha` exponent — the real
  per-iteration step limit is `alpha * max_step_db`, not `max_step_db`.

## Relationship to other projects

pyExCo is standalone, public (`ladisk/pyExCo`) and installable on its own
(`pip install pyexco`). It does not import or depend on any other project.
Downstream projects (for example the SIgMA measurement code) use it as an
ordinary dependency, either from PyPI or from git; developing it against such
a project means pointing that project's `[tool.uv.sources]` entry at a local
editable path.

- README/docstrings describe the intended caller as
  [LDAQ](https://github.com/ladisk/LDAQ) (data-acquisition library by the
  same author group): the caller owns profile design, hardware and the loop;
  the control law lives here.
- `pyproject.toml` deliberately pins `requires-python = ">=3.9"` and nothing
  newer-than-3.9 syntax "so this dependency never forces its dependents up a
  version" — a direct nod to being embedded in other projects' toolchains.
- Because this is a published package, keep changes minimal and
  backwards-compatible; the public API is the four names above.

## Repo layout

```
pyExCo/
  __init__.py       public API re-export + package docstring/example, __version__
  _core.py          the whole implementation: update_psd, error_db, metrics, UpdateInfo
tests/
  test_core.py      full test suite, pure-array, no hardware/scipy
examples/
  basic_example.py  self-contained closed-loop demo against a simulated SDOF plant
docs/
  source/           Sphinx docs (index.rst, getting_started.rst [includes README+CONTRIBUTING], code.rst [automodule])
sync_version.py     keeps version in sync across pyproject.toml, __init__.py, docs/source/conf.py
.readthedocs.yaml   Read the Docs build config (Python 3.12, installs the `docs` extra)
.github/workflows/
  python-package.yml              CI: flake8 + pytest across Python 3.9-3.14
  release-and-publish-to-pypi.yml tag-triggered release: sync_version --set-version, build, GitHub Release, PyPI publish (Trusted Publishing/OIDC)
```

## Dev environment setup

Uses **uv** + **hatchling** as the build backend (`pyproject.toml`:
`[build-system] requires = ["hatchling"]`). Package name on PyPI is `pyexco`.

```bash
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

`dev` extra pulls in `pytest`, `build`, `twine`, `flake8`, and the `docs`
extra. Runtime dependency is `numpy>=1.23` only — no scipy (the package does
no spectral estimation).

## Running tests

```bash
pytest
```

`pytest.ini_options` sets `testpaths = ["tests", "pyExCo"]` and
`addopts = "--doctest-modules"` — the docstring examples in `pyExCo/_core.py`
are collected and run as tests too, alongside `tests/test_core.py`. No
hardware, no scipy needed; the plant is simulated as a per-line power gain.

Run the example script (not part of the test suite, but illustrative) from the
project root:

```bash
python -m examples.basic_example
```

## Building docs

Sphinx docs live under `docs/`, using `sphinx-book-theme` and
`sphinx-copybutton` (install via `pip install -e ".[docs]"`).

```bash
cd docs
make clean
make html
```

Read the Docs builds from `.readthedocs.yaml` (Python 3.12, `pip install .[docs]`,
`fail_on_warning: false`). `getting_started.rst` simply `.. include::`s the
top-level `README.rst` and `CONTRIBUTING.rst`; `code.rst` autodocuments the
four public names via `automodule`.

## Versioning

`sync_version.py` is the source-of-truth syncer: it reads `version` from the
`[project]` table of `pyproject.toml` (treating the file as plain text so
comments/formatting survive) and rewrites `pyExCo/__init__.py`'s
`__version__` and `docs/source/conf.py`'s `version`/`release` to match.

```bash
python sync_version.py --bump patch|minor|major   # bump and sync
python sync_version.py --set-version X.Y.Z         # set and sync
python sync_version.py                             # just sync, no change
```

The release workflow (`.github/workflows/release-and-publish-to-pypi.yml`)
runs `sync_version.py --set-version <tag>` on a `v*` tag push, commits the
sync if needed, builds, creates a GitHub Release, and publishes to PyPI via
Trusted Publishing (no API token secret).

## Contributing conventions (from CONTRIBUTING.rst)

- One bug-fix or enhancement per pull request.
- Standard fork -> branch -> `uv venv` + `uv pip install -e ".[dev]"` -> fix
  -> update/add tests -> `pytest` -> update docs and check `cd docs && make clean && make html`
  -> commit -> push -> PR workflow.
- A PR should preferably be a single commit on top of current `main` (rebase
  and squash before submitting).
- CI (`python-package.yml`) runs flake8 (hard errors on `E9,F63,F7,F82`; style
  checks are informational, `--exit-zero`, max line length 127) and pytest
  across Python 3.9 through 3.14 — keep changes compatible with 3.9 syntax.
