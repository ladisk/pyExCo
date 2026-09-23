"""Tests for the pyExCo control law.

No hardware, no scipy, no signal synthesis: every test drives the update rule
directly with arrays, which is the whole point of keeping the core in the
frequency domain. The plant is applied as a per-line power gain, ``S_response =
G * S_drive``, which is exactly what a linear time-invariant plant does to a PSD.
"""

import numpy as np
import pytest

import pyExCo
from pyExCo import UpdateInfo, error_db, metrics, update_psd


# ----------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------

N = 200
BAND = slice(20, 180)


def band_mask(n=N, sl=BAND):
    m = np.zeros(n, dtype=bool)
    m[sl] = True
    return m


def flat_target(n=N, sl=BAND, level=1.0):
    t = np.zeros(n)
    t[sl] = level
    return t


def sloped_target(n=N, sl=BAND):
    """A non-flat target, so tests cannot pass by accident on symmetry."""
    t = np.zeros(n)
    idx = np.arange(n)[sl]
    t[sl] = (idx / idx[0]) ** 1.7
    return t


def modal_gain(n=N, sl=BAND, seed=1):
    """|H|**2 of the 3-mode shaker/beam plant from 003_get_excitation.py.

    Ported from `simulated_plant`: the force transmitted through the stinger is
    what the armature does not spend accelerating itself, which produces the
    characteristic force drop-out at every resonance. Pure numpy.
    """
    freq = np.linspace(1.0, 1500.0, n)
    omega = 2.0 * np.pi * np.maximum(freq, 1e-6)
    mobility = np.zeros_like(omega, dtype=complex)
    for f_n, zeta in ((71.0, 0.004), (220.0, 0.004), (383.0, 0.005)):
        wn = 2.0 * np.pi * f_n
        mobility += (1j * omega / 0.3) / (wn ** 2 - omega ** 2 + 2j * zeta * wn * omega)
    coil = 1.0 / (1.0 + 1j * freq / 2000.0)
    H = coil / (1.0 + 1j * omega * 0.05 * mobility)
    g = np.abs(H) ** 2
    g[~band_mask(n, sl)] = 0.0
    return freq, g


def run_loop(gain, target, alpha, n_iter, noise_db=0.0, seed=0, **kw):
    """Closed loop against a static per-line power gain. Returns the error history."""
    rng = np.random.default_rng(seed)
    mask = target > 0
    drive = mask.astype(float)                 # flat seed, in band only
    hist = []
    for _ in range(n_iter):
        resp = gain * drive
        if noise_db:
            resp = resp * 10.0 ** (rng.normal(0.0, noise_db, resp.size) / 10.0)
        hist.append(metrics(error_db(resp, target)))
        drive = update_psd(drive, resp, target, alpha, **kw)
    return hist, drive


# ----------------------------------------------------------------------
# the control law
# ----------------------------------------------------------------------

def test_perfect_measurement_is_a_no_op():
    """A response already on target must not move the drive."""
    target = sloped_target()
    drive = np.where(target > 0, 3.0, 0.0)
    # Any level: normalize=True is blind to it.
    for scale in (1.0, 1e-6, 1e6):
        out = update_psd(drive, target * scale, target)
        assert np.allclose(out, drive, rtol=1e-12, atol=0.0)


def test_plant_cancels_in_one_step_at_alpha_one():
    """alpha=1 is the exact Newton step: |H|**2 cancels algebraically."""
    _, gain = modal_gain()
    target = sloped_target()
    hist, _ = run_loop(gain, target, alpha=1.0, n_iter=2,
                       max_step_db=1e3, max_shape_db=None)
    assert hist[0]["rms_dB"] > 5.0            # the plant really is in the way
    assert hist[1]["rms_dB"] < 1e-9           # ...and one step removes it


def test_error_decays_geometrically():
    """The log-error spread is multiplied by (1 - alpha) every iteration.

    Exactly, not approximately: the update is affine in log space and the p95-p5
    spread is invariant to the common offset the median normalisation introduces.
    """
    _, gain = modal_gain()
    target = sloped_target()
    for alpha in (0.3, 0.6, 0.9):
        hist, _ = run_loop(gain, target, alpha=alpha, n_iter=12,
                           max_step_db=1e3, max_shape_db=None)
        spread = np.array([h["spread_dB"] for h in hist])
        # Stop comparing once the spread has decayed into float noise -- at
        # alpha=0.9 that happens by iteration 8, below 1e-6 dB.
        live = spread[:-1] > 1e-6
        ratios = (spread[1:] / spread[:-1])[live]
        assert ratios.size >= 5
        assert np.allclose(ratios, 1.0 - alpha, rtol=1e-9)


def test_step_clamp_is_applied_before_the_relaxation_exponent():
    """The real per-iteration limit is alpha*max_step_db, not max_step_db.

    This pins the ordering inside update_psd: clip(ratio) ** alpha, never
    clip(ratio ** alpha). At the defaults that is 3.6 dB, not 6 dB.
    """
    alpha, max_step_db = 0.6, 6.0
    target = flat_target()
    drive = target.copy()

    for offset_db, sign in ((+40.0, -1.0), (-40.0, +1.0)):
        resp = target.copy()
        resp[100] = 10.0 ** (offset_db / 10.0)
        out = update_psd(drive, resp, target, alpha,
                         max_step_db=max_step_db, max_shape_db=None)
        step = 10.0 * np.log10(out[100] / drive[100])
        assert np.isclose(step, sign * alpha * max_step_db, atol=1e-12)
        # ...and every untouched line stays exactly put.
        others = np.ones(N, dtype=bool)
        others[100] = False
        assert np.allclose(out[others & (target > 0)],
                           drive[others & (target > 0)], rtol=1e-12)


def test_dead_bin_gives_a_finite_bounded_correction():
    """An identically-zero response line must not produce inf or nan."""
    target = flat_target()
    drive = target.copy()
    resp = target.copy()
    resp[100] = 0.0

    # With the clamp wide open the floor alone bounds the step at
    # (1/floor_frac)**alpha.
    out = update_psd(drive, resp, target, 0.6, floor_frac=1e-2,
                     max_step_db=1e3, max_shape_db=None)
    assert np.all(np.isfinite(out))
    assert np.isclose(out[100] / drive[100], 100.0 ** 0.6, rtol=1e-12)

    # With the default clamp, the clamp is what binds.
    out = update_psd(drive, resp, target, 0.6, max_step_db=6.0, max_shape_db=None)
    assert np.isclose(10.0 * np.log10(out[100] / drive[100]), 0.6 * 6.0, atol=1e-12)


def test_anti_windup_bounds_a_line_that_never_responds():
    """100 iterations against a dead line must not run the drive away."""
    _, gain = modal_gain()
    gain[100] = 0.0                       # a line the shaker simply cannot excite
    target = sloped_target()
    max_shape_db = 30.0
    _, drive = run_loop(gain, target, alpha=0.6, n_iter=100,
                        max_shape_db=max_shape_db)

    mask = target > 0
    span = drive[mask].max() / np.median(drive[mask])
    assert span <= 10.0 ** (max_shape_db / 10.0) * (1 + 1e-9)
    assert np.all(np.isfinite(drive))

    # Without the guard the same run diverges by orders of magnitude more.
    _, runaway = run_loop(gain, target, alpha=0.6, n_iter=100, max_shape_db=None)
    assert runaway[mask].max() / np.median(runaway[mask]) > 1e6


def test_out_of_band_lines_are_exactly_zero():
    target = sloped_target()
    drive = np.full(N, 7.0)               # nonzero everywhere, including out of band
    resp = np.full(N, 3.0)
    out = update_psd(drive, resp, target)
    assert np.all(out[target <= 0.0] == 0.0)
    assert np.all(out[target > 0.0] > 0.0)


def test_explicit_mask_overrides_the_target_default():
    target = flat_target()
    drive = np.full(N, 1.0)
    resp = np.full(N, 1.0)
    narrow = np.zeros(N, dtype=bool)
    narrow[50:60] = True
    out = update_psd(drive, resp, target, mask=narrow)
    assert np.all(out[~narrow] == 0.0)
    assert np.all(out[narrow] > 0.0)


# ----------------------------------------------------------------------
# shape-only versus absolute level
# ----------------------------------------------------------------------

def test_normalize_true_is_blind_to_response_level():
    _, gain = modal_gain()
    target = sloped_target()
    drive = (target > 0).astype(float)
    resp = gain * drive
    base = update_psd(drive, resp, target)
    for c in (1e-9, 0.5, 1e9):
        assert np.allclose(update_psd(drive, c * resp, target), base, rtol=1e-12)
        assert np.allclose(error_db(c * resp, target), error_db(resp, target),
                           rtol=1e-12, equal_nan=True)


def test_normalize_false_reaches_the_absolute_level():
    """Absolute-unit control: one Newton step lands the response ON the target."""
    _, gain = modal_gain()
    target = sloped_target() * 4.2e-3          # a level, not just a shape
    mask = target > 0
    drive = mask.astype(float) * 17.0          # deliberately the wrong level

    for _ in range(3):
        resp = gain * drive
        drive = update_psd(drive, resp, target, alpha=1.0, normalize=False,
                           max_step_db=1e3, max_shape_db=None)
    assert np.allclose((gain * drive)[mask], target[mask], rtol=1e-9)

    # ...and it is NOT level-blind, unlike normalize=True.
    resp = gain * drive
    a = update_psd(drive, resp, target, normalize=False)
    b = update_psd(drive, 100.0 * resp, target, normalize=False)
    assert not np.allclose(a[mask], b[mask])


def test_normalize_true_ignores_level_that_normalize_false_chases():
    _, gain = modal_gain()
    target = sloped_target()
    drive = (target > 0).astype(float)
    resp = 1000.0 * target                     # right shape, wrong level by 30 dB
    mask = target > 0
    assert np.allclose(update_psd(drive, resp, target, normalize=True), drive,
                       rtol=1e-9)
    out = update_psd(drive, resp, target, normalize=False, max_step_db=1e3,
                     max_shape_db=None)
    assert out[mask].max() < drive[mask].max()   # pulled down towards the target


# ----------------------------------------------------------------------
# alpha and estimator noise -- why the default is 0.6 and not 1.0
# ----------------------------------------------------------------------

def test_relaxation_keeps_estimator_noise_out_of_the_drive():
    """With a noisy estimator, alpha=1 freezes the noise; alpha<1 averages it out.

    Deterministic (fixed seed), and the margin is wide -- the point is the sign of
    the effect, not the exact ratio alpha/(2-alpha).
    """
    _, gain = modal_gain()
    target = sloped_target()
    mask = target > 0
    residual = {}
    for alpha in (1.0, 0.6):
        _, drive = run_loop(gain, target, alpha=alpha, n_iter=40, noise_db=1.0,
                            seed=7, max_step_db=1e3, max_shape_db=None)
        ideal = target / np.where(mask, gain, 1.0)
        r = np.log10(drive[mask] / ideal[mask])
        residual[alpha] = float(np.std(r - r.mean()))
    assert residual[0.6] < 0.75 * residual[1.0]


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------

def test_update_info():
    target = flat_target()
    drive = target.copy()
    resp = target.copy()
    resp[100] = 1e6            # far above -> step clamp
    resp[101] = 0.0            # dead      -> floor, then step clamp

    out, info = update_psd(drive, resp, target, return_info=True)
    assert isinstance(info, UpdateInfo)
    n_band = int((target > 0).sum())
    assert np.isclose(info.step_clamped_fraction, 2.0 / n_band)
    assert np.isclose(info.floored_fraction, 1.0 / n_band)
    assert info.shape_clamped_fraction == 0.0
    assert np.all(np.isfinite(out))

    # Nothing clamped when nothing needs clamping.
    _, info = update_psd(drive, target, target, return_info=True)
    assert info == UpdateInfo(0.0, 0.0, 0.0)


def test_shape_clamp_is_reported():
    _, gain = modal_gain()
    gain[100] = 0.0
    target = sloped_target()
    _, drive = run_loop(gain, target, alpha=0.6, n_iter=60, max_shape_db=30.0)
    resp = gain * drive
    _, info = update_psd(drive, resp, target, max_shape_db=30.0, return_info=True)
    assert info.shape_clamped_fraction > 0.0


def test_error_db_sign_and_out_of_band_nan():
    target = flat_target()
    resp = target.copy()
    resp[100] = 10.0 ** 0.6          # +6 dB, but the median normalisation moves it
    err = error_db(resp, target, normalize=False)
    assert np.isclose(err[100], 6.0)
    assert np.isnan(err[:20]).all() and np.isnan(err[180:]).all()
    assert np.allclose(err[(target > 0)][:50], 0.0, atol=1e-12)


def test_metrics_values():
    err = np.full(N, np.nan)
    err[BAND] = 0.0
    err[100] = 8.0
    err[101] = -4.0
    freq = np.linspace(0.0, 1000.0, N)
    m = metrics(err, freq=freq, alarm_db=3.0, abort_db=6.0)
    assert m["n_bins"] == 160
    assert m["max_dB"] == 8.0
    assert m["f_max"] == freq[100]
    assert m["n_alarm"] == 2 and m["n_abort"] == 1
    assert np.isclose(m["rms_dB"], np.sqrt((64.0 + 16.0) / 160.0))
    assert m["spread_dB"] >= 0.0
    assert "f_max" not in metrics(err)          # only when freq is supplied


def test_metrics_spread_is_the_usual_stop_criterion():
    """A converged loop must actually report a small spread."""
    _, gain = modal_gain()
    target = sloped_target()
    hist, _ = run_loop(gain, target, alpha=0.6, n_iter=8)
    assert hist[0]["spread_dB"] > 1.5
    assert hist[-1]["spread_dB"] < 1.5          # 003's tol_dB, within its n_iter_max


# ----------------------------------------------------------------------
# hygiene
# ----------------------------------------------------------------------

def test_inputs_are_never_mutated():
    target = sloped_target()
    drive = np.where(target > 0, 2.0, 0.0)
    resp = np.where(target > 0, 5.0, 0.0)
    d0, r0, t0 = drive.copy(), resp.copy(), target.copy()
    update_psd(drive, resp, target)
    error_db(resp, target)
    assert np.array_equal(drive, d0)
    assert np.array_equal(resp, r0)
    assert np.array_equal(target, t0)


def test_integer_input_is_accepted():
    target = np.zeros(N, dtype=int)
    target[BAND] = 1
    drive = target.copy()
    resp = target.copy()
    out = update_psd(drive, resp, target)
    assert out.dtype == np.float64


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(alpha=0.0), "alpha"),
        (dict(alpha=1.5), "alpha"),
        (dict(max_step_db=0.0), "max_step_db"),
        (dict(max_shape_db=-1.0), "max_shape_db"),
        (dict(floor_frac=0.0), "floor_frac"),
        (dict(floor_frac=2.0), "floor_frac"),
    ],
)
def test_rejects_bad_parameters(kwargs, match):
    target = flat_target()
    with pytest.raises(ValueError, match=match):
        update_psd(target, target, target, **kwargs)


def test_rejects_bad_spectra():
    target = flat_target()
    with pytest.raises(ValueError, match="same frequency grid"):
        update_psd(target, target[:-1], target)
    with pytest.raises(ValueError, match="1-D"):
        update_psd(np.ones((4, 4)), target, target)
    with pytest.raises(ValueError, match="power spectrum"):
        update_psd(target, -target, target)
    with pytest.raises(ValueError, match="NaN or inf"):
        update_psd(target, np.full(N, np.nan), target)
    with pytest.raises(ValueError, match="selects no lines"):
        update_psd(target, target, np.zeros(N))
    with pytest.raises(ValueError, match="in-band median of response_psd"):
        update_psd(target, np.zeros(N), target)


def test_public_api_is_four_names():
    assert set(pyExCo.__all__) == {
        "update_psd", "error_db", "metrics", "UpdateInfo", "__version__"
    }
    assert pyExCo.__version__ == "0.1.0"
