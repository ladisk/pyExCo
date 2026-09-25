"""The excitation-control update rule.

One idea, three functions. Given the drive PSD you played, the response PSD you
measured, and the response PSD you wanted, produce the drive PSD to play next::

    S_drive <- S_drive * clip(S_target / S_response) ** alpha

Everything here is per-line arithmetic on three arrays that live on one common
frequency grid. There is deliberately no sampling rate, no frequency vector and
no window length in any signature: the update has no notion of time, so it does
not take a time axis. Estimating the PSDs, designing the target, synthesizing a
waveform from the result and running the loop all belong to the caller.


Conventions
-----------
POWER dB throughout, i.e. ``10*log10(S)``. A PSD is a power quantity, so the
"6 dB" in ``max_step_db`` is a factor of ``10**0.6 = 3.98`` in PSD, not in
amplitude. Mixing this up with the amplitude convention (``20*log10``) is the
classic way to get a controller that moves half as far as you think.

Shape versus level. With ``normalize=True`` (the default) both spectra are
divided by their own in-band median before they are compared, so the loop
controls the *shape* of the response and is blind to its overall level -- the
level is whatever the amplifier gain and the caller's drive scaling make it.
With ``normalize=False`` the loop drives the response to the target in real
engineering units.
"""

from typing import NamedTuple

import numpy as np

__all__ = ["update_psd", "error_db", "metrics", "UpdateInfo"]


class UpdateInfo(NamedTuple):
    """Diagnostics from one :func:`update_psd` call, all in-band fractions in [0, 1].

    Worth printing every iteration. A ``step_clamped_fraction`` that stays high
    means the loop is fighting something structural rather than converging, and a
    non-zero ``shape_clamped_fraction`` means some line is asking for a drive the
    anti-windup guard refuses to give it -- usually a line sitting on the noise
    floor, or a genuine anti-resonance the shaker cannot excite.
    """

    step_clamped_fraction: float
    shape_clamped_fraction: float
    floored_fraction: float


def _as_psd(x, name, n=None):
    """Validate a one-sided PSD: 1-D, real, finite, non-negative, right length."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if n is not None and a.size != n:
        raise ValueError(
            f"{name} has {a.size} lines but the other spectra have {n}; all three "
            "PSDs must be on the same frequency grid"
        )
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    if np.any(a < 0.0):
        raise ValueError(f"{name} contains negative values; it is a power spectrum")
    return a


def _resolve_mask(mask, target, n):
    """In-band boolean mask, defaulting to where the target is non-zero.

    A target profile is zero outside its band by construction, which makes it the
    natural definition of "in band" and is the reason no frequency vector is
    needed here.
    """
    if mask is None:
        m = target > 0.0
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (n,):
            raise ValueError(f"mask must have shape ({n},), got {m.shape}")
    if not m.any():
        raise ValueError(
            "the in-band mask selects no lines (the target is zero everywhere?)"
        )
    return m


def _in_band_median(a, mask, name):
    med = float(np.median(a[mask]))
    if med <= 0.0:
        raise ValueError(
            f"the in-band median of {name} is {med}; more than half its in-band "
            "lines are zero, so it carries no usable level"
        )
    return med


def update_psd(
    drive_psd,
    response_psd,
    target_psd,
    alpha=0.6,
    *,
    mask=None,
    normalize=True,
    max_step_db=6.0,
    max_shape_db=30.0,
    floor_frac=1e-2,
    return_info=False,
):
    """One relaxed, clamped control step: the drive PSD for the next iteration.

    Parameters
    ----------
    drive_psd : array_like, shape (M,)
        The excitation PSD that produced ``response_psd`` [drive_unit**2/Hz].
        Pass either the PSD you *designed* the drive from, or a Welch estimate of
        the waveform that was *actually played* -- see Notes, the two behave
        differently and the second is better conditioned.
    response_psd : array_like, shape (M,)
        The measured response PSD [response_unit**2/Hz], on the same grid.
    target_psd : array_like, shape (M,)
        The desired response PSD, on the same grid, zero outside the control band.
    alpha : float, default 0.6
        Relaxation exponent, in ``(0, 1]``. ``1.0`` is the full Newton step; the
        in-band log-error then decays as ``(1 - alpha)**k`` but the drive inherits
        all of the PSD estimator's noise. See Notes.
    mask : array_like of bool, shape (M,), optional
        Which lines are in band. Defaults to ``target_psd > 0``. Out-of-band lines
        of the returned PSD are exactly zero.
    normalize : bool, default True
        ``True``: divide both spectra by their own in-band median first, so the
        loop controls shape only. ``False``: control the absolute level too.
    max_step_db : float, default 6.0
        Clamp on the per-iteration correction *ratio*, in power dB. See Notes for
        why the resulting step limit is ``alpha * max_step_db``, not this.
    max_shape_db : float or None, default 30.0
        Clamp on the *total* drive shape about its own in-band median, in power
        dB. Anti-windup: without it, a line that never responds demands a
        correction every iteration and the drive runs away. ``None`` disables it.
    floor_frac : float, default 1e-2
        Floor the response at this fraction of its own in-band median before
        inverting it, so an empty bin asks for a large but finite correction.
        ``1e-2`` is -20 dB.
    return_info : bool, default False
        Also return an :class:`UpdateInfo` with the clamp and floor statistics.

    Returns
    -------
    numpy.ndarray, shape (M,)
        The drive PSD for the next iteration. A fresh array; the input is never
        modified.
    UpdateInfo
        Only when ``return_info=True``.

    Notes
    -----
    **Why it converges without a plant model.** For a linear time-invariant plant
    the response is ``S_resp = |H|**2 * S_drive``. Substituting that into the
    update at ``alpha = 1`` gives ``S_drive_new = S_target / |H|**2``, i.e. the
    exact answer in one step -- the unknown ``|H|**2`` cancels algebraically. It
    is Newton's method in the log domain, and the plant never has to be measured,
    stored or inverted. For ``alpha < 1`` the log-error is multiplied by
    ``(1 - alpha)`` each iteration.

    **Why the default is 0.6 and not 1.0.** A Welch estimate has a per-line
    scatter of roughly ``4.343 * sqrt(2/dof)`` dB, and ``alpha = 1`` freezes all
    of it into the drive. Repeated relaxed steps let it average out instead: the
    retained noise power settles at ``alpha / (2 - alpha)`` of the single-estimate
    value, so 0.6 keeps about 43 % of what 1.0 keeps, at the cost of a 0.4-per-
    iteration error decay instead of 0.

    **Clamp ordering.** ``max_step_db`` clamps the ratio *before* the ``alpha``
    exponent is applied, so the largest change the drive can actually make in one
    iteration is ``alpha * max_step_db`` -- 3.6 dB at the defaults, not 6 dB.
    This is deliberate: it keeps the clamp a statement about the *measurement*
    ("no single estimate may be trusted for more than 6 dB") rather than about the
    step, and it means changing ``alpha`` rescales the step limit with it.

    **Designed versus played drive PSD.** One finite realization of a random
    signal has its own chi-square scatter about the PSD it was designed from. If
    you pass the designed PSD, that scatter is attributed to the plant and enters
    the correction. If you instead pass a Welch estimate of the waveform actually
    played, computed over the same record, the scatter appears in both spectra and
    cancels -- the update becomes an H1 estimate of ``1/|H|**2``. That costs one
    extra Welch call and no extra hardware, since the played waveform is the array
    you sent to the DAC.

    **Zeros are absorbing, almost.** The update is multiplicative, so a line whose
    drive PSD is exactly zero can never be revived by the ratio alone. The
    ``max_shape_db`` clamp does revive it, by lifting every in-band line to at
    least ``median / 10**(max_shape_db/10)``. With ``max_shape_db=None`` a zeroed
    in-band line stays zero forever.

    Examples
    --------
    >>> import numpy as np
    >>> target = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    >>> drive = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    >>> response = np.array([0.0, 1.0, 2.0, 1.0, 0.0])   # one line 3 dB hot

    The full Newton step halves the drive on that line and leaves the rest alone:

    >>> np.round(update_psd(drive, response, target, alpha=1.0), 4)
    array([0. , 1. , 0.5, 1. , 0. ])

    Relaxed, it takes 60 % of the correction in the log domain:

    >>> np.round(update_psd(drive, response, target, alpha=0.6), 4)
    array([0.    , 1.    , 0.6598, 1.    , 0.    ])

    Note that ``max_step_db=6.0`` is a factor of ``10**0.6 = 3.98`` in PSD, not
    4 -- a line exactly 4x hot is already past the clamp.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if max_step_db <= 0.0:
        raise ValueError(f"max_step_db must be positive, got {max_step_db}")
    if max_shape_db is not None and max_shape_db <= 0.0:
        raise ValueError(f"max_shape_db must be positive or None, got {max_shape_db}")
    if not 0.0 < floor_frac <= 1.0:
        raise ValueError(f"floor_frac must be in (0, 1], got {floor_frac}")

    drive = _as_psd(drive_psd, "drive_psd")
    resp = _as_psd(response_psd, "response_psd", drive.size)
    targ = _as_psd(target_psd, "target_psd", drive.size)
    band = _resolve_mask(mask, targ, drive.size)

    # Floor the response before inverting it. Written as an absolute floor rather
    # than a floor on the normalised spectrum so that it means the same thing in
    # both `normalize` modes; the two are algebraically identical when normalising.
    resp_med = _in_band_median(resp, band, "response_psd")
    floor = floor_frac * resp_med
    floored = float(np.mean(resp[band] < floor))
    resp_f = np.maximum(resp, floor)

    if normalize:
        targ_med = _in_band_median(targ, band, "target_psd")
        ratio_raw = (targ / targ_med) / (resp_f / resp_med)
    else:
        ratio_raw = targ / resp_f

    r_max = 10.0 ** (max_step_db / 10.0)  # POWER dB -- see module docstring
    ratio = np.clip(ratio_raw, 1.0 / r_max, r_max)
    step_clamped = float(np.mean(ratio_raw[band] != ratio[band]))

    new = drive * ratio ** alpha
    new[~band] = 0.0

    shape_clamped = 0.0
    if max_shape_db is not None:
        med = float(np.median(new[band]))
        if med > 0.0:
            span = 10.0 ** (max_shape_db / 10.0)
            pre = new[band]
            post = np.clip(pre, med / span, med * span)
            shape_clamped = float(np.mean(pre != post))
            new[band] = post

    if return_info:
        return new, UpdateInfo(step_clamped, shape_clamped, floored)
    return new


def error_db(response_psd, target_psd, *, mask=None, normalize=True):
    """Per-line control error ``10*log10(S_response / S_target)`` [power dB].

    Out-of-band lines are ``nan``, which is what makes the result safe to feed
    straight to :func:`metrics` without repeating the mask.

    This uses the same ``normalize`` convention as :func:`update_psd`, and that is
    the reason it lives here rather than in the caller: an error metric computed
    under a different convention silently stops measuring what the controller is
    minimizing, and a stop criterion built on it stops meaning what it says.

    Parameters
    ----------
    response_psd, target_psd : array_like, shape (M,)
        Measured and desired response PSDs on a common grid.
    mask : array_like of bool, shape (M,), optional
        In-band lines. Defaults to ``target_psd > 0``.
    normalize : bool, default True
        ``True``: normalise both spectra by their in-band median first, so the
        error is blind to overall level. ``False``: absolute error.

    Returns
    -------
    numpy.ndarray, shape (M,)
        Error in dB in band, ``nan`` elsewhere. Positive means the response is
        above the target.
    """
    resp = _as_psd(response_psd, "response_psd")
    targ = _as_psd(target_psd, "target_psd", resp.size)
    band = _resolve_mask(mask, targ, resp.size)

    err = np.full(resp.size, np.nan)
    r = resp[band]
    t = targ[band]
    if normalize:
        r = r / _in_band_median(resp, band, "response_psd")
        t = t / _in_band_median(targ, band, "target_psd")
    err[band] = 10.0 * np.log10(np.maximum(r, 1e-30) / np.maximum(t, 1e-30))
    return err


def metrics(err_db, *, freq=None, mask=None, alarm_db=3.0, abort_db=6.0):
    """Summarise an error curve, in the vocabulary the random-vibration standards use.

    Reported, never enforced -- ``alarm_db`` and ``abort_db`` only count lines
    here. Deciding what to do about them is the caller's job.

    Parameters
    ----------
    err_db : array_like, shape (M,)
        Output of :func:`error_db`; ``nan`` marks out-of-band lines.
    freq : array_like, shape (M,), optional
        Frequency vector [Hz]. Supply it and the result gains ``f_max``, the
        frequency of the worst line. This is the only place in the package that
        accepts a frequency axis, and it is used for reporting only.
    mask : array_like of bool, shape (M,), optional
        In-band lines. Defaults to where ``err_db`` is finite.
    alarm_db, abort_db : float
        Thresholds on ``|err_db|`` for the ``n_alarm`` / ``n_abort`` counts.

    Returns
    -------
    dict
        ``spread_dB`` (p95 - p5, the usual convergence criterion), ``rms_dB``,
        ``max_dB``, ``n_alarm``, ``n_abort``, ``n_bins``, and ``f_max`` when
        ``freq`` was given.
    """
    e_full = np.asarray(err_db, dtype=float)
    if e_full.ndim != 1:
        raise ValueError(f"err_db must be 1-D, got shape {e_full.shape}")
    band = np.isfinite(e_full) if mask is None else np.asarray(mask, dtype=bool)
    e = e_full[band]
    if e.size == 0:
        raise ValueError("no in-band lines in err_db")
    if not np.all(np.isfinite(e)):
        raise ValueError("err_db has non-finite values inside the mask")

    out = {
        "spread_dB": float(np.percentile(e, 95) - np.percentile(e, 5)),
        "rms_dB": float(np.sqrt(np.mean(e ** 2))),
        "max_dB": float(np.max(np.abs(e))),
        "n_alarm": int(np.count_nonzero(np.abs(e) > alarm_db)),
        "n_abort": int(np.count_nonzero(np.abs(e) > abort_db)),
        "n_bins": int(e.size),
    }
    if freq is not None:
        f = np.asarray(freq, dtype=float)
        if f.shape != e_full.shape:
            raise ValueError(
                f"freq must have shape {e_full.shape}, got {f.shape}"
            )
        out["f_max"] = float(f[band][np.argmax(np.abs(e))])
    return out
