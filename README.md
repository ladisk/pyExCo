# pyExCo

**Excitation control for closed-loop random vibration testing: the PSD update
rule, and nothing else.**

Given the drive PSD you played, the response PSD you measured, and the response
PSD you wanted, `update_psd` returns the drive PSD to play next:

$$S_\text{drive} \leftarrow S_\text{drive}\left(\frac{S_\text{target}}{S_\text{response}}\right)^{\alpha}$$

That is the package. It does not estimate spectra, design target profiles,
synthesize waveforms, talk to hardware, or run the loop — those belong to the
caller, which for the intended user is [LDAQ](https://github.com/ladisk/LDAQ).

```
pip install numpy   # the entire dependency list
```

## Why frequency domain, and why no sample rate

The controller's *state* is a PSD, not a waveform. Each iteration throws the
waveform away and re-synthesizes it with fresh random phase; only the shape
carries across. So the natural interface is PSD in, PSD out — and a
time-signal interface would be strictly worse:

1. **It would re-derive what the caller already knows.** The drive PSD is known
   exactly, because you designed the drive from it. Estimating it back out of the
   waveform adds a second dose of estimator noise for nothing.
2. **It would drag in policy.** Window, segment length, overlap, detrending, the
   RNG, the record length, and the peak scaling that respects your amplifier are
   all the caller's decisions. Every one would have to appear in the signature.
3. **Drive and response need not share a sample rate.** A shaker driven at
   10 kHz while the load cell is sampled at 51.2 kHz is completely ordinary. A
   time-domain interface has to resample internally and guess which grid the
   correction should happen on.
4. **The update has no notion of time,** so it should not take a time axis.

Hence: **no `fs`, no frequency vector, no `nperseg` anywhere in the core.** The
three PSDs arrive on one common grid, and the last piece of frequency knowledge —
which lines are in band — defaults to `target_psd > 0`, since a target profile is
zero outside its band by construction. Pass `mask=` if you want it explicit.

`metrics(..., freq=...)` is the single exception, and only so it can report *at
what frequency* the worst line sits.

## The four public names

```python
update_psd(drive_psd, response_psd, target_psd, alpha=0.6, *, mask=None,
           normalize=True, max_step_db=6.0, max_shape_db=30.0,
           floor_frac=1e-2, return_info=False)   # -> ndarray [, UpdateInfo]

error_db(response_psd, target_psd, *, mask=None, normalize=True)   # -> ndarray

metrics(err_db, *, freq=None, mask=None, alarm_db=3.0, abort_db=6.0)  # -> dict

UpdateInfo(step_clamped_fraction, shape_clamped_fraction, floored_fraction)
```

`error_db` and `metrics` are here rather than at the call site because they carry
the *same normalization convention* as `update_psd`. Roll your own error metric
under a different convention and it silently stops measuring the thing the
controller is minimizing — and the stop criterion built on it stops meaning what
it says.

## A complete loop

The two things this package deliberately does not do are one line each at the
call site:

```python
import numpy as np, scipy.signal, pyExSi, pyExCo

freq = np.fft.rfftfreq(nperseg, 1 / fs_in)
target = my_target_profile(freq)          # zero outside the control band
drive = (target > 0).astype(float)        # flat seed, in band only

for i in range(8):
    sig = pyExSi.random_gaussian(n_out, drive_on_output_grid, fs_out, rg=rng)
    sig *= v_peak_max / np.abs(sig).max()          # amplifier budget: yours
    response = measure(sig)                        # hardware: yours

    _, meas = scipy.signal.welch(response, fs_in, "hann", nperseg,
                                 nperseg // 2, detrend="constant")
    m = pyExCo.metrics(pyExCo.error_db(meas, target), freq=freq)
    print(f"iter {i}: rms {m['rms_dB']:.2f} dB | spread {m['spread_dB']:.2f} dB "
          f"| worst {m['max_dB']:.1f} dB @ {m['f_max']:.0f} Hz")
    if m["spread_dB"] < 1.5:
        break

    drive, info = pyExCo.update_psd(drive, meas, target, alpha=0.6,
                                    return_info=True)
    if info.step_clamped_fraction > 0.3:
        print("  loop is fighting something structural, not converging")
```

## Why it works without a plant model

For a linear time-invariant plant, `S_response = |H|² · S_drive`. Substitute that
into the update at `alpha=1`:

$$S_\text{drive}^\text{new} = S_\text{drive}\cdot\frac{S_\text{target}}{|H|^2 S_\text{drive}} = \frac{S_\text{target}}{|H|^2}$$

The unknown `|H|²` cancels algebraically and you land on the answer in one step.
It is Newton's method in the log domain, and the plant is never measured, stored
or inverted. For `alpha < 1` the in-band log-error is multiplied by `(1 - alpha)`
each iteration — geometric convergence, per line.

**So why is the default 0.6 and not 1.0?** A Welch estimate has a per-line
scatter of roughly `4.343·√(2/dof)` dB, and `alpha = 1` freezes all of it into the
drive. Relaxed steps let it average out instead: the retained noise power settles
at `alpha/(2 - alpha)` of the single-estimate value, so 0.6 keeps about 43 % of
what 1.0 keeps.

| alpha | error decay per iteration | retained estimator noise |
|------:|--------------------------:|-------------------------:|
| 1.0   | 0.0 | 100 % |
| 0.8   | 0.2 | 67 % |
| **0.6** | **0.4** | **43 %** |
| 0.4   | 0.6 | 25 % |

## Conventions and guards

**Power dB throughout** (`10·log10`). A PSD is a power quantity, so
`max_step_db=6.0` is a factor of `10**0.6 = 3.98`, not 4. Confusing this with the
amplitude convention (`20·log10`) is the classic way to get a controller that
moves half as far as you think.

**Shape versus level.** `normalize=True` (default) divides both spectra by their
own in-band median before comparing, so the loop controls shape only and is blind
to overall level — the level is whatever your amplifier gain and drive scaling
make it. `normalize=False` drives the response to the target in real engineering
units (g²/Hz, N²/Hz). In that mode an unreachable target will wind the drive up
without limit; enforcing a voltage ceiling and an abort is the caller's job.

Four guards, each against a specific failure:

| guard | prevents |
|---|---|
| in-band median normalization | the loop fighting the amplifier gain |
| `floor_frac` (default 1e-2) | an empty bin demanding an infinite correction |
| `max_step_db` (default 6) | one bad estimate throwing the drive somewhere the next measurement cannot recover from |
| `max_shape_db` (default 30) | a line that never responds running the drive away over many iterations |

**Clamp ordering is deliberate.** `max_step_db` clamps the ratio *before* the
`alpha` exponent, so the largest change the drive can make in one iteration is
`alpha · max_step_db` — 3.6 dB at the defaults, **not** 6 dB. This keeps the
clamp a statement about the *measurement* ("no single estimate may be trusted for
more than 6 dB") rather than about the step, and it means changing `alpha`
rescales the step limit with it.

## Two tricks worth knowing

**Pass the drive PSD you actually played.** One finite realization of a random
signal has its own χ² scatter about the PSD it was designed from. Pass the
*designed* PSD and that scatter is blamed on the plant and enters the correction.
Pass a Welch estimate of the waveform that was *actually played*, over the same
record, and the scatter appears in both spectra and cancels — the update becomes
an H1 estimate of `1/|H|²`. It costs one extra Welch call and no extra hardware,
since the played waveform is the array you sent to the DAC. Because `drive_psd`
is an argument rather than internal state, the choice is yours.

**Keep the best iterate, not the last.** The loop is stochastic; iteration 8 is
not necessarily better than iteration 6. Track `rms_dB` and keep the drive that
produced the lowest one.

## Assumptions, and where they break

| assumption | breaks when |
|---|---|
| the plant is linear and time-invariant | the structure is driven nonlinearly, or heats/loosens during the run |
| the response is stationary | the fixture rattles, or a joint slips mid-record |
| magnitude only, phase ignored | never, for a single shaker — this is a SISO controller and phase is not controllable |
| the estimator resolves the target's features | a lightly-damped anti-resonance is narrower than your line spacing; the loop then reports "converged" over a dip it cannot see |

SISO only. Several shakers with a cross-spectral matrix is a genuinely different
problem and a different package.

## Tests

```
pip install -e ".[dev]"
pytest
```

28 tests, no hardware, no scipy. They pin the properties the docs claim: exact
plant cancellation at `alpha=1`, geometric `(1-alpha)` decay of the error spread,
the clamp-before-exponent ordering, boundedness against a line that never
responds, level-blindness under `normalize=True`, and absolute-level convergence
under `normalize=False`.
