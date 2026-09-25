"""
A basic use case of pyExCo: the control loop against a simulated plant.

No hardware, no scipy, no signal synthesis. The plant is a lightly damped
single-degree-of-freedom system applied as a per-line power gain,
``S_response = |H|**2 * S_drive``, and the Welch estimator is imitated by
multiplying the true response PSD by chi-square scatter with ``dof`` degrees of
freedom. In a real test those two lines are the hardware and
``scipy.signal.welch``; everything else is what the caller would write.

Run it from the project base directory with::

    python -m examples.basic_example
"""

import numpy as np

import pyExCo

rng = np.random.default_rng(0)

fs, nperseg, dof = 5120.0, 1024, 60
freq = np.fft.rfftfreq(nperseg, 1 / fs)

# Target: flat 1e-3 g**2/Hz from 20 Hz to 1 kHz, zero outside the control band.
target = np.where((freq >= 20.0) & (freq <= 1000.0), 1e-3, 0.0)

# Plant: SDOF, 300 Hz, 2 % damping -- a ~28 dB resonance the drive must notch.
fn, zeta = 300.0, 0.02
r = freq / fn
H2 = 1.0 / ((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2)

drive = (target > 0).astype(float)          # flat seed, in band only
best = (np.inf, drive)

for i in range(12):
    # "Measure": the true response PSD with a Welch-like chi-square scatter.
    meas = H2 * drive * rng.chisquare(dof, freq.size) / dof

    m = pyExCo.metrics(pyExCo.error_db(meas, target), freq=freq)
    print(f"iter {i:2d}: rms {m['rms_dB']:5.2f} dB | spread {m['spread_dB']:5.2f} dB "
          f"| worst {m['max_dB']:5.1f} dB @ {m['f_max']:6.0f} Hz")
    if m["rms_dB"] < best[0]:
        best = (m["rms_dB"], drive)          # keep the best iterate, not the last
    if m["spread_dB"] < 1.5:
        break

    drive, info = pyExCo.update_psd(drive, meas, target, alpha=0.6,
                                    return_info=True)
    if info.step_clamped_fraction > 0.3:
        print("  loop is fighting something structural, not converging")

print(f"best in-band rms error: {best[0]:.2f} dB")
