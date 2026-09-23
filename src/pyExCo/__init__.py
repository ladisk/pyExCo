"""pyExCo -- excitation control for closed-loop random vibration testing.

The control law and nothing else. Given the drive PSD you played, the response
PSD you measured, and the response PSD you wanted, :func:`update_psd` returns the
drive PSD to play next::

    S_drive <- S_drive * clip(S_target / S_response) ** alpha

There is no sampling rate, frequency vector or window length in any signature:
the three PSDs arrive on one common grid and the update is per-line arithmetic.
Estimating those PSDs, designing the target, synthesizing a waveform from the
result, driving the hardware and running the loop are all the caller's job --
which for this package's intended user is LDAQ.

The companion package for the pieces this one deliberately does not do:
``pyExSi`` realizes a PSD as a random-phase Gaussian time signal, and
``scipy.signal.welch`` estimates one back from a measurement.

Example
-------
::

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
        print(f"iter {i}: spread {m['spread_dB']:.2f} dB")
        if m["spread_dB"] < 1.5:
            break
        drive = pyExCo.update_psd(drive, meas, target, alpha=0.6)
"""

from ._core import UpdateInfo, error_db, metrics, update_psd

__version__ = "0.1.0"

__all__ = ["update_psd", "error_db", "metrics", "UpdateInfo", "__version__"]
