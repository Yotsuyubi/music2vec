# from https://github.com/liangliannie/hht-spectrum/blob/master/source/hht.py

import numpy as np
import math
from PyEMD import EEMD



def hilb(s, unwrap=False):
    """
    Performs Hilbert transformation on signal s.
    Returns amplitude and phase of signal.
    Depending on unwrap value phase can be either
    in range [-pi, pi) (unwrap=False) or
    continuous (unwrap=True).
    """
    from scipy.signal import hilbert
    H = hilbert(s)
    amp = np.abs(H)
    phase = np.arctan2(H.imag, H.real)
    if unwrap: phase = np.unwrap(phase)

    return amp, phase


def FAhilbert(imfs, dt):
    """
    Performs Hilbert transformation on imfs.
    Returns frequency and amplitude of signal.
    """
    n_imfs = imfs.shape[0]
    f = []
    a = []
    for i in range(n_imfs - 1):
        # upper, lower = pyhht.utils.get_envelops(imfs[i, :])
        inst_imf = imfs[i, :]  # /upper
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        inst_freq = (2 * math.pi) / np.diff(phase)  #

        inst_freq = np.insert(inst_freq, len(inst_freq), inst_freq[-1])
        inst_amp = np.insert(inst_amp, len(inst_amp), inst_amp[-1])

        f.append(inst_freq)
        a.append(inst_amp)
    return np.asarray(f).T, np.asarray(a).T


def hht(data, freqsol=33, timesol=50):
    """
    hht function for the Hilbert Huang Transform spectrum

    Parameters
    ----------
    data : array-like, shape (n_samples,)
       The input signal.
    time : array-like, shape (n_samples), optional
       Time instants of the signal samples.
       (defaults to `np.arange(1, len(signal))`)
    -------
    `matplotlib.figure.Figure`
       The figure (new or existing) in which the hht spectrum is plotted.

    example:
    --------------------

    .. sourcecode:: ipython
        f = Dataset('./source/obs.nc')
        # read one example data
        fsh = f.variables['FSH']
        time = f.variables['time']
        one_site = np.ma.masked_invalid(fsh[0,:])
        time = time[~one_site.mask]
        data = one_site.compressed()
        hht(data, time)


    ----------------
        """
    #   freqsol give frequency - axis resolution for hilbert - spectrum
    #   timesol give time - axis resolution for hilbert - spectrum
    time = np.arange(1, len(data))
    t0 = time[0]
    t1 = time[-1]
    dt = (t1 - t0) / (len(time) - 1)

    eemd = EEMD(max_imfs=5)
    imfs = eemd.eemd(data)
    freq, amp = FAhilbert(imfs, dt)

    #     fw0 = np.min(np.min(freq)) # maximum frequency
    #     fw1 = np.max(np.max(freq)) # maximum frequency

    #     if fw0 <= 0:
    #         fw0 = np.min(np.min(freq[freq > 0])) # only consider positive frequency

    #     fw = fw1-fw0
    tw = t1 - t0

    bins = np.linspace(0, 12, freqsol)  # np.logspace(0, 10, freqsol, base=2.0)
    p = np.digitize(freq, 2 ** bins)
    t = np.ceil((timesol - 1) * (time - t0) / tw)
    t = t.astype(int)

    hilbert_spectrum = np.zeros([timesol, freqsol])
    for i in range(len(time)):
        for j in range(imfs.shape[0] - 1):
            if p[i, j] >= 0 and p[i, j] < freqsol:
                hilbert_spectrum[t[i], p[i, j]] += amp[i, j]

    hilbert_spectrum = abs(hilbert_spectrum)
    return hilbert_spectrum