import numpy as np
from scipy.signal import detrend
#import nitime.algorithms as tsa
import mne
#from scipy.fft import rfftfreq
from mne.time_frequency.multitaper import _compute_mt_params, _mt_spectra, _psd_from_mt
from mne.parallel import parallel_func
from mne.utils import verbose as verbose_decorator


@verbose_decorator
def psd_array_multitaper2(x, sfreq, fmin=0, fmax=np.inf, bandwidth=None,
                         adaptive=False, low_bias=True, normalization='length',
                         n_jobs=1, verbose=None, NFFT=None):
    # Reshape data so its 2-D for parallelization
    ndim_in = x.ndim
    x = np.atleast_2d(x)
    n_times = x.shape[-1]
    dshape = x.shape[:-1]
    x = x.reshape(-1, n_times)

    dpss, eigvals, adaptive = _compute_mt_params(
        n_times, sfreq, bandwidth, low_bias, adaptive)

    # decide which frequencies to keep
    #freqs = rfftfreq(n_times, 1. / sfreq)
    freqs = np.linspace(0, sfreq / 2, NFFT //  2 + 1)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    psd = np.zeros((x.shape[0], freq_mask.sum()))
    # Let's go in up to 50 MB chunks of signals to save memory
    n_chunk = max(50000000 // (len(freq_mask) * len(eigvals) * 16), n_jobs)
    offsets = np.concatenate((np.arange(0, x.shape[0], n_chunk), [x.shape[0]]))
    for start, stop in zip(offsets[:-1], offsets[1:]):
        x_mt = _mt_spectra(x[start:stop], dpss, sfreq, n_fft=NFFT)[0]
        if not adaptive:
            weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
            psd[start:stop] = _psd_from_mt(x_mt[:, :, freq_mask], weights)
        else:
            n_splits = min(stop - start, n_jobs)
            parallel, my_psd_from_mt_adaptive, n_jobs = \
                parallel_func(_psd_from_mt_adaptive, n_splits)
            out = parallel(my_psd_from_mt_adaptive(x, eigvals, freq_mask)
                           for x in np.array_split(x_mt, n_splits))
            psd[start:stop] = np.concatenate(out)

    if normalization == 'full':
        psd /= sfreq

    # Combining/reshaping to original data shape
    psd.shape = dshape + (-1,)
    if ndim_in == 1:
        psd = psd[0]
    return psd, freqs


def multitaper_spectrogram(EEG, Fs, NW, window_length, window_step, EEG_segs=None):#, dpss=None, eigvals=None):
    """Compute spectrogram using multitaper estimation.

    Arguments:
    EEG -- EEG signal, size=(channel_num, sample_point_num)
    Fs -- sampling frequency in Hz
    NW -- the time-halfbandwidth product
    window_length -- length of windows in seconds
    window_step -- step of windows in seconds

    Outputs:
    psd estimation, size=(window_num, freq_point_num, channel_num)
    frequencies, size=(freq_point_num,)
    """

    #window_length = int(round(window_length*Fs))
    #window_step = int(round(window_step*Fs))

    nfft = max(1<<(window_length-1).bit_length(), window_length)

    #freqs = np.arange(0, Fs, Fs*1.0/nfft)[:nfft//2+1]
    if EEG_segs is None:
        window_starts = np.arange(0,EEG.shape[1]-window_length+1,window_step)
        #window_num = len(window_starts)
        EEG_segs = detrend(EEG[:,list(map(lambda x:np.arange(x,x+window_length), window_starts))], axis=2)
    #freq, pxx, _ = tsa.multi_taper_psd(EEG_segs, Fs=Fs, NW=NW, adaptive=False, jackknife=False, low_bias=True, NFFT=nfft)#, dpss=dpss, eigvals=eigvals)
    pxx, freq = psd_array_multitaper2(#mne.time_frequency.psd_array_multitaper
                EEG_segs, Fs, bandwidth=NW*2./(window_length/Fs),
                normalization='full', verbose=False, NFFT=nfft)

    return pxx.transpose(1,2,0), freq
    
