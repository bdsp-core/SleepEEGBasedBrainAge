import numpy as np


def bandpower(pxx, freqs, band_freq, total_freq_range=None, relative=False, ravel_if_one_band=True):
    """Compute band power from power spectrogram.

    Arguments:
    pxx -- power spectrogram from from function multitaper_spectrogram, size=(window_num, freq_point_num1, channel_num), or a list of them for each band
    freqs -- in Hz, size=(nfft//2+1,), or a list of them for each band
    band_freq -- bands to compute, [[band1_start,band1_end],[band2_start,band2_end],...] in Hz

    Keyword arguments:
    total_freq_range -- default None, total range of frequency in a two-element list in Hz, if None and relative is True, use the maximum range in band_freq
    relative -- default False, whether to compute relative band power w.r.t the total frequency range
    ravel_if_one_band -- default True, whether to only return the first element if one band

    Outputs:
    band power, size=(window_num, channel_num) or a list of them for each band
    indices in freqs for each band
    """
    if not hasattr(band_freq[0],'__iter__'):
        band_freq = [band_freq]
    band_num = len(band_freq)

    if relative and total_freq_range is None:
        total_freq_range = [min(min(bf) for bf in band_freq),max(max(bf) for bf in band_freq)]
    if relative:
        total_findex = np.where(np.logical_and(freqs>=total_freq_range[0], freqs<total_freq_range[1]))[0]

    bp = []
    band_findex = []
    for bi in range(band_num):
        band_findex.append(np.where(np.logical_and(freqs>=band_freq[bi][0], freqs<band_freq[bi][1]))[0])

        if relative:
            bp.append(pxx[:,band_findex[-1],:].sum(axis=1)*1.0/pxx[:,total_findex,:].sum(axis=1))
        else:
            bp.append(pxx[:,band_findex[-1],:].sum(axis=1)*(freqs[1]-freqs[0]))

    if ravel_if_one_band and band_num==1:
        bp = bp[0]
        band_findex = band_findex[0]

    return bp, band_findex
    

