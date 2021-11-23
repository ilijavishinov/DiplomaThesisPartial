import numpy as np
from pywt import wavedec
import numpy as np


def histogram_stds(ordered_iterable, bin_width):
    more_bins_to_create = True
    current_start_position = 0
    bin_stds = list()
    while more_bins_to_create:
        bin_stds.append(
            np.abs(ordered_iterable[current_start_position:current_start_position+bin_width]).std()
        )
        current_start_position += bin_width
        if current_start_position + bin_width > len(ordered_iterable):
            bin_stds.append(sum(ordered_iterable[current_start_position:]))
            more_bins_to_create = False

    return bin_stds


def histogram_sums(ordered_iterable, bin_width):
    more_bins_to_create = True
    current_start_position = 0
    bin_sums = list()
    while more_bins_to_create:
        bin_sums.append(
            sum(ordered_iterable[current_start_position:current_start_position+bin_width])
        )
        current_start_position += bin_width
        if current_start_position + bin_width > len(ordered_iterable):
            bin_sums.append(sum(ordered_iterable[current_start_position:]))
            more_bins_to_create = False

    return bin_sums


def wavelet_transform(signal_array_, levels_, bin_width_):
    features = list()
    decomp = wavedec(signal_array_, 'db2', level = levels_)
    for level in range(levels_+1):
        signal = decomp[level]
        level_bin_sums = histogram_stds(signal, bin_width_)
        features.extend(level_bin_sums)
    return features


def fft_transform(signal_array_, zero_padding_multiple_, num_coefs_to_save_):
    return (np.abs(
                np.fft.fft(
                    signal_array_,
                    n = signal_array_.shape[0] * zero_padding_multiple_
                )) / signal_array_.shape[0]
            )[1:num_coefs_to_save_]