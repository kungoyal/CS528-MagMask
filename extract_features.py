import scipy.signal as sp
import statistics
import numpy as np
import pandas as pd

ZERO_CROSSING_THRESHOLD = 1
WINDOW_SIZE = 20
STRIDE = 10
SAMPLE_RATE = 20


def get_feature_vectors(raw_data):
    data = fix_sr(raw_data)
    fv_arr = []
    for win in range(0, len(data)-WINDOW_SIZE, STRIDE):
        fv = get_window_fv(data[win: win+WINDOW_SIZE])
        fv_arr.append(fv)
    print(f"num windows  = {len(fv_arr)}")
    return np.array(fv_arr)

def fix_sr(raw_data):
    # TODO: use interpolation to ensure sample rate is correct
    return raw_data


def get_window_fv(window):
    fv_funcs = [
        get_mean,
        get_median,
        get_stdev,
        get_peak1_count,
        get_mean_peak1_distance,
        get_mean_peak1_height,
        get_num_zero_corssings
    ]
    x, y, z, t = window.T
    mag = (x**2+y**2+z**2)**0.5

    final_fv = []
    for i in (x, y, z, mag):
        final_fv += [f(i) for f in fv_funcs]
    return final_fv


def get_mean(window):
    return np.mean(window)

def get_median(window):
    return np.median(window)

def get_stdev(window):
    return statistics.stdev(window)

def get_peak1_count(window):
    return len(sp.find_peaks(window)[0])

def get_mean_peak1_distance(window):
    peaks, _ = sp.find_peaks(window)
    diff = peaks[1:] - peaks[-1:]
    return np.mean(diff)

def get_mean_peak1_height(window):
    peaks, _ = sp.find_peaks(window)
    peak_heights = window[peaks]
    return np.mean(peak_heights)

def get_num_zero_corssings(window):
    t = ZERO_CROSSING_THRESHOLD
    zc = (window[-1:] < t) and (window[1:] > t)
    return len(zc)

def get_fv_csv(csv_string):

    lines = csv_string.split("\n")

    data = np.array([l.split(',') for l in lines])
    fv_arr = []
    return_csv_str = ""
    for win_idx in range(0, len(data) - WINDOW_SIZE, STRIDE):
        fv = get_window_fv(data[win_idx:win_idx + WINDOW_SIZE])
        return_csv_str += ','.join(['%.5f' % num for num in fv])
        return_csv_str += "\n"

    return return_csv_str

# for manual  testing
if __name__ == "__main__":
    csv_string = None
    get_fv_csv(csv_string)


