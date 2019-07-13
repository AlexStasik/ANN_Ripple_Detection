import scipy.io as sio
import numpy as np
import scipy
import os
import warnings
import pandas as pd
import pickle

from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector
from ripple_detection.simulate import simulate_time
from scipy import signal

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_out_arrays(data):
    lfp = data['lfp'][0][0]
    run_speed = data['run_speed'][0][0]
    ripple_loc = data['rippleLocs'][0][0].flatten()
    min_val = min(lfp.shape[0], run_speed.shape[0])
    return lfp[:min_val,:], run_speed[:min_val,:], ripple_loc


def generate_data_set_for_animal(data, animal, sf=2.5e3, q=1, length=150):
    lfp, speed, ripple_index = read_out_arrays(data[animal])

    time = simulate_time(lfp.shape[0], sf)
    ripple_times = time[ripple_index]
    lfp = lfp.flatten()

    def perform_high_pass_filter(lfp, low_cut_frequency, high_cut_frequency, sf):
            wn = sf / 2.
            b, a = signal.butter(5, [low_cut_frequency/wn, high_cut_frequency/wn], 'bandpass')
            lfp = signal.filtfilt(b, a, lfp)
            return lfp

    # lfp = perform_high_pass_filter(lfp, 1, 2500, sf)
    lfp = scipy.signal.decimate(lfp.flatten(), q)

    lfp = lfp[:, np.newaxis]
    speed = scipy.signal.decimate(speed.flatten(), q)
    time = simulate_time(lfp.shape[0], sf/q)

    ripple_time_index_sparse = list()
    for t in ripple_times:
        ripple_time_index_sparse.append(np.argmin(np.abs(t-time)))


    ripples = Kay_ripple_detector(time, lfp, speed.flatten(), sf/q)

    ripples['duration'] = ripples.loc[:,'end_time'] - ripples.loc[:,'start_time']
    ripples['center'] = (ripples.loc[:,'end_time'] + ripples.loc[:,'start_time'])/2.
    ripples['start_index'] = ripples.apply(lambda row: int(np.argwhere(time==(row['start_time']))), axis=1)
    ripples['end_index'] = ripples.apply(lambda row: int(np.argwhere(time==(row['end_time']))), axis=1)
    ripples['center_index'] = ripples.apply(lambda row: int((row['end_index']+row['start_index'])/2), axis=1)
    ripples['duration_index'] = ripples.apply(lambda row: row['end_index']-row['start_index'], axis=1)

    ripples['speed'] = ripples.apply(lambda row: speed[int(row['center_index']-length):int(row['center_index']+length)], axis=1)
    ripples['lfp'] = ripples.apply(lambda row: lfp[int(row['center_index']-length):int(row['center_index']+length),:], axis=1)
    ripples['time'] = ripples.apply(lambda row: time[int(row['center_index']-length):int(row['center_index']+length)], axis=1)

    # ripples['speed'] = ripples.apply(lambda row: speed[int(row['start_index']):int(row['end_index'])], axis=1)
    # ripples['lfp'] = ripples.apply(lambda row: lfp[int(row['start_index']):int(row['end_index']),:], axis=1)
    # ripples['time'] = ripples.apply(lambda row: time[int(row['start_index']):int(row['end_index'])], axis=1)

    end_time = np.array(ripples.end_time)
    start_time = np.array(ripples.start_time)
    assert end_time.shape==start_time.shape
    n_ripples = end_time.shape[0]
    label = np.zeros(n_ripples)
    for i in range(n_ripples):
        for j in range(ripple_times.shape[0]):
            if np.logical_and(start_time[i]<=ripple_times[j], ripple_times[j]<=end_time[i]):
                label[i] = 1
    ripples['labels'] = label

    return ripples


def generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat'):
    data = sio.loadmat(data_path)

    for key in data.keys():
        if key.startswith("m"):
            print('Generating', str(key))
            directory = os.path.join('..', 'data', 'processed_data', key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            res = generate_data_set_for_animal(data, key, )
            res.to_csv(os.path.join(directory, 'all_data.csv'))
            res.to_pickle(os.path.join(directory, 'all_data.pkl'))

    return res


if __name__ == '__main__':
    res = generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat')
