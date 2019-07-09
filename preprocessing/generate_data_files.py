import scipy.io as sio
import numpy as np
import scipy
import os
import warnings

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


def generate_data_set_for_animal(data, animal, sf=2.5e3, q=1):
    lfp, speed, ripple_index = read_out_arrays(data[animal])

    time = simulate_time(lfp.shape[0], sf)
    ripple_times = time[ripple_index]

    # lfp = scipy.signal.decimate(lfp.flatten(), q)
    lfp = lfp.flatten()

    def perform_high_pass_filter(lfp, low_cut_frequency, high_cut_frequency, sf):
            wn = sf / 2.
            b, a = signal.butter(5, [low_cut_frequency/wn, high_cut_frequency/wn], 'bandpass')
            lfp = signal.filtfilt(b, a, lfp)
            return lfp

    lfp = perform_high_pass_filter(lfp, 1, 500, sf)

    lfp = lfp[:, np.newaxis]
    speed = scipy.signal.decimate(speed.flatten(), q)
    time = simulate_time(lfp.shape[0], sf/q)

    ripple_time_index_sparse = list()
    for t in ripple_times:
        ripple_time_index_sparse.append(np.argmin(np.abs(t-time)))


    Kay_ripple_times = Kay_ripple_detector(time, lfp, speed.flatten(), sf/q)
    label = np.zeros_like(time)
    label_all = np.zeros_like(time)
    ripple_index = list()
    ripple_index_all = list()



    true_array = np.zeros_like(time)
    true_array[np.array(ripple_time_index_sparse)] = 1
    for i in range(Kay_ripple_times.shape[0]):
        start_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,0]))
        end_index = int(np.argwhere(time==np.array(Kay_ripple_times)[i,1]))
        if (speed[start_index:end_index]).sum()<1:
            label_all[start_index:end_index] = 1
            ripple_index_all.append([start_index, end_index])
            if (true_array[start_index:end_index]).sum()>0:
                label[start_index:end_index] = 1
                ripple_index.append([start_index, end_index])

    res = dict()
    res['sf'] = sf/q
    res['X'] = lfp
    res['y'] = label
    res['y2'] = label_all
    res['speed'] = speed
    res['time'] = time
    res['ripple_times'] = ripple_times
    res['ripple_time_index'] = np.array(ripple_time_index_sparse)
    res['ripple_periods'] = np.array(Kay_ripple_times)
    res['ripple_index'] = np.array(ripple_index)
    res['true_array'] = np.array(true_array)

    return res

def generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat'):
    data = sio.loadmat(data_path)

    for key in data.keys():
        if key.startswith("m"):
            print('Generating', str(key))
            directory = os.path.join('..', 'data', 'processed_data', key)
            if not os.path.exists(directory):
                os.makedirs(directory)
            res = generate_data_set_for_animal(data, key, )
            np.save(os.path.join(directory, 'all.npy'), res)
            np.save(os.path.join(directory, 'X.npy'), res['X'])
            np.save(os.path.join(directory, 'y.npy'), res['y'])
            np.save(os.path.join(directory, 'y2.npy'), res['y2'])
            np.save(os.path.join(directory, 'true_array.npy'), res['true_array'])
            np.save(os.path.join(directory, 'speed.npy'), res['speed'])
            np.save(os.path.join(directory, 'time.npy'), res['time'])
            np.save(os.path.join(directory, 'ripple_periods.npy'), res['ripple_periods'])
            np.save(os.path.join(directory, 'ripple_times.npy'), res['ripple_times'])

    return res


if __name__ == '__main__':
    res = generate_all_outputs(data_path='../data/m4000series_LFP_ripple.mat')
