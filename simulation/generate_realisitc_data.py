import numpy as np
import matplotlib.pyplot as plt
from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ripple_detection.simulate import simulate_time
import scipy.stats


SAMPLING_FREQUENCY = 1000
n_samples = SAMPLING_FREQUENCY * 500
time = simulate_time(n_samples, SAMPLING_FREQUENCY)
white_noise = np.random.normal(size=time.shape)
RIPPLE_FREQUENCY = 200
ripple_signal = np.sin(2 * np.pi * time * RIPPLE_FREQUENCY)
carrier = scipy.stats.norm(loc=1.1, scale=0.025).pdf(time)
carrier /= carrier.max()

ripple_loc = np.random.uniform(np.min(time), np.max(time), 1000)


from ripple_detection.simulate import simulate_LFP

LFPs = simulate_LFP(time, ripple_loc, noise_amplitude=3., ripple_amplitude=1.5)[:, np.newaxis]
speed = np.ones_like(time)


Kay_ripple_times = Kay_ripple_detector(time, LFPs, speed, SAMPLING_FREQUENCY)


y = np.zeros_like(speed)
for ripple in Kay_ripple_times.itertuples():
    mask = np.logical_and(ripple.start_time<=time,
                         ripple.end_time>=time)
    y[mask] = 1


save_dir='../data/fake_data_realistic'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


np.save(os.path.join(save_dir, 't.npy'), time)
np.save(os.path.join(save_dir, 'X.npy'), LFPs)
np.save(os.path.join(save_dir, 'Y.npy'), y)
