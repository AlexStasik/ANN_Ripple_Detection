import os
import numpy as np
from scipy import stats
import quantities as pq
import matplotlib.pyplot as plt


def generate_time_array(t_start, t_stop, sf):
    '''
    This function generates a time sequence from t_start to t_stop with
    sampling frequence sf. All input will be casted to sec and Hz
    '''
    t_start = t_start.rescale(pq.s)
    t_stop = t_stop.rescale(pq.s)
    sf = sf.rescale(pq.Hz)
    t = np.arange(t_start, t_stop, 1/sf)
    return t*pq.s


def generate_ripple(t, center, A_0=1, scale=0.1*pq.s, omega=30*pq.Hz):
    '''
    Generates a Gaussian shaped ripple at loc center with dimensionless
    magnitude A_0 and width scale at frequency omega
    '''
    t = (t.rescale(pq.s)).magnitude
    scale = (scale.rescale(pq.s)).magnitude
    omega = (omega.rescale(pq.Hz)).magnitude

    A = A_0 * stats.norm.pdf(t, loc=center, scale=scale)
    phase = np.exp(2.*np.pi * omega * 1j*t)

    flag = np.zeros_like(A)
    flag[A>np.max(A)/1.5] = 1

    return np.real(A * phase), flag


def generate_single_fake_data(sf=1.e2*pq.Hz, t_start=0.*pq.s, t_stop=1000.*pq.s,
                       noise_level=1., n_spikes=1000):
    '''
    Generates a white noise stream and adds n_spikes non-overlapping ripples
    '''
    t = generate_time_array(t_start, t_stop, sf)
    x = np.zeros((t.shape[0]))
    y = np.zeros((t.shape[0]))

    x += noise_level * np.random.randn(t.shape[0])
    for n in range(n_spikes):
        print(n)
        while True:
            loc = np.random.uniform(t_start, t_stop, 1)
            ripple, truth = generate_ripple(t, loc)
            if np.max(y+truth)==1:
                x += ripple
                y += truth
                break
            
    y = y.astype(int)

    return t, x, y


def generate_fake_data(N, save_dir='../data/fake_data'):
    '''
    Generates a matrix of N time series and the according ground truth
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

#sorry for this hack
#     t, _, _ = generate_single_fake_data()

#     X = np.zeros((N, t.shape[0]))
#     Y = np.zeros((N, t.shape[0]))

#     for i in range(N):
    t,X,Y = generate_single_fake_data()
#     X[i] = x
#     Y[i] = y

    np.save(os.path.join(save_dir, 't.npy'), t)
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'Y.npy'), Y)


def example_plot():
    t, x, y = generate_single_fake_data()

    plt.figure(figsize=(15, 5))
    plt.plot(t, x, )
    plt.plot(t, y, lw=5)
    plt.show()


if __name__ == "__main__":
    generate_fake_data(1)
#    example_plot()
