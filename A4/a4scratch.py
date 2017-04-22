import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.signal import get_window
from scipy.signal import argrelmin
import matplotlib.pyplot as plt
import sys
sys.path.append('../../software/models')
import stft
import utilFunctions as UF

EPS = np.finfo(float).eps
FIG_X = 12
FIG_Y = 8


def pltsac(figure, subplot, title=None):
    '''Plot Select and Clear'''
    plt.figure(figure)
    splt = plt.subplot(subplot)
    splt.cla()
    if not title:
        title = 'Figure [{}], Subplot [{}]'.format(figure, subplot)
    splt.set_title(title)
    return splt


def setup_plotting():
    # setup plotting
    subplots = range(311, 314)
    figures = [1, 2]
    for figure in figures:
        plt.figure(figure, figsize=(FIG_X, FIG_Y))
        plt.title('Scratch Plot [{}]'.format(figure))
        plt.show()
        for subplot in subplots:
            pltsac(figure, subplot)
    plt.figure(1)  # select figure 1


setup_plotting()

def assignment_4_part_1_scratch():
    # Assignment 4, Part 1 Scratch
    M = 100
    window = 'blackmanharris'
    N = 8 * M
    w = get_window(window, M)
    plot = pltsac(1, 311, 'Blackman Harris Window')
    plot.plot(w)
    # Populate FFT buffer (must be even due to 8)
    fftbuffer = np.zeros(N)
    hN = N/2
    hM = M/2
    fftbuffer[hN-hM:hN+hM] = w
    # Take FFT
    X = fftshift(fft(fftbuffer))  # FFT and center
    X[X == 0.] = EPS  # avoid log warnings
    mX = 20 * np.log10(abs(X))
    plot = pltsac(1, 312, 'Blackman Harris mX')
    plot.plot(mX)
    # Main lobe center, then find closest local minima
    mlc = np.argmax(mX)
    localmin = argrelmin(mX)[0]
    left_mllm = localmin[localmin < 400][-1]
    rite_mllm = localmin[localmin > 400][0]
    # Extract the main lobe (include right min so +1)
    main_lobe = mX[left_mllm:rite_mllm+1]
    plot = pltsac(1, 313, 'Blackman Harris mX Main Lobe')
    plot.plot(main_lobe)

# ASsignment 4, Part 2 scratch
inputfile = '../../sounds/piano.wav'
inputwav = UF.wavread(inputfile)
x = inputwav[1]
window = 'blackman'
M = 513
w = get_window(window, M)
N = 2048
H = 128
y = stft.stft(x, w, N, H)
