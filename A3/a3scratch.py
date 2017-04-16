import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

fs = 1000
f = 100
M = 25
x = np.cos(2 * np.pi * f * np.arange(0.0, 1.0, 1.0 / fs))[:M]
plt.figure(1, figsize=(12, 8))
plt.show()
plt.subplot(411)
plt.cla()
plt.plot(x)
spp = fs / f
periods = M / float(spp)
reqperiods = np.ceil(periods)
N = int(reqperiods * spp)
window = np.zeros(N)
midlo = np.floor(M / 2.0)
midhi = np.floor((M + 1) / 2.0)
window[:midhi] = x[midlo:]
window[-midlo:] = x[:midlo]
plt.subplot(412)
plt.cla()
plt.plot(window)
X = fft(window)
plt.subplot(413)
plt.cla()
plt.plot(X)
mX = 20.0 * np.log10(np.abs(X[:(N / 2) + 1]))
plt.subplot(414)
plt.cla()
plt.plot(mX)

import sys
sys.path.append('../../software/models/')
from dftModel import dftAnal, dftSynth
from scipy.signal import get_window

fs = 5000
x40 = np.cos(2 * np.pi * 40 * np.arange(0.0, 1.0, 1.0 / fs))
x100 = np.cos(2 * np.pi * 100 * np.arange(0.0, 1.0, 1.0 / fs))
x200 = np.cos(2 * np.pi * 200 * np.arange(0.0, 1.0, 1.0 / fs))
x1000 = np.cos(2 * np.pi * 1000 * np.arange(0.0, 1.0, 1.0 / fs))
plt.subplot(411)
plt.cla()
plt.plot(x40)
plt.subplot(412)
plt.cla()
plt.plot(x100)
plt.subplot(413)
plt.cla()
plt.plot(x200)
plt.subplot(414)
plt.cla()
plt.plot(x1000)

x = x40 + x100 + x200 + x1000
x = x[:1001]
plt.subplot(411)
plt.cla()
plt.plot(x)
N = 1024
M = len(x)
w = get_window('hamming', M)
outputScaleFactor = sum(w)
plt.subplot(412)
plt.cla()
plt.plot(w)
mX, pX = dftAnal(x, w, N)
fk = fs * np.arange(0, len(mX)) / N
plt.subplot(413)
plt.cla()
plt.plot(fk, mX)
cutoffbucket = len(fk[fk < 70.0])
y = dftSynth(mX, pX, w.size) * outputScaleFactor
plt.subplot(414)
plt.cla()
plt.plot(y)
mX2 = mX.copy()
mX2[0:cutoffbucket] = -120.0
yfilt = dftSynth(mX2, pX, w.size) * outputScaleFactor
plt.subplot(411)
plt.cla()
plt.plot(yfilt)
xfilt = x100 + x200 + x1000
xfilt = xfilt[:1001]
plt.subplot(412)
plt.cla()
plt.plot(xfilt)
