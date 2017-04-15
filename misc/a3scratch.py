import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

fs = 1000
f = 100
M = 25
x = np.cos(2 * np.pi * f * np.arange(0.0, 1.0, 1.0 / fs))[:M]
plt.figure(1, figsize=(16, 12))
plt.show()
plt.subplot(411)
plt.cla()
plt.plot(x)
spp = fs / f
periods = M / float(spp)
reqperiods = np.ceil(periods)
N = int(reqperiods * spp)
print N
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

