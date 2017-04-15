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

x = np.array([2, 3, 4, 3, 2])
M = len(x)
print M
plt.subplot(411)
plt.cla()
plt.plot(x)
dftbuffer = np.zeros(M)
midlo = np.floor(M / 2.0)
midhi = np.floor((M + 1) / 2.0)
dftbuffer[:midhi] = x[midlo:]
dftbuffer[-midlo:] = x[:midlo]
plt.subplot(412)
plt.cla()
plt.plot(dftbuffer)
X = fft(dftbuffer)
plt.subplot(413)
plt.cla()
plt.plot(X)
plt.subplot(414)
plt.cla()
N = len(X)
X_1 = abs(X[1:N/2.0 +1])
X_2 = abs(X[-N/2.0:][::-1])
XT = X_1 - X_2
print XT
