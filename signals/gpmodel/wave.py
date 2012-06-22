import numpy as np
import matplotlib.pyplot as plt

N = 20
a = np.random.rand(N)-0.5

M = 1000
x = np.zeros(M,dtype=complex)
xpos = np.linspace(0,60,M)
for n in range(M):
    x[n] = np.sum([a[k]*np.exp(1j*k*(2*np.pi/N)*xpos[n]) for k in range(N)])
plt.plot(np.abs(x))
plt.show()

aa = np.zeros(N,dtype=complex)
for k in range(N):
    aa[k] = (1.0/N)*np.sum([x[n]*np.exp(-1j*k*(2*np.pi/N)*n) for n in range(N)])

print a
print aa