#QUESTION 2
import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of sinusoids to sum
center_freq = 1.5e9 # RF carrier frequency in Hz
Fs = 50 # sample rate of simulation


t = np.arange(0, 1, 1/Fs) 

recv_list =[]
tx_signal = np.zeros(len(t))

for j in range(len(t)):
    for i in range(N):
    

        y = np.sin(2*np.pi*center_freq*t)
        h = 1/np.sqrt(2)*(np.random.randn(N, len(y)) + 1j*np.random.randn(N,len(y)))
        tx_signal = y*h
        
        noise_ryl = 1/np.sqrt(2)*(np.random.randn(N, len(y)) + 1j*np.random.randn(N,len(y)))
        # noise_ryl = 1/np.sqrt(2)*(np.random.normal(0,1) + 1j*np.random.normal(0,1))
        recv_ryl = tx_signal + noise_ryl
        # recv_ryl_ab = np.abs(recv_ryl)
        # recv_ryl_dB = 10*np.log10(recv_ryl_ab)
        # recv_ryl_dB = 10*np.log10(recv_ryl)
        noise_pr = np.var(noise_ryl)
        recv_pr = np.var(recv_ryl)
        recv_ryl_dB = 10*np.log10(recv_pr/noise_pr)
    recv_list.append(recv_ryl_dB)

# Plot fading 
plt.plot(t,recv_list)
# plt.plot(t,recv_ryl_dB)
plt.legend(['Rayleigh Fading'])
plt.xlim(0,1)
plt.xlabel("Time")
plt.ylabel("SNR of received signal")
plt.show()